// SMP + time: TSC calibration against the PIT, x2APIC helpers, AP release through Limine MP into
// the asm worker loop (engine/smp.asm), worker parking policy and the BSP idle wait.
use crate::{idt, println};
use core::arch::asm;
use core::ptr::{addr_of, addr_of_mut};
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use limine::mp;
use limine::response::MpResponse;

unsafe extern "C" {
    fn ap_limine_entry(cpu: &mp::Cpu) -> !;
    static mut tp_n_threads: u64;
    static tp_n_online: AtomicU32;
    static tp_n_parked: AtomicU32;
    static mut tp_park_enable: u32;
    static mut tp_park_tsc: u64;
    static tp_wakeups: u64;
}

pub static TSC_HZ: AtomicU64 = AtomicU64::new(0);
static X2APIC: AtomicBool = AtomicBool::new(false);
static DEADLINE: AtomicBool = AtomicBool::new(false);
/// Worker threads available (BSP + online APs).
pub static MAX_THREADS: AtomicU64 = AtomicU64::new(1);

pub fn rdtsc() -> u64 {
    let (lo, hi): (u32, u32);
    unsafe { asm!("lfence; rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack)) };
    (hi as u64) << 32 | lo as u64
}

unsafe fn rdmsr(m: u32) -> u64 {
    let (lo, hi): (u32, u32);
    unsafe { asm!("rdmsr", in("ecx") m, out("eax") lo, out("edx") hi, options(nomem, nostack)) };
    (hi as u64) << 32 | lo as u64
}

unsafe fn wrmsr(m: u32, v: u64) {
    unsafe { asm!("wrmsr", in("ecx") m, in("eax") v as u32, in("edx") (v >> 32) as u32, options(nomem, nostack)) };
}

unsafe fn outb(p: u16, v: u8) {
    unsafe { asm!("out dx, al", in("dx") p, in("al") v, options(nomem, nostack)) };
}

unsafe fn inb(p: u16) -> u8 {
    let v: u8;
    unsafe { asm!("in al, dx", in("dx") p, out("al") v, options(nomem, nostack)) };
    v
}

pub fn lapic_id() -> u32 {
    let r = unsafe { core::arch::x86_64::__cpuid(1) };
    r.ebx >> 24
}

/// Busy-wait `ticks` PIT ticks (1..65535) on channel 2, as drivers.asm pit_wait_ticks.
fn pit_wait_ticks(ticks: u16) {
    unsafe {
        let b = inb(0x61) & 0xFC;
        outb(0x61, b);
        outb(0x61, b | 1); // gate CH2 on, speaker off
        outb(0x43, 0xB0); // CH2, lobyte/hibyte, mode 0
        outb(0x42, ticks as u8);
        outb(0x42, (ticks >> 8) as u8);
        while inb(0x61) & 0x20 == 0 {}
    }
}

/// TSC rate from a 20 ms PIT window (CPUID gives no TSC rate on this host).
pub fn calibrate_tsc() -> u64 {
    pit_wait_ticks(1193); // warm-up, 1 ms
    let t0 = rdtsc();
    pit_wait_ticks(23864); // 20.0003 ms
    let dt = rdtsc() - t0;
    let hz = (dt as u128 * 1_193_182 / 23864) as u64;
    TSC_HZ.store(hz, Relaxed);
    hz
}

pub fn us_to_tsc(us: u64) -> u64 {
    (us as u128 * TSC_HZ.load(Relaxed) as u128 / 1_000_000) as u64
}

pub fn tsc_to_us(t: u64) -> u64 {
    (t as u128 * 1_000_000 / TSC_HZ.load(Relaxed).max(1) as u128) as u64
}

fn enable_lapic() {
    if X2APIC.load(Relaxed) {
        unsafe { wrmsr(0x80F, rdmsr(0x80F) | 0x1FF) }; // SVR: enable, spurious vector 0xFF
    }
}

/// Called by ap_limine_entry on each AP's own stack, before enable_avx_fpu and the worker loop.
#[unsafe(no_mangle)]
extern "C" fn ap_init(_tid: u64) {
    idt::load();
    enable_lapic();
}

#[derive(Clone, Copy, PartialEq)]
pub enum Park {
    /// never park: workers spin forever
    Off,
    /// spin during a generation, park between requests (default)
    Idle,
    /// park after this many microseconds of idle spinning, also inside a generation
    Always(u64),
}

pub static mut PARK: Park = Park::Idle;

fn set_park(enable: bool, us: u64) {
    unsafe {
        addr_of_mut!(tp_park_tsc).write_volatile(us_to_tsc(us));
        addr_of_mut!(tp_park_enable).write_volatile(enable as u32);
    }
}

/// Policy at the start (busy = true) and end (busy = false) of a generation.
pub fn park_phase(busy: bool) {
    match unsafe { PARK } {
        Park::Off => set_park(false, 1000),
        Park::Idle => set_park(!busy, 1000),
        Park::Always(us) => set_park(true, us),
    }
}

pub fn parked() -> u32 {
    unsafe { tp_n_parked.load(Relaxed) }
}

pub fn wakeups() -> u64 {
    unsafe { addr_of!(tp_wakeups).read_volatile() }
}

pub fn threads() -> u64 {
    unsafe { addr_of!(tp_n_threads).read_volatile() }
}

pub fn set_threads(n: u64) {
    let n = n.clamp(1, MAX_THREADS.load(Relaxed));
    unsafe { addr_of_mut!(tp_n_threads).write_volatile(n) };
}

/// Release every AP into the asm worker loop. Returns the number of worker threads (BSP included).
pub fn start(mp: &MpResponse) -> u64 {
    let x2 = mp.flags().contains(mp::ResponseFlags::X2APIC);
    X2APIC.store(x2, Relaxed);
    let deadline = unsafe { core::arch::x86_64::__cpuid(1) }.ecx & (1 << 24) != 0;
    DEADLINE.store(x2 && deadline, Relaxed);
    enable_lapic();
    if x2 && deadline {
        unsafe { wrmsr(0x832, 0b10 << 17 | idt::TIMER_VECTOR as u64) }; // LVT timer: TSC-deadline
    }
    if !x2 {
        // shortcut: no xAPIC MMIO mapping yet, so no wake IPI: workers never park. Map the LAPIC for HW without x2APIC.
        unsafe { PARK = Park::Off };
    }
    park_phase(false);

    let mut tid = 1u64;
    for cpu in mp.cpus() {
        if cpu.lapic_id == mp.bsp_lapic_id() {
            continue;
        }
        if tid >= 64 {
            break;
        }
        cpu.extra.store(tid, SeqCst);
        cpu.goto_address.write(ap_limine_entry);
        tid += 1;
    }
    let want = (tid - 1) as u32;
    let t_end = rdtsc() + us_to_tsc(2_000_000);
    while unsafe { tp_n_online.load(Acquire) } < want && rdtsc() < t_end {
        core::hint::spin_loop();
    }
    let online = unsafe { tp_n_online.load(Acquire) } as u64;
    MAX_THREADS.store(online + 1, Relaxed);
    set_threads(online + 1);
    println!(
        "smp:  {} of {} APs online in the worker loop, x2APIC={}, TSC-deadline idle timer={}",
        online, want, x2, x2 && deadline
    );
    online + 1
}

/// BSP idle: sleep until the next interrupt or ~1 ms (TSC deadline), else just pause.
pub fn idle_wait() {
    if DEADLINE.load(Relaxed) {
        unsafe {
            wrmsr(0x6E0, rdtsc() + us_to_tsc(1000));
            asm!("sti; hlt; cli", options(nomem, nostack));
        }
    } else {
        core::hint::spin_loop();
    }
}
