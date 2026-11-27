// System facts shared by the shell builtins and the agent tools (one source for both).
use crate::log::SBuf;
use crate::{mem, smp};
use core::fmt::Write;
use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

const MIB: u64 = 1 << 20;
pub static USABLE: AtomicU64 = AtomicU64::new(0); // usable RAM in the memory map (bytes)
pub static TOTAL: AtomicU64 = AtomicU64::new(0); // usable + bootloader-reclaimable + kernel/modules
pub static MODEL_BYTES: AtomicU64 = AtomicU64::new(0);
pub static mut BRAND: SBuf<64> = SBuf::new();

pub fn meminfo<const N: usize>(o: &mut SBuf<N>) {
    let (usable, total) = (USABLE.load(Relaxed) / MIB, TOTAL.load(Relaxed) / MIB);
    let used = mem::used() / MIB;
    let model = MODEL_BYTES.load(Relaxed) / MIB;
    let pool = crate::snap::pool_bytes() / MIB;
    let heap = crate::mem::HEAP_BYTES as u64 / MIB;
    let _ = write!(
        o,
        "RAM total {} MiB, free {} MiB, used {} MiB (model weights {} MiB, LLM engine arena {} MiB, state snapshot pool {} MiB, kernel heap {} MiB, kernel and firmware {} MiB)",
        total, usable.saturating_sub(used), total - usable.saturating_sub(used), model, used - pool - heap, pool, heap,
        total.saturating_sub(usable + model)
    );
}

pub fn cpuinfo<const N: usize>(o: &mut SBuf<N>) {
    let hz = smp::TSC_HZ.load(Relaxed);
    let _ = write!(
        o,
        "{}; {} cores; TSC {} MHz; features used by the LLM engine: AVX2 FMA F16C",
        unsafe { (*core::ptr::addr_of!(BRAND)).as_str() },
        smp::MAX_THREADS.load(Relaxed),
        hz / 1_000_000
    );
}

pub fn cores<const N: usize>(o: &mut SBuf<N>) {
    let n = smp::MAX_THREADS.load(Relaxed);
    let _ = write!(
        o,
        "{} cores online: core 0 = BSP (shell, agent loop, engine slice), cores 1-{} = engine workers; {} of {} used for inference; {} parked now",
        n, n - 1, smp::threads(), n, smp::parked()
    );
}

pub fn uptime<const N: usize>(o: &mut SBuf<N>) {
    let ms = crate::log::ms_since_boot();
    let s = ms / 1000;
    let _ = write!(o, "up {}h {}m {}s ({}.{:03} s)", s / 3600, s / 60 % 60, s % 60, s, ms % 1000);
}

fn cmos(reg: u8) -> u8 {
    let v: u8;
    unsafe {
        core::arch::asm!("out 0x70, al", in("al") reg, options(nomem, nostack));
        core::arch::asm!("in al, 0x71", out("al") v, options(nomem, nostack));
    }
    v
}

/// Wall clock from the CMOS RTC (QEMU: host time, UTC by default): (year, month, day, hour, min, sec).
/// Read twice until stable (no update in between), BCD or binary and 12/24 h per status B.
pub fn rtc() -> (u32, u32, u32, u32, u32, u32) {
    let read = || {
        let mut k = 0;
        while cmos(0x0A) & 0x80 != 0 && k < 100_000 {
            k += 1; // update in progress (at most ~2 ms)
        }
        [cmos(0), cmos(2), cmos(4), cmos(7), cmos(8), cmos(9), cmos(0x32)]
    };
    let mut a = read();
    for _ in 0..4 {
        let b = read();
        if a == b {
            break;
        }
        a = b;
    }
    let sb = cmos(0x0B);
    let v = |x: u8| if sb & 4 != 0 { x as u32 } else { (x & 0x0f) as u32 + (x >> 4) as u32 * 10 };
    let mut h = v(a[2] & 0x7f);
    if sb & 2 == 0 {
        h = h % 12 + if a[2] & 0x80 != 0 { 12 } else { 0 }; // 12-hour mode
    }
    let c = v(a[6]);
    let y = if (19..=21).contains(&c) { c * 100 } else { 2000 } + v(a[5]);
    (y, v(a[4]), v(a[3]), h, v(a[1]), v(a[0]))
}

/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's days_from_civil).
pub fn days(y: u32, m: u32, d: u32) -> i64 {
    let y = y as i64 - (m <= 2) as i64;
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m as i64 + 9) % 12) + 2) / 5 + d as i64 - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// Seconds since the Unix epoch, from the RTC (record timestamps of agentd).
pub fn unix_time() -> u64 {
    let (y, mo, d, h, mi, s) = rtc();
    (days(y, mo, d) * 86_400) as u64 + (h * 3600 + mi * 60 + s) as u64
}

/// ACPI S5 on QEMU q35 (ICH9 PM1a_CNT at 0x604, SLP_TYP 0 | SLP_EN).
/// shortcut: fixed port, no FADT parsing; real hardware needs the ACPI tables. Halts if it fails.
pub fn poweroff() -> ! {
    crate::hib::on_poweroff(); // M4.7b: the conversation goes to /var/lib/ringzero/hib first
    crate::agentd::shutdown();
    let _ = crate::fs::sync();
    crate::klog!(b'W', "power off");
    unsafe {
        core::arch::asm!("out dx, ax", in("dx") 0x604u16, in("ax") 0x2000u16, options(nomem, nostack));
        loop {
            core::arch::asm!("cli; hlt");
        }
    }
}

/// Reset through the 8042 (pulse CPU reset line), then a triple fault as fallback.
pub fn reboot() -> ! {
    crate::agentd::shutdown();
    let _ = crate::fs::sync();
    crate::klog!(b'W', "reboot");
    unsafe {
        core::arch::asm!("out 0x64, al", in("al") 0xFEu8, options(nomem, nostack));
        let idtr = [0u64; 2];
        core::arch::asm!("lidt [{}]; int3", in(reg) idtr.as_ptr());
        loop {
            core::arch::asm!("cli; hlt");
        }
    }
}
