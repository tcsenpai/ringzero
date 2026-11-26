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

/// ACPI S5 on QEMU q35 (ICH9 PM1a_CNT at 0x604, SLP_TYP 0 | SLP_EN).
/// shortcut: fixed port, no FADT parsing; real hardware needs the ACPI tables. Halts if it fails.
pub fn poweroff() -> ! {
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
