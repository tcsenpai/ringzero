// IDT: CPU exceptions print vector, error code, RIP, CR2 (serial + framebuffer) and halt.
// Vector 0xF0 (wake IPI) and 0xF1 (BSP idle timer) are EOI-only (lapic_eoi_isr in smp.asm),
// 0xFF (LAPIC spurious) is a bare iretq. Same table on every CPU.
use core::arch::{asm, global_asm};
use core::ptr::addr_of;

pub const WAKE_VECTOR: u8 = 0xF0;
pub const TIMER_VECTOR: u8 = 0xF1;

#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    off_lo: u16,
    sel: u16,
    ist: u8,
    attr: u8,
    off_mid: u16,
    off_hi: u32,
    _r: u32,
}

static mut IDT: [Entry; 256] = [Entry { off_lo: 0, sel: 0, ist: 0, attr: 0, off_mid: 0, off_hi: 0, _r: 0 }; 256];

#[repr(C, packed)]
struct Idtr {
    limit: u16,
    base: u64,
}
static mut IDTR: Idtr = Idtr { limit: 0, base: 0 };

// Common exception tail: rsp -> ExcFrame { vector, err, rip, cs, rflags, rsp, ss }.
global_asm!(
    "exc_common:",
    "mov rdi, rsp", // -> ExcFrame
    "and rsp, -16",
    "call exc_handler",
    "2: cli",
    "hlt",
    "jmp 2b",
    "spurious_isr:",
    "iretq",
);

// Exception stub: push a dummy error code where the CPU does not, then the vector.
macro_rules! stub {
    ($name:ident, $v:literal, false) => {
        global_asm!(concat!(".global ", stringify!($name), "\n", stringify!($name), ":\npush 0\npush ", $v, "\njmp exc_common\n"));
    };
    ($name:ident, $v:literal, true) => {
        global_asm!(concat!(".global ", stringify!($name), "\n", stringify!($name), ":\npush ", $v, "\njmp exc_common\n"));
    };
}
stub!(exc0, 0, false);
stub!(exc1, 1, false);
stub!(exc2, 2, false);
stub!(exc3, 3, false);
stub!(exc4, 4, false);
stub!(exc5, 5, false);
stub!(exc6, 6, false);
stub!(exc7, 7, false);
stub!(exc8, 8, true);
stub!(exc9, 9, false);
stub!(exc10, 10, true);
stub!(exc11, 11, true);
stub!(exc12, 12, true);
stub!(exc13, 13, true);
stub!(exc14, 14, true);
stub!(exc15, 15, false);
stub!(exc16, 16, false);
stub!(exc17, 17, true);
stub!(exc18, 18, false);
stub!(exc19, 19, false);
stub!(exc20, 20, false);
stub!(exc21, 21, true);
stub!(exc22, 22, false);
stub!(exc23, 23, false);
stub!(exc24, 24, false);
stub!(exc25, 25, false);
stub!(exc26, 26, false);
stub!(exc27, 27, false);
stub!(exc28, 28, false);
stub!(exc29, 29, true);
stub!(exc30, 30, true);
stub!(exc31, 31, false);

unsafe extern "C" {
    fn exc0(); fn exc1(); fn exc2(); fn exc3(); fn exc4(); fn exc5(); fn exc6(); fn exc7();
    fn exc8(); fn exc9(); fn exc10(); fn exc11(); fn exc12(); fn exc13(); fn exc14(); fn exc15();
    fn exc16(); fn exc17(); fn exc18(); fn exc19(); fn exc20(); fn exc21(); fn exc22(); fn exc23();
    fn exc24(); fn exc25(); fn exc26(); fn exc27(); fn exc28(); fn exc29(); fn exc30(); fn exc31();
    fn spurious_isr();
    fn lapic_eoi_isr();
}

#[repr(C)]
pub struct ExcFrame {
    vector: u64,
    err: u64,
    rip: u64,
    cs: u64,
    rflags: u64,
    rsp: u64,
    ss: u64,
}

const NAMES: [&str; 32] = [
    "#DE divide", "#DB debug", "NMI", "#BP", "#OF", "#BR", "#UD invalid opcode", "#NM no FPU",
    "#DF double fault", "coproc", "#TS bad TSS", "#NP not present", "#SS stack", "#GP general protection",
    "#PF page fault", "reserved", "#MF x87", "#AC alignment", "#MC machine check", "#XM SIMD fp",
    "#VE virtualization", "#CP control protection", "22", "23", "24", "25", "26", "27", "28", "#VC", "#SX", "31",
];

#[unsafe(no_mangle)]
extern "C" fn exc_handler(f: &ExcFrame) -> ! {
    let cr2: u64;
    unsafe { asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack)) };
    let v = f.vector as usize;
    crate::println!(
        "\n*** EXCEPTION vec={} ({}) err={:#x} rip={:#x} cr2={:#x} rsp={:#x} cs={:#x} rflags={:#x} lapic={} -> halt",
        v, NAMES[v & 31], f.err, f.rip, cr2, f.rsp, f.cs, f.rflags, crate::smp::lapic_id()
    );
    loop {
        unsafe { asm!("cli; hlt") };
    }
}

fn set(v: usize, handler: unsafe extern "C" fn(), cs: u16) {
    let a = handler as usize as u64;
    // present, DPL 0, 64-bit interrupt gate (IF cleared on entry)
    unsafe {
        (*core::ptr::addr_of_mut!(IDT))[v] = Entry {
            off_lo: a as u16,
            sel: cs,
            ist: 0,
            attr: 0x8E,
            off_mid: (a >> 16) as u16,
            off_hi: (a >> 32) as u32,
            _r: 0,
        };
    }
}

/// Build the table (BSP, once) and load it.
pub fn init() {
    let cs: u16;
    unsafe { asm!("mov {0:x}, cs", out(reg) cs, options(nomem, nostack)) };
    let ex: [unsafe extern "C" fn(); 32] = [
        exc0, exc1, exc2, exc3, exc4, exc5, exc6, exc7, exc8, exc9, exc10, exc11, exc12, exc13, exc14, exc15,
        exc16, exc17, exc18, exc19, exc20, exc21, exc22, exc23, exc24, exc25, exc26, exc27, exc28, exc29, exc30, exc31,
    ];
    for (v, h) in ex.iter().enumerate() {
        set(v, *h, cs);
    }
    set(WAKE_VECTOR as usize, lapic_eoi_isr, cs);
    set(TIMER_VECTOR as usize, lapic_eoi_isr, cs);
    set(0xFF, spurious_isr, cs);
    unsafe {
        IDTR = Idtr { limit: (core::mem::size_of::<[Entry; 256]>() - 1) as u16, base: addr_of!(IDT) as u64 };
    }
    load();
}

/// lidt on the calling CPU (APs call this from ap_init).
pub fn load() {
    unsafe { asm!("lidt [{}]", in(reg) addr_of!(IDTR), options(readonly, nostack)) };
}
