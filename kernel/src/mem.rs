// Engine memory: a bump arena (sys_mmap_anon) carved from the largest usable memory-map region,
// identity-mapped in the lower half with 2 MiB pages (the engine treats negative pointers from
// sys_mmap_anon as errors, transformer.asm:529, so HHDM addresses cannot be used), plus a
// page-table walk to report the page size behind an address.
use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

static ARENA_CUR: AtomicU64 = AtomicU64::new(0);
static ARENA_END: AtomicU64 = AtomicU64::new(0);
static ARENA_BASE: AtomicU64 = AtomicU64::new(0);
static HHDM: AtomicU64 = AtomicU64::new(0);

const MB2: u64 = 2 << 20;
const ADDR: u64 = 0x000F_FFFF_FFFF_F000;
// page-table pages, taken from the start of the arena region (phys)
static PT_NEXT: AtomicU64 = AtomicU64::new(0);
static PT_END: AtomicU64 = AtomicU64::new(0);

fn pt_alloc() -> u64 {
    let p = PT_NEXT.fetch_add(4096, Relaxed);
    assert!(p + 4096 <= PT_END.load(Relaxed), "page-table pool exhausted");
    unsafe { core::ptr::write_bytes((HHDM.load(Relaxed) + p) as *mut u8, 0, 4096) };
    p
}

/// Identity-map [phys, phys+len) (rounded out to 2 MiB) in the lower half, writable, write-back.
pub fn map_low(phys: u64, len: u64) {
    let hhdm = HHDM.load(Relaxed);
    let mut cr3: u64;
    unsafe { asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack)) };
    let ent = |table: u64, i: u64| (hhdm + (table & ADDR) + i * 8) as *mut u64;
    let mut a = phys & !(MB2 - 1);
    while a < phys + len {
        assert!(a < 1 << 39, "map_low: only the first 512 GiB");
        unsafe {
            let l4 = ent(cr3, a >> 39 & 511);
            if *l4 & 1 == 0 {
                *l4 = pt_alloc() | 3;
            }
            let l3 = ent(*l4, a >> 30 & 511);
            if *l3 & 1 == 0 {
                *l3 = pt_alloc() | 3;
            }
            assert!(*l3 & 0x80 == 0, "map_low: 1 GiB page already there");
            *ent(*l3, a >> 21 & 511) = a | 0x83; // present | writable | 2 MiB
        }
        a += MB2;
    }
    unsafe { asm!("mov {0}, cr3", "mov cr3, {0}", inout(reg) cr3 => _, options(nostack)) };
}

/// Arena = largest usable region. Must run before the APs start (they share this CR3, no shootdown).
pub fn init(hhdm: u64, phys: u64, len: u64) {
    HHDM.store(hhdm, Relaxed);
    PT_NEXT.store(phys, Relaxed);
    PT_END.store(phys + 64 * 1024, Relaxed);
    let start = (phys + 64 * 1024 + MB2 - 1) & !(MB2 - 1);
    let end = (phys + len) & !(MB2 - 1);
    map_low(start, end - start);
    ARENA_BASE.store(start, Relaxed);
    ARENA_CUR.store(start, Relaxed);
    ARENA_END.store(end, Relaxed);
}

pub fn used() -> u64 {
    ARENA_CUR.load(Relaxed) - ARENA_BASE.load(Relaxed)
}

pub fn base() -> u64 {
    ARENA_BASE.load(Relaxed)
}

/// Engine import: 64-byte aligned, zeroed, never freed (same contract as runtime.asm). -12 on OOM.
#[unsafe(no_mangle)]
pub extern "C" fn sys_mmap_anon(size: u64) -> *mut u8 {
    let p = (ARENA_CUR.load(Relaxed) + 63) & !63;
    let n = (size + 63) & !63;
    if p + n > ARENA_END.load(Relaxed) {
        return -12i64 as *mut u8;
    }
    ARENA_CUR.store(p + n, Relaxed);
    unsafe { core::ptr::write_bytes(p as *mut u8, 0, n as usize) };
    p as *mut u8
}

#[unsafe(no_mangle)]
extern "C" fn sys_munmap(_ptr: *mut u8, _size: u64) -> i64 {
    0 // bump arena: freeing is a no-op, same as the asm OS
}

/// Size of the page that maps `va` (4096, 2 MiB or 1 GiB), 0 if unmapped. Walks CR3 through the HHDM.
pub fn page_size(va: u64) -> u64 {
    let hhdm = HHDM.load(Relaxed);
    let mut t: u64;
    unsafe { asm!("mov {}, cr3", out(reg) t, options(nomem, nostack)) };
    t &= 0x000F_FFFF_FFFF_F000;
    for (lvl, shift) in [(4, 39u32), (3, 30), (2, 21), (1, 12)] {
        let e = unsafe { ((hhdm + t) as *const u64).add(((va >> shift) & 511) as usize).read_volatile() };
        if e & 1 == 0 {
            return 0;
        }
        if lvl == 1 || (lvl < 4 && e & 0x80 != 0) {
            return 1 << shift;
        }
        t = e & 0x000F_FFFF_FFFF_F000;
    }
    0
}
