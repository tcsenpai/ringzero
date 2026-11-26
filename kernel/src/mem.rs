// Engine memory: a bump arena (sys_mmap_anon) carved from the largest usable memory-map region,
// identity-mapped in the lower half with 2 MiB pages (the engine treats negative pointers from
// sys_mmap_anon as errors, transformer.asm:529, so HHDM addresses cannot be used), plus a
// page-table walk to report the page size behind an address.
use core::alloc::GlobalAlloc;
use core::arch::asm;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

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
    map(phys, len, 0x83); // present | writable | 2 MiB
}

/// Device MMIO (PCI BARs): identity map like map_low, uncached (PCD | PWT). BSP only, after the APs run:
/// the APs never touch these pages, so no TLB shootdown is needed.
pub fn map_mmio(phys: u64, len: u64) {
    map(phys, len, 0x9b);
}

fn map(phys: u64, len: u64, flags: u64) {
    let hhdm = HHDM.load(Relaxed);
    let mut cr3: u64;
    unsafe { asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack)) };
    let ent = |table: u64, i: u64| (hhdm + (table & ADDR) + i * 8) as *mut u64;
    let mut a = phys & !(MB2 - 1);
    while a < phys + len {
        assert!(a < 1 << 47, "map: lower half only");
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
            *ent(*l3, a >> 21 & 511) = a | flags;
        }
        a += MB2;
    }
    unsafe { asm!("mov {0}, cr3", "mov cr3, {0}", inout(reg) cr3 => _, options(nostack)) };
}

/// Arena = largest usable region. Must run before the APs start (they share this CR3, no shootdown).
pub fn init(hhdm: u64, phys: u64, len: u64) {
    HHDM.store(hhdm, Relaxed);
    PT_NEXT.store(phys, Relaxed);
    PT_END.store(phys + 256 * 1024, Relaxed);
    let start = (phys + 256 * 1024 + MB2 - 1) & !(MB2 - 1);
    let end = (phys + len) & !(MB2 - 1);
    map_low(start, end - start);
    ARENA_BASE.store(start, Relaxed);
    ARENA_CUR.store(start, Relaxed);
    ARENA_END.store(end, Relaxed);
}

// Kernel heap (alloc, for the FAT driver, the file indexer and wasmi): a fixed slice of the arena, taken at boot.
static HEAP: linked_list_allocator::LockedHeap = linked_list_allocator::LockedHeap::empty();
pub const HEAP_BYTES: usize = 64 << 20;

// M5b: the wasm slot, one region of the memory cap (wasm.rs) for one wasm linear memory. wasmi keeps a linear
// memory in a Vec<u8> and grows it by doubling (realloc); in the heap a 128 -> 256 MiB step needs 384 MiB at
// once. While wasm.rs instantiates a module the slot is armed: the first allocation of whole 64 KiB pages (the
// memory's initial size) takes it, and every realloc of it stays in place.
static SLOT: AtomicU64 = AtomicU64::new(0);
static SLOT_LEN: AtomicU64 = AtomicU64::new(0);
static SLOT_ARMED: AtomicBool = AtomicBool::new(false);
static SLOT_BUSY: AtomicBool = AtomicBool::new(false);

pub fn set_wasm_slot(p: *mut u8, len: usize) {
    SLOT_LEN.store(len as u64, Relaxed);
    SLOT.store(p as u64, Relaxed);
}

/// Around Linker::instantiate only (one memory per instance: a second one stays in the heap).
pub fn arm_wasm_slot(on: bool) {
    SLOT_ARMED.store(on, Relaxed);
}

fn in_slot(p: *mut u8) -> bool {
    let (s, n) = (SLOT.load(Relaxed), SLOT_LEN.load(Relaxed));
    s != 0 && (p as u64) >= s && (p as u64) < s + n
}

fn slot_take(size: usize, align: usize) -> Option<*mut u8> {
    let ok = SLOT_ARMED.load(Relaxed) && size % 65536 == 0 && size as u64 <= SLOT_LEN.load(Relaxed) && align <= 4096;
    (ok && SLOT_BUSY.compare_exchange(false, true, Relaxed, Relaxed).is_ok()).then(|| {
        SLOT_ARMED.store(false, Relaxed);
        SLOT.load(Relaxed) as *mut u8
    })
}

struct Alloc;
#[global_allocator]
static ALLOC: Alloc = Alloc;

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, l: core::alloc::Layout) -> *mut u8 {
        match slot_take(l.size(), l.align()) {
            Some(p) => p,
            None => unsafe { HEAP.alloc(l) },
        }
    }
    unsafe fn alloc_zeroed(&self, l: core::alloc::Layout) -> *mut u8 {
        match slot_take(l.size(), l.align()) {
            Some(p) => unsafe {
                core::ptr::write_bytes(p, 0, l.size()); // the slot is reused across runs
                p
            },
            None => unsafe { HEAP.alloc_zeroed(l) },
        }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: core::alloc::Layout) {
        if in_slot(p) {
            SLOT_BUSY.store(false, Relaxed);
        } else {
            unsafe { HEAP.dealloc(p, l) }
        }
    }
    unsafe fn realloc(&self, p: *mut u8, l: core::alloc::Layout, new: usize) -> *mut u8 {
        // shortcut: a doubling step may ask for more than the slot (1088 KiB x 2^8 = 272 MiB > 256); the Vec only
        // writes up to its length, and StoreLimits keeps the memory within the slot. Only the linear memory is here.
        if in_slot(p) { p } else { unsafe { HEAP.realloc(p, l, new) } }
    }
}

pub fn heap_free() -> usize {
    HEAP.lock().free()
}

pub fn heap_init() {
    let p = sys_mmap_anon(HEAP_BYTES as u64);
    assert!((p as i64) > 0, "heap: arena full");
    unsafe { HEAP.lock().init(p, HEAP_BYTES) };
}

pub fn heap_used() -> usize {
    HEAP.lock().used()
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
