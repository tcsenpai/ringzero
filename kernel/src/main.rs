// ringzero kernel (edition "ringzero Qwen35@2b-Qwen3@0.6b"): boot under Limine, IDT, engine arena, APs in the asm worker
// loop, Qwen3.5-2B + Qwen3-Embedding-0.6B on the same engine (context swap, embed.rs), Qwen3.5-2B golden self-test,
// virtio-blk + FAT32 at / (blk.rs, fs.rs), agentd (memory stores, persistent index, context compiler, idle
// consolidation), system-agent prefix (boot snapshot cache or prefill) + snapshot, then the shell on COM1 + PS/2.
#![no_std]
#![no_main]

extern crate alloc;

mod agent;
mod agentd;
mod blk;
mod console;
mod embed;
mod engine;
mod fidx;
mod fs;
mod idt;
mod kbd;
mod llm;
mod log;
mod mem;
mod memidx;
mod mstore;
mod q4test;
mod shell;
mod smp;
mod snap;
mod sys;
mod upy; // M5 MicroPython
mod wasm; // M5b WASI runtime
mod gram; // M4.7a constrained decoding
mod hib; // M4.7b hibernation

use core::arch::asm;
use limine::firmware_type::FirmwareType;
use limine::memory_map::EntryType;
use limine::request::*;
use limine::BaseRevision;
use raw_cpuid::CpuId;

#[used]
#[unsafe(link_section = ".requests")]
static BASE_REVISION: BaseRevision = BaseRevision::new();

#[used]
#[unsafe(link_section = ".requests")]
static BOOTLOADER: BootloaderInfoRequest = BootloaderInfoRequest::new();

#[used]
#[unsafe(link_section = ".requests")]
static FIRMWARE: FirmwareTypeRequest = FirmwareTypeRequest::new();

#[used]
#[unsafe(link_section = ".requests")]
static FRAMEBUFFER: FramebufferRequest = FramebufferRequest::new();

#[used]
#[unsafe(link_section = ".requests")]
static HHDM: HhdmRequest = HhdmRequest::new();

#[used]
#[unsafe(link_section = ".requests")]
static MEMMAP: MemoryMapRequest = MemoryMapRequest::new();

#[used]
#[unsafe(link_section = ".requests")]
static MP: MpRequest = MpRequest::new().with_flags(limine::mp::RequestFlags::X2APIC);

#[used]
#[unsafe(link_section = ".requests")]
static MODULES: ModuleRequest = ModuleRequest::new();

#[used]
#[unsafe(link_section = ".requests_start_marker")]
static _START: RequestsStartMarker = RequestsStartMarker::new();

#[used]
#[unsafe(link_section = ".requests_end_marker")]
static _END: RequestsEndMarker = RequestsEndMarker::new();

const MIB: u64 = 1024 * 1024;
/// M7: `make MODEL=9b` builds the 9B edition (Makefile sets RINGZERO_EDITION)
const EDITION: &str = match option_env!("RINGZERO_EDITION") {
    Some(e) => e,
    None => "ringzero Qwen35@2b-Qwen3@0.6b (QweQwe)",
};

fn xgetbv0() -> u64 {
    let (lo, hi): (u32, u32);
    unsafe { asm!("xgetbv", in("ecx") 0, out("eax") lo, out("edx") hi, options(nomem, nostack)) };
    (hi as u64) << 32 | lo as u64
}

#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    let tsc_entry = smp::rdtsc();
    assert!(BASE_REVISION.is_supported());

    let fb = FRAMEBUFFER.get_response().and_then(|r| r.framebuffers().next()).map(|f| console::Fb {
        addr: f.addr(),
        width: f.width() as usize,
        height: f.height() as usize,
        pitch: f.pitch() as usize,
        bpp: f.bpp() as usize,
    });
    let fb_desc = fb.as_ref().map(|f| (f.width, f.height, f.bpp));
    console::init(fb);
    idt::init();

    klog!(b'I', "==============================================");
    klog!(b'I', " {}  (Rust no_std kernel + NASM LLM engine in ring 0)", EDITION);
    klog!(b'I', "==============================================");
    if let Some(b) = BOOTLOADER.get_response() {
        klog!(b'I', "boot: {} {}, base revision {:?}", b.name(), b.version(), BASE_REVISION.loaded_revision());
    }
    if let Some(f) = FIRMWARE.get_response() {
        let fw = f.firmware_type();
        let name = if fw == FirmwareType::X86_BIOS { "BIOS" } else if fw == FirmwareType::UEFI_64 { "UEFI x64" } else { "other" };
        klog!(b'I', "boot: firmware {}", name);
    }
    match fb_desc {
        Some((w, h, bpp)) => klog!(b'I', "fb:   {}x{} {} bpp", w, h, bpp),
        None => klog!(b'I', "fb:   none"),
    }

    // ---- CPU: check features, enable SSE/AVX for the engine ----
    let cpuid = CpuId::new();
    let fi = cpuid.get_feature_info().expect("cpuid leaf 1");
    let ef = cpuid.get_extended_feature_info().expect("cpuid leaf 7");
    let (avx, avx2, fma, f16c, xsave) = (fi.has_avx(), ef.has_avx2(), fi.has_fma(), fi.has_f16c(), fi.has_xsave());
    if let Some(b) = cpuid.get_processor_brand_string() {
        klog!(b'I', "cpu:  {}", b.as_str().trim());
        unsafe { (*core::ptr::addr_of_mut!(sys::BRAND)).push(b.as_str().trim().as_bytes()) };
    }
    klog!(b'I', "cpu:  AVX={} AVX2={} FMA={} F16C={} XSAVE={}", avx, avx2, fma, f16c, xsave);
    if !(avx && avx2 && fma && f16c && xsave) {
        panic!("engine needs AVX2+FMA+F16C+XSAVE");
    }
    unsafe { engine::enable_avx_fpu() };
    let osxsave = CpuId::new().get_feature_info().unwrap().has_oxsave();
    klog!(b'I', "cpu:  enabled: CR4.OSXSAVE={} XCR0={:#x} (want 0x7)", osxsave, xgetbv0());

    let hz = smp::calibrate_tsc();
    klog!(b'I', "time: TSC {}.{:03} MHz (PIT ch2, 20 ms window), entry TSC={} (~{} ms after TSC 0)",
        hz / 1_000_000, hz / 1000 % 1000, tsc_entry, tsc_entry / (hz / 1000).max(1));

    // ---- memory map ----
    let hhdm = HHDM.get_response().expect("hhdm").offset();
    let mm = MEMMAP.get_response().expect("memmap");
    let (mut usable, mut reclaim, mut modk, mut top) = (0u64, 0u64, 0u64, 0u64);
    let mut largest = (0u64, 0u64);
    for e in mm.entries() {
        let ty = match e.entry_type {
            EntryType::USABLE => "usable",
            EntryType::BOOTLOADER_RECLAIMABLE => "bootloader-reclaimable",
            EntryType::EXECUTABLE_AND_MODULES => "kernel+modules",
            EntryType::FRAMEBUFFER => "framebuffer",
            EntryType::ACPI_RECLAIMABLE => "acpi-reclaimable",
            EntryType::ACPI_NVS => "acpi-nvs",
            EntryType::BAD_MEMORY => "bad",
            _ => "reserved",
        };
        if e.length >= MIB {
            klog!(b'I', "mmap: {:#011x}-{:#011x} {:>5} MiB {}", e.base, e.base + e.length, e.length / MIB, ty);
        }
        match e.entry_type {
            EntryType::USABLE => {
                usable += e.length;
                if e.length > largest.1 {
                    largest = (e.base, e.length);
                }
            }
            EntryType::BOOTLOADER_RECLAIMABLE => reclaim += e.length,
            EntryType::EXECUTABLE_AND_MODULES => modk += e.length,
            _ => {}
        }
        if e.entry_type == EntryType::USABLE || e.entry_type == EntryType::EXECUTABLE_AND_MODULES {
            top = top.max(e.base + e.length);
        }
    }
    klog!(b'I', 
        "mem:  {} entries, usable {} MiB, bootloader-reclaimable {} MiB, kernel+modules {} MiB, top {:#x}",
        mm.entries().len(),
        usable / MIB,
        reclaim / MIB,
        modk / MIB,
        top
    );
    klog!(b'I', "mem:  largest usable run {:#x} + {} MiB, HHDM offset {:#x}", largest.0, largest.1 / MIB, hhdm);

    sys::USABLE.store(usable, core::sync::atomic::Ordering::Relaxed);
    sys::TOTAL.store(usable + reclaim + modk, core::sync::atomic::Ordering::Relaxed);
    mem::init(hhdm, largest.0, largest.1);
    klog!(b'I', "mem:  engine arena identity-mapped at {:#x} ({} MiB), page size {} KiB (HHDM view: {} KiB)",
        mem::base(), largest.1 / MIB, mem::page_size(mem::base()) / 1024, mem::page_size(hhdm + mem::base()) / 1024);
    // the model too: lower-half identity map with 2 MiB pages, before the APs start (shared CR3)
    let mods = MODULES.get_response().map(|r| r.modules()).unwrap_or(&[]);
    let named = |s: &[u8]| mods.iter().find(|m| m.path().to_bytes().ends_with(s));
    let (model, emb) = (named(b"/model.asmbin").or(mods.first()), named(b"/embed.asmbin"));
    let map = |m: &&limine::file::File| {
        let p = m.addr() as u64 - hhdm;
        mem::map_low(p, m.size());
        p
    };
    let model_phys = model.map(map);
    let emb_phys = emb.map(map);

    // ---- SMP ----
    match MP.get_response() {
        Some(mp) => {
            klog!(b'I', "smp:  {} CPUs reported by Limine MP, BSP LAPIC id {}", mp.cpus().len(), mp.bsp_lapic_id());
            smp::start(mp);
        }
        None => klog!(b'I', "smp:  no MP response, running on the BSP only"),
    }

    // ---- modules (the model) ----
    klog!(b'I', "mod:  {} module(s)", mods.len());
    for m in mods {
        let p = m.addr();
        let h = unsafe { core::slice::from_raw_parts(p as *const u8, 64) };
        let u32_at = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let magic = core::str::from_utf8(&h[0..4]).unwrap_or("????");
        klog!(b'I', 
            "mod:  {} size {} B ({} MiB) at virt {:p} phys {:#x}",
            m.path().to_str().unwrap_or("?"),
            m.size(),
            m.size() / MIB,
            p,
            p as u64 - hhdm
        );
        klog!(b'I', 
            "mod:  magic \"{}\" header version {} dim {} layers {} vocab {}",
            magic,
            u32_at(4),
            u32_at(8),
            u32_at(16),
            u32_at(32)
        );
    }

    // ---- kernel heap, virtio disk, FAT32 at / ----
    mem::heap_init();
    klog!(b'I', "mem:  kernel heap {} MiB (linked_list_allocator, from the arena)", mem::HEAP_BYTES >> 20);
    fs::init();
    upy::init(); // M5: MicroPython heap + stack, before the snapshot pool takes the arena
    wasm::init(); // M5b: the wasm memory slot, same reason
    klog!(b'I', "wasm: command-line splitter self-test {}", if wasm::selftest() { "PASS" } else { "FAIL" });

    // ---- engine ABI test, model, golden self-test, REPL ----
    let ok = engine::selftest();
    klog!(b'I', "engine: M0 ABI self-test {}", if ok { "PASS" } else { "FAIL" });
    let (Some(m), Some(mp)) = (model, model_phys) else { panic!("no model module") };
    klog!(b'I', "mem:  model identity-mapped at {:#x}, page size {} KiB (HHDM view: {} KiB)",
        mp, mem::page_size(mp) / 1024, mem::page_size(m.addr() as u64) / 1024);
    sys::MODEL_BYTES.store(m.size() + emb.map_or(0, |e| e.size()), core::sync::atomic::Ordering::Relaxed);
    if llm::init(mp as *const u8, m.size()) {
        if let Some(q4) = q4test::selftest() {
            klog!(b'I', "q4: Q4_0 kernel self-test {}", if q4 { "PASS" } else { "FAIL" });
        }
        // the embedder gets its own g_ctx (embed.rs) before the snapshot pool takes the rest of the arena
        match (emb, emb_phys) {
            (Some(e), Some(ep)) => {
                klog!(b'I', "mem:  embedder identity-mapped at {:#x} ({} MiB)", ep, e.size() / MIB);
                if embed::init(ep as *const u8, e.size()) {
                    embed::boot_check();
                }
            }
            _ => klog!(b'W', "embed: no embed.asmbin module, /embed /index /search and recall are off"),
        }
        klog!(b'I', "index: memory index self-test (fp16, Q8_0 rows, ln, identifier gate, retain) {}", if memidx::selftest() { "PASS" } else { "FAIL" });
        klog!(b'I', "index: file indexer self-test (chunker, path checks) {}", if fidx::selftest() { "PASS" } else { "FAIL" });
        klog!(b'I', "agentd: self-test (record codec + torn tail, fact validation, grounding, calendar) {}",
            if mstore::selftest() && agentd::selftest() { "PASS" } else { "FAIL" });
        agentd::init(); // stores recovered, index loaded or rebuilt, files compared with the saved index
        llm::selftest(); // the 2B's golden test runs after the context swap round trip of the boot check
        klog!(b'I', "bench: 64-token decode, prompt \"Explain how a CPU cache works in 3 concise sentences.\"");
        llm::bench();
        klog!(b'I', "agent: tool-call parser self-test {}", if agent::selftest() { "PASS" } else { "FAIL" });
        gram::init();
        klog!(b'I', "gram: tool-call grammar self-test {}", if gram::selftest() { "PASS" } else { "FAIL" });
        gram::bench();
        klog!(b'I', "kbd:  PS/2 layout {} (change with '!kbd us|it')", kbd::layout());
        agent::init();
        hib::boot(); // M4.7b: resume the hibernated conversation, if any
        agentd::booted();
        shell::run();
    }
    loop {
        unsafe { asm!("cli; hlt") };
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    klog!(b'E', "PANIC: {}", info);
    loop {
        unsafe { asm!("cli; hlt") };
    }
}
