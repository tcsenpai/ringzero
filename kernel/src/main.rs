// AIOS kernel, milestone M0: boot under Limine, report what the bootloader handed us,
// enable AVX on the BSP and prove the NASM engine links and runs.
#![no_std]
#![no_main]

mod console;
mod engine;

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
static MP: MpRequest = MpRequest::new();

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

fn rdtsc() -> u64 {
    let (lo, hi): (u32, u32);
    unsafe { asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack)) };
    (hi as u64) << 32 | lo as u64
}

fn xgetbv0() -> u64 {
    let (lo, hi): (u32, u32);
    unsafe { asm!("xgetbv", in("ecx") 0, out("eax") lo, out("edx") hi, options(nomem, nostack)) };
    (hi as u64) << 32 | lo as u64
}

#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    let tsc_entry = rdtsc();
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

    println!("==============================================");
    println!(" AIOS M0  (Rust no_std kernel + NASM LLM engine)");
    println!("==============================================");
    if let Some(b) = BOOTLOADER.get_response() {
        println!("boot: {} {}, base revision {:?}", b.name(), b.version(), BASE_REVISION.loaded_revision());
    }
    if let Some(f) = FIRMWARE.get_response() {
        let fw = f.firmware_type();
        let name = if fw == FirmwareType::X86_BIOS { "BIOS" } else if fw == FirmwareType::UEFI_64 { "UEFI x64" } else { "other" };
        println!("boot: firmware {}", name);
    }
    match fb_desc {
        Some((w, h, bpp)) => println!("fb:   {}x{} {} bpp", w, h, bpp),
        None => println!("fb:   none"),
    }

    // ---- CPU: check features, enable SSE/AVX for the engine ----
    let cpuid = CpuId::new();
    let fi = cpuid.get_feature_info().expect("cpuid leaf 1");
    let ef = cpuid.get_extended_feature_info().expect("cpuid leaf 7");
    let (avx, avx2, fma, f16c, xsave) = (fi.has_avx(), ef.has_avx2(), fi.has_fma(), fi.has_f16c(), fi.has_xsave());
    if let Some(b) = cpuid.get_processor_brand_string() {
        println!("cpu:  {}", b.as_str().trim());
    }
    println!("cpu:  AVX={} AVX2={} FMA={} F16C={} XSAVE={}", avx, avx2, fma, f16c, xsave);
    if !(avx && avx2 && fma && f16c && xsave) {
        panic!("engine needs AVX2+FMA+F16C+XSAVE");
    }
    unsafe { engine::enable_avx_fpu() };
    let osxsave = CpuId::new().get_feature_info().unwrap().has_oxsave();
    println!("cpu:  enabled: CR4.OSXSAVE={} XCR0={:#x} (want 0x7)", osxsave, xgetbv0());

    // TSC at entry ~ time since VM power-on (firmware + Limine, incl. module load), if the TSC rate is known.
    let tsc_hz = cpuid
        .get_tsc_info()
        .and_then(|t| t.tsc_frequency())
        .or_else(|| cpuid.get_processor_frequency_info().map(|p| p.processor_base_frequency() as u64 * 1_000_000))
        .filter(|&hz| hz > 0);
    match tsc_hz {
        Some(hz) => println!("time: TSC {} MHz, kernel entry at {} ms after TSC 0", hz / 1_000_000, tsc_entry / (hz / 1000)),
        None => println!("time: TSC rate unknown, entry TSC={}", tsc_entry),
    }

    // ---- memory map ----
    let hhdm = HHDM.get_response().expect("hhdm").offset();
    let mm = MEMMAP.get_response().expect("memmap");
    let (mut usable, mut reclaim, mut modk, mut top) = (0u64, 0u64, 0u64, 0u64);
    let mut largest = (0u64, 0u64);
    for e in mm.entries() {
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
    println!(
        "mem:  {} entries, usable {} MiB, bootloader-reclaimable {} MiB, kernel+modules {} MiB, top {:#x}",
        mm.entries().len(),
        usable / MIB,
        reclaim / MIB,
        modk / MIB,
        top
    );
    println!("mem:  largest usable run {:#x} + {} MiB, HHDM offset {:#x}", largest.0, largest.1 / MIB, hhdm);

    // ---- SMP ----
    match MP.get_response() {
        Some(mp) => println!("smp:  {} CPUs reported by Limine MP, BSP LAPIC id {}", mp.cpus().len(), mp.bsp_lapic_id()),
        None => println!("smp:  no MP response"),
    }

    // ---- modules (the model) ----
    let mods = MODULES.get_response().map(|r| r.modules()).unwrap_or(&[]);
    println!("mod:  {} module(s)", mods.len());
    for m in mods {
        let p = m.addr();
        let h = unsafe { core::slice::from_raw_parts(p as *const u8, 64) };
        let u32_at = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let magic = core::str::from_utf8(&h[0..4]).unwrap_or("????");
        println!(
            "mod:  {} size {} B ({} MiB) at virt {:p} phys {:#x}",
            m.path().to_str().unwrap_or("?"),
            m.size(),
            m.size() / MIB,
            p,
            p as u64 - hhdm
        );
        println!(
            "mod:  magic \"{}\" header version {} dim {} layers {} vocab {}",
            magic,
            u32_at(4),
            u32_at(8),
            u32_at(16),
            u32_at(32)
        );
    }

    // ---- engine link + ABI test ----
    let ok = engine::selftest();
    println!("M0 {}", if ok { "PASS" } else { "FAIL" });

    loop {
        unsafe { asm!("cli; hlt") };
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("PANIC: {}", info);
    loop {
        unsafe { asm!("cli; hlt") };
    }
}
