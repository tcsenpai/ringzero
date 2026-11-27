// T54: /proc, a read-only virtual directory generated at read time (fs.rs routes every path under /proc here),
// so the shell, the agent's ls/cat tools and WASI programs see the same files. Familiar Linux names, plain
// "key: value" content (not byte-exact Linux formats: the model reads it).
use crate::{fs, smp, sys};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use core::sync::atomic::Ordering::Relaxed;

/// Every entry under /proc; a trailing '/' is a directory.
const ENTRIES: [&str; 8] = ["cpuinfo", "meminfo", "uptime", "version", "devices", "bus/", "bus/pci/", "bus/pci/devices"];

/// `rel` is under /proc ("proc" or "proc/..."), relative to / as fs::check returns it.
pub fn is_proc(rel: &str) -> bool {
    rel == "proc" || rel.starts_with("proc/")
}

/// (is_dir, size) of a /proc path, or None if it does not exist.
pub fn entry(rel: &str) -> Option<(bool, u64)> {
    let r = rel.strip_prefix("proc").unwrap_or(rel).trim_start_matches('/');
    if r.is_empty() || ENTRIES.iter().any(|e| e.strip_suffix('/') == Some(r)) {
        return Some((true, 0));
    }
    read(rel).map(|v| (false, v.len() as u64))
}

/// The entries of a /proc directory as (name, is_dir, size).
pub fn list(rel: &str) -> Option<Vec<(String, bool, u64)>> {
    if entry(rel)? != (true, 0) {
        return None;
    }
    let r = rel.strip_prefix("proc").unwrap_or(rel).trim_start_matches('/');
    let pre = if r.is_empty() { String::new() } else { alloc::format!("{}/", r) };
    Some(ENTRIES.iter().filter_map(|e| {
        let rest = e.strip_prefix(pre.as_str())?;
        let (name, dir) = match rest.strip_suffix('/') {
            Some(d) => (d, true),
            None => (rest, false),
        };
        if name.is_empty() || name.contains('/') {
            return None;
        }
        let size = if dir { 0 } else { read(&alloc::format!("proc/{}", e)).map_or(0, |v| v.len() as u64) };
        Some((String::from(name), dir, size))
    }).collect())
}

/// The content of a /proc file, generated now.
pub fn read(rel: &str) -> Option<Vec<u8>> {
    let mut o = String::new();
    match rel.strip_prefix("proc/")? {
        "cpuinfo" => cpuinfo(&mut o),
        "meminfo" => meminfo(&mut o),
        "uptime" => {
            let ms = crate::log::ms_since_boot();
            let _ = writeln!(o, "{}.{:02}", ms / 1000, ms % 1000 / 10);
        }
        "version" => {
            let _ = writeln!(o, "ringzero version {} {} ({}), Rust kernel, x86_64, not Linux", crate::VERSION, crate::CHANNEL,
                crate::EDITION);
        }
        "devices" => devices(&mut o),
        "bus/pci/devices" => pci(&mut o),
        _ => return None,
    }
    Some(o.into_bytes())
}

/// CPUID feature flags, with Linux /proc/cpuinfo names.
fn flags(o: &mut String) {
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    let (l1, l7) = unsafe { (__cpuid(1), __cpuid_count(7, 0)) };
    let bits: [(u32, u32, &str); 16] = [
        (l1.edx, 0, "fpu"), (l1.edx, 4, "tsc"), (l1.edx, 25, "sse"), (l1.edx, 26, "sse2"), (l1.ecx, 0, "pni"),
        (l1.ecx, 9, "ssse3"), (l1.ecx, 12, "fma"), (l1.ecx, 19, "sse4_1"), (l1.ecx, 20, "sse4_2"), (l1.ecx, 23, "popcnt"),
        (l1.ecx, 28, "avx"), (l1.ecx, 29, "f16c"), (l1.ecx, 30, "rdrand"), (l7.ebx, 5, "avx2"), (l7.ebx, 8, "bmi2"),
        (l7.ebx, 16, "avx512f"),
    ];
    for (reg, bit, name) in bits {
        if reg >> bit & 1 != 0 {
            let _ = write!(o, " {}", name);
        }
    }
}

fn cpuinfo(o: &mut String) {
    let n = smp::MAX_THREADS.load(Relaxed);
    let mhz = smp::TSC_HZ.load(Relaxed) / 1000;
    let mut fl = String::new();
    flags(&mut fl);
    for i in 0..n {
        let _ = writeln!(o, "processor\t: {}\nmodel name\t: {}\ncpu MHz\t\t: {}.{:03}\ncpu cores\t: {}\nrole\t\t: {}\nflags\t\t:{}\n",
            i, unsafe { (*core::ptr::addr_of!(sys::BRAND)).as_str() }, mhz / 1000, mhz % 1000, n,
            if i == 0 { "BSP: shell, agent loop, engine slice" } else { "engine worker" }, fl);
    }
}

fn meminfo(o: &mut String) {
    let (usable, total) = (sys::USABLE.load(Relaxed) >> 10, sys::TOTAL.load(Relaxed) >> 10);
    let free = usable.saturating_sub(crate::mem::used() >> 10);
    let rows: [(&str, u64); 6] = [
        ("MemTotal", total), ("MemFree", free), ("MemAvailable", free),
        ("ModelWeights", sys::MODEL_BYTES.load(Relaxed) >> 10), ("SnapshotPool", crate::snap::pool_bytes() >> 10),
        ("KernelHeap", crate::mem::HEAP_BYTES as u64 >> 10),
    ];
    for (k, v) in rows {
        let _ = writeln!(o, "{:<16}{:>10} kB", alloc::format!("{}:", k), v);
    }
}

/// The drivers this kernel runs, one per line: "<kind> <driver>: <what it drives>".
fn devices(o: &mut String) {
    let _ = writeln!(o, "Active drivers:");
    let (serial, fb) = crate::console::devices();
    if serial {
        let _ = writeln!(o, "serial       16550 UART: COM1 at I/O port 0x3f8, the console");
    }
    let _ = writeln!(o, "keyboard     PS/2 (i8042): layout {}", crate::kbd::layout());
    if let Some((w, h, bpp)) = fb {
        let _ = writeln!(o, "framebuffer  Limine framebuffer: {}x{} {} bpp, text console", w, h, bpp);
    }
    match crate::blk::location() {
        Some((d, f, mib)) => {
            let _ = writeln!(o, "block        virtio-blk: PCI 00:{:02x}.{}, {} MiB", d, f, mib);
        }
        None => {
            let _ = writeln!(o, "block        none (no virtio-blk disk)");
        }
    }
    if fs::mounted() {
        let mut d = String::new();
        fs::df(&mut d);
        let _ = writeln!(o, "filesystem   fatfs: {} on virtio-blk, mounted at /", d);
    }
    let _ = writeln!(o, "filesystem   proc: this directory, read-only, generated at read time");
    let _ = writeln!(o, "\nPCI devices: /proc/bus/pci/devices");
}

fn class(c: u8, s: u8) -> &'static str {
    match (c, s) {
        (0x01, 0x06) => "SATA controller",
        (0x01, _) => "storage controller",
        (0x02, _) => "network controller",
        (0x03, _) => "display controller",
        (0x06, 0x00) => "host bridge",
        (0x06, 0x01) => "ISA bridge",
        (0x06, _) => "bridge",
        (0x0c, 0x05) => "SMBus controller",
        (0x0c, _) => "serial bus controller",
        _ => "other",
    }
}

/// PCI bus 0 through the legacy config ports: "bus:dev.fn vendor:device class, driver".
fn pci(o: &mut String) {
    use virtio_drivers::transport::pci::bus::PciRoot;
    use virtio_drivers::transport::pci::virtio_device_type;
    let blk = crate::blk::location();
    let root = PciRoot::new(crate::blk::PortCam);
    for (df, info) in root.enumerate_bus(0) {
        let drv = match virtio_device_type(&info) {
            Some(virtio_drivers::transport::DeviceType::Block) if blk.is_some_and(|(d, f, _)| (d, f) == (df.device, df.function)) => "virtio-blk",
            Some(_) => "virtio, no driver",
            None => "no driver",
        };
        let _ = writeln!(o, "{:02x}:{:02x}.{} {:04x}:{:04x} {} (class {:02x}{:02x}), {}", df.bus, df.device, df.function,
            info.vendor_id, info.device_id, class(info.class, info.subclass), info.class, info.subclass, drv);
    }
}

/// Boot self-check of the path table (no hardware involved).
pub fn selftest() -> bool {
    entry("proc") == Some((true, 0)) && entry("proc/bus/pci") == Some((true, 0)) && entry("proc/nope").is_none()
        && list("proc").is_some_and(|v| v.iter().any(|e| e.0 == "bus" && e.1) && v.iter().any(|e| e.0 == "uptime" && !e.1))
        && list("proc/bus").is_some_and(|v| v.len() == 1 && v[0].0 == "pci")
        && list("proc/uptime").is_none() && read("proc/version").is_some()
}
