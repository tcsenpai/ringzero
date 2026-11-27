// T54: /proc, a read-only virtual directory generated at read time (fs.rs mount_of routes every path under /proc
// here, as the path inside /proc), so the shell, the agent's ls/cat tools and WASI programs see the same files.
// Familiar Linux names, plain "key: value" content (not byte-exact Linux formats: the model reads it).
use crate::{fs, smp, sys};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use core::sync::atomic::Ordering::Relaxed;

/// Every entry under /proc; a trailing '/' is a directory.
const ENTRIES: [&str; 10] = ["cpuinfo", "meminfo", "uptime", "version", "devices", "kmsg", "mounts", "bus/", "bus/pci/", "bus/pci/devices"];

/// (is_dir, size) of a path inside /proc ("" = /proc), or None if it does not exist.
pub fn entry(r: &str) -> Option<(bool, u64)> {
    fs::tree_entry(&ENTRIES, r, read)
}

/// The entries of a /proc directory as (name, is_dir, size).
pub fn list(r: &str) -> Option<Vec<(String, bool, u64)>> {
    fs::tree_list(&ENTRIES, r, read)
}

/// The content of a /proc file, generated now.
pub fn read(r: &str) -> Option<Vec<u8>> {
    let mut o = String::new();
    match r {
        "cpuinfo" => cpuinfo(&mut o),
        "meminfo" => meminfo(&mut o),
        "uptime" => {
            // T55: the seconds stay the first field for programs, the rest is for the model
            let ms = crate::log::ms_since_boot();
            let s = ms / 1000;
            let _ = writeln!(o, "{}.{:02} seconds ({} h {} min {} s)", s, ms % 1000 / 10, s / 3600, s / 60 % 60, s % 60);
        }
        "kmsg" => crate::log::last(usize::MAX, |l| {
            o.push_str(l);
            o.push('\n');
        }),
        "mounts" => {
            if fs::mounted() {
                o.push_str("/dev/vda / vfat rw\n");
            }
            o.push_str("proc /proc proc ro\nsysfs /sys sysfs ro\ndevfs /dev devfs rw\n");
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
        // T55: the flags once (about 70 tokens per core in a cat)
        let _ = writeln!(o, "processor\t: {}\nmodel name\t: {}\ncpu MHz\t\t: {}.{:03}\ncpu cores\t: {}\nrole\t\t: {}\nflags\t\t:{}\n",
            i, unsafe { (*core::ptr::addr_of!(sys::BRAND)).as_str() }, mhz / 1000, mhz % 1000, n, crate::sysfs::role(i),
            if i == 0 { fl.as_str() } else { " as processor 0" });
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
    let _ = writeln!(o, "filesystem   sysfs: /sys, read-only (but /sys/power/state); devfs: /dev");
    let _ = writeln!(o, "\nPCI devices: /proc/bus/pci/devices and /sys/bus/pci/devices");
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

/// One function on PCI bus 0: (device, function, vendor, device id, class, subclass, prog-if, driver).
pub type Pci = (u8, u8, u16, u16, u8, u8, u8, &'static str);

/// PCI bus 0 through the legacy config ports (also /sys/bus/pci/devices, sysfs.rs).
pub fn pci_scan() -> Vec<Pci> {
    use virtio_drivers::transport::pci::bus::PciRoot;
    use virtio_drivers::transport::pci::virtio_device_type;
    let blk = crate::blk::location();
    let root = PciRoot::new(crate::blk::PortCam);
    root.enumerate_bus(0).map(|(df, info)| {
        let drv = match virtio_device_type(&info) {
            Some(virtio_drivers::transport::DeviceType::Block) if blk.is_some_and(|(d, f, _)| (d, f) == (df.device, df.function)) => "virtio-blk",
            Some(_) => "virtio, no driver",
            None => "no driver",
        };
        (df.device, df.function, info.vendor_id, info.device_id, info.class, info.subclass, info.prog_if, drv)
    }).collect()
}

/// "bus:dev.fn vendor:device class, driver".
fn pci(o: &mut String) {
    for (d, f, v, id, c, sc, _, drv) in pci_scan() {
        let _ = writeln!(o, "00:{:02x}.{} {:04x}:{:04x} {} (class {:02x}{:02x}), {}", d, f, v, id, class(c, sc), c, sc, drv);
    }
}

/// Boot self-check of the path table (no hardware involved).
pub fn selftest() -> bool {
    entry("") == Some((true, 0)) && entry("bus/pci") == Some((true, 0)) && entry("nope").is_none()
        && list("").is_some_and(|v| v.iter().any(|e| e.0 == "bus" && e.1) && v.iter().any(|e| e.0 == "uptime" && !e.1))
        && list("bus").is_some_and(|v| v.len() == 1 && v[0].0 == "pci")
        && list("uptime").is_none() && read("version").is_some() && read("mounts").is_some()
        && fs::mount_of("proc/bus") == (fs::Mount::Proc, "bus") && fs::mount_of("SYS") == (fs::Mount::Sys, "")
        && fs::mount_of("etc/passwd").0 == fs::Mount::Etc && fs::mount_of("etc/hostname").0 == fs::Mount::Fat
        && fs::mount_of("system").0 == fs::Mount::Fat && crate::sysfs::selftest()
}
