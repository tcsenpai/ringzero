// T55: /sys, a read-only virtual sysfs-like tree generated at read time (fs.rs mount_of routes every path under
// /sys here, as the path inside /sys): one value per file, familiar Linux paths. /sys/power/state is the one file
// that takes a write (`disk` hibernates and powers off, as on Linux). ringzero's own data is in /var/lib/ringzero.
use crate::{fs, procfs, smp};
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::sync::atomic::Ordering::Relaxed;

/// The writable file, as a path inside /sys.
pub const POWER: &str = "power/state";

fn addr(d: u8, f: u8) -> String {
    format!("0000:00:{:02x}.{}", d, f)
}

/// Every path of the tree (directories end with '/'), generated from the hardware now.
fn nodes() -> Vec<String> {
    let mut v: Vec<String> = ["class/", "class/block/", "bus/", "bus/pci/", "bus/pci/devices/", "devices/", "devices/system/",
        "devices/system/cpu/", "devices/system/cpu/online", "devices/system/cpu/possible", "devices/system/cpu/inference",
        "power/", POWER].iter().map(|s| String::from(*s)).collect();
    if crate::blk::location().is_some() {
        v.extend(["class/block/vda/", "class/block/vda/size", "class/block/vda/ro", "class/block/vda/device"].map(String::from));
    }
    for (d, f, ..) in procfs::pci_scan() {
        let a = addr(d, f);
        v.push(format!("bus/pci/devices/{}/", a));
        v.extend(["vendor", "device", "class", "driver"].map(|k| format!("bus/pci/devices/{}/{}", a, k)));
    }
    for i in 0..smp::MAX_THREADS.load(Relaxed) {
        v.push(format!("devices/system/cpu/cpu{}/", i));
        v.push(format!("devices/system/cpu/cpu{}/role", i));
    }
    v
}

/// (is_dir, size) of a path inside /sys ("" = /sys), or None if it does not exist.
pub fn entry(r: &str) -> Option<(bool, u64)> {
    fs::tree_entry(&nodes(), r, read)
}

/// The entries of a /sys directory as (name, is_dir, size).
pub fn list(r: &str) -> Option<Vec<(String, bool, u64)>> {
    fs::tree_list(&nodes(), r, read)
}

/// What core i does (also the role line of /proc/cpuinfo).
pub fn role(i: u64) -> &'static str {
    match i {
        0 => "BSP: shell, agent loop, engine slice",
        _ if i < smp::threads() => "engine worker",
        _ => "engine worker, not used for inference now (!set cores)",
    }
}

fn range(n: u64) -> String {
    if n <= 1 { String::from("0") } else { format!("0-{}", n - 1) }
}

/// The content of a /sys file, generated now (one value and a newline).
pub fn read(r: &str) -> Option<Vec<u8>> {
    let cpus = smp::MAX_THREADS.load(Relaxed);
    let v: String = match r {
        "devices/system/cpu/online" | "devices/system/cpu/possible" => range(cpus),
        "devices/system/cpu/inference" => range(smp::threads()),
        POWER => String::from("disk"),
        "class/block/vda/size" => format!("{}", crate::blk::size()? / 512),
        "class/block/vda/ro" => String::from("0"),
        "class/block/vda/device" => {
            let (d, f, _) = crate::blk::location()?;
            addr(d, f)
        }
        _ => {
            if let Some(n) = r.strip_prefix("devices/system/cpu/cpu").and_then(|x| x.strip_suffix("/role")) {
                let i: u64 = n.parse().ok().filter(|&i| i < cpus)?;
                String::from(role(i))
            } else {
                let rest = r.strip_prefix("bus/pci/devices/")?;
                let (a, k) = rest.split_once('/')?;
                let (d, f, ven, dev, c, sc, pi, drv) = procfs::pci_scan().into_iter().find(|p| addr(p.0, p.1) == a)?;
                let _ = (d, f);
                match k {
                    "vendor" => format!("0x{:04x}", ven),
                    "device" => format!("0x{:04x}", dev),
                    "class" => format!("0x{:02x}{:02x}{:02x}", c, sc, pi),
                    "driver" => String::from(match drv {
                        "virtio-blk" => "virtio-blk",
                        _ => "none",
                    }),
                    _ => return None,
                }
            }
        }
    };
    let mut o = v.into_bytes();
    o.push(b'\n');
    Some(o)
}

/// `echo disk > /sys/power/state`: hibernate (the conversation goes to /var/lib/ringzero/hib) and power off.
/// Linux's other states have no ringzero counterpart: suspend to RAM needs ACPI S3 and device resume.
pub fn power_write(data: &[u8]) -> Result<(), &'static str> {
    match core::str::from_utf8(data).unwrap_or("").trim() {
        "disk" => crate::hib::power_disk(),
        "mem" | "freeze" | "standby" => Err("not supported: ringzero has no suspend to RAM (write disk to hibernate and power off)"),
        _ => Err("invalid argument: /sys/power/state takes disk"),
    }
}

/// Boot self-check of the path table.
pub fn selftest() -> bool {
    entry("") == Some((true, 0)) && entry("devices/system/cpu") == Some((true, 0)) && entry("nope").is_none()
        && read("devices/system/cpu/online").is_some_and(|v| v.ends_with(b"\n")) && range(8) == "0-7"
        && list("power").is_some_and(|v| v == vec![(String::from("state"), false, 5)])
        && power_write(b"mem").is_err() && power_write(b"x").is_err()
}
