// T55: /dev, the virtual device directory (fs.rs mount_of routes every path under /dev here). Programs read and
// write the devices through wasm.rs (Fd::Dev, Fd::Tty); the shell's and the agent's cat refuse them (fs::DEVICE)
// but /dev/null, and !write / write_file to a data device is accepted and discarded (the console prints it).
use alloc::string::String;
use alloc::vec::Vec;

pub const NAMES: [&str; 7] = ["console", "null", "random", "tty", "urandom", "vda", "zero"];

/// (is_dir, size) of a path inside /dev: devices list with size 0.
pub fn entry(r: &str) -> Option<(bool, u64)> {
    if r.is_empty() {
        return Some((true, 0));
    }
    NAMES.contains(&r).then_some((false, 0))
}

pub fn list(r: &str) -> Option<Vec<(String, bool, u64)>> {
    r.is_empty().then(|| NAMES.iter().map(|n| (String::from(*n), false, 0)).collect())
}

/// A write is accepted (everything but the disk).
pub fn writable(r: &str) -> bool {
    NAMES.contains(&r) && r != "vda"
}

pub fn is_console(r: &str) -> bool {
    r == "tty" || r == "console"
}
