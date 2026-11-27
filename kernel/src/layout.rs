// T55: the disk layout. ringzero's own files live under /var/lib/ringzero (agentd's stores and index, the
// hibernation image, the boot snapshot cache), off the Linux paths. init() runs from fs::init right after the
// mount, before agentd, the agent and hib touch the disk: it migrates a disk written by an older kernel (FAT
// /sys/...), makes /etc (hostname) and /tmp. The migration never renames (fatfs frees the old directory entry
// before it writes the new one, so a crash in between would lose the file): copy, read back and verify (size and
// FNV-1a 64), commit marker, then remove the sources. A crash at any step ends, at a later boot, in the same state.
use crate::fs;
use crate::klog;
use crate::log::SBuf;
use crate::mstore::{fnv, FNV0};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub const ROOT: &str = "/var/lib/ringzero";
pub const JOURNAL: &str = "/var/lib/ringzero/mem/journal.log";
pub const FACTS: &str = "/var/lib/ringzero/mem/facts.log";
pub const EPISODES: &str = "/var/lib/ringzero/mem/episodes.log";
pub const INDEX: &str = "/var/lib/ringzero/mem/index.bin";
pub const HIB: &str = "/var/lib/ringzero/hib/image.bin";
pub const BOOTSNAP: &str = "/var/lib/ringzero/bootsnap.bin";
pub const LEGACY: &str = "/var/lib/ringzero/legacy";
const MARK: &str = "/var/lib/ringzero/layout";
const MARK_TEXT: &[u8] = b"ringzero-layout 2\n";
/// The old place of each store (layout 1) and its new one.
const MOVES: [(&str, &str); 4] = [("/sys/mem/journal.log", JOURNAL), ("/sys/mem/facts.log", FACTS),
    ("/sys/mem/episodes.log", EPISODES), ("/sys/mem/index.bin", INDEX)];
/// Not copied: a cache, and an image whose key (the boot prefix) this kernel cannot match.
const DROPS: [&str; 2] = ["/sys/bootsnap.bin", "/sys/hib/image.bin"];
/// Crash test builds (`make iso LAYOUT_STOP=N`): halt after the Nth migration step (a copy, the commit, a removal).
const STOP: u32 = crate::agent::env_num(option_env!("RINGZERO_LAYOUT_STOP"), 0, 99);

static mut PENDING: Option<&'static str> = None;
static mut HOST: SBuf<64> = SBuf::new();

/// Why the migration could not finish this boot (agentd then keeps its stores in RAM), or None.
pub fn pending() -> Option<&'static str> {
    unsafe { PENDING }
}

/// /etc/hostname's first line, read at boot.
pub fn hostname() -> &'static str {
    unsafe { (*core::ptr::addr_of!(HOST)).as_str() }
}

/// fs::init, after the mount.
pub fn init() {
    unsafe { fs::RAW = true };
    let r = migrate();
    unsafe { fs::RAW = false };
    if let Err(e) = r {
        unsafe { PENDING = Some(e) };
        klog!(b'W', "layout: migration not finished ({}): every old file is kept, agentd runs without disk stores this boot, the next boot retries", e);
    }
    let _ = fs::mkdir("/tmp");
    if fs::entry("/etc/hostname").is_err() {
        let _ = fs::write("/etc/hostname", b"ringzero\n");
    }
    let h = fs::read("/etc/hostname", 0, 256).map(|(b, _)| b).unwrap_or_default();
    let name = core::str::from_utf8(&h).unwrap_or("").lines().next().unwrap_or("").trim();
    unsafe { (*core::ptr::addr_of_mut!(HOST)).push(if name.is_empty() { b"ringzero" } else { name.as_bytes() }) };
    klog!(b'I', "host: {} (/etc/hostname)", hostname());
}

/// Virtual files in the real /etc: /etc/os-release and /etc/passwd (read-only).
pub fn etc_virtual(name: &str) -> Option<Vec<u8>> {
    match name {
        "os-release" => Some(format!("NAME=ringzero\nID=ringzero\nID_LIKE=unix\nVERSION=\"{} {}\"\nVERSION_ID={}\nPRETTY_NAME=\"{}\"\n",
            crate::VERSION, crate::CHANNEL, crate::VERSION, crate::EDITION).into_bytes()),
        "passwd" => Some(b"root:x:0:0:root:/:/bin/sh\n".to_vec()),
        _ => None,
    }
}

/// hib::boot when no image resumes: /tmp starts empty (a resumed conversation keeps it).
pub fn tmp_clean() {
    let Ok(ents) = fs::list("/tmp") else { return };
    let n = ents.iter().filter(|(name, ..)| fs::rm(&format!("/tmp/{}", name), true).is_ok()).count();
    if n > 0 {
        klog!(b'I', "tmp: {} entries removed from /tmp (fresh boot)", n);
    }
}

fn halt_at(step: &mut u32) {
    *step += 1;
    if STOP != 0 && *step == STOP {
        klog!(b'W', "layout: crash test build: halting after migration step {}", STOP);
        loop {
            unsafe { core::arch::asm!("cli; hlt") };
        }
    }
}

/// Every file under a FAT path (a file is itself).
fn files(p: &str, out: &mut Vec<String>) -> Result<(), &'static str> {
    if !fs::entry(p)?.0 {
        out.push(String::from(p));
        return Ok(());
    }
    for (name, _, _) in fs::list(p)? {
        files(&format!("{}/{}", p, name), out)?;
    }
    Ok(())
}

/// `t` holds exactly `data` (size and FNV-1a 64 read back from the disk).
fn same(data: &[u8], t: &str) -> bool {
    fs::read(t, 0, usize::MAX).is_ok_and(|(b, _)| b.len() == data.len() && fnv(FNV0, &b) == fnv(FNV0, data))
}

fn migrate() -> Result<(), &'static str> {
    let marked = fs::read(MARK, 0, 64).is_ok_and(|(b, _)| b.starts_with(&MARK_TEXT[..17]));
    let olds: Vec<(&str, u32)> = ["sys", "proc", "dev"].into_iter()
        .filter_map(|d| fs::entry(&format!("/{}", d)).ok().map(|(_, _, mt)| (d, mt))).collect();
    if olds.is_empty() {
        if !marked {
            fs::mkdir("/var/lib/ringzero/mem")?;
            fs::write(MARK, MARK_TEXT)?;
            fs::sync()?;
            klog!(b'I', "layout: new disk, ringzero's data goes to {}", ROOT);
        }
        return Ok(());
    }
    let mut srcs = Vec::new();
    for (d, _) in &olds {
        files(&format!("/{}", d), &mut srcs)?;
    }
    klog!(b'I', "layout: {} files in FAT {} ({}): moving them to {}", srcs.len(),
        olds.iter().map(|(d, _)| format!("/{}", d)).collect::<Vec<_>>().join(", "),
        if marked { "the commit marker exists: an interrupted migration ends, or an older kernel wrote them" } else { "the layout of ringzero 0.1" }, ROOT);
    // (source, target or None = drop)
    let mut plan: Vec<(String, Option<String>)> = Vec::new();
    let (mut step, mut copied, mut bytes, mut legacy) = (0u32, 0u32, 0u64, None::<String>);
    for src in srcs {
        let low = src.to_ascii_lowercase();
        if DROPS.contains(&low.as_str()) {
            plan.push((src, None));
            continue;
        }
        let prim = match MOVES.iter().find(|(o, _)| *o == low) {
            Some((_, t)) => String::from(*t),
            None => format!("{}{}", LEGACY, src),
        };
        let data = fs::read(&src, 0, usize::MAX)?.0;
        let t = if marked && !same(&data, &prim) {
            // an older kernel wrote it after the commit: kept apart, never merged on its own (/agentd merge-legacy)
            let (first, rest) = src[1..].split_once('/').unwrap_or((&src[1..], ""));
            let mt = olds.iter().find(|(d, _)| d.eq_ignore_ascii_case(first)).map_or(0, |o| o.1);
            let dir = format!("{}/{}-{}", LEGACY, first.to_ascii_lowercase(), fs::unix_of(mt));
            let t = format!("{}/{}", dir, rest);
            legacy.get_or_insert(dir);
            t
        } else {
            prim
        };
        if !same(&data, &t) {
            fs::write_parts(&t, &[&data])?;
            if !same(&data, &t) {
                return Err("a copy does not read back the same");
            }
            klog!(b'I', "layout: copied {} -> {} ({} B, verified)", src, t, data.len());
            copied += 1;
            bytes += data.len() as u64;
            halt_at(&mut step);
        }
        plan.push((src, Some(t)));
    }
    if !marked {
        fs::write(MARK, MARK_TEXT)?;
        fs::sync()?;
        klog!(b'I', "layout: commit marker {} written", MARK);
        halt_at(&mut step);
    }
    // after the commit an error only leaves old files behind: the next boot finds them verified and removes them
    let mut kept = 0;
    for (src, t) in &plan {
        let ok = match t {
            Some(t) => fs::read(src, 0, usize::MAX).is_ok_and(|(d, _)| same(&d, t)),
            None => true,
        };
        if ok && fs::rm(src, false).is_ok() {
            halt_at(&mut step);
        } else {
            kept += 1;
        }
    }
    if kept == 0 {
        for (d, _) in &olds {
            let _ = fs::rm(&format!("/{}", d), true);
        }
    }
    let _ = fs::sync();
    let dropped = plan.iter().filter(|p| p.1.is_none()).count();
    klog!(b'I', "layout: migrated {} files, {} KiB, {} old caches dropped (boot cache, hibernation image){}", copied, bytes >> 10, dropped,
        if kept > 0 { ", some old files are still in place (removed next boot)" } else { "" });
    if let Some(l) = legacy {
        klog!(b'W', "layout: files an older kernel wrote are kept in {} (not merged; /agentd merge-legacy merges them)", l);
    }
    Ok(())
}
