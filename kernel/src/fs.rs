// FAT32 on the virtio disk (fatfs 0.4, no_std + alloc), mounted at / at boot. One volume, no partition
// table (`make disk` formats the whole image). Every path from the shell or the agent goes through check():
// absolute, no "." or ".." components, bounded length. Writes are write-through (blk.rs); `!sync`,
// poweroff and reboot flush the FS info sector and the device cache. File times come from the CMOS RTC
// (sys::rtc), so a file changed on the host (mtools) is told apart by its size and mtime (agentd, M4.5).
// T54: /proc is a virtual read-only directory (procfs.rs) in front of the volume: ls, read, entry and list serve
// it, every write path fails with ROFS. T55: one mount table (mount_of): /proc, /sys (sysfs.rs), /dev (devfs.rs)
// and /etc/os-release, /etc/passwd (layout.rs) are virtual, everything else is the FAT volume.
use crate::blk::{self, Disk, DiskErr};
use crate::klog;
use alloc::string::String;
use alloc::vec::Vec;
use fatfs::{Error, FileSystem, FsOptions, Read, Seek, SeekFrom, Write};

pub const PATH_MAX: usize = 255;
/// Largest file written by !write / write_file.
pub const WRITE_MAX: usize = 64 << 10;

/// fatfs time provider: the CMOS RTC (QEMU: host time). Years outside FAT's 1980..2107 are clamped.
#[derive(Debug, Clone, Copy)]
pub struct RtcTime;
impl fatfs::TimeProvider for RtcTime {
    fn get_current_date(&self) -> fatfs::Date {
        self.get_current_date_time().date
    }
    fn get_current_date_time(&self) -> fatfs::DateTime {
        let (y, mo, d, h, mi, s) = crate::sys::rtc();
        fatfs::DateTime::new(fatfs::Date::new(y.clamp(1980, 2107) as u16, mo as u16, d as u16), fatfs::Time::new(h as u16, mi as u16, s as u16, 0))
    }
}

type Fs = FileSystem<Disk, RtcTime, fatfs::LossyOemCpConverter>;
fn opts() -> FsOptions<RtcTime, fatfs::LossyOemCpConverter> {
    FsOptions::new().time_provider(RtcTime)
}
static mut FS: Option<Fs> = None;

fn fs() -> Result<&'static Fs, &'static str> {
    unsafe { (*core::ptr::addr_of!(FS)).as_ref() }.ok_or("no disk mounted")
}

fn err(e: Error<DiskErr>) -> &'static str {
    match e {
        Error::NotFound => "not found",
        Error::AlreadyExists => "already exists",
        Error::DirectoryIsNotEmpty => "directory is not empty",
        Error::NotEnoughSpace => "disk full",
        Error::InvalidFileNameLength => "name empty or too long",
        Error::UnsupportedFileNameCharacter => "unsupported character in a name",
        Error::InvalidInput => "not a directory, or a directory where a file was expected",
        Error::CorruptedFileSystem => "corrupted file system",
        _ => "disk I/O error",
    }
}

/// Boot: find the virtio disk and mount it.
pub fn init() {
    let Some(b) = blk::probe() else {
        klog!(b'W', "disk: no virtio-blk device (make disk, then a run target attaches build/disk.img); file commands are off");
        return;
    };
    match FileSystem::new(Disk::new(b), opts()) {
        Ok(f) => {
            unsafe { FS = Some(f) };
            let mut o = crate::log::SBuf::<160>::new();
            df(&mut o);
            klog!(b'I', "disk: mounted at /: {}", o.as_str());
            crate::layout::init(); // T55: data move, /etc and /tmp, before agentd, agent and hib touch the disk
        }
        Err(e) => klog!(b'E', "disk: mount failed ({}); not FAT? `make disk` formats a new image", err(e)),
    }
}

pub fn mounted() -> bool {
    fs().is_ok()
}

/// Validate an absolute path; returns it relative to / ("" = the root), trailing '/' dropped.
pub fn check(path: &str) -> Result<&str, &'static str> {
    if !path.starts_with('/') {
        return Err("path must be absolute (start with /)");
    }
    if path.len() > PATH_MAX {
        return Err("path too long");
    }
    let rel = path.trim_matches('/');
    for c in rel.split('/').filter(|_| !rel.is_empty()) {
        match c {
            "" => return Err("empty path component (//)"),
            "." | ".." => return Err("'.' and '..' are not allowed in paths"),
            _ if c.bytes().any(|b| b < 0x20 || b == b'\\') => return Err("control character or backslash in path"),
            _ => {}
        }
    }
    Ok(rel)
}

/// T54: /proc is virtual (procfs.rs) and read-only: every change under it fails with this error.
pub const ROFS: &str = "read-only file system";
/// T55: cat of a device (the shell's and the agent's): an endless /dev/zero never reaches a tool result.
pub const DEVICE: &str = "is a device: read it with a program, e.g. head -c 16 /dev/urandom";

/// T55: the mounts of the tree.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mount {
    Proc,
    Sys,
    Dev,
    Etc, // the two virtual files in the real /etc
    Fat,
}

/// T55: set by layout.rs while it migrates an old disk, so the FAT directories /sys, /proc and /dev are reachable.
pub static mut RAW: bool = false;

/// The mount of a checked path (relative to /) and the path inside it ("" = the mount's root).
pub fn mount_of(rel: &str) -> (Mount, &str) {
    if unsafe { RAW } {
        return (Mount::Fat, rel);
    }
    let (first, rest) = rel.split_once('/').unwrap_or((rel, ""));
    for (n, m) in [("proc", Mount::Proc), ("sys", Mount::Sys), ("dev", Mount::Dev)] {
        if first.eq_ignore_ascii_case(n) {
            return (m, rest);
        }
    }
    if first.eq_ignore_ascii_case("etc") && crate::layout::etc_virtual(rest).is_some() {
        return (Mount::Etc, rest);
    }
    (Mount::Fat, rel)
}

/// (is_dir, size) of a path on a virtual mount.
fn v_entry(m: Mount, r: &str) -> Option<(bool, u64)> {
    match m {
        Mount::Proc => crate::procfs::entry(r),
        Mount::Sys => crate::sysfs::entry(r),
        Mount::Dev => crate::devfs::entry(r),
        Mount::Etc => crate::layout::etc_virtual(r).map(|v| (false, v.len() as u64)),
        Mount::Fat => None,
    }
}

/// The content of a virtual file (devices: empty, fs::read refuses them).
fn v_read(m: Mount, r: &str) -> Option<Vec<u8>> {
    match m {
        Mount::Proc => crate::procfs::read(r),
        Mount::Sys => crate::sysfs::read(r),
        Mount::Etc => crate::layout::etc_virtual(r),
        _ => None,
    }
}

/// A virtual file's content; a virtual directory is the same error fatfs gives for one, a device the DEVICE one.
pub fn virt_read(path: &str) -> Result<Vec<u8>, &'static str> {
    let (m, r) = mount_of(check(path)?);
    match v_entry(m, r) {
        Some((true, _)) => Err("not a directory, or a directory where a file was expected"),
        Some(_) if m == Mount::Dev => if r == "null" { Ok(Vec::new()) } else { Err(DEVICE) },
        Some(_) => v_read(m, r).ok_or("not found"),
        None => Err("not found"),
    }
}

/// T55: a virtual tree given as its paths (directories end with '/'): (is_dir, size) of `r`.
pub fn tree_entry<S: AsRef<str>>(nodes: &[S], r: &str, read: impl Fn(&str) -> Option<Vec<u8>>) -> Option<(bool, u64)> {
    if r.is_empty() || nodes.iter().any(|e| e.as_ref().strip_suffix('/') == Some(r)) {
        return Some((true, 0));
    }
    nodes.iter().any(|e| e.as_ref() == r).then(|| read(r).map_or(0, |v| v.len() as u64)).map(|n| (false, n))
}

/// The entries of the directory `r` of a virtual tree as (name, is_dir, size).
pub fn tree_list<S: AsRef<str>>(nodes: &[S], r: &str, read: impl Fn(&str) -> Option<Vec<u8>>) -> Option<Vec<(String, bool, u64)>> {
    if !tree_entry(nodes, r, &read)?.0 {
        return None;
    }
    let pre = if r.is_empty() { String::new() } else { alloc::format!("{}/", r) };
    Some(nodes.iter().filter_map(|e| {
        let e = e.as_ref();
        let rest = e.strip_prefix(pre.as_str())?;
        let (name, dir) = match rest.strip_suffix('/') {
            Some(d) => (d, true),
            None => (rest, false),
        };
        if name.is_empty() || name.contains('/') {
            return None;
        }
        Some((String::from(name), dir, if dir { 0 } else { read(e).map_or(0, |v| v.len() as u64) }))
    }).collect())
}

/// `check` plus: Err(ROFS) for a path on a virtual mount.
fn check_rw(path: &str) -> Result<&str, &'static str> {
    let rel = check(path)?;
    if mount_of(rel).0 != Mount::Fat { Err(ROFS) } else { Ok(rel) }
}

/// T55: /tmp, /var/lib/ringzero (ringzero's data), /lib (CPython's stdlib) and the virtual mounts are not the
/// user's documents: the file indexer skips them.
pub fn indexed(path: &str) -> bool {
    let Ok(rel) = check(path) else { return false };
    let under = |d: &str| rel.len() >= d.len() && rel[..d.len()].eq_ignore_ascii_case(d) && matches!(rel.as_bytes().get(d.len()), None | Some(b'/'));
    mount_of(rel).0 == Mount::Fat && !["tmp", "lib", "var/lib/ringzero"].iter().any(|d| under(d))
}

/// The checked path in one spelling ("/a/b"), the key the indexer uses.
pub fn canon(path: &str) -> Result<String, &'static str> {
    Ok(alloc::format!("/{}", check(path)?))
}

/// `ls`: one entry per line, directories end with '/'.
pub fn ls(path: &str, o: &mut impl core::fmt::Write) -> Result<usize, &'static str> {
    let mut n = 0;
    for (name, dir, size) in list(path)? {
        let _ = if dir { write!(o, "{}{}/", if n > 0 { "\n" } else { "" }, name) } else {
            write!(o, "{}{}  {} B", if n > 0 { "\n" } else { "" }, name, size)
        };
        n += 1;
    }
    Ok(n)
}

/// Read at most `max` bytes from `off`; returns (bytes, file size).
pub fn read(path: &str, off: u64, max: usize) -> Result<(Vec<u8>, u64), &'static str> {
    let rel = check(path)?;
    if rel.is_empty() {
        return Err("/ is a directory");
    }
    if mount_of(rel).0 != Mount::Fat {
        let v = virt_read(path)?;
        let (size, a) = (v.len() as u64, (off as usize).min(v.len()));
        return Ok((v[a..(a + max).min(v.len())].to_vec(), size));
    }
    let mut f = fs()?.root_dir().open_file(rel).map_err(err)?;
    let size = f.seek(SeekFrom::End(0)).map_err(err)?;
    f.seek(SeekFrom::Start(off.min(size))).map_err(err)?;
    let mut v = alloc::vec![0u8; max.min(size.saturating_sub(off) as usize)];
    f.read_exact(&mut v).map_err(err)?;
    Ok((v, size))
}

/// Create or replace a file; missing parent directories are created. T55: /sys/power/state and the data
/// devices take a write (the shell's !write and the agent's write_file, which asks first for the power file).
pub fn write(path: &str, data: &[u8]) -> Result<(), &'static str> {
    match mount_of(check(path)?) {
        (Mount::Sys, r) if r == crate::sysfs::POWER => return crate::sysfs::power_write(data),
        (Mount::Dev, r) if crate::devfs::writable(r) => {
            if crate::devfs::is_console(r) {
                crate::console::write_bytes(data);
            }
            return Ok(());
        }
        _ => {}
    }
    let rel = check_rw(path)?;
    if rel.is_empty() {
        return Err("/ is a directory");
    }
    if data.len() > WRITE_MAX {
        return Err("file too large (64 KiB at most)");
    }
    if let Some((parent, _)) = rel.rsplit_once('/') {
        mkdirs(parent)?;
    }
    let mut f = fs()?.root_dir().create_file(rel).map_err(err)?;
    f.truncate().map_err(err)?;
    f.write_all(data).map_err(err)?;
    f.flush().map_err(err)
}

/// mkdir -p. Ok(false) if it was already there.
pub fn mkdir(path: &str) -> Result<bool, &'static str> {
    let rel = check_rw(path)?;
    if rel.is_empty() {
        return Ok(false);
    }
    if fs()?.root_dir().open_dir(rel).is_ok() {
        return Ok(false);
    }
    mkdirs(rel)?;
    Ok(true)
}

/// Create each level of rel (fatfs create_dir makes only the last one; it opens existing ones).
fn mkdirs(rel: &str) -> Result<(), &'static str> {
    let root = fs()?.root_dir();
    for (i, _) in rel.match_indices('/').chain([(rel.len(), "")]) {
        root.create_dir(&rel[..i]).map_err(err)?;
    }
    Ok(())
}

/// What rm would delete: Ok((is_dir, size or entries inside, recursively)).
pub fn stat(path: &str) -> Result<(bool, u64), &'static str> {
    let rel = check_rw(path)?;
    if rel.is_empty() {
        return Err("refusing to remove /");
    }
    let root = fs()?.root_dir();
    match root.open_dir(rel) {
        Ok(d) => Ok((true, count(&d)?)),
        Err(_) => {
            let mut f = root.open_file(rel).map_err(err)?;
            Ok((false, f.seek(SeekFrom::End(0)).map_err(err)?))
        }
    }
}

type Dir = fatfs::Dir<'static, Disk, RtcTime, fatfs::LossyOemCpConverter>;

fn count(d: &Dir) -> Result<u64, &'static str> {
    let mut n = 0;
    for e in d.iter() {
        let e = e.map_err(err)?;
        let name = e.file_name();
        if name != "." && name != ".." {
            n += 1 + if e.is_dir() { count(&e.to_dir())? } else { 0 };
        }
    }
    Ok(n)
}

fn rm_tree(d: &Dir) -> Result<(), &'static str> {
    let names: Vec<(String, bool)> = d.iter().filter_map(|e| e.ok()).map(|e| (e.file_name(), e.is_dir()))
        .filter(|(n, _)| n != "." && n != "..").collect();
    for (name, dir) in names {
        if dir {
            rm_tree(&d.open_dir(&name).map_err(err)?)?;
        }
        d.remove(&name).map_err(err)?;
    }
    Ok(())
}

/// Remove a file or a directory (recursive = with everything in it).
pub fn rm(path: &str, recursive: bool) -> Result<(), &'static str> {
    let rel = check_rw(path)?;
    if rel.is_empty() {
        return Err("refusing to remove /");
    }
    let root = fs()?.root_dir();
    if recursive {
        if let Ok(d) = root.open_dir(rel) {
            rm_tree(&d)?;
        }
    }
    root.remove(rel).map_err(err)
}

/// Read from `off` until buf is full or the file ends; returns (bytes read, file size). No heap.
pub fn read_into(path: &str, off: u64, buf: &mut [u8]) -> Result<(usize, u64), &'static str> {
    let mut f = open(path)?;
    let size = len(&mut f)?;
    Ok((read_at(&mut f, off, buf)?, size))
}

/// Create or replace a file from several pieces (no size limit: agentd's index and the boot snapshot).
pub fn write_parts(path: &str, parts: &[&[u8]]) -> Result<u64, &'static str> {
    let rel = check_rw(path)?;
    if let Some((parent, _)) = rel.rsplit_once('/') {
        mkdirs(parent)?;
    }
    let mut f = fs()?.root_dir().create_file(rel).map_err(err)?;
    f.truncate().map_err(err)?;
    let mut n = 0;
    for p in parts {
        f.write_all(p).map_err(err)?;
        n += p.len() as u64;
    }
    f.flush().map_err(err)?;
    Ok(n)
}

/// Append to a file (created with its parents if missing); returns the size after the append.
pub fn append(path: &str, data: &[u8]) -> Result<u64, &'static str> {
    let rel = check_rw(path)?;
    if let Some((parent, _)) = rel.rsplit_once('/') {
        mkdirs(parent)?;
    }
    let root = fs()?.root_dir();
    let mut f = match root.open_file(rel) {
        Ok(f) => f,
        Err(Error::NotFound) => root.create_file(rel).map_err(err)?,
        Err(e) => return Err(err(e)),
    };
    let end = f.seek(SeekFrom::End(0)).map_err(err)?;
    f.write_all(data).map_err(err)?;
    f.flush().map_err(err)?;
    Ok(end + data.len() as u64)
}

/// Cut a file at `len` bytes (recovery of a torn append).
pub fn truncate(path: &str, len: u64) -> Result<(), &'static str> {
    let mut f = fs()?.root_dir().open_file(check_rw(path)?).map_err(err)?;
    f.seek(SeekFrom::Start(len)).map_err(err)?;
    f.truncate().map_err(err)?;
    f.flush().map_err(err)
}

/// FAT mtime packed as DOS date << 16 | time (2 s resolution): only compared for equality.
fn mtime(d: fatfs::DateTime) -> u32 {
    let (dt, t) = (d.date, d.time);
    ((dt.year.saturating_sub(1980) as u32) << 25) | (dt.month as u32) << 21 | (dt.day as u32) << 16
        | (t.hour as u32) << 11 | (t.min as u32) << 5 | (t.sec as u32 / 2)
}

/// A FAT mtime (`mtime` packing, local time from the RTC) as seconds since the epoch (0 stays 0).
pub fn unix_of(mt: u32) -> u64 {
    if mt == 0 {
        return 0;
    }
    let d = crate::sys::days((mt >> 25) + 1980, mt >> 21 & 15, mt >> 16 & 31);
    (d * 86_400) as u64 + (mt >> 11 & 31) as u64 * 3600 + (mt >> 5 & 63) as u64 * 60 + (mt & 31) as u64 * 2
}

/// (size, mtime) of a file.
pub fn meta(path: &str) -> Result<(u64, u32), &'static str> {
    match entry(path)? {
        (false, size, mt) => Ok((size, mt)),
        _ => Err("not found"),
    }
}

/// (is_dir, size, mtime) of a file or a directory; / is a directory.
pub fn entry(path: &str) -> Result<(bool, u64, u32), &'static str> {
    let rel = check(path)?;
    if rel.is_empty() {
        fs()?;
        return Ok((true, 0, 0));
    }
    let (m, r) = mount_of(rel);
    if m != Mount::Fat {
        return v_entry(m, r).map(|(d, n)| (d, n, 0)).ok_or("not found");
    }
    let (dir, name) = rel.rsplit_once('/').unwrap_or(("", rel));
    let root = fs()?.root_dir();
    let d = if dir.is_empty() { root } else { root.open_dir(dir).map_err(err)? };
    for e in d.iter() {
        let e = e.map_err(err)?;
        if e.file_name().eq_ignore_ascii_case(name) {
            return Ok((e.is_dir(), e.len(), mtime(e.modified())));
        }
    }
    Err("not found")
}

// ---- M5b: what the WASI runtime (wasm.rs) needs beyond the shell commands ----

/// A directory's entries as (name, is_dir, size).
pub fn list(path: &str) -> Result<Vec<(String, bool, u64)>, &'static str> {
    let rel = check(path)?;
    let (m, r) = mount_of(rel);
    if m != Mount::Fat {
        return match v_entry(m, r) {
            Some((true, _)) => match m {
                Mount::Proc => crate::procfs::list(r),
                Mount::Sys => crate::sysfs::list(r),
                _ => crate::devfs::list(r),
            }.ok_or("not found"),
            Some(_) => Err("not a directory, or a directory where a file was expected"),
            None => Err("not found"),
        };
    }
    let root = fs()?.root_dir();
    let d = if rel.is_empty() { root } else { root.open_dir(rel).map_err(err)? };
    let mut v = Vec::new();
    for e in d.iter() {
        let e = e.map_err(err)?;
        let name = e.file_name();
        if name != "." && name != ".." {
            v.push((name, e.is_dir(), e.len()));
        }
    }
    if rel.is_empty() && !unsafe { RAW } {
        // T54, T55: the virtual mounts show in `ls /` (a FAT directory of the same name is moved away by layout.rs)
        v.retain(|e| !["proc", "sys", "dev"].iter().any(|n| e.0.eq_ignore_ascii_case(n)));
        v.extend(["dev", "proc", "sys"].map(|n| (String::from(n), true, 0)));
    }
    Ok(v)
}

/// Open or create a file (parents must exist), emptied if `trunc`.
pub fn touch(path: &str, trunc: bool) -> Result<(), &'static str> {
    let rel = check_rw(path)?;
    let mut f = fs()?.root_dir().create_file(rel).map_err(err)?;
    if trunc {
        f.truncate().map_err(err)?;
    }
    f.flush().map_err(err)
}

// M8: an open file, kept by the WASI runtime while the program has the fd open (sqlite's random IO would
// otherwise walk the path and the cluster chain on every call). The caller drops it before anything else changes
// the same file (rename, unlink, a write through another handle); a run never spans a !sync.
pub type File = fatfs::File<'static, Disk, RtcTime, fatfs::LossyOemCpConverter>;

pub fn open(path: &str) -> Result<File, &'static str> {
    let rel = check(path)?;
    if rel.is_empty() {
        return Err("/ is a directory");
    }
    if mount_of(rel).0 != Mount::Fat {
        return Err(ROFS); // callers read virtual files through read() (wasm.rs keeps the bytes in the fd)
    }
    fs()?.root_dir().open_file(rel).map_err(err)
}

pub fn len(f: &mut File) -> Result<u64, &'static str> {
    f.seek(SeekFrom::End(0)).map_err(err)
}

/// Read from `off` until buf is full or the file ends.
pub fn read_at(f: &mut File, off: u64, buf: &mut [u8]) -> Result<usize, &'static str> {
    let size = len(f)?;
    f.seek(SeekFrom::Start(off.min(size))).map_err(err)?;
    let mut n = 0;
    while n < buf.len() {
        let k = f.read(&mut buf[n..]).map_err(err)?;
        if k == 0 {
            break;
        }
        n += k;
    }
    Ok(n)
}

/// Cut or extend (zero-filled) to `n` bytes.
pub fn set_len(f: &mut File, n: u64) -> Result<(), &'static str> {
    if n >= len(f)? {
        return write_file_at(f, n, &[]).map(|_| ());
    }
    f.seek(SeekFrom::Start(n)).map_err(err)?;
    f.truncate().map_err(err)?;
    f.flush().map_err(err)
}

/// Write at `off` (a gap past the end is zero-filled), directory entry flushed; returns the size after.
pub fn write_file_at(f: &mut File, off: u64, data: &[u8]) -> Result<u64, &'static str> {
    let size = len(f)?;
    let mut gap = off.saturating_sub(size);
    if gap == 0 {
        f.seek(SeekFrom::Start(off)).map_err(err)?;
    }
    while gap > 0 {
        let k = gap.min(512);
        f.write_all(&[0u8; 512][..k as usize]).map_err(err)?;
        gap -= k;
    }
    f.write_all(data).map_err(err)?;
    f.flush().map_err(err)?;
    Ok(size.max(off + data.len() as u64))
}

/// Rename or move a file or a directory (the target must not exist).
pub fn rename(from: &str, to: &str) -> Result<(), &'static str> {
    let (a, b) = (check_rw(from)?, check_rw(to)?);
    if a.is_empty() || b.is_empty() {
        return Err("refusing to rename /");
    }
    let root = fs()?.root_dir();
    root.rename(a, &root, b).map_err(err)
}

/// Every file on the volume the indexer reads (T55: not under /var/lib/ringzero, /tmp or /lib, see `indexed`),
/// as (path, size, mtime).
pub fn walk(f: &mut dyn FnMut(&str, u64, u32)) -> Result<(), &'static str> {
    fn rec(d: &Dir, path: &mut String, f: &mut dyn FnMut(&str, u64, u32)) -> Result<(), &'static str> {
        let ents: Vec<(String, bool, u64, u32)> = d.iter().filter_map(|e| e.ok())
            .map(|e| (e.file_name(), e.is_dir(), e.len(), mtime(e.modified())))
            .filter(|(n, _, _, _)| n != "." && n != "..").collect();
        for (name, dir, size, mt) in ents {
            let k = path.len();
            path.push('/');
            path.push_str(&name);
            if !dir {
                f(path, size, mt);
            } else if indexed(path) {
                rec(&d.open_dir(&name).map_err(err)?, path, f)?;
            }
            path.truncate(k);
        }
        Ok(())
    }
    rec(&fs()?.root_dir(), &mut String::new(), f)
}

/// Free bytes on the FAT volume (docstate.rs keeps a reserve before it writes a cache file).
pub fn free_bytes() -> Result<u64, &'static str> {
    let s = fs()?.stats().map_err(err)?;
    Ok(s.free_clusters() as u64 * s.cluster_size() as u64)
}

pub fn df(o: &mut impl core::fmt::Write) {
    let Ok(f) = fs() else {
        let _ = write!(o, "no disk mounted");
        return;
    };
    match f.stats() {
        Ok(s) => {
            let c = s.cluster_size() as u64;
            let (tot, free) = (s.total_clusters() as u64 * c, s.free_clusters() as u64 * c);
            let _ = write!(o, "{:?} '{}', {} MiB, used {} KiB, free {} MiB (cluster {} KiB)", f.fat_type(),
                f.volume_label().trim_end(), tot >> 20, (tot - free) >> 10, free >> 20, c >> 10);
        }
        Err(e) => {
            let _ = write!(o, "stats: {}", err(e));
        }
    }
}

/// Unmount (FS info sector, clean flag) + device cache flush, then mount again. Writes are already on the
/// device (write-through); this makes the volume clean for fsck if the machine stops after it. A volume
/// that was already dirty at mount stays marked dirty (fatfs keeps the flag it found).
pub fn sync() -> Result<(), &'static str> {
    let f = unsafe { (*core::ptr::addr_of_mut!(FS)).take() }.ok_or("no disk mounted")?;
    let r = f.unmount().map_err(err).and_then(|_| Disk.flush().map_err(|_| "disk I/O error"));
    match FileSystem::new(Disk, opts()) {
        Ok(f) => unsafe { FS = Some(f) },
        Err(e) => {
            klog!(b'E', "disk: mount after sync failed ({})", err(e));
            return Err("mount after sync failed");
        }
    }
    r
}

/// Text test for the indexer: valid UTF-8 with no NUL.
pub fn is_text(b: &[u8]) -> bool {
    !b.contains(&0) && core::str::from_utf8(b).is_ok()
}
