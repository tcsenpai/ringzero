// FAT32 on the virtio disk (fatfs 0.4, no_std + alloc), mounted at / at boot. One volume, no partition
// table (`make disk` formats the whole image). Every path from the shell or the agent goes through check():
// absolute, no "." or ".." components, bounded length. Writes are write-through (blk.rs); `!sync`,
// poweroff and reboot flush the FS info sector and the device cache. File times come from the CMOS RTC
// (sys::rtc), so a file changed on the host (mtools) is told apart by its size and mtime (agentd, M4.5).
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

/// The checked path in one spelling ("/a/b"), the key the indexer uses.
pub fn canon(path: &str) -> Result<String, &'static str> {
    Ok(alloc::format!("/{}", check(path)?))
}

/// `ls`: one entry per line, directories end with '/'.
pub fn ls(path: &str, o: &mut impl core::fmt::Write) -> Result<usize, &'static str> {
    let rel = check(path)?;
    let root = fs()?.root_dir();
    let d = if rel.is_empty() { root } else { root.open_dir(rel).map_err(err)? };
    let mut n = 0;
    for e in d.iter() {
        let e = e.map_err(err)?;
        let name = e.file_name();
        if name == "." || name == ".." {
            continue;
        }
        let _ = if e.is_dir() { write!(o, "{}{}/", if n > 0 { "\n" } else { "" }, name) } else {
            write!(o, "{}{}  {} B", if n > 0 { "\n" } else { "" }, name, e.len())
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
    let mut f = fs()?.root_dir().open_file(rel).map_err(err)?;
    let size = f.seek(SeekFrom::End(0)).map_err(err)?;
    f.seek(SeekFrom::Start(off.min(size))).map_err(err)?;
    let mut v = alloc::vec![0u8; max.min(size.saturating_sub(off) as usize)];
    f.read_exact(&mut v).map_err(err)?;
    Ok((v, size))
}

/// Create or replace a file; missing parent directories are created.
pub fn write(path: &str, data: &[u8]) -> Result<(), &'static str> {
    let rel = check(path)?;
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
    let rel = check(path)?;
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
    let rel = check(path)?;
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
    let rel = check(path)?;
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
    let rel = check(path)?;
    let mut f = fs()?.root_dir().open_file(rel).map_err(err)?;
    let size = f.seek(SeekFrom::End(0)).map_err(err)?;
    f.seek(SeekFrom::Start(off.min(size))).map_err(err)?;
    let mut n = 0;
    while n < buf.len() {
        let k = f.read(&mut buf[n..]).map_err(err)?;
        if k == 0 {
            break;
        }
        n += k;
    }
    Ok((n, size))
}

/// Create or replace a file from several pieces (no size limit: agentd's index and the boot snapshot).
pub fn write_parts(path: &str, parts: &[&[u8]]) -> Result<u64, &'static str> {
    let rel = check(path)?;
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
    let rel = check(path)?;
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
    let mut f = fs()?.root_dir().open_file(check(path)?).map_err(err)?;
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
    Ok(v)
}

/// Open or create a file (parents must exist), emptied if `trunc`.
pub fn touch(path: &str, trunc: bool) -> Result<(), &'static str> {
    let rel = check(path)?;
    let mut f = fs()?.root_dir().create_file(rel).map_err(err)?;
    if trunc {
        f.truncate().map_err(err)?;
    }
    f.flush().map_err(err)
}

/// Write at `off` in an existing file (a gap past the end is zero-filled); returns the size after.
pub fn write_at(path: &str, off: u64, data: &[u8]) -> Result<u64, &'static str> {
    let mut f = fs()?.root_dir().open_file(check(path)?).map_err(err)?;
    let size = f.seek(SeekFrom::End(0)).map_err(err)?;
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
    let (a, b) = (check(from)?, check(to)?);
    if a.is_empty() || b.is_empty() {
        return Err("refusing to rename /");
    }
    let root = fs()?.root_dir();
    root.rename(a, &root, b).map_err(err)
}

/// Every file on the volume outside /sys (agentd's own files), as (path, size, mtime).
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
            } else if path != "/sys" {
                rec(&d.open_dir(&name).map_err(err)?, path, f)?;
            }
            path.truncate(k);
        }
        Ok(())
    }
    rec(&fs()?.root_dir(), &mut String::new(), f)
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
