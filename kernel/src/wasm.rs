// M5b: WASI preview1 runtime in ring 0. wasmi 2.0 (an interpreter, no_std + alloc) runs wasm32-wasip1 programs
// from the FAT32 disk: `!wasm PATH [ARGS...] [< FILE]` (a bare NAME means /bin/NAME.wasm, or the coreutils tool
// NAME; `!run` is the same command) and the agent's run_wasm tool. Every run is a fresh Engine + Store; the linear memory lives in the
// wasm slot (mem.rs), capped at RINGZERO_WASM_MIB (build time, default 256). The program runs in fuel slices:
// between two slices the kernel checks the time budget and Ctrl-C and stops the call (no hard abort needed,
// the interpreter state is simply dropped). The host functions map WASI onto fs.rs: one preopened dir "/" as
// fd 3, paths resolved under it ('..' above / is refused), /sys (agentd's files) read-only. stdout/stderr go
// to the console, or into a buffer for the agent. WASI calls we do not implement return ENOSYS, logged once.
// M8: a shell run has the console as its terminal: stdin is typed lines (REPLs), /dev/tty is raw input (editors),
// isatty() is true. Open files keep their fatfs handle. The agent's output is cut to TOK_CAP tokens. A kernel
// heap running low stops the run instead of the kernel.
use crate::log::SBuf;
use crate::{console, fs, klog, println, shell, smp};
use alloc::{format, string::String, vec, vec::Vec};
use core::fmt::Write;
use wasmi::{Caller, Config, CustomFuelCosts, Engine, Error, ExternType, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TypedResumableCall, Val};

const W: &str = "wasi_snapshot_preview1";
const SHELL_MS: u64 = 60_000;
const TOOL_MS: u64 = 10_000;
/// Fuel (about one wasm instruction each) between two checks of the budget and Ctrl-C.
const SLICE: u64 = 10_000_000;
/// Captured output for the agent (the tool result cap is 8 KiB, the status line needs some room).
const OUT_CAP: usize = 6144;
/// M8: ... and at most this many tokens of it (6 KiB of digits is 6000 tokens: 36 s of prefill on the 2B).
const TOK_CAP: usize = 1500;
/// Largest .wasm file (CPython is 31 MB): the bytes and wasmi's module live in the kernel heap (mem.rs).
const MOD_MAX: u64 = 48 << 20;
const READ_MAX: usize = 1 << 20; // per fd_read call

// errno (WASI preview1)
const ACCES: i32 = 2;
const BADF: i32 = 8;
const EXIST: i32 = 20;
const FAULT: i32 = 21;
const INVAL: i32 = 28;
const IO: i32 = 29;
const ISDIR: i32 = 31;
const NOENT: i32 = 44;
const NOSPC: i32 = 51;
const NOSYS: i32 = 52;
const NOTDIR: i32 = 54;
const NOTEMPTY: i32 = 55;
const SPIPE: i32 = 70;
const NOTCAPABLE: i32 = 76;
const RIGHTS: u64 = 0x1fff_ffff; // all 29 rights
const FD_WRITE: u64 = 1 << 6;
const SEEK: u64 = 1 << 2;
const TELL: u64 = 1 << 5;

static mut CAP: usize = 0;
static mut RDRAND: bool = false;
static mut OUT: SBuf<OUT_CAP> = SBuf::new();
static mut CAPTURE: bool = false;
static mut DROPPED: usize = 0;
static mut DEADLINE: u64 = 0;
static mut NEXT_POLL: u64 = 0;
static mut WHY: Why = Why::None;

#[derive(Clone, Copy, PartialEq)]
enum Why {
    None,
    Intr,
    Budget,
    Mem, // M8: the kernel heap ran low (mem.rs), the run is stopped instead of the kernel
}

/// Boot: the wasm slot (the memory cap) from the arena, before the snapshot pool takes the rest.
pub fn init() {
    let mib = option_env!("RINGZERO_WASM_MIB").and_then(|s| s.parse::<usize>().ok()).unwrap_or(256).clamp(16, 2048);
    let p = crate::mem::sys_mmap_anon((mib << 20) as u64);
    if (p as i64) <= 0 {
        klog!(b'W', "wasm: arena full, the WASI runtime is off");
        return;
    }
    crate::mem::set_wasm_slot(p, mib << 20);
    unsafe {
        CAP = mib << 20;
        RDRAND = raw_cpuid::CpuId::new().get_feature_info().is_some_and(|f| f.has_rdrand());
    }
    klog!(b'I', "wasm: wasmi 2.0.3 (WASI preview1), memory cap {} MiB per instance, random_get from {}", mib,
        if unsafe { RDRAND } { "RDRAND" } else { "a TSC-seeded xorshift (no RDRAND)" });
}

fn out(b: &[u8]) {
    unsafe {
        if CAPTURE {
            DROPPED += (*core::ptr::addr_of_mut!(OUT)).push_cut(b);
        } else {
            console::write_bytes(b);
        }
    }
}

/// True once the run must stop (Ctrl-C or past the deadline). Input is polled every 10 ms (a port read is a VM exit).
fn stop() -> bool {
    let now = smp::rdtsc();
    unsafe {
        if WHY == Why::None && now >= NEXT_POLL {
            NEXT_POLL = now + smp::us_to_tsc(10_000);
            if shell::poll_intr() {
                WHY = Why::Intr;
            }
        }
        if WHY == Why::None && now >= DEADLINE {
            WHY = Why::Budget;
        }
        if WHY == Why::None && crate::mem::heap_low().0 {
            WHY = Why::Mem;
        }
        WHY != Why::None
    }
}

// ---- host state ----

struct OpenFile {
    path: String,
    pos: u64,
    append: bool,
    write: bool,
    h: Option<fs::File>, // M8: the open fatfs file, opened on first use (see drop_handles)
}

impl OpenFile {
    fn new(path: String, append: bool, write: bool) -> Fd {
        Fd::File(OpenFile { path, pos: 0, append, write, h: None })
    }
    fn handle(&mut self) -> R<&mut fs::File> {
        if self.h.is_none() {
            self.h = Some(fs::open(&self.path).map_err(errno)?);
        }
        Ok(self.h.as_mut().unwrap())
    }
}

enum Fd {
    Null,           // stdin of the agent's runs (no `<`): always at EOF
    Con,            // stdout, stderr
    Tty(bool),      // M8: console input, from the shell only: fd 0 (line mode) or /dev/tty (raw = true)
    Dir(String, bool), // path, preopened
    File(OpenFile),
}

struct Host {
    mem: Option<Memory>,
    limits: StoreLimits,
    args: Vec<Vec<u8>>, // NUL-terminated
    env: Vec<Vec<u8>>,
    fds: Vec<Option<Fd>>,
    t0: u64,
    epoch_ns: u64,
    rng: u64,
    nosys: Vec<String>,
    tty: bool,       // M8: a shell run: the console is a terminal (isatty), stdin is typed lines
    line: Vec<u8>,   // typed input not read yet
    last_cr: bool,   // the last typed byte was \r (a following \n is the same Enter)
}

type C<'a> = Caller<'a, Host>;
type R<T = ()> = Result<T, i32>;

fn ret(r: R) -> i32 {
    r.err().unwrap_or(0)
}

fn mh<'a>(c: &'a mut C<'_>) -> R<(&'a mut [u8], &'a mut Host)> {
    let m = c.data().mem.ok_or(FAULT)?;
    Ok(m.data_and_store_mut(c))
}

fn at(m: &[u8], p: u32, n: usize) -> R<core::ops::Range<usize>> {
    let p = p as usize;
    match p.checked_add(n) {
        Some(e) if e <= m.len() => Ok(p..e),
        _ => Err(FAULT),
    }
}
fn r32(m: &[u8], p: u32) -> R<u32> {
    Ok(u32::from_le_bytes(m[at(m, p, 4)?].try_into().unwrap()))
}
fn r64(m: &[u8], p: u32) -> R<u64> {
    Ok(u64::from_le_bytes(m[at(m, p, 8)?].try_into().unwrap()))
}
fn put(m: &mut [u8], p: u32, b: &[u8]) -> R {
    let r = at(m, p, b.len())?;
    m[r].copy_from_slice(b);
    Ok(())
}
fn w32(m: &mut [u8], p: u32, v: u32) -> R {
    put(m, p, &v.to_le_bytes())
}
fn w64(m: &mut [u8], p: u32, v: u64) -> R {
    put(m, p, &v.to_le_bytes())
}
fn iovs(m: &[u8], p: u32, n: u32) -> R<Vec<(u32, usize)>> {
    (0..n).map(|i| Ok((r32(m, p + 8 * i)?, r32(m, p + 8 * i + 4)? as usize))).collect()
}
fn gather(m: &[u8], p: u32, n: u32) -> R<Vec<u8>> {
    let mut v = Vec::new();
    for (b, l) in iovs(m, p, n)? {
        v.extend_from_slice(&m[at(m, b, l)?]);
    }
    Ok(v)
}
fn scatter(m: &mut [u8], p: u32, n: u32, mut data: &[u8]) -> R {
    for (b, l) in iovs(m, p, n)? {
        let k = l.min(data.len());
        put(m, b, &data[..k])?;
        data = &data[k..];
    }
    Ok(())
}
fn guest_str(m: &[u8], p: u32, n: u32) -> R<&str> {
    core::str::from_utf8(&m[at(m, p, n as usize)?]).map_err(|_| INVAL)
}

/// fs.rs error -> errno.
fn errno(e: &str) -> i32 {
    match e {
        "not found" => NOENT,
        "already exists" => EXIST,
        "directory is not empty" => NOTEMPTY,
        "disk full" => NOSPC,
        "not a directory, or a directory where a file was expected" => NOTDIR,
        "disk I/O error" | "corrupted file system" | "no disk mounted" => IO,
        _ => INVAL,
    }
}

impl Host {
    fn fd(&mut self, fd: i32) -> R<&mut Fd> {
        self.fds.get_mut(fd as usize).and_then(|f| f.as_mut()).ok_or(BADF)
    }
    fn file(&mut self, fd: i32) -> R<&mut OpenFile> {
        match self.fd(fd)? {
            Fd::File(f) => Ok(f),
            Fd::Dir(..) => Err(ISDIR),
            _ => Err(SPIPE),
        }
    }
    /// `path` relative to the directory `fd`, as an absolute fs.rs path. '..' may not climb above /.
    fn resolve(&mut self, fd: i32, path: &str) -> R<String> {
        let Fd::Dir(base, _) = self.fd(fd)? else { return Err(NOTDIR) };
        if path.starts_with('/') {
            return Err(NOTCAPABLE);
        }
        let mut parts: Vec<&str> = base.split('/').filter(|c| !c.is_empty()).collect();
        for c in path.split('/') {
            match c {
                "" | "." => {}
                ".." => {
                    if parts.pop().is_none() {
                        klog!(b'W', "wasm: sandbox: '{}' climbs above the preopened /, refused", path);
                        return Err(NOTCAPABLE);
                    }
                }
                c => parts.push(c),
            }
        }
        let p = format!("/{}", parts.join("/"));
        fs::check(&p).map_err(|_| INVAL)?;
        Ok(p)
    }
    fn new_fd(&mut self, f: Fd) -> u32 {
        match self.fds.iter().skip(3).position(|f| f.is_none()) {
            Some(i) => {
                self.fds[i + 3] = Some(f);
                (i + 3) as u32
            }
            None => {
                self.fds.push(Some(f));
                (self.fds.len() - 1) as u32
            }
        }
    }
    fn rand(&mut self) -> u64 {
        if unsafe { RDRAND } {
            for _ in 0..10 {
                let (v, ok): (u64, u8);
                unsafe { core::arch::asm!("rdrand {}", "setc {}", out(reg) v, out(reg_byte) ok, options(nomem, nostack)) };
                if ok == 1 {
                    return v;
                }
            }
        }
        // xorshift64*, seeded from the TSC at the start of the run
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn now(&self, id: u32) -> R<u64> {
        let el = smp::tsc_to_us(smp::rdtsc() - self.t0) * 1000;
        match id {
            0 => Ok(self.epoch_ns + el),                       // realtime: RTC at the start + TSC
            1 => Ok(smp::tsc_to_us(smp::rdtsc()) * 1000), // monotonic: since boot
            2 | 3 => Ok(el),                                   // process / thread CPU time: since the start
            _ => Err(INVAL),
        }
    }
    /// Close the fatfs handles on `path` (all fds but `keep`): before the file changes by path or through another
    /// fd, so no handle keeps a stale size or cluster chain. They reopen on the next use.
    fn drop_handles(&mut self, path: &str, keep: Option<i32>) {
        for (i, f) in self.fds.iter_mut().enumerate() {
            if let Some(Fd::File(f)) = f {
                if keep != Some(i as i32) && f.path.eq_ignore_ascii_case(path) {
                    f.h = None;
                }
            }
        }
    }
}

/// M8: typed input for a program run from the shell. Line mode (fd 0): echo, backspace, Enter ends the line,
/// Ctrl-D ends the input (a read of 0 bytes; an unfinished line is returned without its newline first). Raw mode
/// (/dev/tty, editors): one byte as typed, no echo. Ctrl-C stops the run (shell.rs takes it out of
/// the input). The time spent waiting is added to the deadline: a REPL may wait for the user as long as it likes.
fn tty_read(h: &mut Host, raw: bool, want: usize) -> Result<Vec<u8>, Error> {
    let t0 = smp::rdtsc();
    let r = tty_wait(h, raw, want);
    unsafe { DEADLINE += smp::rdtsc() - t0 };
    r.ok_or_else(|| {
        unsafe { WHY = Why::Intr };
        Error::new("interrupted while reading the console")
    })
}

fn tty_wait(h: &mut Host, raw: bool, want: usize) -> Option<Vec<u8>> {
    let next = || loop {
        if let Some(b) = shell::getb() {
            return Some(b);
        }
        if shell::poll_intr() {
            return None;
        }
        smp::idle_wait();
    };
    if raw {
        // one byte per read: kibi wraps /dev/tty in a new BufReader for each read, so more would be lost
        if want == 0 {
            return Some(Vec::new());
        }
        return next().map(|b| vec![b]);
    }
    if h.line.is_empty() {
        let mut cur: Vec<u8> = Vec::new();
        let mut esc = 0u8; // 1: after ESC, 2: inside ESC [ ... (ended by a byte in 0x40..=0x7e)
        loop {
            let b = next()?;
            let cr = core::mem::replace(&mut h.last_cr, b == b'\r');
            match (esc, b) {
                (0, 0x1b) => esc = 1,
                (1, b'[' | b'O') => esc = 2,
                (1, _) => esc = 0,
                (2, 0x40..=0x7e) => esc = 0, // arrows and other keys: ignored
                (2, _) => {}
                (_, b'\n') if cr => {}
                (_, b'\r' | b'\n') => {
                    console::write_bytes(b"\n");
                    cur.push(b'\n');
                    break;
                }
                (_, 4) => break, // Ctrl-D: what was typed so far, or 0 bytes = end of input
                (_, 8 | 127) => {
                    if !cur.is_empty() {
                        while cur.pop().is_some_and(|c| c & 0xC0 == 0x80) {} // a whole UTF-8 character
                        console::write_bytes(b"\x08 \x08");
                    }
                }
                (_, b) if b >= 0x20 || b == b'\t' => {
                    cur.push(b);
                    console::write_bytes(&[b]);
                }
                _ => {}
            }
        }
        h.line = cur;
    }
    let k = want.min(h.line.len());
    Some(h.line.drain(..k).collect())
}

/// /sys holds agentd's stores and the boot snapshot: a wasm program may read it, never change it.
fn sys_ro(p: &str) -> R {
    let first = p.trim_start_matches('/').split('/').next().unwrap_or("");
    if first.eq_ignore_ascii_case("sys") { Err(ACCES) } else { Ok(()) }
}

/// FAT mtime (fs.rs packing, local time from the RTC) -> ns since the epoch.
fn ns(mt: u32) -> u64 {
    if mt == 0 {
        return 0;
    }
    let d = crate::sys::days((mt >> 25) + 1980, mt >> 21 & 15, mt >> 16 & 31);
    ((d * 86_400) as u64 + (mt >> 11 & 31) as u64 * 3600 + (mt >> 5 & 63) as u64 * 60 + (mt & 31) as u64 * 2) * 1_000_000_000
}

fn filestat(m: &mut [u8], p: u32, ft: u8, size: u64, mt: u32) -> R {
    let mut b = [0u8; 64];
    b[16] = ft;
    b[24..32].copy_from_slice(&1u64.to_le_bytes());
    b[32..40].copy_from_slice(&size.to_le_bytes());
    for o in [40, 48, 56] {
        b[o..o + 8].copy_from_slice(&ns(mt).to_le_bytes());
    }
    put(m, p, &b)
}

fn stat_path(m: &mut [u8], p: u32, path: &str) -> R {
    let (dir, size, mt) = fs::entry(path).map_err(errno)?;
    filestat(m, p, if dir { 3 } else { 4 }, size, mt)?;
    // M8: dev 1, inode = FNV-1a of the lowercased path (FAT has no inodes, its names are case-insensitive). With
    // 0 for every file, `cp /a /b` said "'/a' and '/b' are the same file"
    let ino = path.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, c| (h ^ c.to_ascii_lowercase() as u64).wrapping_mul(0x100_0000_01b3));
    put(m, p, &1u64.to_le_bytes())?;
    put(m, p + 8, &ino.to_le_bytes())
}

// ---- WASI preview1 ----

fn fd_write(c: &mut C, fd: i32, iov: u32, n: u32, nw: u32, off: Option<u64>) -> R {
    let (m, h) = mh(c)?;
    let data = gather(m, iov, n)?;
    let path = match h.fd(fd)? {
        Fd::Con | Fd::Tty(_) => {
            out(&data);
            None
        }
        Fd::File(f) if f.write => {
            let (append, pos) = (f.append, f.pos);
            let fh = f.handle()?;
            let o = match (off, append) {
                (Some(o), _) => o,
                (None, true) => fs::len(fh).map_err(errno)?,
                (None, false) => pos,
            };
            fs::write_file_at(fh, o, &data).map_err(errno)?;
            if off.is_none() {
                f.pos = o + data.len() as u64;
            }
            Some(f.path.clone())
        }
        _ => return Err(BADF),
    };
    if let Some(p) = path {
        h.drop_handles(&p, Some(fd));
    }
    w32(m, nw, data.len() as u32)
}

fn fd_read(c: &mut C, fd: i32, iov: u32, n: u32, nr: u32, off: Option<u64>) -> Result<R, Error> {
    let (m, h) = match mh(c) {
        Ok(x) => x,
        Err(e) => return Ok(Err(e)),
    };
    let want = match iovs(m, iov, n) {
        Ok(v) => v.iter().map(|x| x.1).sum::<usize>().min(READ_MAX),
        Err(e) => return Ok(Err(e)),
    };
    let r: R<usize> = match h.fd(fd) {
        Err(e) => Err(e),
        Ok(Fd::Null) => Ok(0),
        Ok(Fd::Tty(raw)) => {
            let raw = *raw;
            let data = tty_read(h, raw, want)?; // Ctrl-C: a trap that ends the run
            scatter(m, iov, n, &data).map(|_| data.len())
        }
        Ok(Fd::File(f)) => (|| {
            let mut buf = vec![0u8; want];
            let at = off.unwrap_or(f.pos);
            let k = fs::read_at(f.handle()?, at, &mut buf).map_err(errno)?;
            scatter(m, iov, n, &buf[..k])?;
            if off.is_none() {
                f.pos += k as u64;
            }
            Ok(k)
        })(),
        Ok(Fd::Dir(..)) => Err(ISDIR),
        Ok(Fd::Con) => Err(BADF),
    };
    Ok(r.and_then(|k| w32(m, nr, k as u32)))
}

fn fd_seek(c: &mut C, fd: i32, off: i64, whence: i32, p: u32) -> R {
    let (m, h) = mh(c)?;
    let f = h.file(fd)?;
    let base = match whence {
        0 => 0,
        1 => f.pos as i64,
        2 => fs::len(f.handle()?).map_err(errno)? as i64,
        _ => return Err(INVAL),
    };
    let to = base.checked_add(off).filter(|&x| x >= 0).ok_or(INVAL)?;
    f.pos = to as u64;
    w64(m, p, to as u64)
}

fn fd_fdstat_get(c: &mut C, fd: i32, p: u32) -> R {
    let (m, h) = mh(c)?;
    let tty = h.tty;
    let (ft, fl, rights) = match h.fd(fd)? {
        // character device; wasi-libc's isatty() is "no seek/tell right": a terminal only for shell runs
        Fd::Null | Fd::Con | Fd::Tty(_) if tty => (2u8, 0u8, RIGHTS & !(SEEK | TELL)),
        Fd::Null | Fd::Con | Fd::Tty(_) => (2, 0, RIGHTS),
        Fd::Dir(..) => (3, 0, RIGHTS),
        Fd::File(f) => (4, f.append as u8, RIGHTS),
    };
    let mut b = [0u8; 24];
    b[0] = ft;
    b[2] = fl;
    b[8..16].copy_from_slice(&rights.to_le_bytes());
    b[16..24].copy_from_slice(&RIGHTS.to_le_bytes());
    put(m, p, &b)
}

fn fd_filestat_get(c: &mut C, fd: i32, p: u32) -> R {
    let (m, h) = mh(c)?;
    match h.fd(fd)? {
        Fd::Null | Fd::Con | Fd::Tty(_) => filestat(m, p, 2, 0, 0),
        Fd::Dir(path, _) | Fd::File(OpenFile { path, .. }) => {
            let path = path.clone();
            stat_path(m, p, &path)
        }
    }
}

fn fd_filestat_set_size(c: &mut C, fd: i32, size: u64) -> R {
    let h = c.data_mut();
    let f = h.file(fd)?;
    if !f.write {
        return Err(BADF);
    }
    fs::set_len(f.handle()?, size).map_err(errno)?;
    let p = f.path.clone();
    h.drop_handles(&p, Some(fd));
    Ok(())
}

fn fd_prestat_get(c: &mut C, fd: i32, p: u32) -> R {
    let (m, h) = mh(c)?;
    match h.fd(fd)? {
        Fd::Dir(_, true) => put(m, p, &[0, 0, 0, 0, 1, 0, 0, 0]), // tag dir, name "/" (1 byte)
        _ => Err(BADF),
    }
}

fn fd_prestat_dir_name(c: &mut C, fd: i32, p: u32, n: u32) -> R {
    let (m, h) = mh(c)?;
    match h.fd(fd)? {
        Fd::Dir(_, true) => put(m, p, &b"/"[..n.min(1) as usize]),
        _ => Err(BADF),
    }
}

fn fd_readdir(c: &mut C, fd: i32, buf: u32, len: u32, cookie: u64, used: u32) -> R {
    let (m, h) = mh(c)?;
    let Fd::Dir(path, _) = h.fd(fd)? else { return Err(NOTDIR) };
    let ents = fs::list(path).map_err(errno)?;
    let mut out = Vec::new();
    for (i, (name, dir, _)) in ents.iter().enumerate().skip(cookie as usize) {
        if out.len() >= len as usize {
            break;
        }
        let mut d = [0u8; 24];
        d[0..8].copy_from_slice(&(i as u64 + 1).to_le_bytes()); // d_next: the cookie of the next entry
        d[8..16].copy_from_slice(&(i as u64 + 1).to_le_bytes());
        d[16..20].copy_from_slice(&(name.len() as u32).to_le_bytes());
        d[20] = if *dir { 3 } else { 4 };
        out.extend_from_slice(&d);
        out.extend_from_slice(name.as_bytes());
    }
    out.truncate(len as usize); // a cut last entry (bufused == len) makes libc call again with a bigger buffer
    put(m, buf, &out)?;
    w32(m, used, out.len() as u32)
}

#[allow(clippy::too_many_arguments)]
fn path_open(c: &mut C, dirfd: i32, pp: u32, pl: u32, oflags: i32, rights: u64, fdflags: i32, fdp: u32) -> R {
    let (m, h) = mh(c)?;
    let path = h.resolve(dirfd, guest_str(m, pp, pl)?)?;
    let (creat, dir_only, excl, trunc) = (oflags & 1 != 0, oflags & 2 != 0, oflags & 4 != 0, oflags & 8 != 0);
    let append = fdflags & 1 != 0;
    let write = rights & FD_WRITE != 0 || trunc || append;
    if path.eq_ignore_ascii_case("/dev/tty") {
        // M8: the terminal, raw (editors open it: kibi); only a shell run has one
        if !h.tty {
            return Err(NOENT);
        }
        let n = h.new_fd(Fd::Tty(true));
        return w32(m, fdp, n);
    }
    if trunc {
        h.drop_handles(&path, None);
    }
    let f = match fs::entry(&path) {
        Ok(_) if creat && excl => return Err(EXIST),
        Ok((true, ..)) if write => return Err(ISDIR),
        Ok((true, ..)) => Fd::Dir(path, false),
        Ok(_) if dir_only => return Err(NOTDIR),
        Ok(_) => {
            if write {
                sys_ro(&path)?;
            }
            if trunc {
                fs::truncate(&path, 0).map_err(errno)?;
            }
            OpenFile::new(path, append, write)
        }
        Err("not found") if creat && !dir_only => {
            sys_ro(&path)?;
            fs::touch(&path, false).map_err(errno)?;
            OpenFile::new(path, append, true)
        }
        Err(e) => return Err(errno(e)),
    };
    let n = h.new_fd(f);
    w32(m, fdp, n)
}

fn path_op(c: &mut C, fd: i32, pp: u32, pl: u32, what: u8) -> R {
    let (m, h) = mh(c)?;
    let path = h.resolve(fd, guest_str(m, pp, pl)?)?;
    sys_ro(&path)?;
    h.drop_handles(&path, None);
    match (what, fs::entry(&path)) {
        (b'm', Ok(_)) => Err(EXIST),
        (b'm', Err("not found")) => {
            let parent = path.rsplit_once('/').map(|x| x.0).filter(|p| !p.is_empty()).unwrap_or("/");
            match fs::entry(parent).map_err(errno)? {
                (true, ..) => fs::mkdir(&path).map(|_| ()).map_err(errno),
                _ => Err(NOTDIR),
            }
        }
        (b'd', Ok((false, ..))) => Err(NOTDIR),
        (b'u', Ok((true, ..))) => Err(ISDIR),
        (b'd' | b'u', Ok(_)) if path == "/" => Err(ACCES),
        (b'd' | b'u', Ok(_)) => fs::rm(&path, false).map_err(errno),
        (_, Err(e)) => Err(errno(e)),
        _ => Err(INVAL),
    }
}

fn path_rename(c: &mut C, fd: i32, op: u32, ol: u32, fd2: i32, np: u32, nl: u32) -> R {
    let (m, h) = mh(c)?;
    let from = h.resolve(fd, guest_str(m, op, ol)?)?;
    let to = h.resolve(fd2, guest_str(m, np, nl)?)?;
    sys_ro(&from)?;
    sys_ro(&to)?;
    h.drop_handles(&from, None);
    h.drop_handles(&to, None);
    let src_dir = fs::entry(&from).map_err(errno)?.0;
    match fs::entry(&to) {
        Ok(_) if from.eq_ignore_ascii_case(&to) => return Ok(()),
        Ok((true, ..)) if !src_dir => return Err(ISDIR),
        Ok((false, ..)) if src_dir => return Err(NOTDIR),
        Ok(_) => fs::rm(&to, false).map_err(errno)?, // POSIX rename replaces the target (an empty dir for a dir)
        Err(_) => {}
    }
    fs::rename(&from, &to).map_err(errno)
}

fn poll_oneoff(c: &mut C, inp: u32, outp: u32, nsub: u32, nev: u32) -> Result<R, Error> {
    let (m, h) = match mh(c) {
        Ok(x) => x,
        Err(e) => return Ok(Err(e)),
    };
    let res: R<u32> = (|| {
        let mut evs: Vec<(u64, u8)> = Vec::new();
        let mut clocks: Vec<(u64, u64)> = Vec::new();
        for i in 0..nsub {
            let s = inp + 48 * i;
            let ud = r64(m, s)?;
            match m[at(m, s + 8, 1)?.start] {
                0 => {
                    let (id, t, abs) = (r32(m, s + 16)?, r64(m, s + 24)?, m[at(m, s + 40, 1)?.start] & 1 != 0);
                    clocks.push((ud, if abs { t.saturating_sub(h.now(id)?) } else { t }));
                }
                t @ (1 | 2) => evs.push((ud, t)), // fd_read / fd_write: always ready (files, console, EOF stdin)
                _ => return Err(INVAL),
            }
        }
        if evs.is_empty() {
            if let Some(&min) = clocks.iter().map(|(_, t)| t).min() {
                let end = smp::rdtsc() + smp::us_to_tsc(min / 1000);
                while smp::rdtsc() < end {
                    if stop() {
                        return Err(-1);
                    }
                    core::hint::spin_loop();
                }
                evs.extend(clocks.iter().filter(|(_, t)| *t <= min).map(|(ud, _)| (*ud, 0u8)));
            }
        }
        for (i, (ud, ty)) in evs.iter().enumerate() {
            let mut e = [0u8; 32];
            e[0..8].copy_from_slice(&ud.to_le_bytes());
            e[10] = *ty;
            put(m, outp + 32 * i as u32, &e)?;
        }
        Ok(evs.len() as u32)
    })();
    match res {
        Err(-1) => Err(Error::new("stopped while sleeping in poll_oneoff")),
        Err(e) => Ok(Err(e)),
        Ok(n) => Ok(w32(m, nev, n)),
    }
}

fn sizes(c: &mut C, env: bool, cp: u32, bp: u32) -> R {
    let (m, h) = mh(c)?;
    let v = if env { &h.env } else { &h.args };
    w32(m, cp, v.len() as u32)?;
    w32(m, bp, v.iter().map(|a| a.len() as u32).sum())
}

fn strings(c: &mut C, env: bool, ptrs: u32, buf: u32) -> R {
    let (m, h) = mh(c)?;
    let mut b = buf;
    for (i, a) in (if env { &h.env } else { &h.args }).iter().enumerate() {
        w32(m, ptrs + 4 * i as u32, b)?;
        put(m, b, a)?;
        b += a.len() as u32;
    }
    Ok(())
}

/// Names defined below; every other wasi_snapshot_preview1 import gets an ENOSYS stub.
const NAMES: &[&str] = &["args_get", "args_sizes_get", "environ_get", "environ_sizes_get", "clock_res_get", "clock_time_get",
    "fd_advise", "fd_close", "fd_datasync", "fd_fdstat_get", "fd_fdstat_set_flags", "fd_filestat_get", "fd_filestat_set_size",
    "fd_filestat_set_times", "fd_pread", "fd_prestat_get", "fd_prestat_dir_name", "fd_pwrite", "fd_read", "fd_readdir",
    "fd_renumber", "fd_seek", "fd_sync", "fd_tell", "fd_write", "path_create_directory", "path_filestat_get",
    "path_filestat_set_times", "path_open", "path_readlink", "path_remove_directory", "path_rename", "path_unlink_file",
    "poll_oneoff", "proc_exit", "random_get", "sched_yield"];

fn link(l: &mut Linker<Host>, module: &Module) -> Result<(), Error> {
    l.func_wrap(W, "args_sizes_get", |mut c: C, a: u32, b: u32| ret(sizes(&mut c, false, a, b)))?;
    l.func_wrap(W, "args_get", |mut c: C, a: u32, b: u32| ret(strings(&mut c, false, a, b)))?;
    l.func_wrap(W, "environ_sizes_get", |mut c: C, a: u32, b: u32| ret(sizes(&mut c, true, a, b)))?;
    l.func_wrap(W, "environ_get", |mut c: C, a: u32, b: u32| ret(strings(&mut c, true, a, b)))?;
    l.func_wrap(W, "clock_res_get", |mut c: C, id: u32, p: u32| {
        ret(mh(&mut c).and_then(|(m, h)| h.now(id).and_then(|_| w64(m, p, 1000))))
    })?;
    l.func_wrap(W, "clock_time_get", |mut c: C, id: u32, _prec: u64, p: u32| {
        ret(mh(&mut c).and_then(|(m, h)| h.now(id).and_then(|t| w64(m, p, t))))
    })?;
    l.func_wrap(W, "fd_write", |mut c: C, fd: i32, iov: u32, n: u32, nw: u32| ret(fd_write(&mut c, fd, iov, n, nw, None)))?;
    l.func_wrap(W, "fd_pwrite", |mut c: C, fd: i32, iov: u32, n: u32, off: u64, nw: u32| ret(fd_write(&mut c, fd, iov, n, nw, Some(off))))?;
    l.func_wrap(W, "fd_read", |mut c: C, fd: i32, iov: u32, n: u32, nr: u32| fd_read(&mut c, fd, iov, n, nr, None).map(ret))?;
    l.func_wrap(W, "fd_pread", |mut c: C, fd: i32, iov: u32, n: u32, off: u64, nr: u32| fd_read(&mut c, fd, iov, n, nr, Some(off)).map(ret))?;
    l.func_wrap(W, "fd_seek", |mut c: C, fd: i32, off: i64, wh: i32, p: u32| ret(fd_seek(&mut c, fd, off, wh, p)))?;
    l.func_wrap(W, "fd_tell", |mut c: C, fd: i32, p: u32| ret(fd_seek(&mut c, fd, 0, 1, p)))?;
    l.func_wrap(W, "fd_close", |mut c: C, fd: i32| {
        ret(c.data_mut().fd(fd).map(|_| ()).map(|_| c.data_mut().fds[fd as usize] = None))
    })?;
    l.func_wrap(W, "fd_renumber", |mut c: C, from: i32, to: i32| {
        let h = c.data_mut();
        let ok = h.fd(from).is_ok() && h.fd(to).is_ok();
        if ok {
            h.fds[to as usize] = h.fds[from as usize].take();
        }
        if ok { 0 } else { BADF }
    })?;
    l.func_wrap(W, "fd_fdstat_get", |mut c: C, fd: i32, p: u32| ret(fd_fdstat_get(&mut c, fd, p)))?;
    l.func_wrap(W, "fd_fdstat_set_flags", |mut c: C, fd: i32, fl: i32| {
        ret(c.data_mut().fd(fd).map(|f| if let Fd::File(f) = f { f.append = fl & 1 != 0 }))
    })?;
    l.func_wrap(W, "fd_filestat_get", |mut c: C, fd: i32, p: u32| ret(fd_filestat_get(&mut c, fd, p)))?;
    l.func_wrap(W, "fd_filestat_set_size", |mut c: C, fd: i32, n: u64| ret(fd_filestat_set_size(&mut c, fd, n)))?;
    // FAT keeps one mtime that the kernel sets on every write: setting times is accepted and ignored
    l.func_wrap(W, "fd_filestat_set_times", |mut c: C, fd: i32, _a: u64, _m: u64, _f: i32| ret(c.data_mut().fd(fd).map(|_| ())))?;
    l.func_wrap(W, "path_filestat_set_times", |_: C, _fd: i32, _fl: i32, _p: u32, _l: u32, _a: u64, _m: u64, _f: i32| 0i32)?;
    for name in ["fd_sync", "fd_datasync"] {
        l.func_wrap(W, name, |mut c: C, fd: i32| ret(c.data_mut().fd(fd).map(|_| ())))?; // writes are write-through
    }
    l.func_wrap(W, "fd_advise", |mut c: C, fd: i32, _o: u64, _l: u64, _a: i32| ret(c.data_mut().fd(fd).map(|_| ())))?;
    l.func_wrap(W, "fd_prestat_get", |mut c: C, fd: i32, p: u32| ret(fd_prestat_get(&mut c, fd, p)))?;
    l.func_wrap(W, "fd_prestat_dir_name", |mut c: C, fd: i32, p: u32, n: u32| ret(fd_prestat_dir_name(&mut c, fd, p, n)))?;
    l.func_wrap(W, "fd_readdir", |mut c: C, fd: i32, b: u32, n: u32, ck: u64, u: u32| ret(fd_readdir(&mut c, fd, b, n, ck, u)))?;
    l.func_wrap(W, "path_open", |mut c: C, fd: i32, _df: i32, p: u32, n: u32, of: i32, rb: u64, _ri: u64, ff: i32, out: u32| {
        ret(path_open(&mut c, fd, p, n, of, rb, ff, out))
    })?;
    l.func_wrap(W, "path_filestat_get", |mut c: C, fd: i32, _fl: i32, p: u32, n: u32, out: u32| {
        ret(mh(&mut c).and_then(|(m, h)| {
            let path = h.resolve(fd, guest_str(m, p, n)?)?;
            stat_path(m, out, &path)
        }))
    })?;
    l.func_wrap(W, "path_create_directory", |mut c: C, fd: i32, p: u32, n: u32| ret(path_op(&mut c, fd, p, n, b'm')))?;
    l.func_wrap(W, "path_remove_directory", |mut c: C, fd: i32, p: u32, n: u32| ret(path_op(&mut c, fd, p, n, b'd')))?;
    l.func_wrap(W, "path_unlink_file", |mut c: C, fd: i32, p: u32, n: u32| ret(path_op(&mut c, fd, p, n, b'u')))?;
    l.func_wrap(W, "path_rename", |mut c: C, fd: i32, op: u32, ol: u32, fd2: i32, np: u32, nl: u32| {
        ret(path_rename(&mut c, fd, op, ol, fd2, np, nl))
    })?;
    // FAT has no symlinks: nothing is a link
    l.func_wrap(W, "path_readlink", |_: C, _fd: i32, _p: u32, _n: u32, _b: u32, _bl: u32, _u: u32| INVAL)?;
    l.func_wrap(W, "poll_oneoff", |mut c: C, i: u32, o: u32, n: u32, ne: u32| poll_oneoff(&mut c, i, o, n, ne).map(ret))?;
    l.func_wrap(W, "proc_exit", |_: C, code: i32| -> Result<(), Error> { Err(Error::i32_exit(code)) })?;
    l.func_wrap(W, "sched_yield", |_: C| 0i32)?;
    l.func_wrap(W, "random_get", |mut c: C, p: u32, n: u32| {
        ret(mh(&mut c).and_then(|(m, h)| {
            let r = at(m, p, n as usize)?;
            for ch in m[r].chunks_mut(8) {
                let v = h.rand().to_le_bytes();
                ch.copy_from_slice(&v[..ch.len()]);
            }
            Ok(())
        }))
    })?;
    for imp in module.imports() {
        if let (W, ExternType::Func(ty)) = (imp.module(), imp.ty()) {
            if NAMES.contains(&imp.name()) {
                continue;
            }
            let name = String::from(imp.name());
            l.func_new(W, imp.name(), ty.clone(), move |mut c, _, res| {
                let h = c.data_mut();
                if !h.nosys.contains(&name) {
                    klog!(b'W', "wasm: {} is not implemented (ENOSYS)", name);
                    h.nosys.push(name.clone());
                }
                res.iter_mut().for_each(|r| *r = Val::I32(NOSYS));
                Ok(())
            })?;
        }
    }
    Ok(())
}

// ---- a run ----

pub struct Run {
    pub code: Option<i32>, // exit status (None: trap, error or stopped)
    pub msg: String,       // trap or error
    pub us: u64,
    pub fuel: u64,
    pub mem: u64,
    heap: u64, // M8: peak kernel heap use during the run
    why: Why,
}

impl Run {
    pub fn status<const N: usize>(&self, o: &mut SBuf<N>, budget_ms: u64) {
        let _ = match (self.why, self.code) {
            (Why::Intr, _) => write!(o, "interrupted (Ctrl-C)"),
            (Why::Budget, _) => write!(o, "stopped: time budget exceeded"),
            (Why::Mem, _) => write!(o, "stopped: the kernel heap ran low ({} MiB in all)", crate::mem::HEAP_BYTES >> 20),
            (_, Some(c)) => write!(o, "exit {}", c),
            _ => write!(o, "trap: {}", self.msg),
        };
        let _ = write!(o, " in {} ms (budget {} ms), {} M fuel, memory {} KiB (cap {} MiB), kernel heap peak {} MiB",
            self.us / 1000, budget_ms, self.fuel / 1_000_000, self.mem >> 10, unsafe { CAP } >> 20, self.heap >> 20);
    }
}

/// (path, argv[0]) of a program: a path, or NAME = /bin/NAME, /bin/NAME.wasm, else /bin/coreutils.wasm with
/// argv[0] NAME (uutils is a multicall binary: `ls` runs coreutils' ls), like the links a Unix /bin would have.
fn program(name: &str) -> (String, String) {
    let base = name.rsplit('/').next().unwrap_or(name).trim_end_matches(".wasm");
    let p = if name.contains('/') { String::from(name) } else { format!("/bin/{}", name) };
    let in_bin = p.rsplit_once('/').is_some_and(|(d, _)| d.eq_ignore_ascii_case("/bin"));
    for c in [p.clone(), p.clone() + ".wasm"] {
        if fs::entry(&c).is_ok() || !in_bin {
            return (c, String::from(base));
        }
    }
    (String::from("/bin/coreutils.wasm"), String::from(base))
}

/// Words of a command line: spaces split, "..." keeps spaces, `< FILE` (or `<FILE`) is stdin.
/// M8: '...' too (the model writes `jq '.a' /x.json`), each kind keeps the other quote as it is; \" and \\ escape.
pub fn split(s: &str) -> Result<(Vec<String>, Option<String>), &'static str> {
    let (mut words, mut stdin, mut cur, mut redir) = (Vec::new(), None, String::new(), false);
    let mut quote: Option<char> = None;
    let mut word_done = |w: String, redir: &mut bool| {
        if core::mem::take(redir) { stdin = Some(w) } else { words.push(w) }
    };
    let mut any = false;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        let quoted = quote.is_some();
        match ch {
            // \" and \\ outside single quotes, as in a shell (`\d` stays as it is, for regexes)
            '\\' if quote != Some('\'') && matches!(chars.peek(), Some('"' | '\\')) => {
                cur.extend(chars.next());
                any = true;
            }
            '"' | '\'' if quote.is_none() || quote == Some(ch) => {
                quote = if quoted { None } else { Some(ch) };
                any = true;
            }
            ' ' | '\t' if !quoted => {
                if any {
                    word_done(core::mem::take(&mut cur), &mut redir);
                    any = false;
                }
            }
            '<' if !quoted && !any => redir = true,
            ch => {
                cur.push(ch);
                any = true;
            }
        }
    }
    if quote.is_some() {
        return Err("unterminated quote");
    }
    if any {
        word_done(cur, &mut redir);
    }
    if redir {
        return Err("'<' needs a file name");
    }
    Ok((words, stdin))
}

fn run(words: &[String], stdin: Option<&str>, budget_ms: u64, capture: bool) -> Result<Run, String> {
    if unsafe { CAP } == 0 {
        return Err("the WASI runtime is off (no arena memory at boot)".into());
    }
    let (path, argv0) = program(&words[0]);
    let path = fs::canon(&path).map_err(String::from)?;
    let (dir, size, _) = fs::entry(&path).map_err(|e| format!("{}: {}", path, e))?;
    if dir {
        return Err(format!("{}: is a directory", path));
    }
    if size > MOD_MAX {
        return Err(format!("{}: {} KiB, larger than the {} KiB limit for a module", path, size >> 10, MOD_MAX >> 10));
    }
    // M8: loading takes the file bytes, wasmi's copy and Vec growth steps (both buffers at once) in single big
    // allocations, which the low-heap flag cannot stop: they must fit. CPython (31 MB) panicked the kernel in a
    // 100 MiB heap with 96 MiB free, so 4x the file plus the low mark is asked for.
    let (need, free) = (4 * size + crate::mem::HEAP_LOW as u64, crate::mem::heap_free() as u64);
    if need > free {
        return Err(format!("{}: loading it needs about {} MiB of kernel heap, {} MiB is free ({} MiB in all, RINGZERO_HEAP_MIB)",
            path, need >> 20, free >> 20, crate::mem::HEAP_BYTES >> 20));
    }
    let stdin_fd = match stdin {
        None if capture => Fd::Null,
        None => Fd::Tty(false),
        Some(p) => {
            let p = fs::canon(p).map_err(String::from)?;
            match fs::entry(&p) {
                Ok((false, ..)) => OpenFile::new(p, false, false),
                Ok(_) => return Err(format!("{}: is a directory", p)),
                Err(e) => return Err(format!("{}: {}", p, e)),
            }
        }
    };
    let bytes = fs::read(&path, 0, size as usize).map_err(|e| format!("{}: {}", path, e))?.0;
    if !bytes.starts_with(b"\0asm") {
        // M8: wasmi's own message dumps both magic numbers, long and useless for the agent
        return Err(format!("{}: not a wasm program (the programs are in /bin: coreutils tools, rg, jq, sqlite3, lua, python3, qjs)", path));
    }
    let t0 = smp::rdtsc();
    unsafe {
        CAPTURE = capture;
        (*core::ptr::addr_of_mut!(OUT)).clear();
        DROPPED = 0;
        WHY = Why::None;
        DEADLINE = t0 + smp::us_to_tsc(budget_ms * 1000);
        NEXT_POLL = 0;
    }
    let mut cfg = Config::default();
    cfg.consume_fuel(true);
    // lazy translation charges fuel too, and running out there is not resumable (a trap, wasmi 2.0.3): only the
    // executed code pays; bytes_copied_per_fuel keeps wasmi's default
    cfg.fuel_cost(CustomFuelCosts { bytes_copied_per_fuel: 64, fuel_per_bytes_translated: 0, fuel_per_bytes_validated: 0 });
    let engine = Engine::new(&cfg);
    crate::mem::heap_watch();
    let module = Module::new(&engine, &bytes[..]).map_err(|e| format!("{}: not a valid wasm module: {}", path, e))?;
    drop(bytes);
    if crate::mem::heap_low().0 {
        return Err(format!("{}: loading it left the kernel heap almost full ({} MiB in all), not run", path, crate::mem::HEAP_BYTES >> 20));
    }
    let z = |s: &str| [s.as_bytes(), b"\0"].concat();
    let host = Host {
        mem: None,
        limits: StoreLimitsBuilder::new().memory_size(unsafe { CAP }).build(),
        args: core::iter::once(z(&argv0)).chain(words[1..].iter().map(|a| z(a))).collect(),
        // PYTHON_BASIC_REPL: CPython's new REPL needs termios; trying it first costs 9 s of imports
        env: ["HOME=/", "PWD=/", "PATH=/bin", "TERM=dumb", "PYTHON_BASIC_REPL=1"].iter().map(|e| z(e)).collect(),
        fds: vec![Some(stdin_fd), Some(Fd::Con), Some(Fd::Con), Some(Fd::Dir(String::from("/"), true))],
        t0,
        epoch_ns: crate::sys::unix_time() * 1_000_000_000,
        rng: t0 | 1,
        nosys: Vec::new(),
        tty: !capture,
        line: Vec::new(),
        last_cr: false,
    };
    let mut store = Store::new(&engine, host);
    store.limiter(|h| &mut h.limits);
    let mut linker = Linker::new(&engine);
    link(&mut linker, &module).map_err(|e| format!("linker: {}", e))?;
    crate::mem::arm_wasm_slot(true);
    let inst = linker.instantiate_and_start(&mut store, &module);
    crate::mem::arm_wasm_slot(false);
    let inst = inst.map_err(|e| format!("instantiate: {}", e))?;
    store.data_mut().mem = inst.get_memory(&store, "memory");
    let start = inst.get_typed_func::<(), ()>(&store, "_start").map_err(|_| String::from("no _start: not a WASI command"))?;
    store.set_fuel(SLICE).map_err(|e| format!("{}", e))?;
    let mut given = SLICE;
    let mut call = start.call_resumable(&mut store, ());
    let end = |e: &Error| match e.i32_exit_status() {
        Some(c) => (Some(c), String::new()),
        None => (None, format!("{}", e)),
    };
    let (code, msg) = loop {
        match call {
            Ok(TypedResumableCall::Finished(())) => break (Some(0), String::new()),
            Ok(TypedResumableCall::OutOfFuel(r)) => {
                if stop() {
                    break (None, String::new());
                }
                let more = SLICE.max(r.required_fuel());
                given += more - store.get_fuel().unwrap_or(0);
                let _ = store.set_fuel(more);
                call = r.resume(&mut store);
            }
            Ok(TypedResumableCall::HostTrap(t)) => break end(t.host_error()),
            Err(e) => break end(&e),
        }
    };
    let mem = inst.get_memory(&store, "memory").map_or(0, |m| m.data_size(&store) as u64);
    let heap = (crate::mem::HEAP_BYTES as u64).saturating_sub(crate::mem::heap_low().1);
    let fuel = given - store.get_fuel().unwrap_or(0);
    Ok(Run { code, msg, us: smp::tsc_to_us(smp::rdtsc() - t0), fuel, mem, heap, why: unsafe { WHY } })
}

/// run() on the private stack (upy::on_stack), with the log line; capture is reset after.
fn go(words: &[String], stdin: Option<&str>, budget_ms: u64, capture: bool) -> Result<Run, String> {
    let mut r = Err(String::new());
    crate::upy::on_stack(&mut || r = run(words, stdin, budget_ms, capture));
    unsafe { CAPTURE = false };
    let mut l = SBuf::<300>::new();
    let _ = write!(l, "wasm {}: ", words[0]);
    match &r {
        Ok(x) => x.status(&mut l, budget_ms),
        Err(e) => {
            let _ = write!(l, "{}", e);
        }
    }
    crate::log::record(b'I', l.as_str());
    r
}

/// `!wasm PATH|NAME [ARGS...] [< FILE]` (and `!run`): output straight to the console.
pub fn shell_cmd(arg: &str) {
    let (words, stdin) = match split(arg) {
        Ok((w, _)) if w.is_empty() => return println!("usage: !wasm PATH|NAME [ARGS...] [< FILE]  (NAME = /bin/NAME.wasm, else the coreutils tool NAME)"),
        Ok(x) => x,
        Err(e) => return println!("[wasm] {}", e),
    };
    match go(&words, stdin.as_deref(), SHELL_MS, false) {
        Ok(r) => {
            let mut o = SBuf::<320>::new();
            r.status(&mut o, SHELL_MS);
            println!("\n[wasm] {}", o.as_str());
        }
        Err(e) => println!("[wasm] {}", e),
    }
}

/// The agent's run_wasm tool: captured stdout + stderr and one status line.
pub fn tool<const N: usize>(path: &str, args: &str, o: &mut SBuf<N>) {
    // M8: the 2B often puts the whole command in path (path=seq 1 3000): the words after the first are arguments
    let (path, args) = match path.trim().split_once(' ') {
        Some((p, rest)) => (p, format!("{} {}", rest, args)),
        None => (path.trim(), String::from(args)),
    };
    let r = split(&args).map_err(String::from).and_then(|(mut w, stdin)| {
        // ... or repeats the program in args (path=sqlite3, args=sqlite3 /a.db ...): that word is dropped.
        // shortcut: `echo echo` loses its argument too, revisit if a program needs its own name as argv[1]
        let stem = |s: &str| String::from(s.rsplit('/').next().unwrap_or(s).trim_end_matches(".wasm"));
        if w.first().is_some_and(|a| stem(a).eq_ignore_ascii_case(&stem(path))) {
            w.remove(0);
        }
        w.insert(0, String::from(path));
        go(&w, stdin.as_deref(), TOOL_MS, true)
    });
    match r {
        Ok(r) => {
            let out = unsafe { &*core::ptr::addr_of!(OUT) };
            let all = out.as_str().trim_end().as_bytes();
            let keep = tok_cut(all);
            o.push(&all[..keep]);
            let dropped = unsafe { DROPPED } + all.len() - keep;
            if dropped > 0 {
                let _ = write!(o, "\n[... {} more bytes of output dropped (the tool keeps about {} tokens): narrow the command, e.g. head, wc -l, rg -c]",
                    dropped, TOK_CAP);
            }
            // the full status (time, fuel, memory) is in the log: the model gets the outcome only
            let _ = write!(o, "{}[", if keep > 0 { "\n" } else { "" });
            let _ = match (r.why, r.code) {
                (Why::None, Some(c)) => write!(o, "exit {}", c),
                (Why::None, None) => write!(o, "trap: {}", r.msg),
                _ => {
                    let mut s = SBuf::<320>::new();
                    r.status(&mut s, TOOL_MS);
                    write!(o, "{}", s.as_str().split(" in ").next().unwrap_or(""))
                }
            };
            o.push(b"]");
        }
        Err(e) => {
            let _ = write!(o, "error: {}", e);
        }
    }
}

/// M8: how many bytes of `b` fit in TOK_CAP tokens of the loaded model's tokenizer (all of it when they fit),
/// cut at a line end when one is in the second half, never inside a UTF-8 character.
fn tok_cut(b: &[u8]) -> usize {
    static mut TOKS: [u32; OUT_CAP] = [0; OUT_CAP]; // a BPE token is at least one byte: OUT_CAP bytes fit
    let toks = |n: usize| crate::llm::encode_plain(&b[..n], unsafe { &mut *core::ptr::addr_of_mut!(TOKS) });
    if b.len() > OUT_CAP || toks(b.len()) <= TOK_CAP {
        return b.len().min(OUT_CAP);
    }
    let (mut lo, mut hi) = (0, b.len()); // toks(lo) <= TOK_CAP < toks(hi)
    while hi - lo > 16 {
        let mid = (lo + hi) / 2;
        if toks(mid) <= TOK_CAP { lo = mid } else { hi = mid }
    }
    while lo > 0 && b[lo] & 0xC0 == 0x80 {
        lo -= 1;
    }
    match b[..lo].iter().rposition(|&c| c == b'\n') {
        Some(i) if i > lo / 2 => i,
        _ => lo,
    }
}

/// Boot self-test of the command-line splitter (quotes, `<`).
pub fn selftest() -> bool {
    let ok = |s: &str, w: &[&str], i: Option<&str>| split(s).is_ok_and(|(a, b)| a.iter().map(|x| x.as_str()).eq(w.iter().copied()) && b.as_deref() == i);
    ok("rg -n \"fn main\" /src", &["rg", "-n", "fn main", "/src"], None)
        && ok("wc -l < /a.txt", &["wc", "-l"], Some("/a.txt"))
        && ok("cat <in x", &["cat", "x"], Some("in"))
        && ok("echo \"\"", &["echo", ""], None)
        && ok("jq '.a b' \"it's\" '\"x\"'", &["jq", ".a b", "it's", "\"x\""], None)
        && ok("python3 -c \"print(\\\"hi\\\")\" '\\\"' rg \"\\d+\"", &["python3", "-c", "print(\"hi\")", "\\\"", "rg", "\\d+"], None)
        && split("a 'b").is_err()
        && split("a \"b").is_err()
        && split("a <").is_err()
}
