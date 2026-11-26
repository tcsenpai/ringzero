// M5b: WASI preview1 runtime in ring 0. wasmi 2.0 (an interpreter, no_std + alloc) runs wasm32-wasip1 programs
// from the FAT32 disk: `!wasm PATH [ARGS...] [< FILE]` (a bare NAME means /bin/NAME.wasm, or the coreutils tool
// NAME; `!run` is the same command) and the agent's run_wasm tool. Every run is a fresh Engine + Store; the linear memory lives in the
// wasm slot (mem.rs), capped at RINGZERO_WASM_MIB (build time, default 256). The program runs in fuel slices:
// between two slices the kernel checks the time budget and Ctrl-C and stops the call (no hard abort needed,
// the interpreter state is simply dropped). The host functions map WASI onto fs.rs: one preopened dir "/" as
// fd 3, paths resolved under it ('..' above / is refused), /sys (agentd's files) read-only. stdout/stderr go
// to the console, or into a buffer for the agent. WASI calls we do not implement return ENOSYS, logged once.
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
/// Largest .wasm file: the bytes and wasmi's module live in the 64 MiB kernel heap.
const MOD_MAX: u64 = 24 << 20;
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
        WHY != Why::None
    }
}

// ---- host state ----

struct OpenFile {
    path: String,
    pos: u64,
    append: bool,
    write: bool,
}

enum Fd {
    Null,           // stdin without `<`: always at EOF
    Con,            // stdout, stderr
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
    filestat(m, p, if dir { 3 } else { 4 }, size, mt)
}

// ---- WASI preview1 ----

fn fd_write(c: &mut C, fd: i32, iov: u32, n: u32, nw: u32, off: Option<u64>) -> R {
    let (m, h) = mh(c)?;
    let data = gather(m, iov, n)?;
    match h.fd(fd)? {
        Fd::Con => out(&data),
        Fd::File(f) if f.write => {
            let o = match (off, f.append) {
                (Some(o), _) => o,
                (None, true) => fs::entry(&f.path).map_err(errno)?.1,
                (None, false) => f.pos,
            };
            fs::write_at(&f.path, o, &data).map_err(errno)?;
            if off.is_none() {
                f.pos = o + data.len() as u64;
            }
        }
        _ => return Err(BADF),
    }
    w32(m, nw, data.len() as u32)
}

fn fd_read(c: &mut C, fd: i32, iov: u32, n: u32, nr: u32, off: Option<u64>) -> R {
    let (m, h) = mh(c)?;
    let want = iovs(m, iov, n)?.iter().map(|x| x.1).sum::<usize>().min(READ_MAX);
    let k = match h.fd(fd)? {
        Fd::Null => 0,
        Fd::File(f) => {
            let mut buf = vec![0u8; want];
            let (k, _) = fs::read_into(&f.path, off.unwrap_or(f.pos), &mut buf).map_err(errno)?;
            scatter(m, iov, n, &buf[..k])?;
            if off.is_none() {
                f.pos += k as u64;
            }
            k
        }
        Fd::Dir(..) => return Err(ISDIR),
        Fd::Con => return Err(BADF),
    };
    w32(m, nr, k as u32)
}

fn fd_seek(c: &mut C, fd: i32, off: i64, whence: i32, p: u32) -> R {
    let (m, h) = mh(c)?;
    let f = h.file(fd)?;
    let base = match whence {
        0 => 0,
        1 => f.pos as i64,
        2 => fs::entry(&f.path).map_err(errno)?.1 as i64,
        _ => return Err(INVAL),
    };
    let to = base.checked_add(off).filter(|&x| x >= 0).ok_or(INVAL)?;
    f.pos = to as u64;
    w64(m, p, to as u64)
}

fn fd_fdstat_get(c: &mut C, fd: i32, p: u32) -> R {
    let (m, h) = mh(c)?;
    let (ft, fl) = match h.fd(fd)? {
        Fd::Null | Fd::Con => (2u8, 0u8), // character device
        Fd::Dir(..) => (3, 0),
        Fd::File(f) => (4, f.append as u8),
    };
    let mut b = [0u8; 24];
    b[0] = ft;
    b[2] = fl;
    b[8..16].copy_from_slice(&RIGHTS.to_le_bytes());
    b[16..24].copy_from_slice(&RIGHTS.to_le_bytes());
    put(m, p, &b)
}

fn fd_filestat_get(c: &mut C, fd: i32, p: u32) -> R {
    let (m, h) = mh(c)?;
    match h.fd(fd)? {
        Fd::Null | Fd::Con => filestat(m, p, 2, 0, 0),
        Fd::Dir(path, _) | Fd::File(OpenFile { path, .. }) => {
            let path = path.clone();
            stat_path(m, p, &path)
        }
    }
}

fn fd_filestat_set_size(c: &mut C, fd: i32, size: u64) -> R {
    let f = c.data_mut().file(fd)?;
    if !f.write {
        return Err(BADF);
    }
    let cur = fs::entry(&f.path).map_err(errno)?.1;
    if size < cur { fs::truncate(&f.path, size) } else { fs::write_at(&f.path, size, &[]).map(|_| ()) }.map_err(errno)
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
            Fd::File(OpenFile { path, pos: 0, append, write })
        }
        Err("not found") if creat && !dir_only => {
            sys_ro(&path)?;
            fs::touch(&path, false).map_err(errno)?;
            Fd::File(OpenFile { path, pos: 0, append, write: true })
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
    l.func_wrap(W, "fd_read", |mut c: C, fd: i32, iov: u32, n: u32, nr: u32| ret(fd_read(&mut c, fd, iov, n, nr, None)))?;
    l.func_wrap(W, "fd_pread", |mut c: C, fd: i32, iov: u32, n: u32, off: u64, nr: u32| ret(fd_read(&mut c, fd, iov, n, nr, Some(off))))?;
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
    why: Why,
}

impl Run {
    pub fn status<const N: usize>(&self, o: &mut SBuf<N>, budget_ms: u64) {
        let _ = match (self.why, self.code) {
            (Why::Intr, _) => write!(o, "interrupted (Ctrl-C)"),
            (Why::Budget, _) => write!(o, "stopped: time budget exceeded"),
            (_, Some(c)) => write!(o, "exit {}", c),
            _ => write!(o, "trap: {}", self.msg),
        };
        let _ = write!(o, " in {} ms (budget {} ms), {} M fuel, memory {} KiB (cap {} MiB)", self.us / 1000, budget_ms,
            self.fuel / 1_000_000, self.mem >> 10, unsafe { CAP } >> 20);
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
pub fn split(s: &str) -> Result<(Vec<String>, Option<String>), &'static str> {
    let (mut words, mut stdin, mut cur, mut quoted, mut redir) = (Vec::new(), None, String::new(), false, false);
    let mut word_done = |w: String, redir: &mut bool| {
        if core::mem::take(redir) { stdin = Some(w) } else { words.push(w) }
    };
    let mut any = false;
    for ch in s.chars() {
        match ch {
            '"' => {
                quoted = !quoted;
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
    if quoted {
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
    let max = MOD_MAX.min(crate::mem::heap_free() as u64 / 3);
    if size > max {
        return Err(format!("{}: {} KiB, larger than the {} KiB the kernel heap can take now", path, size >> 10, max >> 10));
    }
    let stdin_fd = match stdin {
        None => Fd::Null,
        Some(p) => {
            let p = fs::canon(p).map_err(String::from)?;
            match fs::entry(&p) {
                Ok((false, ..)) => Fd::File(OpenFile { path: p, pos: 0, append: false, write: false }),
                Ok(_) => return Err(format!("{}: is a directory", p)),
                Err(e) => return Err(format!("{}: {}", p, e)),
            }
        }
    };
    let bytes = fs::read(&path, 0, size as usize).map_err(|e| format!("{}: {}", path, e))?.0;
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
    let module = Module::new(&engine, &bytes[..]).map_err(|e| format!("{}: not a valid wasm module: {}", path, e))?;
    drop(bytes);
    let z = |s: &str| [s.as_bytes(), b"\0"].concat();
    let host = Host {
        mem: None,
        limits: StoreLimitsBuilder::new().memory_size(unsafe { CAP }).build(),
        args: core::iter::once(z(&argv0)).chain(words[1..].iter().map(|a| z(a))).collect(),
        env: ["HOME=/", "PWD=/", "PATH=/bin", "TERM=dumb"].iter().map(|e| z(e)).collect(),
        fds: vec![Some(stdin_fd), Some(Fd::Con), Some(Fd::Con), Some(Fd::Dir(String::from("/"), true))],
        t0,
        epoch_ns: crate::sys::unix_time() * 1_000_000_000,
        rng: t0 | 1,
        nosys: Vec::new(),
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
    Ok(Run { code, msg, us: smp::tsc_to_us(smp::rdtsc() - t0), fuel: given - store.get_fuel().unwrap_or(0), mem, why: unsafe { WHY } })
}

/// run() on the private stack (upy::on_stack), with the log line; capture is reset after.
fn go(words: &[String], stdin: Option<&str>, budget_ms: u64, capture: bool) -> Result<Run, String> {
    let mut r = Err(String::new());
    crate::upy::on_stack(&mut || r = run(words, stdin, budget_ms, capture));
    unsafe { CAPTURE = false };
    let mut l = SBuf::<200>::new();
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
    let r = split(args).map_err(String::from).and_then(|(mut w, stdin)| {
        w.insert(0, String::from(path.trim()));
        go(&w, stdin.as_deref(), TOOL_MS, true)
    });
    match r {
        Ok(r) => {
            let out = unsafe { &*core::ptr::addr_of!(OUT) };
            o.push(out.as_str().trim_end().as_bytes());
            let dropped = unsafe { DROPPED };
            if dropped > 0 {
                let _ = write!(o, "\n[... {} more bytes of output dropped]", dropped);
            }
            let _ = write!(o, "{}[wasm: ", if out.n > 0 { "\n" } else { "" });
            r.status(o, TOOL_MS);
            o.push(b"]");
        }
        Err(e) => {
            let _ = write!(o, "error: {}", e);
        }
    }
}

/// Boot self-test of the command-line splitter (quotes, `<`).
pub fn selftest() -> bool {
    let ok = |s: &str, w: &[&str], i: Option<&str>| split(s).is_ok_and(|(a, b)| a.iter().map(|x| x.as_str()).eq(w.iter().copied()) && b.as_deref() == i);
    ok("rg -n \"fn main\" /src", &["rg", "-n", "fn main", "/src"], None)
        && ok("wc -l < /a.txt", &["wc", "-l"], Some("/a.txt"))
        && ok("cat <in x", &["cat", "x"], Some("in"))
        && ok("echo \"\"", &["echo", ""], None)
        && split("a \"b").is_err()
        && split("a <").is_err()
}
