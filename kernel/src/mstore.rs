// agentd's persistent stores on the FAT volume (/sys/mem): append-only files of records
//   'R' kind:u8 len:u32 ts:u64 sum:u32 payload[len]   (little endian, ts = Unix seconds from the RTC)
// sum = folded FNV-1a over kind, ts and payload. FAT has no journaling, so a crash in the middle of an append
// can leave a torn tail: load() scans to the last good record and truncates the file there (logged).
use crate::fs;
use alloc::vec::Vec;

pub const JOURNAL: &str = "/sys/mem/journal.log";
pub const FACTS: &str = "/sys/mem/facts.log";
pub const EPISODES: &str = "/sys/mem/episodes.log";
const HDR: usize = 18;
pub const MAX_REC: usize = 16 << 10;
pub const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

pub fn fnv(mut h: u64, b: &[u8]) -> u64 {
    for &x in b {
        h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Hash of a large module (model file) from a sample: size, first and last 64 KiB, 256 x 4 KiB spread out.
/// shortcut: a sample, not the whole 2 GB (that is seconds of boot); a full sha256 belongs to the build.
pub fn sample_hash(p: *const u8, size: u64) -> u64 {
    let b = unsafe { core::slice::from_raw_parts(p, size as usize) };
    let mut h = fnv(FNV0, &size.to_le_bytes());
    let k = b.len().min(64 << 10);
    h = fnv(h, &b[..k]);
    h = fnv(h, &b[b.len() - k..]);
    let step = (b.len() / 256).max(1);
    for i in 0..256 {
        let o = (i * step).min(b.len().saturating_sub(4096));
        h = fnv(h, &b[o..(o + 4096).min(b.len())]);
    }
    h
}

fn sum(kind: u8, ts: u64, p: &[u8]) -> u32 {
    let h = fnv(fnv(fnv(FNV0, &[kind]), &ts.to_le_bytes()), p);
    (h ^ (h >> 32)) as u32
}

pub fn encode(kind: u8, ts: u64, p: &[u8], out: &mut Vec<u8>) {
    out.push(b'R');
    out.push(kind);
    out.extend_from_slice(&(p.len() as u32).to_le_bytes());
    out.extend_from_slice(&ts.to_le_bytes());
    out.extend_from_slice(&sum(kind, ts, p).to_le_bytes());
    out.extend_from_slice(p);
}

/// Append one record (payload cut to MAX_REC bytes by the caller); returns the store size after it.
pub fn append(path: &str, kind: u8, p: &str) -> Result<u64, &'static str> {
    let p = &p[..crate::agent::floor_char(p, MAX_REC)];
    let mut v = Vec::with_capacity(HDR + p.len());
    encode(kind, crate::sys::unix_time(), p.as_bytes(), &mut v);
    fs::append(path, &v)
}

pub struct Rec<'a> {
    pub kind: u8,
    pub ts: u64,
    pub off: u64,
    pub p: &'a str,
}

/// Call f on each good record of buf, in order; returns the end of the last good one.
pub fn parse(buf: &[u8], mut f: impl FnMut(Rec)) -> usize {
    let mut o = 0;
    while o + HDR <= buf.len() {
        let b = &buf[o..];
        let len = u32::from_le_bytes([b[2], b[3], b[4], b[5]]) as usize;
        if b[0] != b'R' || len > MAX_REC || HDR + len > b.len() {
            break;
        }
        let ts = u64::from_le_bytes(b[6..14].try_into().unwrap());
        let p = &b[HDR..HDR + len];
        if sum(b[1], ts, p) != u32::from_le_bytes([b[14], b[15], b[16], b[17]]) {
            break;
        }
        let Ok(p) = core::str::from_utf8(p) else { break };
        f(Rec { kind: b[1], ts, off: o as u64, p });
        o += HDR + len;
    }
    o
}

/// Boot: read a store, truncate a torn tail (logged), call f on every good record.
/// Returns (records, good bytes, bytes cut). A missing store is empty.
pub fn load(path: &str, mut f: impl FnMut(Rec)) -> (usize, u64, u64) {
    let Ok((buf, size)) = fs::read(path, 0, usize::MAX) else { return (0, 0, 0) };
    let mut n = 0;
    let good = parse(&buf, |r| {
        n += 1;
        f(r)
    }) as u64;
    if good < size {
        let r = fs::truncate(path, good);
        crate::dlog!("recover {}: {} good records, torn tail of {} bytes at offset {} {}", path, n, size - good, good,
            if r.is_ok() { "truncated" } else { "NOT truncated (I/O error)" });
    }
    (n, good, size - good)
}

/// Boot self-check of the record codec and the torn-tail scan (no disk).
pub fn selftest() -> bool {
    let mut v = Vec::new();
    encode(b'U', 1, b"mi chiamo Cris", &mut v);
    encode(b'A', 2, "ciao è".as_bytes(), &mut v);
    let two = v.len();
    encode(b'U', 3, b"third", &mut v);
    let mut k = 0;
    let all = parse(&v, |r| k += (r.kind == b'U' && r.p == "third") as usize) == v.len() && k == 1;
    let torn = parse(&v[..v.len() - 3], |_| {}) == two; // an append cut short
    let mut bad = v.clone();
    bad[two + HDR] ^= 1; // a flipped payload byte in the last record
    let flip = parse(&bad, |_| {}) == two;
    all && torn && flip && fnv(FNV0, b"a") == 0xaf63_dc4c_8601_ec8c
}
