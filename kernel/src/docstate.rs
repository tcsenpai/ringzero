// M4.8c (T27): precomputed document states. `/read PATH` puts a text file into the conversation (a user turn with
// the file, then a fixed kernel reply) and keeps the engine state those tokens leave, the KV rows of their
// positions and the whole DeltaNet block R, in /var/lib/ringzero/cache/docs. The state of a token depends on every
// token before it, so an entry is keyed by the boot key (model, boot prefix, engine, KV geometry), the token ids of
// the conversation before the file and the file's own token ids: it hits only at the same position after the same
// context, token for token (always true for the first /read of a fresh conversation), and then the live state is
// the one the prefill would leave, bit for bit. A changed file has other token ids, so another key; the old entry
// of the same path and context is removed when the new one is written. Size cap CAP with LRU eviction (a hit
// rewrites the magic, which refreshes the mtime) and a disk reserve. load() checks the header, the size and the
// token ids before it reads the state and a checksum after; a refused entry is deleted, the caller restores the
// current point and prefills.
use crate::mstore::{fnv, FNV0};
use crate::{agent, fs, klog, llm, println, smp, snap};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

const DIR: &str = "/var/lib/ringzero/cache/docs";
const MAGIC: &[u8; 4] = b"RZDS";
const VER: u32 = 1;
/// 80 bytes of fields, then the file's path (256 bytes, NUL padded).
const HDR: usize = 336;
/// /read takes files up to 64 KiB.
const MAX_BYTES: usize = 64 * 1024;
/// Shorter documents are prefilled every time (their state file would be mostly R: 19-50 MiB).
const MIN_TOK: usize = 512;
/// The cache's size cap and the free space it leaves on the disk.
const CAP: u64 = 512 << 20;
const RESERVE: u64 = 64 << 20;

/// FNV-1a over 64-bit words (the snapshot hashes), the byte tail folded in.
fn hw(mut h: u64, b: &[u8]) -> u64 {
    let w = b.chunks_exact(8);
    let tail = w.remainder();
    for c in w {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x100_0000_01b3);
    }
    fnv(h, tail)
}
fn ids(t: &[u32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(t.as_ptr() as *const u8, t.len() * 4) }
}
fn u64_at(h: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(h[o..o + 8].try_into().unwrap())
}
fn u32_at(h: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(h[o..o + 4].try_into().unwrap())
}

/// What a cache file must match: boot key, context key, document key, base position, token count.
struct Key {
    boot: u64,
    ctx: u64,
    doc: u64,
    base: u64,
    n: u64,
}

fn header(k: &Key, path: &str, sum: u64) -> [u8; HDR] {
    let (hb, heads, rb) = snap::geometry();
    let mut h = [0u8; HDR];
    h[..4].copy_from_slice(MAGIC);
    h[4..8].copy_from_slice(&VER.to_le_bytes());
    for (i, v) in [k.boot, k.ctx, k.doc, k.base, k.n].iter().enumerate() {
        h[8 + i * 8..16 + i * 8].copy_from_slice(&v.to_le_bytes());
    }
    h[48..52].copy_from_slice(&hb.to_le_bytes());
    h[52..56].copy_from_slice(&heads.to_le_bytes());
    h[56..64].copy_from_slice(&rb.to_le_bytes());
    h[64..72].copy_from_slice(&sum.to_le_bytes());
    let p = &path.as_bytes()[..path.len().min(HDR - 80)];
    h[72..76].copy_from_slice(&(p.len() as u32).to_le_bytes());
    h[80..80 + p.len()].copy_from_slice(p);
    h
}

fn file_bytes(n: u64) -> u64 {
    let (hb, heads, rb) = snap::geometry();
    HDR as u64 + n * 4 + 2 * heads as u64 * n * hb as u64 + rb
}

/// Write the live state of [base, base + n) (just prefilled) to `file`. Returns the bytes written.
fn save(file: &str, k: &Key, path: &str) -> Result<u64, &'static str> {
    let parts = snap::state_parts(k.base as usize, k.n as usize);
    let sum = parts.iter().fold(FNV0, |h, b| hw(h, b));
    let h = header(k, path, sum);
    let mut all: Vec<&[u8]> = alloc::vec![&h[..]];
    all.extend(parts.iter().map(|b| &b[..]));
    fs::write_parts(file, &all)
}

/// Load `file` into the live state of [base, base + n). Err((why, touched)): touched = R was overwritten (the
/// caller restores the current point); rows and token ids above the live position are not live state.
fn load(file: &str, k: &Key, toks: &[u32]) -> Result<u64, (&'static str, bool)> {
    let mut f = fs::open(file).map_err(|_| ("no entry", false))?;
    let mut h = [0u8; HDR];
    let io = |_| ("read error", false);
    if fs::read_at(&mut f, 0, &mut h).map_err(io)? != HDR || &h[..4] != MAGIC || u32_at(&h, 4) != VER {
        return Err(("bad magic or version", false));
    }
    let want = header(k, "", 0);
    if h[8..48] != want[8..48] {
        return Err(("key differs", false)); // a 64-bit name collision, or a file renamed by hand
    }
    if h[48..64] != want[48..64] {
        return Err(("KV geometry differs", false));
    }
    if fs::len(&mut f).map_err(io)? != file_bytes(k.n) {
        return Err(("size does not match the header", false));
    }
    let parts = snap::state_parts(k.base as usize, k.n as usize);
    let last = parts.len() - 1;
    let (mut off, mut sum) = (HDR as u64, FNV0);
    for (i, b) in parts.into_iter().enumerate() {
        let got = fs::read_at(&mut f, off, b).map_err(|_| ("read error", i == last))?;
        if got != b.len() {
            return Err(("short read", i == last));
        }
        if i == 0 && &*b != ids(toks) {
            return Err(("token ids differ", false));
        }
        sum = hw(sum, b);
        off += got as u64;
    }
    if sum != u64_at(&h, 64) {
        return Err(("checksum mismatch", true));
    }
    let _ = fs::write_file_at(&mut f, 0, MAGIC); // LRU: the mtime says when the entry was last used
    Ok(off)
}

struct Entry {
    file: String,
    size: u64,
    mtime: u32,
    boot: u64,
    ctx: u64,
    base: u64,
    n: u64,
    path: String,
}

/// Every file in the cache directory, with its header fields (zero for a file whose header does not parse).
fn entries() -> Vec<Entry> {
    let mut v = Vec::new();
    for (name, dir, size) in fs::list(DIR).unwrap_or_default() {
        if dir {
            continue;
        }
        let file = format!("{}/{}", DIR, name);
        let mtime = fs::meta(&file).map(|m| m.1).unwrap_or(0);
        let mut h = [0u8; HDR];
        let ok = fs::read_into(&file, 0, &mut h).is_ok_and(|(k, _)| k == HDR) && &h[..4] == MAGIC;
        let (boot, ctx, base, n, path) = if ok {
            let pl = (u32_at(&h, 72) as usize).min(HDR - 80);
            (u64_at(&h, 8), u64_at(&h, 16), u64_at(&h, 32), u64_at(&h, 40), String::from(core::str::from_utf8(&h[80..80 + pl]).unwrap_or("?")))
        } else {
            (0, 0, 0, 0, String::from("?"))
        };
        v.push(Entry { file, size, mtime, boot, ctx, base, n, path });
    }
    v
}

/// Before `file` is written: drop the older entries of the same path and context (the file changed), then evict
/// entries of another model or engine and the least recently used ones until `need` fits the cap and the reserve.
fn make_room(file: &str, need: u64, k: &Key, path: &str) -> Result<(), String> {
    if need > CAP {
        return Err(format!("{} MiB, more than the {} MiB cap", need >> 20, CAP >> 20));
    }
    let mut ents = entries();
    ents.retain(|e| {
        let stale = e.file != file && e.path == path && e.ctx == k.ctx && e.boot == k.boot;
        if stale && fs::rm(&e.file, false).is_ok() {
            klog!(b'I', "doc: {} changed, old cache entry {} removed", path, e.file);
        }
        !stale
    });
    loop {
        let total: u64 = ents.iter().map(|e| e.size).sum();
        let free = fs::free_bytes().map_err(String::from)?;
        if total + need <= CAP && free >= need + RESERVE {
            return Ok(());
        }
        let i = ents.iter().position(|e| e.boot != k.boot)
            .or_else(|| (0..ents.len()).min_by_key(|&i| ents[i].mtime))
            .ok_or_else(|| format!("{} MiB free on the disk, {} MiB needed plus a {} MiB reserve", free >> 20, need >> 20, RESERVE >> 20))?;
        let e = ents.swap_remove(i);
        fs::rm(&e.file, false).map_err(String::from)?;
        klog!(b'I', "doc: evicted {} ({}, {} MiB)", e.file, e.path, e.size >> 20);
    }
}

fn secs(us: u64) -> (u64, u64) {
    (us / 1_000_000, us / 1000 % 1000)
}

/// `/read PATH | clear` (no argument: the cache)
pub fn cmd(arg: &str) {
    match arg {
        "" => list(),
        "clear" => {
            let ents = entries();
            let mib = ents.iter().map(|e| e.size).sum::<u64>() >> 20;
            match fs::rm(DIR, true) {
                Ok(()) => println!("[read] cache cleared: {} entries, {} MiB", ents.len(), mib),
                Err(e) => println!("[read] nothing to clear ({})", e),
            }
        }
        p => read(p),
    }
}

fn list() {
    let ents = entries();
    let mib = ents.iter().map(|e| e.size).sum::<u64>() >> 20;
    let free = fs::free_bytes().map(|b| b >> 20).unwrap_or(0);
    println!("[read] document cache {}: {} entries, {} MiB of {} MiB, {} MiB free on the disk", DIR, ents.len(), mib, CAP >> 20, free);
    let boot = agent::boot_key();
    for e in &ents {
        println!("  {}  {}  {} tok at ctx {}  {} MiB{}", e.file.rsplit('/').next().unwrap_or(""), e.path, e.n, e.base, e.size >> 20,
            if e.boot != boot { "  (another model or engine: evicted first)" } else { "" });
    }
}

fn read(arg: &str) {
    let p = match fs::canon(arg) {
        Ok(p) => p,
        Err(e) => return println!("[read] {}: {}", arg, e),
    };
    let (b, size) = match fs::read(&p, 0, MAX_BYTES + 1) {
        Ok(r) => r,
        Err(e) => return println!("[read] {}: {}", p, e),
    };
    if size as usize > MAX_BYTES {
        return println!("[read] {} has {} bytes; /read takes files up to {} bytes", p, size, MAX_BYTES);
    }
    if !fs::is_text(&b) {
        return println!("[read] {}: not a text file", p);
    }
    if snap::unsaved() {
        return println!("[read] the last turn is not saved (snapshot pool full): /drop something first");
    }
    let base = snap::ensure_live();
    // the file's text is encoded without special tokens: a "<|im_end|>" inside it stays text
    let mut head = String::new();
    if snap::at_root() {
        head.push_str("<|im_end|>\n"); // a fresh conversation: close the system turn, as agent::ask does
    }
    let _ = write!(head, "<|im_start|>user\nHere is the file {} ({} bytes). Read it, my questions about it follow.\n\n", p, size);
    let ack = format!("I have read {}.", p);
    let tail = format!("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n", ack);
    let mut toks = alloc::vec![0u32; head.len() + b.len() + tail.len() + 16];
    let mut n = llm::encode(head.as_bytes(), &mut toks);
    n += llm::encode_plain(&b, &mut toks[n..]);
    n += llm::encode(tail.as_bytes(), &mut toks[n..]);
    toks.truncate(n);
    let max_seq = llm::ctx_max_seq();
    if base + n as u64 + 600 >= max_seq {
        return println!("[read] {} is {} tokens: it does not fit in the context ({}/{} in use; /clear starts over)", p, n, base, max_seq);
    }
    let root = snap::root_pos() as usize;
    let boot = agent::boot_key();
    let k = Key { boot, ctx: hw(boot, ids(&llm::rec()[root..base as usize])), doc: hw(FNV0, ids(&toks)), base, n: n as u64 };
    let file = format!("{}/{:016x}.bin", DIR, fnv(k.ctx, &k.doc.to_le_bytes()));
    let cache = n >= MIN_TOK && fs::mounted() && crate::layout::pending().is_none();
    let t0 = smp::rdtsc();
    let mut hit = None;
    if cache {
        match load(&file, &k, &toks) {
            Ok(bytes) => hit = Some(bytes),
            Err(("no entry", _)) => {}
            Err((e, touched)) => {
                klog!(b'W', "doc: cache entry {} for {} refused ({}), deleted; prefilling", file, p, e);
                let _ = fs::rm(&file, false);
                if touched {
                    snap::rollback();
                }
            }
        }
    }
    let t1 = smp::rdtsc();
    if let Some(bytes) = hit {
        let (s, ms) = secs(smp::tsc_to_us(t1 - t0));
        klog!(b'I', "doc: {} {} tok at ctx {}: cache hit, {} MiB of state loaded in {}.{:03} s instead of a prefill", p, n, base, bytes >> 20, s, ms);
    } else {
        smp::park_phase(true);
        let mut pos = base;
        let ok = llm::feed(&toks, &mut pos);
        smp::park_phase(false);
        if !ok {
            let at = snap::rollback();
            return println!("^C [read] interrupted, the conversation is back at ctx {}", at);
        }
        let t2 = smp::rdtsc();
        let (s, ms) = secs(smp::tsc_to_us(t2 - t1));
        let mut why = String::from("not cached: ");
        if !cache {
            why.push_str(if n < MIN_TOK { "shorter than 512 tokens" } else { "no disk, or the layout migration is pending" });
        } else {
            let need = file_bytes(k.n);
            why = match make_room(&file, need, &k, &p).and_then(|_| save(&file, &k, &p).map_err(String::from)) {
                Ok(w) => {
                    let (ws, wms) = secs(smp::tsc_to_us(smp::rdtsc() - t2));
                    format!("cached as {} ({} MiB, written in {}.{:03} s)", file, w >> 20, ws, wms)
                }
                Err(e) => {
                    let _ = fs::rm(&file, false); // no half file
                    format!("not cached ({})", e)
                }
            };
        }
        klog!(b'I', "doc: {} {} tok at ctx {}: prefilled in {}.{:03} s ({} tok/s), {}", p, n, base, s, ms,
            n as u64 * 1_000_000 / smp::tsc_to_us(t2 - t1).max(1), why);
    }
    let t3 = smp::rdtsc();
    snap::commit(base + n as u64, &format!("/read {}", p), &ack, false, "", 0);
    let (s, ms) = secs(smp::tsc_to_us(smp::rdtsc() - t0));
    println!("[read] {} is in the context ({} tokens, ctx {}), {}.{:03} s in all (snapshot {} ms); ask about it", p, n, snap::pos(), s, ms,
        smp::tsc_to_us(smp::rdtsc() - t3) / 1000);
    agent::note(&format!("[read] {} ({} tokens) put into the context", p, n));
}
