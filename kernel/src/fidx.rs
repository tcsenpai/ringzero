// File indexer: text files written through the shell (!write) or the agent (write_file) are queued here and
// embedded into the memory index (memidx.rs) in idle time at the shell prompt, one chunk (~1000 bytes,
// ~250 tokens) per embedding, stored as "[/path] text". The work runs in slices between forward calls of the
// embedder (embed::begin/resume): any input or Ctrl-C ends a slice within one forward call (~6 ms), and a
// slice never runs longer than SLICE_MS, so the shell never waits for the indexer. Rewriting or removing a
// file drops its old chunks first. In RAM only, like the index (persistence is M4.5).
use crate::log::SBuf;
use crate::{embed, fs, memidx, smp};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

/// Target chunk size in bytes (Italian / English text: ~3.5-4.5 bytes per token, so ~220-290 tokens).
pub const CHUNK: usize = 1000;
const SLICE_MS: u64 = 100;

struct Job {
    path: String,
    off: u64, // next chunk starts here
    size: u64,
}

struct St {
    q: Vec<Job>,
    // chunk being embedded: embed job id, its text, the offset after it
    cur: Option<(u64, SBuf<{ CHUNK + 300 }>, u64)>,
    chunks: u64,
    files: u64,
    us: u64,
}

static mut ST: St = St { q: Vec::new(), cur: None, chunks: 0, files: 0, us: 0 };

fn st() -> &'static mut St {
    unsafe { &mut *core::ptr::addr_of_mut!(ST) }
}

fn under(p: &str, root: &str) -> bool {
    p == root || p.strip_prefix(root).is_some_and(|r| r.starts_with('/'))
}

/// A file (or a whole directory) is gone or about to be rewritten: drop its chunks and its queue entry.
pub fn forget(path: &str) -> usize {
    let s = st();
    let path = path.trim_end_matches('/');
    s.q.retain(|j| !under(&j.path, path));
    s.cur = None; // the chunk in flight may be this file's: the head starts its chunk over
    memidx::retain(|t| !(t.starts_with('[') && t[1..].split_once("] ").is_some_and(|(p, _)| under(p, path))))
}

/// Queue a file written through the shell or the agent (text files only, checked here).
pub fn enqueue(path: &str, data: &[u8]) {
    forget(path);
    if !embed::ready() || !fs::is_text(data) || data.iter().all(|b| b.is_ascii_whitespace()) {
        return;
    }
    let s = st();
    s.q.push(Job { path: String::from(path), off: 0, size: data.len() as u64 });
    s.cur = None; // the head may have changed
}

pub fn pending() -> bool {
    !st().q.is_empty()
}

/// Cut text at a chunk boundary: at most CHUNK bytes, preferring a newline, then a space, past CHUNK/2.
pub fn cut(b: &[u8]) -> usize {
    if b.len() <= CHUNK {
        return b.len();
    }
    let w = &b[..CHUNK];
    let at = |c: u8| w[CHUNK / 2..].iter().rposition(|&x| x == c).map(|i| CHUNK / 2 + i + 1);
    let mut k = at(b'\n').or_else(|| at(b' ')).unwrap_or(CHUNK);
    while k > 0 && (b[k] & 0xC0) == 0x80 {
        k -= 1; // UTF-8 boundary
    }
    k
}

/// Next chunk of the head job into a fresh embedding job. false = nothing left to start.
fn start_chunk() -> bool {
    let s = st();
    while let Some(j) = s.q.first_mut() {
        if j.off >= j.size {
            s.q.remove(0);
            s.files += 1;
            continue;
        }
        let Ok((b, size)) = fs::read(&j.path, j.off, CHUNK + 8) else {
            s.q.remove(0); // removed or unreadable meanwhile
            continue;
        };
        j.size = size;
        let k = cut(&b);
        let mut t = SBuf::new();
        let _ = write!(t, "[{}] {}", j.path, core::str::from_utf8(&b[..k]).unwrap_or(""));
        if memidx::room(t.as_str()).is_err() {
            crate::klog!(b'W', "index: full, file queue dropped ({} files)", s.q.len());
            s.q.clear();
            return false;
        }
        match embed::begin(t.as_str(), false) {
            Ok(id) => {
                let next = j.off + k.max(1) as u64;
                s.cur = Some((id, t, next));
                return true;
            }
            Err(_) => return false,
        }
    }
    false
}

/// Called by the shell's line editor while it waits for input. Runs at most one slice; true if it worked.
pub fn idle(stop: &mut dyn FnMut() -> bool) -> bool {
    if !pending() || !fs::mounted() {
        return false;
    }
    let s = st();
    let t0 = smp::rdtsc();
    let deadline = t0 + smp::us_to_tsc(SLICE_MS * 1000);
    smp::park_phase(true);
    let mut v = [0.0f32; embed::DIM];
    while !stop() && smp::rdtsc() < deadline {
        if s.cur.is_none() && !start_chunk() {
            break;
        }
        let Some((id, _, _)) = s.cur.as_ref() else { break };
        match embed::resume(*id, memidx::dim(), &mut v, || stop() || smp::rdtsc() >= deadline) {
            Ok(None) => break, // stopped: resume next time
            Ok(Some(info)) => {
                let (_, t, next) = s.cur.take().unwrap();
                if memidx::store(t.as_str(), &v[..memidx::dim()], info).is_ok() {
                    s.chunks += 1;
                }
                if let Some(j) = s.q.first_mut() {
                    j.off = next;
                }
            }
            Err(_) => s.cur = None, // another embedding ran in between: this chunk starts over
        }
    }
    smp::park_phase(false);
    s.us += smp::tsc_to_us(smp::rdtsc() - t0);
    true
}

pub fn stats<const N: usize>(o: &mut SBuf<N>) {
    let s = st();
    let left: u64 = s.q.iter().map(|j| j.size.saturating_sub(j.off)).sum();
    let _ = write!(o, "files: {} queued ({} bytes left{}), {} indexed ({} chunks, {} ms of idle time)", s.q.len(), left,
        if let Some(j) = s.q.first() { alloc::format!(", now {}", j.path) } else { String::new() }, s.files, s.chunks, s.us / 1000);
}

/// Boot self-check of the chunker and the path checks (no disk or model involved).
pub fn selftest() -> bool {
    let mut big = [b'a'; 2500];
    big[700] = b' ';
    big[900] = b'\n';
    let c1 = cut(&big) == 901 && cut(b"short") == 5;
    let mut nosp = [b'x'; 1500];
    nosp[999] = 0xC3; // a 2-byte char across the cut
    nosp[1000] = 0xA8;
    let c2 = cut(&nosp) == 999;
    let p = fs::check("/notes/todo.txt") == Ok("notes/todo.txt") && fs::check("/") == Ok("") && fs::check("/a/").is_ok()
        && fs::check("notes").is_err() && fs::check("/a/../b").is_err() && fs::check("/a//b").is_err() && fs::check("/./a").is_err();
    let u = under("/notes/a.txt", "/notes") && !under("/notesx/a", "/notes") && under("/a", "/a");
    c1 && c2 && p && u
}
