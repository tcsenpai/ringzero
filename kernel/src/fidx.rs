// File indexer: text files written through the shell (!write) or the agent (write_file), and at boot every
// file that changed on the disk since the saved index (agentd), are queued here and chunked into the memory
// index (memidx.rs), one chunk (~1000 bytes, ~250 tokens) per embedding, stored as "[/path] text". agentd's
// idle scheduler runs the embeddings in slices (next_chunk / advance). Rewriting or removing a file drops its
// old chunks first. `table` lists the files whose chunks are in the index, with the size and mtime they had
// then: saved with the index, compared with the disk at the next boot. T72 (design A): a .csv or .tsv file is cut by
// rows instead, one row per chunk with the header line in front ("[/path] header\nrow").
use crate::log::SBuf;
use crate::{embed, fs, memidx};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

/// Target chunk size in bytes (Italian / English text: ~3.5-4.5 bytes per token, so ~220-290 tokens).
pub const CHUNK: usize = 1000;
/// Saved with the index: a change in cut() or the "[/path] " text format means a rebuild. 2 (T72): CSV by rows; an
/// index cut with version 1 still loads and agentd queues its .csv / .tsv files again (TABLE_VERSIONS).
pub const VERSION: u32 = 2;
pub const OLDEST: u32 = 1;

struct Job {
    path: String,
    off: u64, // next chunk starts here
    size: u64,
    chunks: u32,
    head: Option<String>, // a table's header line, once read
}

struct St {
    q: Vec<Job>,
    table: Vec<(String, u64, u32)>,
    gen: u64, // bumped when the head job may have changed (a chunk in flight is then dropped)
    chunks: u64,
    files: u64,
}

static mut ST: St = St { q: Vec::new(), table: Vec::new(), gen: 0, chunks: 0, files: 0 };

fn st() -> &'static mut St {
    unsafe { &mut *core::ptr::addr_of_mut!(ST) }
}

fn under(p: &str, root: &str) -> bool {
    p == root || p.strip_prefix(root).is_some_and(|r| r.starts_with('/'))
}

/// A file (or a whole directory) is gone or about to be rewritten: drop its chunks, queue entry and table row.
pub fn forget(path: &str) -> usize {
    let s = st();
    let path = path.trim_end_matches('/');
    s.q.retain(|j| !under(&j.path, path));
    s.table.retain(|(p, _, _)| !under(p, path));
    s.gen += 1;
    memidx::retain(|t| !(t.starts_with('[') && t[1..].split_once("] ").is_some_and(|(p, _)| under(p, path))))
}

/// Queue a file written through the shell or the agent (text files only, checked here).
pub fn enqueue(path: &str, data: &[u8]) {
    forget(path);
    if !embed::ready() || !fs::indexed(path) || !fs::is_text(data) || data.iter().all(|b| b.is_ascii_whitespace()) {
        return;
    }
    push(path, data.len() as u64);
}

/// Queue a file found on the disk (boot scan); its first 4 KiB decide whether it is text.
pub fn enqueue_path(path: &str) -> bool {
    forget(path);
    let mut b = [0u8; 4096];
    let Ok((k, size)) = fs::read_into(path, 0, &mut b) else { return false };
    let k = match core::str::from_utf8(&b[..k]) {
        Err(e) if e.error_len().is_none() => e.valid_up_to(), // a UTF-8 character cut by the 4 KiB window
        _ => k,
    };
    if !embed::ready() || !fs::is_text(&b[..k]) || b[..k].iter().all(|c| c.is_ascii_whitespace()) {
        return false;
    }
    push(path, size);
    true
}

fn push(path: &str, size: u64) {
    let s = st();
    s.q.push(Job { path: String::from(path), off: 0, size, chunks: 0, head: None });
    s.gen += 1;
}

pub fn pending() -> bool {
    !st().q.is_empty()
}
pub fn gen() -> u64 {
    st().gen
}
/// Indexed files as (path, size, mtime).
pub fn table() -> &'static [(String, u64, u32)] {
    &st().table
}
pub fn table_add(path: &str, size: u64, mtime: u32) {
    st().table.push((String::from(path), size, mtime));
}
pub fn clear() {
    let s = st();
    s.q.clear();
    s.table.clear();
    s.gen += 1;
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

/// A table file: cut by rows.
pub fn table_file(p: &str) -> bool {
    let l = p.len();
    l > 4 && (p[l - 4..].eq_ignore_ascii_case(".csv") || p[l - 4..].eq_ignore_ascii_case(".tsv"))
}

/// The first non-blank line of b as (start, end without the line break, bytes consumed with it); None = only
/// blank lines (consumed = all of b). A line longer than the window is cut like a chunk.
pub fn next_line(b: &[u8], eof: bool) -> (Option<(usize, usize)>, usize) {
    let mut i = 0;
    while i < b.len() {
        let e = b[i..].iter().position(|&c| c == b'\n').map(|k| i + k);
        let (end, next) = match e {
            Some(e) => (e, e + 1),
            None if eof => (b.len(), b.len()),
            None if i == 0 => (cut(b).max(1), cut(b).max(1)),
            None => return (None, i), // the rest of the line is past the window: read again from i
        };
        let l = &b[i..end];
        let l = if l.last() == Some(&b'\r') { &l[..l.len() - 1] } else { l };
        if l.iter().any(|c| !c.is_ascii_whitespace()) {
            return (Some((i, i + l.len())), next);
        }
        i = next;
    }
    (None, i)
}

/// Text of the head job's next chunk and the offset after it. Finished files move to the table (logged);
/// None = nothing left (or the index is full: the queue is dropped).
pub fn next_chunk() -> Option<(SBuf<{ CHUNK + 300 }>, u64)> {
    let s = st();
    while let Some(j) = s.q.first_mut() {
        if j.off >= j.size {
            let j = s.q.remove(0);
            s.files += 1;
            let (size, mt) = fs::meta(&j.path).unwrap_or((j.size, 0));
            s.table.retain(|(p, _, _)| *p != j.path);
            s.table.push((j.path.clone(), size, mt));
            crate::dlog!("index: file {} done, {} chunks ({} B)", j.path, j.chunks, size);
            continue;
        }
        let Ok((b, size)) = fs::read(&j.path, j.off, CHUNK + 8) else {
            crate::dlog!("index: {} unreadable, dropped from the queue", j.path);
            s.q.remove(0);
            continue;
        };
        j.size = size;
        let mut t = SBuf::new();
        let k = if table_file(&j.path) {
            let eof = j.off + b.len() as u64 >= size;
            let (l, used) = next_line(&b, eof);
            let Some((a, e)) = l else {
                j.off = if used == 0 { size } else { j.off + used as u64 };
                continue;
            };
            let line = core::str::from_utf8(&b[a..e]).unwrap_or("");
            if j.head.is_none() {
                j.head = Some(String::from(line)); // the header: no chunk of its own
                j.off += used as u64;
                continue;
            }
            let _ = write!(t, "[{}] {}\n{}", j.path, j.head.as_deref().unwrap_or(""), line);
            used
        } else {
            let k = cut(&b);
            let _ = write!(t, "[{}] {}", j.path, core::str::from_utf8(&b[..k]).unwrap_or(""));
            k
        };
        if memidx::room(t.as_str()).is_err() {
            crate::dlog!("index: full, file queue dropped ({} files)", s.q.len());
            s.q.clear();
            return None;
        }
        return Some((t, j.off + k.max(1) as u64));
    }
    None
}

/// The chunk that ended at `next` is stored.
pub fn advance(next: u64) {
    let s = st();
    s.chunks += 1;
    if let Some(j) = s.q.first_mut() {
        j.off = next;
        j.chunks += 1;
    }
}

pub fn stats<const N: usize>(o: &mut SBuf<N>) {
    let s = st();
    let left: u64 = s.q.iter().map(|j| j.size.saturating_sub(j.off)).sum();
    let _ = write!(o, "files: {} queued ({} bytes left{}), {} indexed this boot ({} chunks), {} in the table", s.q.len(), left,
        if let Some(j) = s.q.first() { alloc::format!(", now {}", j.path) } else { String::new() }, s.files, s.chunks, s.table.len());
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
    let csv = b"\nname,age\r\n\nAnna,30\nLuca";
    let r = next_line(csv, true) == (Some((1, 9)), 11) && next_line(&csv[11..], true) == (Some((1, 8)), 9)
        && next_line(b"Luca", true) == (Some((0, 4)), 4) && next_line(b"  \n", true) == (None, 3)
        && table_file("/demo/people.CSV") && !table_file("/a.txt");
    c1 && c2 && p && u && r
}
