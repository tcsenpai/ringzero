// agentd (M4.5): the deterministic kernel subsystem that owns the agent's context and memory. The 2B model is
// small, so the OS does the bookkeeping; the model reads and writes memory only through typed tools (remember,
// forget, recall). Every decision is logged: kernel log lines at level D, and /agentd log.
//  - stores (mstore.rs, /sys/mem): journal of turns and tool results, facts (key -> value, source, time),
//    episodes (session summaries); torn tails are cut at boot
//  - memory index persistence: memidx saved to /sys/mem/index.bin with the indexed-file table and the store
//    offsets it covers; at boot it is loaded, and only what changed is embedded again
//  - idle scheduler: one priority queue on the BSP, run in slices from the shell's line editor that stop
//    between engine steps on any input: 1 embed memory texts, 2 embed file chunks, 3 save the index,
//    4 consolidate (after CONS_IDLE_S idle seconds; it ages, so a long file queue cannot starve it)
//  - context compiler: at the start of a conversation and after a compaction, a slot of at most SLOT_BUDGET
//    tokens (facts, then notes recalled for the user's first message) is put in front of that message
//  - consolidation: the 2B (greedy, non-thinking, fixed instruction) summarizes the session into an episode
//    and proposes facts; the kernel validates, dedups and stores them. It runs on top of the current
//    snapshot point and restores that point after, so the user's conversation is untouched.
use crate::engine::{CAP, CAPTURE, ECHO};
use crate::log::SBuf;
use crate::mstore::{self, EPISODES, FACTS, JOURNAL};
use crate::{embed, fidx, fs, llm, memidx, smp, snap};
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

pub const INDEX: &str = "/sys/mem/index.bin";
const STORES: [&str; 3] = [JOURNAL, FACTS, EPISODES];
/// index text prefix of each store's entries
const KIND: [&str; 3] = ["[said] ", "[fact] ", "[episode "];
const SLICE_MS: u64 = 100;
/// Token budget of the memory slot (`!set slot N`, 0 = no slot).
pub static mut SLOT_BUDGET: usize = 200;
const FACT_BUDGET: usize = 120;
const RECALL_K: usize = 4;
// T35: what a recalled note needs to enter the slot (tuned on the slot's logged candidates, docs/M45_RESULT.md).
// Unrelated earlier prompts and short files scored 0.45-0.55 against almost any query.
/// embedding cosine for a note found by the embedding alone
const RECALL_MIN: f32 = 0.55;
/// cosine floor for a note the BM25 gate brought in (a shared word, e.g. a name, with no meaning in common)
const RECALL_MIN_BM25: f32 = 0.40;
/// shortest note body, in characters ("ciao", "hello", "comprare il latte" carry nothing for a new message)
const NOTE_MIN: usize = 20;
/// recalled lines at most (facts are pinned and not counted)
const SLOT_LINES: usize = 3;
const MAX_FACTS: usize = 64;
const CONS_MAX: u64 = 220;
/// Idle seconds at the prompt before a session is consolidated (0 = only /consolidate). `!set consolidate N`.
pub static mut CONS_IDLE_S: u64 = 60;
const SLOT_HEAD: &str = "[memory from earlier conversations, kept by the kernel; use it to answer]\n";
/// T22: the built-in manual, one index entry per `## ` section (docs/MANUAL.md, compiled into the kernel)
const MANUAL: &str = include_str!("../../docs/MANUAL.md");
const MAN: &str = "[manual: ";
const MAN_HEAD: &str = "[from the ringzero manual, kept by the kernel; use it to answer]\n";
/// a manual section's retrieval line (Italian and English wordings): embedded, not shown to the model
const ASKED: &str = "Asked as:";
/// Plain "FACT: key = value" lines. Two other wordings were measured and were worse with the 2B (format
/// examples: swapped keys and values; remember tool calls: no EPISODE line), see docs/M45_RESULT.md.
const CONS_ASK: &str = "[kernel] Memory consolidation (internal: the user does not see this reply). Write what is \
worth keeping from this conversation, in exactly this format:\nEPISODE: one or two sentences on what the user asked \
and what happened.\nFACT: key = value\nOne FACT line per stable fact the user stated about themselves, their \
preferences or this machine (key in English snake_case, value short and in the user's own words), at most 6, none \
if there are none. Nothing else.";

// ---- decision log ----

const DL: usize = 128;
const DW: usize = 168;
static mut DRING: [[u8; DW]; DL] = [[0; DW]; DL];
static mut DLEN: [u8; DL] = [0; DL];
static mut DN: usize = 0;
static mut BOOTING: bool = true;

/// Log one decision: kernel log (level D) and the agentd ring; on the console too while booting.
pub fn decide(a: core::fmt::Arguments) {
    let mut s = SBuf::<DW>::new();
    let t = crate::log::ms_since_boot();
    let _ = write!(s, "[{:5}.{:03}] ", t / 1000, t % 1000);
    let head = s.n;
    let _ = s.write_fmt(a);
    let msg = &s.as_str()[head..];
    crate::log::record(b'D', msg);
    if unsafe { BOOTING } {
        crate::println!("[agentd] {}", msg);
    }
    unsafe {
        let i = DN % DL;
        DRING[i] = s.b;
        DLEN[i] = s.n as u8;
        DN += 1;
    }
}

#[macro_export]
macro_rules! dlog {
    ($($t:tt)*) => { $crate::agentd::decide(format_args!($($t)*)) };
}

pub fn log_last(n: usize) {
    let k = n.min(unsafe { DN }).min(DL);
    for j in unsafe { DN } - k..unsafe { DN } {
        let i = j % DL;
        crate::println!("{}", unsafe { core::str::from_utf8_unchecked(&DRING[i][..DLEN[i] as usize]) });
    }
}

// ---- state ----

pub struct Fact {
    pub key: String,
    pub value: String,
    pub source: String,
    pub ts: u64,
}

#[derive(Clone, Copy, PartialEq)]
enum Job {
    Mem,
    File,
    Save,
    Cons,
}
impl Job {
    fn name(self) -> &'static str {
        match self {
            Job::Mem => "embed memory texts",
            Job::File => "embed file chunks",
            Job::Save => "save the index",
            Job::Cons => "consolidate the session",
        }
    }
}

enum Src {
    Mem,
    File(u64, u64), // offset after the chunk, fidx gen at its start
}
struct Emb {
    id: u64,
    text: SBuf<{ fidx::CHUNK + 300 }>,
    src: Src,
}

struct Cons {
    toks: Vec<u32>,
    i: usize,
    pos: u64,
    n_gen: u64,
    next: u64,
    t0: u64,
    temp: u32,
    why: &'static str,
}

struct St {
    on: bool,
    facts: Vec<Fact>,
    memq: VecDeque<String>,
    cur: Option<Emb>,
    cons: Option<Cons>,
    saved_gen: u64,
    size: [u64; 3], // store sizes now
    off: [u64; 3],  // store sizes covered by the saved index
    n_ep: u64,
    n_rec: [u64; 3],
    session: String, // what the user said since the last consolidation (grounding for facts)
    session_turns: u32,
    n_cons: u64,
    after_compact: bool,
    last_input: u64,
    last: Option<Job>,
    waited: u64, // slices a due consolidation waited behind other jobs
    ms: [u64; 4],
    preempted: u64,
    said: Option<String>, // T35: the last user statement, indexed as [said] once its turn ends without acting
    acted: bool,          // that turn called a tool other than remember / forget / recall
    howto: bool,          // T22: the current user message asks how to do something
}

static mut ST: St = St {
    on: false, facts: Vec::new(), memq: VecDeque::new(), cur: None, cons: None, saved_gen: 0, size: [0; 3],
    off: [0; 3], n_ep: 0, n_rec: [0; 3], session: String::new(), session_turns: 0, n_cons: 0, after_compact: false,
    last_input: 0, last: None, waited: 0, ms: [0; 4], preempted: 0, said: None, acted: false,
    howto: false,
};

fn st() -> &'static mut St {
    unsafe { &mut *core::ptr::addr_of_mut!(ST) }
}

pub fn facts() -> &'static [Fact] {
    &st().facts
}

// ---- stores: replay and append ----

fn drop_entries(prefix: &str) {
    let s = st();
    memidx::retain(|t| !t.starts_with(prefix));
    s.memq.retain(|t| !t.starts_with(prefix));
}

fn fact_text(k: &str, v: &str) -> String {
    format!("[fact] {}: {}", k, v)
}

/// Apply one record: `state` = update the RAM facts, `queue` = queue its index text. `ep` = episode number.
fn apply(store: usize, r: &mstore::Rec, state: bool, queue: bool, ep: u64) {
    let s = st();
    match (store, r.kind) {
        (0, b'U') => said_begin(r.p, queue),
        (0, b'T') => said_tool(r.p.split(" -> ").next().unwrap_or("")),
        (1, b'S') => {
            let mut it = r.p.splitn(3, '\t');
            let (k, v, src) = (it.next().unwrap_or(""), it.next().unwrap_or(""), it.next().unwrap_or(""));
            if state {
                s.facts.retain(|f| f.key != k);
                s.facts.push(Fact { key: k.into(), value: v.into(), source: src.into(), ts: r.ts });
            }
            if queue {
                drop_entries(&format!("[fact] {}: ", k));
                s.memq.push_back(fact_text(k, v));
            }
        }
        (1, b'X') => {
            if state {
                s.facts.retain(|f| f.key != r.p);
            }
            if queue {
                drop_entries(&format!("[fact] {}: ", r.p));
            }
        }
        (2, b'E') if queue => s.memq.push_back(format!("[episode {}] {}", ep, r.p)),
        _ => {}
    }
}

fn append(store: usize, kind: u8, p: &str) {
    let s = st();
    if !s.on {
        return;
    }
    match mstore::append(STORES[store], kind, p) {
        Ok(n) => {
            s.size[store] = n;
            s.n_rec[store] += 1;
        }
        Err(e) => crate::dlog!("store {}: append failed ({}), record lost", STORES[store], e),
    }
}

/// Replay the stores from disk (recovering torn tails). Records at or past from[i] are queued for the index.
fn replay(state: bool, from: [u64; 3]) {
    let s = st();
    let each = |i: usize, state: bool, from: u64| {
        let mut ep = 0;
        let r = mstore::load(STORES[i], |r| {
            ep += (r.kind == b'E') as u64;
            apply(i, &r, state, r.off >= from, ep)
        });
        said_end();
        (r, ep)
    };
    for i in 0..3 {
        let ((n, good, _), ep) = each(i, state, from[i]);
        if i == 2 {
            s.n_ep = ep;
        }
        s.size[i] = good;
        s.n_rec[i] = n as u64;
        if good < from[i] {
            crate::dlog!("index: {} is shorter than the index covers ({} < {}): its entries are rebuilt", STORES[i], good, from[i]);
            drop_entries(KIND[i]);
            each(i, false, 0);
        }
    }
}

/// Compare the disk with the indexed-file table: queue new and changed files, drop removed ones.
fn scan_files() {
    let mut seen: Vec<(String, u64, u32)> = Vec::new();
    if let Err(e) = fs::walk(&mut |p, size, mt| seen.push((p.into(), size, mt))) {
        crate::dlog!("scan: walk failed ({})", e);
        return;
    }
    let (mut same, mut new, mut changed, mut gone) = (0, 0, 0, 0);
    let table: Vec<(String, u64, u32)> = fidx::table().to_vec();
    for (p, size, mt) in &seen {
        match table.iter().find(|(q, _, _)| q == p) {
            Some((_, s0, m0)) if s0 == size && m0 == mt => same += 1,
            Some((_, s0, m0)) => {
                changed += 1;
                let q = fidx::enqueue_path(p);
                crate::dlog!("scan: {} changed on disk (size {} -> {}, mtime {:#x} -> {:#x}){}", p, s0, size, m0, mt,
                    if q { ": re-embedding it" } else { ": not text, chunks dropped" });
            }
            None => {
                if fidx::enqueue_path(p) {
                    new += 1;
                    if new <= 8 {
                        crate::dlog!("scan: {} ({} B) is not in the index: queued", p, size);
                    }
                }
            }
        }
    }
    for (p, _, _) in table.iter().filter(|(q, _, _)| !seen.iter().any(|(p, _, _)| p == q)) {
        gone += 1;
        let k = fidx::forget(p);
        crate::dlog!("scan: {} is gone from the disk: {} chunks dropped", p, k);
    }
    crate::dlog!("scan: {} files on disk: {} unchanged (kept, not re-embedded), {} changed, {} new, {} removed", seen.len(), same, changed, new, gone);
}

/// Throw the index away and queue everything again (header mismatch, /index clear).
pub fn rebuild(why: &str) {
    let s = st();
    crate::dlog!("index: rebuild ({}): every file and memory record is embedded again", why);
    memidx::clear();
    fidx::clear();
    s.memq.clear();
    s.cur = None;
    s.off = [0; 3];
    replay(false, [0; 3]);
    scan_files();
    queue_manual();
}

/// Boot, after the embedder is up: recover the stores, load the index (or rebuild it), scan the files.
pub fn init() {
    let s = st();
    if !fs::mounted() {
        crate::dlog!("no disk: memory stores and index persistence are off (RAM only)");
        if embed::ready() {
            queue_manual();
        }
        return;
    }
    s.on = true;
    let _ = fs::mkdir("/sys/mem");
    let t0 = smp::rdtsc();
    let mut why: Option<&str> = None;
    if !embed::ready() {
        why = Some("no embedder");
    } else {
        match memidx::load(INDEX, embed::hash(), fidx::VERSION) {
            Ok(extra) => {
                for l in core::str::from_utf8(&extra).unwrap_or("").lines() {
                    let f: Vec<&str> = l.split('\t').collect();
                    match (f[0], f.len()) {
                        ("S", 4) => (0..3).for_each(|i| s.off[i] = f[i + 1].parse().unwrap_or(0)),
                        ("F", 4) => fidx::table_add(f[1], f[2].parse().unwrap_or(0), f[3].parse().unwrap_or(0)),
                        _ => {}
                    }
                }
                let us = smp::tsc_to_us(smp::rdtsc() - t0);
                crate::dlog!("index: loaded {} entries from {} in {}.{:03} ms ({} files in the table), no re-embedding",
                    memidx::len(), INDEX, us / 1000, us % 1000, fidx::table().len());
            }
            Err(e) => why = Some(e),
        }
    }
    s.saved_gen = memidx::gen();
    if let Some(w) = why {
        crate::dlog!("index: not loaded ({}): rebuilding", w);
        memidx::clear();
        fidx::clear();
        s.off = [0; 3];
    }
    replay(true, s.off);
    if embed::ready() {
        scan_files();
        queue_manual();
    } else {
        s.memq.clear();
    }
    append(0, b'B', "boot");
    crate::dlog!("boot: {} facts, {} episodes, journal {} records ({} KiB); queued for the index: {} memory texts, files: {}",
        s.facts.len(), s.n_ep, s.n_rec[0], s.size[0] >> 10, s.memq.len(), if fidx::pending() { "yes" } else { "none" });
    s.last_input = smp::rdtsc();
}

/// The shell is up: decisions stop printing on the console.
pub fn booted() {
    unsafe { BOOTING = false };
}

fn index_extra() -> String {
    let s = st();
    let mut o = format!("S\t{}\t{}\t{}\n", s.size[0], s.size[1], s.size[2]);
    for (p, size, mt) in fidx::table() {
        let _ = write!(o, "F\t{}\t{}\t{}\n", p, size, mt);
    }
    o
}

fn queue_empty() -> bool {
    let s = st();
    s.memq.is_empty() && s.cur.is_none() && !fidx::pending()
}

fn save_index(why: &str) {
    let s = st();
    let t0 = smp::rdtsc();
    match memidx::save(INDEX, embed::hash(), fidx::VERSION, index_extra().as_bytes()) {
        Ok(b) => crate::dlog!("index: saved {} entries ({} KiB) in {} ms ({})", memidx::len(), b >> 10, smp::tsc_to_us(smp::rdtsc() - t0) / 1000, why),
        Err(e) => crate::dlog!("index: save failed ({}), kept in RAM only", e),
    }
    s.saved_gen = memidx::gen();
    s.off = s.size;
}

/// Before poweroff / reboot: save the index if it changed and nothing is still queued.
pub fn shutdown() {
    said_end();
    let s = st();
    if !s.on || !embed::ready() {
        return;
    }
    append(0, b'P', "poweroff");
    if memidx::gen() == s.saved_gen {
        crate::dlog!("shutdown: index unchanged since the last save");
    } else if queue_empty() {
        save_index("shutdown");
    } else {
        crate::dlog!("shutdown: index not saved, {} texts and files still queued; the next boot embeds them from the last saved index", s.memq.len());
    }
}

// ---- T22: the built-in manual ----

/// The manual's sections as index texts, `[manual: TITLE] body` (the intro before the first `## ` is not indexed).
pub fn manual() -> impl Iterator<Item = String> {
    MANUAL.split("\n## ").skip(1).map(|sec| {
        let (t, b) = sec.split_once('\n').unwrap_or((sec, ""));
        format!("{}{}] {}", MAN, t.trim(), b.trim())
    })
}

/// Boot and rebuild: an indexed section whose text is not in this kernel's manual any more is dropped, a
/// section not in the index is queued (so a changed section is embedded again, the others are kept).
fn queue_manual() {
    let s = st();
    let cur: Vec<String> = manual().collect();
    let dropped = memidx::retain(|t| !t.starts_with(MAN) || cur.iter().any(|c| c == t));
    s.memq.retain(|t| !t.starts_with(MAN));
    let n = cur.len();
    let rows = baked(&cur);
    let (mut q, mut b) = (0, 0);
    for (i, t) in cur.iter().enumerate().filter(|(_, t)| !memidx::contains(t)) {
        match &rows {
            Ok(r) if memidx::store_row(t, &r[i * memidx::row_len()..(i + 1) * memidx::row_len()]).is_ok() => b += 1,
            _ => {
                s.memq.push_back(t.clone());
                q += 1;
            }
        }
    }
    crate::dlog!("manual: {} sections, {} already in the index, {} from the vectors in the kernel (no embedder work), {} queued, {} old entries dropped{}",
        n, n - q - b, b, q, dropped, rows.err().map_or(String::new(), |e| format!(" (vectors in the kernel not used: {})", e)));
}

// ---- T38: manual vectors baked into the kernel (MANUAL_VECTORS=1, tools/manvec.py) ----
// Blob: "RZMV", u32 version, u64 key, u32 n, u32 dim, u32 row bytes, then n Q8_0 rows in manual() order.
// Empty when built with MANUAL_VECTORS=0. The rows come from `/man dump` of a kernel with the same key.
const BAKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/manvec.bin"));
const MV_VER: u32 = 1;
const MV_HDR: usize = 28;

/// What the vectors depend on: every section text (with its Asked-as line), the embedder file, the engine
/// build, the chunker version and the index dim.
fn man_key(cur: &[String]) -> u64 {
    let h = cur.iter().fold(crate::mstore::FNV0, |h, t| crate::mstore::fnv(crate::mstore::fnv(h, t.as_bytes()), &[0]));
    let h = crate::mstore::fnv(h, &embed::hash().to_le_bytes());
    let h = crate::mstore::fnv(h, env!("RINGZERO_ENGINE_HASH").as_bytes());
    let h = crate::mstore::fnv(h, &fidx::VERSION.to_le_bytes());
    crate::mstore::fnv(h, &(memidx::dim() as u32).to_le_bytes())
}

fn mv_head(cur: &[String]) -> Vec<u8> {
    let mut h = Vec::from(&b"RZMV"[..]);
    h.extend_from_slice(&MV_VER.to_le_bytes());
    h.extend_from_slice(&man_key(cur).to_le_bytes());
    for v in [cur.len(), memidx::dim(), memidx::row_len()] {
        h.extend_from_slice(&(v as u32).to_le_bytes());
    }
    h
}

/// The baked rows if their header matches this kernel's manual, embedder, engine, chunker and dim.
fn baked(cur: &[String]) -> Result<&'static [u8], &'static str> {
    if BAKED.is_empty() {
        return Err("not built in (MANUAL_VECTORS=0)");
    }
    if !embed::ready() {
        return Err("no embedder");
    }
    if BAKED.len() < MV_HDR || BAKED[..MV_HDR] != mv_head(cur)[..] {
        return Err("key differs (manual, embedder, engine, chunker or dim changed)");
    }
    let rows = &BAKED[MV_HDR..];
    if rows.len() != cur.len() * memidx::row_len() {
        return Err("size does not match the header");
    }
    Ok(rows)
}

/// `/man dump`: header + every section's Q8_0 row (embedded now, like the idle indexer), as hex lines
/// `MV <hex>` for tools/manvec.py, then `MV end <fnv>`. `/man check`: the same rows against the baked ones.
fn man_vectors(check: bool) {
    let cur: Vec<String> = manual().collect();
    let mut v = [0.0f32; embed::DIM];
    let dim = memidx::dim();
    let mut blob = mv_head(&cur);
    let base = if check { baked(&cur) } else { Ok(&[][..]) };
    if let Err(e) = base {
        crate::println!("[man] vectors in the kernel: {}", e);
        return;
    }
    let (t0, mut same) = (smp::rdtsc(), 0);
    for (i, t) in cur.iter().enumerate() {
        if let Err(e) = embed::embed(t, false, dim, &mut v) {
            crate::println!("[man] embed failed: {}", e);
            return;
        }
        let r = memidx::row_of(&v[..dim]);
        if check {
            same += (base.unwrap()[i * r.len()..(i + 1) * r.len()] == r[..]) as usize;
        }
        blob.extend_from_slice(&r);
    }
    let ms = smp::tsc_to_us(smp::rdtsc() - t0) / 1000;
    if check {
        crate::println!("[man] vectors in the kernel: {}/{} rows byte-identical to a fresh embedding ({} ms)", same, cur.len(), ms);
        return;
    }
    for c in blob.chunks(64) {
        let mut l = String::from("MV ");
        c.iter().for_each(|b| { let _ = write!(l, "{:02x}", b); });
        crate::println!("{}", l);
    }
    crate::println!("MV end {:016x} {} bytes, {} sections in {} ms", crate::mstore::fnv(crate::mstore::FNV0, &blob), blob.len(), cur.len(), ms);
}

/// `/man [TOPIC]`: the best two manual sections for TOPIC, in full; alone: the section titles.
pub fn man(q: &str) {
    if q == "dump" || q == "check" {
        return man_vectors(q == "check");
    }
    if q.is_empty() {
        let t: Vec<String> = manual().map(|t| String::from(&t[MAN.len()..t.find("] ").unwrap_or(MAN.len())])).collect();
        crate::println!("manual sections ({}): {}", t.len(), t.join(", "));
        crate::println!("/man TOPIC searches them, for example /man undo");
        return;
    }
    if !embed::ready() {
        crate::println!("[man] no embedder: the manual cannot be searched");
        return;
    }
    let mut h = [memidx::Hit { id: 0, cos: 0.0, bm25: 0.0, via: memidx::Via::Emb }; 32];
    let n = hits(q, &mut h);
    let mut k = 0;
    for x in h[..n].iter().filter(|x| memidx::text(x.id).starts_with(MAN)).take(2) {
        crate::println!("(score {:.2}) {}\n", x.cos, render(memidx::text(x.id), 0));
        k += 1;
    }
    if k == 0 {
        crate::println!("[man] nothing found for '{}'", q);
    }
    crate::dlog!("man '{}': {} sections", &q[..crate::agent::floor_char(q, 60)], k);
}

// ---- hooks from agent.rs ----

/// User turns that go into the index as "[said]": statements, not questions (a question states no fact and
/// only crowds the slot with the same question asked again), that speak about the user (T35: a first-person
/// word, Italian or English). "create notes.txt with hello" that the 2B answered without a tool came back as
/// "the user said earlier" for every /notes request; "mi chiamo Cris", "mia sorella vive a Torino" stay.
fn said(q: &str) -> bool {
    const SELF: [&str; 16] = ["io", "mi", "mio", "mia", "miei", "mie", "me", "sono", "ho", "ricordati", "ricorda",
        "i", "my", "me", "mine", "remember"];
    !q.trim_end().ends_with('?')
        && q.to_lowercase().split(|c: char| !c.is_alphanumeric()).any(|w| SELF.contains(&w))
}

// T35: a statement becomes "[said]" only when its turn ends without acting on the machine. A request the agent
// carried out with a tool ("crea la cartella /home/x", "leggi /notes/todo.txt") is done: recalled later it only
// pulled the model toward the old request (docs/M45_RESULT.md). Turns that only used remember / forget / recall
// are statements about the user and stay. The same rule runs on the journal replay (U, then its T records).
fn said_begin(q: &str, queue: bool) {
    said_end();
    let s = st();
    s.said = (queue && said(q)).then(|| format!("[said] {}", &q[..crate::agent::floor_char(q, 600)]));
    s.acted = false;
}
fn said_tool(name: &str) {
    if !matches!(name, "remember" | "forget" | "recall") {
        st().acted = true;
    }
}
/// The turn ended (or the next one starts): index its statement unless the turn acted.
fn said_end() {
    let s = st();
    if let Some(t) = s.said.take() {
        if s.acted {
            crate::dlog!("said: not indexed (the turn ran a tool): {}", &t[7..crate::agent::floor_char(&t, 70)]);
        } else {
            s.memq.push_back(t);
        }
    }
}

/// A user turn starts (journal + grounding text for facts).
pub fn on_user(q: &str) {
    let s = st();
    append(0, b'U', q);
    said_begin(q, s.on && embed::ready());
    s.howto = howto(q);
    s.session.push_str(q);
    s.session.push('\n');
    s.session_turns += 1;
}
pub fn on_tool(name: &str, res: &str) {
    said_tool(name);
    append(0, b'T', &format!("{} -> {}", name, &res[..crate::agent::floor_char(res, 1024)]));
}
pub fn on_answer(a: &str) {
    if !a.is_empty() {
        append(0, b'A', a);
    }
    said_end();
}
pub fn on_compact(summary: &str) {
    append(0, b'C', summary);
    st().after_compact = true;
    crate::dlog!("compaction: the next user turn gets a memory slot");
}
pub fn take_after_compact() -> bool {
    core::mem::replace(&mut st().after_compact, false)
}

// ---- facts: validation (the model proposes, the kernel decides) ----

/// snake_case key: a-z, 0-9 and _, starting with a letter, 2..32 chars (spaces, '-', '.' become _).
pub fn norm_key(k: &str) -> Option<String> {
    let k: String = k.trim().chars().map(|c| match c {
        ' ' | '-' | '.' => '_',
        c => c.to_ascii_lowercase(),
    }).collect();
    let ok = (2..=32).contains(&k.len()) && k.as_bytes()[0].is_ascii_lowercase()
        && k.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    ok.then_some(k)
}

fn norm_value(v: &str) -> Result<&str, &'static str> {
    let v = v.trim().trim_end_matches('.').trim_matches(|c| c == '"' || c == '\'').trim();
    if v.is_empty() || v.len() > 120 {
        return Err("value empty or longer than 120 bytes");
    }
    if v.contains(['<', '\n', '\t', '|']) {
        return Err("markup or control characters in the value");
    }
    if ["unknown", "none", "n/a", "null", "?", "-", "nessuno", "sconosciuto"].iter().any(|x| v.eq_ignore_ascii_case(x)) {
        return Err("placeholder value");
    }
    Ok(v)
}

/// The value comes from what the user said: every number in it appears there, and at least half of its
/// words of 3+ characters (at least one).
pub fn grounded(v: &str, said: &str) -> bool {
    let said = said.to_lowercase();
    let v = v.to_lowercase();
    let (mut sig, mut hit) = (0, 0);
    for w in v.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
        let digits = w.bytes().any(|b| b.is_ascii_digit());
        if digits && !said.contains(w) {
            return false;
        }
        if digits || w.chars().count() >= 3 {
            sig += 1;
            hit += said.contains(w) as usize;
        }
    }
    sig > 0 && hit * 2 >= sig
}

/// Store a fact. `ground`: the value must come from the session's user text (model-proposed facts).
pub fn remember(key: &str, value: &str, source: &str, ground: bool) -> Result<String, String> {
    let s = st();
    let Some(k) = norm_key(key) else {
        crate::dlog!("fact rejected (key '{}' is not snake_case, 2..32 chars), source {}", key, source);
        return Err(format!("error: key '{}' must be snake_case, 2..32 characters", key));
    };
    let v = match norm_value(value) {
        Ok(v) => v,
        Err(e) => {
            crate::dlog!("fact rejected {}: {} ({}), source {}", k, value, e, source);
            return Err(format!("error: {}", e));
        }
    };
    if k.replace('_', " ").eq_ignore_ascii_case(v) {
        crate::dlog!("fact rejected {} = {} (the key only repeats the value), source {}", k, v, source);
        return Err(format!("error: the key must name the fact (e.g. user_name), not repeat '{}'", v));
    }
    let old = s.facts.iter().find(|f| f.key == k).map(|f| f.value.clone());
    if old.as_deref() == Some(v) {
        crate::dlog!("fact {} = {} already known (dedup), source {}", k, v, source);
        return Ok(format!("already saved: {} = {}", k, v));
    }
    if ground && !grounded(v, &s.session) {
        crate::dlog!("fact rejected {} = {} (not grounded: not in what the user said), source {}", k, v, source);
        return Err(format!("error: '{}' is not something the user said in this session", v));
    }
    if old.is_none() && s.facts.len() >= MAX_FACTS {
        crate::dlog!("fact rejected {} (store full, {} facts)", k, MAX_FACTS);
        return Err(format!("error: the fact store is full ({} facts); forget one first", MAX_FACTS));
    }
    let rec = format!("{}\t{}\t{}", k, v, source);
    let r = mstore::Rec { kind: b'S', ts: crate::sys::unix_time(), off: 0, p: &rec };
    apply(1, &r, true, s.on && embed::ready(), 0);
    append(1, b'S', &rec);
    crate::dlog!("fact {} {} = {}{} (source {})", if old.is_some() { "updated" } else { "saved" }, k, v,
        old.map(|o| format!(", was {}", o)).unwrap_or_default(), source);
    Ok(format!("saved: {} = {}", k, v))
}

pub fn forget(key: &str) -> Result<String, String> {
    let s = st();
    let k = norm_key(key).unwrap_or_default();
    if !s.facts.iter().any(|f| f.key == k) {
        let keys: Vec<&str> = s.facts.iter().map(|f| f.key.as_str()).collect();
        return Err(format!("error: no fact '{}'; saved keys: {}", key, if keys.is_empty() { "none".into() } else { keys.join(", ") }));
    }
    let r = mstore::Rec { kind: b'X', ts: 0, off: 0, p: &k };
    apply(1, &r, true, true, 0);
    append(1, b'X', &k);
    crate::dlog!("fact forgotten: {}", k);
    Ok(format!("forgotten: {}", k))
}

// ---- recall and the context compiler ----

/// An index text as a line for the model.
fn render(t: &str, max: usize) -> String {
    let (lead, rest) = if let Some(r) = t.strip_prefix("[said] ") {
        ("the user said earlier: ", r)
    } else if let Some(r) = t.strip_prefix("[fact] ") {
        ("fact: ", r)
    } else if let Some((sec, r)) = t.strip_prefix(MAN).and_then(|r| r.split_once("] ")) {
        // a section is short and self-contained: shown whole (the token budget of the slot still applies)
        let body: Vec<&str> = r.lines().filter(|l| !l.starts_with(ASKED)).collect();
        return format!("ringzero manual, {}: {}", sec, body.join(" "));
    } else if let Some((n, r)) = t.strip_prefix("[episode ").and_then(|r| r.split_once("] ")) {
        return format!("earlier conversation {}: {}", n, &r[..crate::agent::floor_char(r, max)]);
    } else if let Some((p, r)) = t.strip_prefix('[').and_then(|r| r.split_once("] ")) {
        return format!("file {}: {}", p, r[..crate::agent::floor_char(r, max)].replace('\n', " "));
    } else {
        ("note: ", t)
    };
    format!("{}{}", lead, &rest[..crate::agent::floor_char(rest, max)].replace('\n', " "))
}

/// Embed the queued memory texts now (a slot or a recall must see the last turns). Ctrl-C stops it.
fn flush_mem() {
    let s = st();
    if s.memq.is_empty() && !matches!(s.cur, Some(Emb { src: Src::Mem, .. })) {
        return;
    }
    if matches!(s.cur, Some(Emb { src: Src::File(..), .. })) {
        s.cur = None; // the file chunk starts over later
    }
    let (n, t0) = (s.memq.len(), smp::rdtsc());
    embed_slice(&mut crate::shell::poll_intr, true);
    crate::dlog!("recall: embedded {} queued memory texts first ({} ms)", n - s.memq.len(), smp::tsc_to_us(smp::rdtsc() - t0) / 1000);
}

fn hits(q: &str, h: &mut [memidx::Hit]) -> usize {
    flush_mem();
    memidx::search(q, h).map(|f| f.n).unwrap_or(0)
}

const HIT0: memidx::Hit = memidx::Hit { id: 0, cos: 0.0, bm25: 0.0, via: memidx::Via::Emb };

/// One search, split (T22): the best `own.len()` entries that are not manual sections (facts, past turns,
/// episodes, files: the manual must not crowd them out) and the best manual section.
fn hits_split(q: &str, own: &mut [memidx::Hit]) -> (usize, Option<memidx::Hit>) {
    let mut h = [HIT0; 24];
    let n = hits(q, &mut h);
    let (mut k, mut man) = (0, None);
    for x in &h[..n] {
        if memidx::text(x.id).starts_with(MAN) {
            man = man.or(Some(*x));
        } else if k < own.len() {
            own[k] = *x;
            k += 1;
        }
    }
    (k, man)
}

/// T22: the message asks how to do something (Italian or English), so a manual section may answer it.
/// "come" + a first-person verb opens it (come torno, come faccio, come cambio), "come mi chiamo" does not.
pub fn howto(q: &str) -> bool {
    const P: [&str; 16] = ["how do i ", "how can i ", "how to ", "how do you ", "how should i ", "how does ",
        "what command", "which command", "is there a command", "quale comando", "che comando", "un comando per",
        "in che modo", "come si fa", "come si usa", "come posso"];
    let l = q.to_lowercase();
    let w: Vec<&str> = l.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    P.iter().any(|p| l.contains(p))
        || w.windows(2).any(|p| p[0] == "come" && p[1].chars().count() >= 3 && p[1].ends_with('o') && p[1] != "sono")
}

/// The recall tool: top 3 of facts, past conversations and files, then the best manual section when the
/// user's message (or the query) asks how to do something and it scores RECALL_MIN or more.
pub fn recall<const N: usize>(q: &str, o: &mut SBuf<N>) {
    let mut h = [HIT0; 4];
    let (mut n, man) = if embed::ready() && memidx::len() + st().memq.len() > 0 { hits_split(q, &mut h[..3]) } else { (0, None) };
    if let Some(m) = man.filter(|m| m.cos >= RECALL_MIN && (st().howto || howto(q))) {
        h[n] = m;
        n += 1;
    }
    for (i, x) in h[..n].iter().enumerate() {
        let _ = write!(o, "{}{}. (score {:.2}) {}", if i > 0 { "\n" } else { "" }, i + 1, x.cos, render(memidx::text(x.id), 300));
    }
    if n == 0 {
        let _ = write!(o, "nothing found in memory");
    }
    crate::dlog!("recall '{}': {} hits{}", &q[..crate::agent::floor_char(q, 60)], n,
        h[..n].iter().map(|x| format!(" #{} {:.2}", x.id, x.cos)).collect::<String>());
}

/// T22: the manual line for a how-to message: the best section, if it scores RECALL_MIN or more. Logged.
fn man_line(man: Option<memidx::Hit>) -> Option<String> {
    let m = man?;
    let line = format!("- {}\n", render(memidx::text(m.id), 0));
    let drop = if m.cos < RECALL_MIN { Some("below the threshold") } else { None };
    crate::dlog!("slot manual #{} cos {:.3} {}: {} | {}", m.id, m.cos, m.via.name(), drop.unwrap_or("kept"),
        &line[2..crate::agent::floor_char(&line, 60)].trim_end());
    drop.is_none().then_some(line)
}

/// T22: a how-to message later in a conversation (no memory slot there) gets the manual line alone.
pub fn man_slot(q: &str, o: &mut String) {
    if unsafe { SLOT_BUDGET } == 0 || !howto(q) || !embed::ready() {
        return;
    }
    let t0 = smp::rdtsc();
    let (_, man) = hits_split(q, &mut []);
    let Some(line) = man_line(man) else { return };
    if tokens(MAN_HEAD) + tokens(&line) > unsafe { SLOT_BUDGET } {
        crate::dlog!("slot (how-to): manual line over the token budget, not added");
        return;
    }
    o.push_str(MAN_HEAD);
    o.push_str(&line);
    o.push('\n');
    crate::dlog!("slot (how-to): manual line = {} tok, built in {} ms", tokens(o), smp::tsc_to_us(smp::rdtsc() - t0) / 1000);
}

fn tokens(t: &str) -> usize {
    static mut TK: [u32; 2048] = [0; 2048];
    llm::encode(t.as_bytes(), unsafe { &mut *core::ptr::addr_of_mut!(TK) })
}

/// The memory slot for the first user message of a conversation (or after a compaction): facts, newest
/// first, within FACT_BUDGET tokens, then at most SLOT_LINES notes recalled for q (of the top RECALL_K: cos >=
/// RECALL_MIN, or >= RECALL_MIN_BM25 for a BM25 match; NOTE_MIN characters or more; not q itself) up to
/// SLOT_BUDGET tokens in all. Empty when there is nothing to say. Every candidate and its verdict is logged.
pub fn slot(q: &str, why: &str, o: &mut String) {
    let s = st();
    if unsafe { SLOT_BUDGET } == 0 {
        crate::dlog!("slot ({}): off (!set slot 0)", why);
        return;
    }
    said_end();
    let t0 = smp::rdtsc();
    let mut body = String::new();
    let mut nf = 0;
    for f in s.facts.iter().rev() {
        let line = format!("- {}: {}\n", f.key, f.value);
        if tokens(&body) + tokens(&line) > FACT_BUDGET.min(unsafe { SLOT_BUDGET }) {
            break;
        }
        body.push_str(&line);
        nf += 1;
    }
    let mut rec = String::new();
    let mut nr = 0;
    if embed::ready() && memidx::len() + s.memq.len() > 0 {
        let mut h = [HIT0; RECALL_K];
        let (n, man) = hits_split(q, &mut h);
        // T22: a how-to message gets the best manual section first (one line, same token budget)
        if let Some(line) = howto(q).then(|| man_line(man)).flatten() {
            if tokens(SLOT_HEAD) + tokens(&body) + tokens(&line) <= unsafe { SLOT_BUDGET } {
                body.push_str(&line);
                let _ = write!(rec, " manual #{} {:.2}", man.map_or(0, |m| m.id), man.map_or(0.0, |m| m.cos));
            } else {
                crate::dlog!("slot: manual line over the token budget, not added");
            }
        }
        for x in &h[..n] {
            let t = memidx::text(x.id);
            let line = format!("- {}\n", render(t, 240));
            let note = t.split_once("] ").map_or(t, |x| x.1).trim();
            let floor = if x.via == memidx::Via::Emb { RECALL_MIN } else { RECALL_MIN_BM25 };
            let drop = if t.starts_with("[fact] ") {
                Some("a fact, pinned above")
            } else if x.cos < floor {
                Some(if x.via == memidx::Via::Emb { "below the threshold" } else { "keyword match, below the bm25 floor" })
            } else if note.chars().count() < NOTE_MIN {
                Some("too short")
            } else if t.starts_with("[said] ") && note.eq_ignore_ascii_case(q.trim()) {
                Some("the message itself")
            } else if nr >= SLOT_LINES {
                Some("line cap")
            } else if tokens(SLOT_HEAD) + tokens(&body) + tokens(&line) > unsafe { SLOT_BUDGET } {
                Some("over the token budget")
            } else {
                None
            };
            crate::dlog!("slot cand #{} cos {:.3} {}: {} | {}", x.id, x.cos, x.via.name(), drop.unwrap_or("kept"),
                &line[2..crate::agent::floor_char(&line, 60)].trim_end());
            if drop.is_some() {
                continue;
            }
            body.push_str(&line);
            nr += 1;
            let _ = write!(rec, " #{} {:.2}", x.id, x.cos);
        }
    }
    let ms = smp::tsc_to_us(smp::rdtsc() - t0) / 1000;
    if body.is_empty() {
        crate::dlog!("slot ({}): empty (no facts, nothing recalled passed the filters), {} ms", why, ms);
        return;
    }
    o.push_str(SLOT_HEAD);
    o.push_str(&body);
    o.push('\n');
    crate::dlog!("slot ({}): {} facts + {} recalled [{}] = {} tok, built in {} ms", why, nf, nr, rec.trim(), tokens(o), ms);
}

// ---- idle scheduler ----

fn cons_due() -> bool {
    let s = st();
    let n = unsafe { CONS_IDLE_S };
    s.on && n > 0 && s.cons.is_none() && s.session_turns > 0 && !snap::unsaved()
        && smp::tsc_to_us(smp::rdtsc() - s.last_input) >= n * 1_000_000
}

/// Highest priority runnable job (lower number first; a due consolidation gains 1 every 50 waited slices).
fn pick() -> Option<(Job, &'static str)> {
    let s = st();
    if let Some(e) = &s.cur {
        return Some((if matches!(e.src, Src::Mem) { Job::Mem } else { Job::File }, "finish the embedding in flight"));
    }
    let cons = s.cons.is_some() || cons_due();
    let cp = 4u64.saturating_sub(s.waited / 50);
    let mut best: Option<(u64, Job, &'static str)> = None;
    let mut cand = |p: u64, j: Job, why: &'static str| {
        if best.map_or(true, |(b, _, _)| p < b) {
            best = Some((p, j, why));
        }
    };
    if !s.memq.is_empty() {
        cand(1, Job::Mem, "memory texts queued (turns, facts, episodes)");
    }
    if fidx::pending() {
        cand(2, Job::File, "files queued");
    }
    if cons {
        cand(cp, Job::Cons, if s.cons.is_some() { "resume the consolidation in flight" } else { "idle long enough, new turns since the last one" });
    }
    if memidx::gen() != s.saved_gen && queue_empty() && s.on {
        cand(3, Job::Save, "the index changed and nothing is queued");
    }
    let (p, j, why) = best?;
    if cons && j != Job::Cons {
        s.waited += 1;
    } else if j == Job::Cons && p < 4 && s.waited > 0 {
        s.waited = 0;
        return Some((j, "it waited too long behind other jobs (aging)"));
    }
    Some((j, why))
}

/// One slice of the best job; called by the shell's line editor while it waits for input. true = it worked.
pub fn idle(stop: &mut dyn FnMut() -> bool) -> bool {
    said_end(); // at the prompt: the last turn is over (it may have ended without an answer)
    let s = st();
    let Some((job, why)) = pick() else {
        s.last = None;
        return false;
    };
    if s.last != Some(job) {
        crate::dlog!("sched: run '{}' ({})", job.name(), why);
        s.last = Some(job);
    }
    let t0 = smp::rdtsc();
    let deadline = t0 + smp::us_to_tsc(SLICE_MS * 1000);
    let mut input = false;
    let mut halt = || {
        input = input || stop();
        input || smp::rdtsc() >= deadline
    };
    smp::park_phase(true);
    match job {
        Job::Mem | Job::File => embed_slice(&mut halt, false),
        Job::Save => save_index("idle"),
        Job::Cons => {
            if s.cons.is_none() {
                if let Err(e) = cons_begin("idle") {
                    crate::dlog!("consolidate: skipped ({}), the session's turns are dropped from the queue", e);
                    s.session_turns = 0;
                }
            }
            if s.cons.is_some() {
                cons_run(&mut halt);
            }
        }
    }
    smp::park_phase(false);
    let i = job as usize;
    s.ms[i] += smp::tsc_to_us(smp::rdtsc() - t0) / 1000;
    if input && pick().is_some() {
        s.preempted += 1;
    }
    true
}

/// Input arrived at the prompt: the engine goes back to the user. A consolidation in flight is stopped and
/// the current snapshot point restored (an embedding in flight just waits: it uses the embedder's context).
pub fn preempt() {
    let s = st();
    s.last_input = smp::rdtsc();
    if s.cons.is_some() {
        cons_abort("input at the prompt");
    }
}

fn embed_slice(stop: &mut dyn FnMut() -> bool, mem_only: bool) {
    let s = st();
    let mut v = [0.0f32; embed::DIM];
    let dim = memidx::dim();
    while !stop() {
        if s.cur.is_none() {
            let (text, src) = if let Some(t) = s.memq.pop_front() {
                if memidx::contains(&t) {
                    continue; // the same text is already there (a repeated question)
                }
                let mut b = SBuf::new();
                b.push(t.as_bytes());
                (b, Src::Mem)
            } else if mem_only {
                break;
            } else if let Some((t, next)) = fidx::next_chunk() {
                (t, Src::File(next, fidx::gen()))
            } else {
                break;
            };
            if memidx::room(text.as_str()).is_err() {
                crate::dlog!("index: full, {} memory texts dropped", s.memq.len() + 1);
                s.memq.clear();
                break;
            }
            match embed::begin(text.as_str(), false) {
                Ok(id) => s.cur = Some(Emb { id, text, src }),
                Err(e) => {
                    crate::dlog!("index: embed failed ({}), queue dropped", e);
                    s.memq.clear();
                    break;
                }
            }
        }
        let e = s.cur.as_mut().unwrap();
        match embed::resume(e.id, dim, &mut v, &mut *stop) {
            Ok(None) => break,
            Ok(Some(info)) => {
                let e = s.cur.take().unwrap();
                match e.src {
                    Src::File(_, g) if g != fidx::gen() => {} // the file changed meanwhile: its queue entry starts over
                    Src::File(next, _) => {
                        if memidx::store(e.text.as_str(), &v[..dim], info).is_ok() {
                            fidx::advance(next);
                        }
                    }
                    Src::Mem => {
                        let _ = memidx::store(e.text.as_str(), &v[..dim], info);
                    }
                }
            }
            Err(_) => match embed::begin(e.text.as_str(), false) {
                Ok(id) => e.id = id, // another embedding ran in between: this one starts over
                Err(_) => s.cur = None,
            },
        }
    }
}

// ---- consolidation ----

fn cons_begin(why: &'static str) -> Result<(), &'static str> {
    let s = st();
    if s.session_turns == 0 {
        return Err("no new turns since the last consolidation");
    }
    if snap::unsaved() {
        return Err("the live state is not saved (snapshot pool full)");
    }
    let p = snap::ensure_live();
    if snap::at_root() {
        s.session.clear();
        return Err("fresh conversation: the session's turns are not in the context any more");
    }
    let t = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", CONS_ASK);
    let mut toks = alloc::vec![0u32; 1024];
    let n = llm::encode(t.as_bytes(), &mut toks);
    toks.truncate(n);
    if p + n as u64 + CONS_MAX + 8 >= llm::ctx_max_seq() {
        return Err("no room left in the context");
    }
    let temp = unsafe { core::mem::replace(&mut *core::ptr::addr_of_mut!(llm::TEMP_BITS), 0) }; // greedy: same session, same result
    unsafe { (*core::ptr::addr_of_mut!(CAP)).clear() };
    s.cons = Some(Cons { toks, i: 0, pos: p, n_gen: 0, next: 0, t0: smp::rdtsc(), temp, why });
    crate::dlog!("consolidate: start ({}) on top of the current point (ctx {}), {} new turns, instruction {} tok", why, p, s.session_turns, n);
    Ok(())
}

/// Engine steps until done or stop(). Output goes to CAP only (not on the screen).
fn cons_run(stop: &mut dyn FnMut() -> bool) {
    let s = st();
    let max_seq = llm::ctx_max_seq();
    unsafe {
        CAPTURE = true;
        ECHO = false;
    }
    let mut done = false;
    while let Some(c) = s.cons.as_mut() {
        if stop() {
            break;
        }
        if c.i < c.toks.len() {
            let e = (c.i + llm::prefill_block().max(1) as usize).min(c.toks.len()); // one prefill block per step
            llm::fwd_block(&c.toks[c.i..e], c.pos, e == c.toks.len());
            c.pos += (e - c.i) as u64;
            c.i = e;
            if c.i == c.toks.len() {
                c.next = llm::pick_next();
            }
            continue;
        }
        if llm::is_eos(c.next) || c.n_gen >= CONS_MAX || c.pos + 3 >= max_seq {
            done = true;
            break;
        }
        unsafe { crate::engine::tok_write_stdout(c.next) };
        llm::fwd(c.next as u32, c.pos, 1);
        c.pos += 1;
        c.n_gen += 1;
        c.next = llm::pick_next();
    }
    unsafe {
        CAPTURE = false;
        ECHO = true;
    }
    if done {
        cons_finish();
    }
}

/// Back to the user's point: the rows above it were scratch, R comes back from the point (snap.rs).
fn cons_restore(c: &Cons) -> u64 {
    let t0 = smp::rdtsc();
    snap::rollback();
    unsafe { llm::TEMP_BITS = c.temp };
    smp::tsc_to_us(smp::rdtsc() - t0)
}

fn cons_abort(why: &str) {
    let s = st();
    let Some(c) = s.cons.take() else { return };
    let us = cons_restore(&c);
    unsafe {
        CAPTURE = false;
        ECHO = true;
    }
    crate::dlog!("consolidate: stopped by {} after {} tok, current point restored (ctx {}) in {} us; retried later", why,
        c.n_gen, snap::pos(), us);
}

fn cons_finish() {
    let s = st();
    let Some(c) = s.cons.take() else { return };
    let us = cons_restore(&c);
    let out = String::from(unsafe { (*core::ptr::addr_of!(CAP)).as_str() });
    let ms = smp::tsc_to_us(smp::rdtsc() - c.t0) / 1000;
    s.n_cons += 1;
    crate::dlog!("consolidate #{} ({}): {} tok in {} ms, current point restored (ctx {}) in {} us", s.n_cons, c.why, c.n_gen, ms,
        snap::pos(), us);
    for l in out.lines().map(str::trim).filter(|l| !l.is_empty()) {
        crate::dlog!("consolidate #{}: model wrote: {}", s.n_cons, &l[..crate::agent::floor_char(l, 140)]);
    }
    let src = format!("consolidation (episode {})", s.n_ep + 1);
    let (mut ep, mut ok, mut bad) = (false, 0, 0);
    for l in out.lines().map(str::trim) {
        if let Some(e) = l.strip_prefix("EPISODE:").map(str::trim).filter(|e| !e.is_empty() && !ep) {
            ep = true;
            let e = &e[..crate::agent::floor_char(e, 600)];
            s.n_ep += 1;
            append(2, b'E', e);
            if s.on && embed::ready() {
                s.memq.push_back(format!("[episode {}] {}", s.n_ep, e));
            }
            crate::dlog!("episode {} saved: {}", s.n_ep, e);
        } else if let Some((k, v)) = l.strip_prefix("FACT:").and_then(|f| f.split_once('=')) {
            if ok + bad < 8 {
                match remember(k, v, &src, true) {
                    Ok(_) => ok += 1,
                    Err(_) => bad += 1,
                }
            }
        }
    }

    if !ep {
        crate::dlog!("consolidate #{}: no EPISODE line in the output, no episode saved", s.n_cons);
    }
    crate::dlog!("consolidate #{}: {} facts accepted or already known, {} rejected; session reset", s.n_cons, ok, bad);
    s.session.clear();
    s.session_turns = 0;
    s.waited = 0;
}

/// `/consolidate`: run it now in the foreground (Ctrl-C stops it and restores the point).
pub fn consolidate_now() {
    let s = st();
    if s.cons.is_some() {
        cons_abort("/consolidate");
    }
    let n0 = (s.facts.len(), s.n_ep);
    if let Err(e) = cons_begin("/consolidate") {
        crate::dlog!("consolidate: skipped ({})", e);
        crate::println!("[agentd] nothing to consolidate: {}", e);
        return;
    }
    crate::println!("[agentd] consolidating the session (Ctrl-C stops)...");
    smp::park_phase(true);
    let mut intr = false;
    cons_run(&mut || {
        intr = intr || crate::shell::poll_intr();
        intr
    });
    smp::park_phase(false);
    if intr {
        cons_abort("Ctrl-C");
        crate::println!("^C [agentd] consolidation stopped, conversation restored (ctx {})", snap::pos());
        return;
    }
    crate::println!("[agentd] done: {} facts ({} new or updated), {} episodes; /facts lists them, /agentd log shows each decision",
        s.facts.len(), s.facts.len() as i64 - n0.0 as i64, s.n_ep);
}

// ---- status ----

pub fn status() {
    let s = st();
    crate::println!("[agentd] stores {}: journal {} records ({} KiB), {} facts, {} episodes",
        if s.on { "/sys/mem" } else { "off (no disk)" }, s.n_rec[0], s.size[0] >> 10, s.facts.len(), s.n_ep);
    crate::println!("[agentd] index {} entries, {}; queue: {} memory texts, files {}",
        memidx::len(), if memidx::gen() == s.saved_gen { "saved" } else { "changed since the last save" }, s.memq.len(),
        if fidx::pending() { "pending" } else { "none" });
    crate::println!("[agentd] consolidation: after {} s idle{}, {} session turns pending, {} done, {}",
        unsafe { CONS_IDLE_S }, if unsafe { CONS_IDLE_S } == 0 { " (off: /consolidate only)" } else { "" }, s.session_turns, s.n_cons,
        if s.cons.is_some() { "one in flight" } else { "none in flight" });
    crate::println!("[agentd] idle time used: memory texts {} ms, file chunks {} ms, index saves {} ms, consolidation {} ms; {} slices stopped by input",
        s.ms[0], s.ms[1], s.ms[2], s.ms[3], s.preempted);
}

pub fn list_facts() {
    let s = st();
    for f in &s.facts {
        let d = f.ts / 86_400;
        let (y, m, dd) = civil(d as i64);
        crate::println!("  {} = {}   ({}, {}-{:02}-{:02} {:02}:{:02} UTC)", f.key, f.value, f.source, y, m, dd, f.ts % 86_400 / 3600, f.ts % 3600 / 60);
    }
    crate::println!("[facts] {} of {} (every fact goes into the memory slot of a new conversation, newest first, {} tokens at most)",
        s.facts.len(), MAX_FACTS, FACT_BUDGET);
}

/// Date of a day number since 1970-01-01 (H. Hinnant's civil_from_days).
fn civil(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + (m <= 2) as i64, m, d)
}

/// Boot self-check: fact validation, grounding, calendar math (no disk, no model).
pub fn selftest() -> bool {
    let k = norm_key("User Name").as_deref() == Some("user_name") && norm_key("x").is_none() && norm_key("1abc").is_none()
        && norm_key("a<b").is_none();
    let v = norm_value(" \"Cris\". ") == Ok("Cris") && norm_value("unknown").is_err() && norm_value("<tool_call>").is_err();
    let said = "mi chiamo Cris e il mio numero preferito e 4711\n";
    let g = grounded("Cris", said) && grounded("4711", said) && !grounded("4712", said) && !grounded("Roma", said)
        && !grounded("Cris 1234", said);
    let d = crate::sys::days(1970, 1, 1) == 0 && crate::sys::days(2000, 3, 1) == 11_017 && civil(11_017) == (2000, 3, 1)
        && civil(crate::sys::days(2026, 11, 26)) == (2026, 11, 26);
    let r = render("[said] ciao", 100) == "the user said earlier: ciao" && render("[/a/b.txt] x\ny", 100) == "file /a/b.txt: x y";
    let sd = self::said("mi chiamo Cris e il mio numero preferito e 4711") && self::said("my sister lives in Turin")
        && !self::said("create notes.txt with hello") && !self::said("come mi chiamo?") && !self::said("imposta la lingua su italiano");
    let man: Vec<String> = manual().collect();
    let m = man.len() >= 20 && man.iter().all(|t| t.len() <= fidx::CHUNK + 300 && t.contains("] ") && !t.ends_with("] "))
        && render("[manual: T] Asked as: x\nbody", 0) == "ringzero manual, T: body";
    let ht = ["come torno indietro di un turno?", "how do I run python?", "come faccio a ibernare?", "come cambio layout tastiera?",
        "how can I list the programs in /bin?"].iter().all(|q| howto(q))
        && !["come mi chiamo?", "come si chiama mia sorella e dove vive?", "how many cores do I have?", "hi, how are you?",
        "quanto fa 2**100? usa python", "come stai?"].iter().any(|q| howto(q));
    k && v && g && d && r && sd && m && ht
}
