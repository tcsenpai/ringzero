// Engine state snapshots, v1 of docs/design/SNAPSHOTS.md. Every finished turn is saved as a point (#N)
// in a tree of delta nodes kept in a pool of 1 MiB chunks. A node holds the KV rows its turn added,
// [base, pos) of every full-attention plane, their token ids, and the DeltaNet block R (19.27 MiB,
// rewritten on every token, so it is stored whole). Invariant: the live KV rows 0..LIVE.pos belong to
// the chain of LIVE, so going back along the chain copies R only, and a switch copies the rows above the
// common ancestor. Branches and saves are names on nodes; the current branch (HEAD) has a tip and CUR
// sits behind it after an undo. `parent` is the storage chain, `back` the history (a compaction node is
// stored under the root but its history continues from the uncompacted point). Only the model goes back:
// tool side effects stay, and /undo prints them. Copies run on the BSP between turns (workers idle).
use crate::engine::{ctx, kv_elt_bytes, transformer_forward};
use crate::log::SBuf;
use crate::{llm, print, println, smp};
use core::fmt::Write;

// engine/constants.inc (QWEN35)
const CTX_MAX_SEQ: usize = 160;
const CTX_N_KV_HEADS: usize = 120;
const CTX_HEAD_DIM: usize = 128;
const CTX_BUF_K_CACHE: usize = 328;
const CTX_BUF_V_CACHE: usize = 336;
const CTX_BUF_DN_STATE: usize = 384;
const CTX_N_FULL: usize = 416;
const CTX_DN_ZERO_BYTES: usize = 496;

const CHUNK: usize = 1 << 20;
const MAX_CHUNKS: usize = 1024;
const POOL_MIB: usize = 1024;
const NIL: u16 = u16::MAX;
const ROOT: u16 = 0;
const MAX_NODES: usize = 256;
const MAX_NAMES: usize = 64;
/// Points on the current line that are never thinned (the checkpoint ring).
const RING: usize = 16;
/// Free chunks kept before a save; below this the pool is evicted (doc 4.3).
const RESERVE: usize = 64;
/// Anonymous lines whose tip was last visited this many turns ago may be dropped.
const DROP_AGE: u64 = 32;
const TXT: usize = 2048;

#[derive(Clone, Copy)]
struct Node {
    used: bool,
    id: u32,
    parent: u16, // storage: rows below `base` belong to the parent chain
    back: u16,   // history: where /undo goes (= parent except after a compaction)
    base: u64,
    pos: u64,
    kv: u16, // chunk list of the rows [base, pos), NIL if none
    r: u16,  // chunk list of R, NIL = thinned (restore replays the token ids)
    rng: u64,
    visit: u64,
    ex: bool,     // q/a is a finished exchange (re-prefilled after a compaction)
    compact: u32, // compaction number of a compaction node, 0 otherwise
    fnv: u64,     // hash of R, taken when thinned (checked after a replay)
}
const NODE0: Node = Node { used: false, id: 0, parent: NIL, back: NIL, base: 0, pos: 0, kv: NIL, r: NIL, rng: 0, visit: 0, ex: false, compact: 0, fnv: 0 };

#[derive(Clone, Copy)]
struct Name {
    used: bool,
    s: [u8; 24],
    n: u8,
    node: u16,
    save: bool,   // /save (or pre-compact-N), else a branch
    auto: bool,   // made by the kernel (main@N, pre-compact-N): may be dropped when old
    pinned: bool, // pre-compact-N until the next compaction
}
const NAME0: Name = Name { used: false, s: [0; 24], n: 0, node: NIL, save: false, auto: false, pinned: false };
impl Name {
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.s[..self.n as usize]).unwrap_or("?")
    }
}

struct State {
    pool: *mut u8,
    n_chunks: usize,
    cap: usize, // usable chunks (`!set snappool`)
    used: [u64; MAX_CHUNKS / 64],
    next: [u16; MAX_CHUNKS],
    nodes: [Node; MAX_NODES],
    names: [Name; MAX_NAMES],
    head: usize,   // index of the current branch in names
    cur: u16,      // current point
    live: u16,     // node whose chain holds the live KV rows 0..pos (NIL after /bench, /fill, /gen)
    pos: u64,      // live position (= cur.pos between turns, unless the last save was refused)
    unsaved: bool, // the live state is ahead of cur: its save was refused (pool full)
    tick: u64,     // turns saved so far (age for eviction)
    next_id: u32,
    save_us: u64,
    save_bytes: u64,
    restore_us: u64,
    restore_bytes: u64,
}
static mut S: State = State {
    pool: core::ptr::null_mut(), n_chunks: 0, cap: 0, used: [0; MAX_CHUNKS / 64], next: [NIL; MAX_CHUNKS],
    nodes: [NODE0; MAX_NODES], names: [NAME0; MAX_NAMES], head: 0, cur: ROOT, live: NIL, pos: 0, unsaved: false,
    tick: 0, next_id: 0, save_us: 0, save_bytes: 0, restore_us: 0, restore_bytes: 0,
};
static mut QS: [SBuf<TXT>; MAX_NODES] = [const { SBuf::new() }; MAX_NODES];
static mut AS: [SBuf<TXT>; MAX_NODES] = [const { SBuf::new() }; MAX_NODES];
static mut FX: [SBuf<160>; MAX_NODES] = [const { SBuf::new() }; MAX_NODES];
static mut MARK: [u8; MAX_NODES] = [0; MAX_NODES];

fn st() -> &'static mut State {
    unsafe { &mut *core::ptr::addr_of_mut!(S) }
}
fn nd(n: u16) -> &'static mut Node {
    &mut st().nodes[n as usize]
}
fn q_of(n: u16) -> &'static mut SBuf<TXT> {
    unsafe { &mut (*core::ptr::addr_of_mut!(QS))[n as usize] }
}
fn a_of(n: u16) -> &'static mut SBuf<TXT> {
    unsafe { &mut (*core::ptr::addr_of_mut!(AS))[n as usize] }
}
fn fx_of(n: u16) -> &'static mut SBuf<160> {
    unsafe { &mut (*core::ptr::addr_of_mut!(FX))[n as usize] }
}
fn mark() -> &'static mut [u8; MAX_NODES] {
    unsafe { &mut *core::ptr::addr_of_mut!(MARK) }
}

// ---- chunk pool: bitmap of 1 MiB chunks, a node's data is a linked list of chunks ----

fn n_used() -> usize {
    st().used.iter().map(|w| w.count_ones() as usize).sum()
}
fn free_chunks() -> usize {
    st().cap.saturating_sub(n_used())
}
fn alloc(k: usize) -> Option<u16> {
    if k == 0 {
        return Some(NIL);
    }
    if free_chunks() < k {
        return None;
    }
    let s = st();
    let (mut head, mut got) = (NIL, 0);
    for i in (0..s.n_chunks).rev() {
        if got == k {
            break;
        }
        if s.used[i / 64] & 1 << (i % 64) == 0 {
            s.used[i / 64] |= 1 << (i % 64);
            s.next[i] = head;
            head = i as u16;
            got += 1;
        }
    }
    Some(head)
}
fn free_list(mut c: u16) {
    let s = st();
    while c != NIL {
        s.used[c as usize / 64] &= !(1 << (c as usize % 64));
        let nx = s.next[c as usize];
        s.next[c as usize] = NIL;
        c = nx;
    }
}
fn chunk(c: u16) -> *mut u8 {
    unsafe { st().pool.add(c as usize * CHUNK) }
}

// ---- engine layout: head-major K/V planes cache[l][kv_h][pos][head_dim], FP16 or FP32 ----

struct Geo {
    hb: usize,     // bytes per row of one plane
    stride: usize, // plane stride (max_seq rows)
    heads: usize,  // planes per cache (n_full * n_kv_heads)
    rpc: usize,    // rows per chunk (all planes + the token id)
}
fn geo() -> Geo {
    let hb = ctx(CTX_HEAD_DIM) as usize * unsafe { kv_elt_bytes } as usize;
    let heads = (ctx(CTX_N_FULL) * ctx(CTX_N_KV_HEADS)) as usize;
    Geo { hb, stride: ctx(CTX_MAX_SEQ) as usize * hb, heads, rpc: CHUNK / (2 * heads * hb + 4) }
}
fn plane(g: &Geo, p: usize) -> *mut u8 {
    let base = if p < g.heads { ctx(CTX_BUF_K_CACHE) } else { ctx(CTX_BUF_V_CACHE) };
    (base as usize + (p % g.heads) * g.stride) as *mut u8
}
fn r_bytes() -> usize {
    ctx(CTX_DN_ZERO_BYTES) as usize
}
fn r_chunks() -> usize {
    r_bytes().div_ceil(CHUNK)
}
fn kv_chunks(rows: u64) -> usize {
    (rows as usize).div_ceil(geo().rpc)
}

/// Copy node n's rows between its chunks and the live cache (plane-major in the chunk, then the token
/// ids). Returns the bytes copied.
fn copy_rows(n: u16, to_pool: bool) -> u64 {
    let g = geo();
    let rec = llm::rec();
    let (base, pos) = (nd(n).base as usize, nd(n).pos as usize);
    let (mut row, mut c) = (base, nd(n).kv);
    while row < pos {
        let k = g.rpc.min(pos - row);
        let ch = chunk(c);
        unsafe {
            for p in 0..2 * g.heads {
                let live = plane(&g, p).add(row * g.hb);
                let sv = ch.add(p * g.rpc * g.hb);
                let (src, dst) = if to_pool { (live, sv) } else { (sv, live) };
                core::ptr::copy_nonoverlapping(src, dst, k * g.hb);
            }
            let t = ch.add(2 * g.heads * g.rpc * g.hb) as *mut u32;
            let r = rec.as_mut_ptr().add(row);
            let (src, dst) = if to_pool { (r, t) } else { (t, r) };
            core::ptr::copy_nonoverlapping(src, dst, k);
        }
        row += k;
        c = st().next[c as usize];
    }
    ((pos - base) * 2 * g.heads * g.hb) as u64
}

/// Copy R (DeltaNet S + conv history, one contiguous block) between a chunk list and the engine.
fn copy_r(mut c: u16, to_pool: bool) {
    let live = ctx(CTX_BUF_DN_STATE) as *mut u8;
    let mut off = 0;
    while off < r_bytes() {
        let k = CHUNK.min(r_bytes() - off);
        unsafe {
            let (src, dst) = if to_pool { (live.add(off), chunk(c)) } else { (chunk(c), live.add(off)) };
            core::ptr::copy_nonoverlapping(src, dst, k);
        }
        off += k;
        c = st().next[c as usize];
    }
}

fn hash_words(h: &mut u64, p: *const u8, bytes: usize) {
    let w = unsafe { core::slice::from_raw_parts(p as *const u64, bytes / 8) };
    w.iter().for_each(|&x| *h = (*h ^ x).wrapping_mul(0x100_0000_01b3));
}
fn hash_r_pool(mut c: u16) -> u64 {
    let (mut h, mut off) = (0xcbf2_9ce4_8422_2325u64, 0);
    while off < r_bytes() {
        let k = CHUNK.min(r_bytes() - off);
        hash_words(&mut h, chunk(c), k);
        off += k;
        c = st().next[c as usize];
    }
    h
}
fn hash_r_live() -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let live = ctx(CTX_BUF_DN_STATE) as *const u8;
    let mut off = 0;
    while off < r_bytes() {
        let k = CHUNK.min(r_bytes() - off);
        hash_words(&mut h, unsafe { live.add(off) }, k);
        off += k;
    }
    h
}

// ---- names and the current line ----

fn tip() -> u16 {
    st().names[st().head].node
}
fn find_name(s: &str) -> Option<usize> {
    st().names.iter().position(|x| x.used && x.as_str() == s)
}
fn add_name(s: &str, node: u16, save: bool, auto: bool) -> Option<usize> {
    let i = st().names.iter().position(|x| !x.used)?;
    let mut x = NAME0;
    let k = s.len().min(24);
    x.s[..k].copy_from_slice(&s.as_bytes()[..k]);
    (x.used, x.n, x.node, x.save, x.auto) = (true, k as u8, node, save, auto);
    st().names[i] = x;
    Some(i)
}
fn names_on(n: u16) -> impl Iterator<Item = &'static Name> {
    st().names.iter().filter(move |x| x.used && x.node == n)
}
/// Is a on the history of n (n itself included)?
fn is_anc(a: u16, mut n: u16) -> bool {
    while n != NIL {
        if n == a {
            return true;
        }
        n = nd(n).back;
    }
    false
}
fn steps(a: u16, mut n: u16) -> usize {
    let mut k = 0;
    while n != a && n != NIL {
        n = nd(n).back;
        k += 1;
    }
    k
}
/// The old tip is about to lose its branch name: name it "<branch>@<id>" if nothing else points at it.
fn keep(t: u16) {
    if t == ROOT || names_on(t).any(|x| !core::ptr::eq(x, &st().names[st().head])) {
        return;
    }
    let mut s = SBuf::<24>::new();
    let _ = write!(s, "{}@{}", st().names[st().head].as_str(), nd(t).id);
    add_name(s.as_str(), t, false, true);
}

/// MARK bit 1 = on the current history (ancestors of the tip or of cur), bit 2 = one of the last RING
/// points of it.
fn mark_path() {
    let m = mark();
    m.fill(0);
    for start in [tip(), st().cur] {
        let (mut n, mut k) = (start, 0);
        while n != NIL {
            m[n as usize] |= if k < RING { 3 } else { 1 };
            k += 1;
            n = nd(n).back;
        }
    }
}
fn is_leaf(n: u16) -> bool {
    !st().nodes.iter().any(|x| x.used && x.parent == n)
}

// ---- eviction (doc 4.3): thin (free R, keep the rows), then drop old anonymous leaves, never names ----

fn thin_candidate() -> Option<u16> {
    mark_path();
    let s = st();
    (1..MAX_NODES as u16)
        .filter(|&n| {
            let x = nd(n);
            x.used && x.r != NIL && n != s.cur && mark()[n as usize] & 2 == 0 && names_on(n).next().is_none()
        })
        .min_by_key(|&n| nd(n).visit)
}
/// A leaf off the current history, unnamed or with only unpinned kernel names (unnamed_only: no names).
fn drop_candidate(min_age: u64, unnamed_only: bool) -> Option<u16> {
    mark_path();
    let s = st();
    (1..MAX_NODES as u16)
        .filter(|&n| {
            let x = nd(n);
            x.used && n != s.cur && mark()[n as usize] & 1 == 0 && is_leaf(n) && s.tick - x.visit >= min_age
                && names_on(n).all(|m| m.auto && !m.pinned && !unnamed_only)
        })
        .min_by_key(|&n| nd(n).visit)
}
fn thin(n: u16) {
    let x = nd(n);
    x.fnv = hash_r_pool(x.r);
    free_list(x.r);
    x.r = NIL;
    let mut l = SBuf::<96>::new();
    let _ = write!(l, "snap: thinned #{} (R freed, rows kept), {} free chunks", x.id, free_chunks());
    crate::log::record(b'A', l.as_str());
}
fn drop_node(n: u16) {
    let s = st();
    for m in s.names.iter_mut().filter(|m| m.used && m.node == n) {
        m.used = false;
    }
    for i in 0..MAX_NODES {
        if s.nodes[i].used && s.nodes[i].back == n {
            s.nodes[i].back = s.nodes[i].parent;
        }
    }
    let x = nd(n);
    free_list(x.kv);
    free_list(x.r);
    let id = x.id;
    *x = NODE0;
    let mut l = SBuf::<64>::new();
    let _ = write!(l, "snap: dropped #{}, {} free chunks", id, free_chunks());
    crate::log::record(b'A', l.as_str());
}
fn make_room(need: usize) -> bool {
    while free_chunks() < need + RESERVE {
        if let Some(n) = thin_candidate() {
            thin(n);
        } else if let Some(n) = drop_candidate(DROP_AGE, false) {
            drop_node(n);
        } else {
            break;
        }
    }
    free_chunks() >= need
}
fn free_slot() -> Option<u16> {
    loop {
        if let Some(i) = st().nodes.iter().position(|x| !x.used) {
            return Some(i as u16);
        }
        drop_node(drop_candidate(DROP_AGE, false)?);
    }
}
/// After a name is removed: drop the leaves nothing points at any more (the line only it used).
fn gc() {
    while let Some(n) = drop_candidate(0, true) {
        drop_node(n);
    }
}

// ---- save and restore ----

/// Save the live state as a child of `parent` (rows [parent.pos, pos) + R). None = no room.
fn save(parent: u16, pos: u64) -> Option<u16> {
    let t0 = smp::rdtsc();
    let base = if parent == NIL { 0 } else { nd(parent).pos };
    let kc = kv_chunks(pos - base);
    if !make_room(kc + r_chunks()) {
        return None;
    }
    let slot = free_slot()?;
    let kv = alloc(kc)?;
    let Some(r) = alloc(r_chunks()) else {
        free_list(kv);
        return None;
    };
    let s = st();
    s.next_id += 1;
    s.nodes[slot as usize] = Node { used: true, id: s.next_id - 1, parent, back: parent, base, pos, kv, r, rng: unsafe { llm::RNG }, visit: s.tick, ex: false, compact: 0, fnv: 0 };
    q_of(slot).clear();
    a_of(slot).clear();
    fx_of(slot).clear();
    let bytes = copy_rows(slot, true) + r_bytes() as u64;
    copy_r(r, true);
    (s.save_us, s.save_bytes) = (smp::tsc_to_us(smp::rdtsc() - t0), bytes);
    let mut l = SBuf::<128>::new();
    let _ = write!(l, "snap: saved #{} rows {}..{} + R, {} KiB copied in {} us, {} chunks", s.next_id - 1, base, pos,
        bytes >> 10, s.save_us, kc + r_chunks());
    crate::log::record(b'A', l.as_str());
    Some(slot)
}

/// Make the live engine state equal to node t: copy the rows of t's chain above the common ancestor with
/// the live chain, then R (or the nearest stored R and a replay of the token ids for a thinned point).
/// false = the replay was interrupted (the previous current point is restored instead).
fn restore(t: u16) -> bool {
    let t0 = smp::rdtsc();
    let s = st();
    let m = mark();
    m.fill(0);
    let mut n = s.live;
    while n != NIL {
        m[n as usize] = 4;
        n = nd(n).parent;
    }
    let mut bytes = 0;
    n = t;
    while n != NIL && m[n as usize] != 4 {
        bytes += copy_rows(n, false);
        n = nd(n).parent;
    }
    let mut a = t;
    while nd(a).r == NIL {
        a = nd(a).parent; // the root is never thinned
    }
    copy_r(nd(a).r, false);
    bytes += r_bytes() as u64;
    s.live = t;
    if a != t {
        let (from, to) = (nd(a).pos, nd(t).pos);
        println!("[snap] #{} was thinned: replaying {} tokens from #{} (about {} s, Ctrl-C stops)", nd(t).id, to - from, nd(a).id, (to - from) / 45 + 1);
        smp::park_phase(true);
        let rec = llm::rec();
        let mut ok = true;
        for p in from..to {
            if (p - from) % 4 == 0 && crate::shell::poll_intr() {
                ok = false;
                break;
            }
            unsafe { transformer_forward(rec[p as usize], p, 0) };
        }
        smp::park_phase(false);
        if !ok {
            let prev = s.cur;
            println!("^C [snap] replay interrupted, staying at #{}", nd(prev).id);
            restore(prev);
            return false;
        }
        let h = hash_r_live();
        let x = nd(t);
        println!("[snap] replay done, R hash {} the stored one", if x.fnv == 0 { "(none stored)" } else if h == x.fnv { "matches" } else { "DIFFERS from" });
        let mut l = SBuf::<96>::new();
        let _ = write!(l, "snap: replayed #{} {} tok, R fnv {:#x} stored {:#x}", x.id, to - from, h, x.fnv);
        crate::log::record(b'A', l.as_str());
        if make_room(r_chunks()) {
            if let Some(r) = alloc(r_chunks()) {
                copy_r(r, true);
                nd(t).r = r; // restorable again
            }
        }
    }
    s.pos = nd(t).pos;
    s.unsaved = false;
    nd(t).visit = s.tick;
    llm::set_rng(nd(t).rng);
    (s.restore_us, s.restore_bytes) = (smp::tsc_to_us(smp::rdtsc() - t0), bytes);
    let mut l = SBuf::<96>::new();
    let _ = write!(l, "snap: restored #{} (ctx {}), {} KiB copied in {} us", nd(t).id, s.pos, bytes >> 10, s.restore_us);
    crate::log::record(b'A', l.as_str());
    true
}

/// Go to point n on the current branch. If n is not on the branch's history, the branch moves to n and
/// its old tip keeps a "<branch>@<id>" name.
fn goto(n: u16) -> bool {
    if !restore(n) {
        return false;
    }
    let s = st();
    let t = tip();
    if !is_anc(n, t) {
        keep(t);
        s.names[s.head].node = n;
    }
    s.cur = n;
    true
}

// ---- API for agent.rs ----

/// Boot: allocate the pool (touched once here) and save the prefilled prefix as root #0 on "main".
pub fn init_root(pos: u64) -> u64 {
    let s = st();
    let mut mib = POOL_MIB;
    while mib >= 64 {
        let p = crate::mem::sys_mmap_anon((mib * CHUNK + CHUNK) as u64);
        if (p as i64) > 0 {
            s.pool = ((p as usize + CHUNK - 1) & !(CHUNK - 1)) as *mut u8; // chunk-aligned
            break;
        }
        mib /= 2;
    }
    assert!(!s.pool.is_null(), "snapshot pool: arena full");
    (s.n_chunks, s.cap) = (mib, mib);
    let slot = save(NIL, pos).expect("snapshot pool: no room for the prefix");
    assert!(slot == ROOT);
    add_name("main", ROOT, false, false);
    (s.head, s.cur, s.live, s.pos) = (0, ROOT, ROOT, pos);
    s.save_bytes
}

pub fn pos() -> u64 {
    st().pos
}
pub fn root_pos() -> u64 {
    nd(ROOT).pos
}
pub fn at_root() -> bool {
    st().cur == ROOT && !st().unsaved
}
pub fn pool_bytes() -> u64 {
    (st().n_chunks * CHUNK) as u64
}

/// /bench, /fill, /gen overwrite the cache from position 0.
pub fn invalidate() {
    st().live = NIL;
}
/// Before a turn: the live state must be the current point (after an invalidate). Returns the position.
pub fn ensure_live() -> u64 {
    if st().live == NIL {
        restore(st().cur);
    }
    st().pos
}
/// Ctrl-C or an interrupted compaction: back to the current point (R only when the rows are intact).
pub fn rollback() -> u64 {
    restore(st().cur);
    st().pos
}
/// Live state = the root (the boot prefix, system turn still open): compaction rebuilds from it.
pub fn restore_root() -> u64 {
    restore(ROOT);
    st().live = ROOT;
    nd(ROOT).pos
}

/// Save a finished turn (or, with compact = N, the rebuilt context of compaction N) as the new current
/// point and branch tip. fx = tool calls with side effects in this turn.
pub fn commit(pos: u64, q: &str, a: &str, ex: bool, fx: &str, compact: u32) {
    let s = st();
    let (cur, parent) = (s.cur, s.live);
    s.pos = pos;
    if parent == NIL || (pos == nd(parent).pos && compact == 0) {
        return;
    }
    let Some(n) = save(parent, pos) else {
        s.unsaved = true;
        println!("[snap] pool full: this turn is not saved and Ctrl-C would lose it. /drop something or /save less");
        return;
    };
    let x = nd(n);
    (x.back, x.ex, x.compact) = (cur, ex, compact);
    let _ = write!(q_of(n), "{}", &q[..crate::agent::floor_char(q, TXT)]);
    let _ = write!(a_of(n), "{}", &a[..crate::agent::floor_char(a, TXT)]);
    let _ = write!(fx_of(n), "{}", fx);
    if compact != 0 {
        for m in s.names.iter_mut().filter(|m| m.used && m.pinned) {
            m.pinned = false; // the previous uncompacted line becomes evictable
        }
        let mut nm = SBuf::<24>::new();
        let _ = write!(nm, "pre-compact-{}", compact);
        if let Some(i) = add_name(nm.as_str(), cur, true, true) {
            s.names[i].pinned = true;
        }
    }
    if cur != tip() {
        keep(tip()); // a new turn after an undo: the old line stays reachable
    }
    s.names[s.head].node = n;
    (s.cur, s.live, s.unsaved) = (n, n, false);
    s.tick += 1;
    x.visit = s.tick;
}

/// /clear: a new conversation on the current branch (the old line keeps a name).
pub fn new_line() {
    let s = st();
    if tip() != ROOT {
        keep(tip());
    }
    s.names[s.head].node = ROOT;
    s.cur = ROOT;
    s.unsaved = false;
    if s.live == NIL || s.live != ROOT {
        restore(ROOT);
    }
}

/// The last k exchanges on the current history (oldest first), for the compaction re-prefill.
pub fn exchanges(k: usize, mut f: impl FnMut(&str, &str)) {
    let mut v = [NIL; 8];
    let (mut n, mut c) = (st().cur, 0);
    while n != NIL && c < k.min(8) {
        if nd(n).ex {
            v[c] = n;
            c += 1;
        }
        n = nd(n).back;
    }
    for i in (0..c).rev() {
        f(q_of(v[i]).as_str(), a_of(v[i]).as_str());
    }
}
/// (exchanges on the current history, exchanges since the last compaction)
pub fn ex_count() -> (usize, usize) {
    let (mut n, mut all, mut since, mut past) = (st().cur, 0, 0, false);
    while n != NIL {
        past |= nd(n).compact != 0;
        if nd(n).ex {
            all += 1;
            since += !past as usize;
        }
        n = nd(n).back;
    }
    (all, since)
}
/// The question of the current point (for /retry).
pub fn last_question<const N: usize>(o: &mut SBuf<N>) -> bool {
    let c = st().cur;
    if c == ROOT || nd(c).compact != 0 || q_of(c).n == 0 {
        return false;
    }
    let _ = write!(o, "{}", q_of(c).as_str());
    true
}

/// "ringzero[main]> " or "ringzero[main~2]> ".
pub fn prompt<const N: usize>(o: &mut SBuf<N>) {
    let s = st();
    let _ = write!(o, "ringzero[{}", s.names[s.head].as_str());
    let k = steps(s.cur, tip());
    if k > 0 {
        let _ = write!(o, "~{}", k);
    }
    let _ = write!(o, "{}]> ", if s.unsaved { "*" } else { "" });
}

// ---- shell commands ----

fn label<const N: usize>(n: u16, o: &mut SBuf<N>) {
    let x = nd(n);
    if n == ROOT {
        let _ = write!(o, "boot prefix");
    } else if x.compact != 0 {
        let _ = write!(o, "[compact {}] summary", x.compact);
    } else {
        let q = q_of(n).as_str();
        let mut k = 0;
        for ch in q.chars() {
            if k == 32 {
                break;
            }
            let _ = o.write_char(if ch.is_control() { ' ' } else { ch });
            k += 1;
        }
    }
}
fn describe(n: u16) {
    let mut l = SBuf::<64>::new();
    label(n, &mut l);
    print!("#{} \"{}\" (ctx {}/{})", nd(n).id, l.as_str(), nd(n).pos, ctx(CTX_MAX_SEQ));
}
fn ms() -> (u64, u64) {
    (st().restore_us / 1000, st().restore_us % 1000 / 10)
}
fn by_id(arg: &str) -> Option<u16> {
    let id = arg.strip_prefix('#')?.parse::<u32>().ok()?;
    (0..MAX_NODES as u16).find(|&n| nd(n).used && nd(n).id == id)
}
fn valid_name(s: &str) -> bool {
    (1..=16).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Notes for the points stepped over by an undo (from `from` back to `to`): their tool side effects stay.
fn fx_notes(from: u16, to: u16) {
    let mut n = from;
    while n != to && n != NIL {
        if fx_of(n).n > 0 {
            println!("[undo] note: #{} ran {}; the machine keeps it.", nd(n).id, fx_of(n).as_str());
        }
        n = nd(n).back;
    }
}

pub fn undo(k: usize) {
    let s = st();
    let from = s.cur;
    let mut n = from;
    for _ in 0..k.max(1) {
        if nd(n).back == NIL {
            break;
        }
        n = nd(n).back;
    }
    if n == from {
        println!("[undo] nothing to undo (at the boot prefix)");
        return;
    }
    if !goto(n) {
        return;
    }
    print!("[undo] back to ");
    describe(n);
    let (a, b) = ms();
    println!(", {}.{:02} ms. /redo brings #{} back.", a, b, nd(from).id);
    fx_notes(from, n);
    let mut h = SBuf::<64>::new();
    let _ = write!(h, "[undo] back to #{}", nd(n).id);
    crate::agent::note(h.as_str());
}

pub fn redo(k: usize) {
    let s = st();
    let behind = steps(s.cur, tip());
    if behind == 0 {
        println!("[redo] nothing to redo (at the tip of {})", s.names[s.head].as_str());
        return;
    }
    let mut n = tip();
    for _ in 0..behind - k.clamp(1, behind) {
        n = nd(n).back;
    }
    if !goto(n) {
        return;
    }
    print!("[redo] forward to ");
    describe(n);
    let (a, b) = ms();
    println!(", {}.{:02} ms", a, b);
    let mut h = SBuf::<64>::new();
    let _ = write!(h, "[redo] forward to #{}", nd(n).id);
    crate::agent::note(h.as_str());
}

pub fn branch(name: &str) {
    if !valid_name(name) {
        println!("usage: /branch NAME  (1-16 of a-z 0-9 _ -)");
        return;
    }
    if find_name(name).is_some() {
        println!("[branch] '{}' already exists (/tree lists the names)", name);
        return;
    }
    let s = st();
    let from = s.names[s.head].as_str();
    let mut f = SBuf::<24>::new();
    let _ = write!(f, "{}", from);
    match add_name(name, s.cur, false, false) {
        Some(i) => {
            s.head = i;
            println!("[branch] {} starts at #{} (from {}); new turns go to {}", name, nd(s.cur).id, f.as_str(), name);
        }
        None => println!("[branch] name table full, /drop a name first"),
    }
}

pub fn save_cmd(name: &str) {
    if !valid_name(name) {
        println!("usage: /save NAME  (1-16 of a-z 0-9 _ -)");
        return;
    }
    if find_name(name).is_some() {
        println!("[save] '{}' already exists (/drop {} first)", name, name);
        return;
    }
    let c = st().cur;
    match add_name(name, c, true, false) {
        Some(_) => {
            print!("[save] {} = ", name);
            describe(c);
            println!(", kept until /drop {}", name);
        }
        None => println!("[save] name table full, /drop a name first"),
    }
}

/// /switch and /load: a branch name switches branch; a save or #ID moves the current branch there.
pub fn switch(arg: &str, cmd: &str) {
    let s = st();
    let (n, branch) = match (find_name(arg), by_id(arg)) {
        (Some(i), _) if !s.names[i].save => (s.names[i].node, Some(i)),
        (Some(i), _) => (s.names[i].node, None),
        (None, Some(n)) => (n, None),
        _ => {
            println!("[{}] no branch, save or point '{}' (/tree lists them)", cmd, arg);
            return;
        }
    };
    if let Some(i) = branch {
        if !restore(n) {
            return;
        }
        (s.head, s.cur) = (i, n);
        if s.names[i].auto {
            s.names[i].auto = false; // switched to on purpose: a real branch now, never dropped
        }
    } else if !goto(n) {
        return;
    }
    print!("[{}] {} at ", cmd, s.names[s.head].as_str());
    describe(n);
    let (a, b) = ms();
    println!(", {}.{:02} ms", a, b);
    let mut h = SBuf::<64>::new();
    let _ = write!(h, "[{}] to #{}", cmd, nd(n).id);
    crate::agent::note(h.as_str());
}

pub fn drop_cmd(arg: &str) {
    let s = st();
    if let Some(i) = find_name(arg) {
        if i == s.head {
            println!("[drop] {} is the current branch: /switch to another one first", arg);
            return;
        }
        s.names[i].used = false;
        let before = n_used();
        gc();
        println!("[drop] {} removed, {} MiB freed", arg, before - n_used());
        return;
    }
    let Some(n) = by_id(arg) else {
        println!("[drop] no branch, save or point '{}' (/tree lists them)", arg);
        return;
    };
    mark_path();
    if n == ROOT || n == s.cur || mark()[n as usize] & 1 != 0 || !is_leaf(n) || names_on(n).next().is_some() {
        println!("[drop] #{} is on the current line, has later points or has names: drop those first", nd(n).id);
        return;
    }
    let before = n_used();
    drop_node(n);
    gc();
    println!("[drop] #{} removed, {} MiB freed", arg.trim_start_matches('#'), before - n_used());
}

/// Compact tree: a linear run stays in one column, a fork indents. Without `all`, the inner points of a
/// run (no names, not current) collapse to "... N more".
pub fn tree(all: bool) {
    static mut STACK: [(u16, u8, bool); MAX_NODES] = [(0, 0, false); MAX_NODES]; // node, depth, last child
    let stack = unsafe { &mut *core::ptr::addr_of_mut!(STACK) };
    let s = st();
    let mut pre = [false; 64]; // pre[d] = an open sibling at depth d (draw '|')
    let mut sp = 1;
    stack[0] = (ROOT, 0, true);
    let kids = |n: u16, f: &mut dyn FnMut(u16)| {
        (0..MAX_NODES as u16).filter(|&c| nd(c).used && nd(c).parent == n && c != n).for_each(|c| f(c));
    };
    while sp > 0 {
        sp -= 1;
        let (top, depth, last) = stack[sp];
        let d = (depth as usize).min(63);
        if d > 0 {
            pre[d] = !last;
        }
        let mut n = top;
        let mut first = true;
        let mut skipped = 0;
        loop {
            let mut nk = 0;
            let mut only = NIL;
            kids(n, &mut |c| {
                nk += 1;
                only = c;
            });
            let show = all || first || nk != 1 || n == s.cur || names_on(n).next().is_some();
            if show {
                let indent = |conn: bool| {
                    for i in 1..d {
                        print!("{}", if pre[i] { "|  " } else { "   " });
                    }
                    if d > 0 {
                        print!("{}", if conn { if last { "`- " } else { "|- " } } else if last { "   " } else { "|  " });
                    }
                };
                if skipped > 0 {
                    indent(false);
                    println!("... {} more", skipped);
                    skipped = 0;
                }
                indent(first);
                let mut l = SBuf::<64>::new();
                label(n, &mut l);
                let mut idl = SBuf::<8>::new();
                let _ = write!(idl, "#{}", nd(n).id);
                print!("{:<5} {:<32} {:>6}", idl.as_str(), l.as_str(), nd(n).pos);
                for (i, m) in names_on(n).enumerate() {
                    print!("{}{}", if i == 0 { "  " } else { ", " }, m.as_str());
                }
                if nd(n).r == NIL {
                    print!("  (thinned)");
                }
                if n == s.cur {
                    print!("  <- here");
                }
                println!("");
            } else {
                skipped += 1;
            }
            first = false;
            if nk == 1 {
                n = only;
                continue;
            }
            // fork (or leaf): push the children, the first one on top
            let mut cs = [NIL; MAX_NODES];
            let mut k = 0;
            kids(n, &mut |c| {
                cs[k] = c;
                k += 1;
            });
            for j in (0..k).rev() {
                if sp < MAX_NODES {
                    stack[sp] = (cs[j], (d + 1) as u8, j == k - 1);
                    sp += 1;
                }
            }
            break;
        }
    }
    status_line();
}

fn status_line() {
    let s = st();
    let (mut pts, mut thinned) = (0, 0);
    s.nodes.iter().filter(|x| x.used).for_each(|x| {
        pts += 1;
        thinned += (x.r == NIL) as usize;
    });
    let br = s.names.iter().filter(|m| m.used && !m.save).count();
    let sv = s.names.iter().filter(|m| m.used && m.save).count();
    println!("pool {}/{} MiB, {} points ({} thinned), {} branches, {} saves", n_used(), s.cap, pts, thinned, br, sv);
}

/// /snap: where we are, pool use, last timings.
pub fn status() {
    let s = st();
    print!("[snap] at ");
    describe(s.cur);
    println!(" on {}{}", s.names[s.head].as_str(), if s.unsaved { ", live state NOT saved (pool full)" } else { "" });
    print!("[snap] ");
    status_line();
    println!("[snap] last save {} KiB in {} us, last restore {} KiB in {} us; R {} KiB per point, {} KV rows per MiB chunk ({} B/row)",
        s.save_bytes >> 10, s.save_us, s.restore_bytes >> 10, s.restore_us, r_bytes() >> 10, geo().rpc, 2 * geo().heads * geo().hb);
}

/// `!set snappool MiB`: cap the usable pool (to test eviction). Never below what is in use.
pub fn set_cap(mib: usize) -> bool {
    if mib < 64 || mib > st().n_chunks {
        return false;
    }
    st().cap = mib.max(n_used());
    true
}
pub fn cap_mib() -> usize {
    st().cap
}
pub fn max_mib() -> usize {
    st().n_chunks
}
