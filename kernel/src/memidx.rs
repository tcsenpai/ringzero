// In-RAM memory index: texts embedded with Qwen3-Embedding (embed.rs) and stored as Q8_0 rows (int8 + fp16
// block scales, the engine's weight layout), so a search is one tp_matmul_q8_0 of the query against all rows on
// every core. Plus BM25 over the same texts (stopwords removed) and the gated fusion from the spike
// (docs/spikes/EMBEDDER.md): BM25 joins the ranking (RRF) only when the query looks like an identifier or path,
// or when its top BM25 hit is strong; otherwise the embedding order stands. API (add, search, text, len, clear,
// set_dim, save, load) is what /index, /search, the agent's recall tool and agentd use.
// Persistence (M4.5): agentd saves the index to /var/lib/ringzero/mem/index.bin with a header (embedder hash, dim,
// Q8_0 row layout, chunker version, a checksum) and loads it at boot; any mismatch means a rebuild by re-embedding.
// Texts carry their source: "[/path] chunk" (fidx.rs), "[fact] key: value", "[episode N] ...", "[said] ..." (agentd).
// T72 (design A, docs/M_MEMA_RESULT.md): every entry has a date (Unix seconds, saved since index version 2), a hub
// score (mean cosine to its HUB_K nearest generic probe questions, memrank.rs) and the search ranks by the cosine
// minus a caller's penalty; Hit.cos stays the raw cosine, the one the floors read.
use crate::embed::{self, DIM};
use crate::engine::{ctx, matmul_set_fp16, tp_matmul_q8_0};
use crate::smp;

pub const CAP: usize = 4096;
const TEXT_BYTES: usize = 1 << 20;
/// one stored text at most (longer ones are cut at a char boundary)
pub const TEXT_MAX: usize = 2048;
const ROW_MAX: usize = DIM + DIM / 16;
const QT: usize = 32; // query terms used by BM25
const K1: f64 = 1.5;
const B: f64 = 0.75;
const RRF_K: f64 = 60.0;
const CTX_FP16_SCALES: usize = 368; // engine/constants.inc

struct Idx {
    n: usize,
    dim: usize,
    text_n: usize,
    off: [u32; CAP + 1],
    embed_us: u64, // sum over adds
    embed_tok: u64,
    gen: u64, // bumped on every change (agentd saves the index when it differs from the saved one)
}

static mut IX: Idx = Idx { n: 0, dim: DIM, text_n: 0, off: [0; CAP + 1], embed_us: 0, embed_tok: 0, gen: 0 };
/// The rows go to tp_matmul_q8_0 as one weight matrix: bit 0 of that pointer tags Q4_0 (matmul.asm, M7), so a plain
/// u8 array (align 1) at an odd address was read as Q4_0 and every cos came out NaN. 64 like the model's tensors.
#[repr(C, align(64))]
struct Rows([u8; CAP * ROW_MAX]);
static mut ROWS: Rows = Rows([0; CAP * ROW_MAX]);
static mut TXT: [u8; TEXT_BYTES] = [0; TEXT_BYTES];
static mut COS: [f32; CAP] = [0.0; CAP];
static mut BM: [f32; CAP] = [0.0; CAP];
static mut ORD: [u16; CAP] = [0; CAP];
static mut RANK_E: [u16; CAP] = [0; CAP];
static mut TF: [[u16; QT]; CAP] = [[0; QT]; CAP];
static mut DL: [u16; CAP] = [0; CAP];
static mut VEC: [f32; DIM] = [0.0; DIM];
/// T72: date of each entry, Unix seconds (0 = not known)
static mut TS: [u64; CAP] = [0; CAP];
static mut SCORE: [f32; CAP] = [0.0; CAP];
/// T72: hub score per entry, valid for the first HUB_N entries (appends extend it, retain / clear / load reset it)
static mut HUB: [f32; CAP] = [0.0; CAP];
static mut HUB_N: usize = 0;
/// the probe questions' Q8_0 rows (memrank::PROBES order), as one weight matrix like ROWS
#[repr(C, align(64))]
struct Probes([u8; crate::memrank::NP * ROW_MAX]);
static mut PROBE: Probes = Probes([0; crate::memrank::NP * ROW_MAX]);
static mut PROBE_DIM: usize = 0; // 0 = no probes yet (no hub score)

fn ix() -> &'static mut Idx {
    unsafe { &mut *core::ptr::addr_of_mut!(IX) }
}
fn stride(dim: usize) -> usize {
    dim + dim / 16
}

pub fn len() -> usize {
    ix().n
}
pub fn dim() -> usize {
    ix().dim
}
pub fn text(id: usize) -> &'static str {
    let x = ix();
    let t = unsafe { &*core::ptr::addr_of!(TXT) };
    core::str::from_utf8(&t[x.off[id] as usize..x.off[id + 1] as usize]).unwrap_or("")
}
/// An entry with exactly this text exists (dedup of repeated user turns).
pub fn contains(t: &str) -> bool {
    let t = t.trim();
    (0..ix().n).any(|i| text(i) == t)
}
pub fn gen() -> u64 {
    ix().gen
}
pub fn clear() {
    let x = ix();
    (x.n, x.text_n, x.embed_us, x.embed_tok) = (0, 0, 0, 0);
    x.gen += 1;
    unsafe { HUB_N = 0 };
}
/// T72: date of an entry (Unix seconds, 0 = not known).
pub fn ts(id: usize) -> u64 {
    unsafe { TS[id] }
}
/// T72: set the date of an entry (the migration of a version 1 index).
pub fn set_ts(id: usize, t: u64) {
    if id < ix().n {
        unsafe { TS[id] = t };
        ix().gen += 1;
    }
}
/// Matryoshka dim (multiple of 128, 128..1024); only on an empty index (stored rows cannot change dim).
pub fn set_dim(d: usize) -> Result<(), &'static str> {
    if d < 128 || d > DIM || d % 128 != 0 {
        return Err("dim must be a multiple of 128 from 128 to 1024");
    }
    if ix().n > 0 && d != ix().dim {
        return Err("the index is not empty: /index clear first");
    }
    ix().dim = d;
    Ok(())
}

// ---- fp16 and Q8_0 (soft-float Rust: no F16C here) ----

fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e <= 0 {
        if e < -10 {
            return sign as u16;
        }
        return (sign | ((m | 0x80_0000) >> (14 - e))) as u16; // subnormal, truncated
    }
    if e >= 31 {
        return (sign | 0x7c00) as u16;
    }
    let h = sign | ((e as u32) << 10) | (m >> 13);
    (h + ((m >> 12) & 1)) as u16 // round half up (a carry into the exponent is still correct)
}

fn f16_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let v = match e {
        0 => return if s != 0 { -(m as f32 * 5.960_464_5e-8) } else { m as f32 * 5.960_464_5e-8 },
        31 => s | 0x7f80_0000 | (m << 13),
        _ => s | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(v)
}

/// One row back to floats (the probes and supersession need an entry's vector, not only its cosine to a query).
fn dequant(row: &[u8], dim: usize, out: &mut [f32]) {
    for b in 0..dim / 32 {
        let s = f16_f32(u16::from_le_bytes([row[dim + 2 * b], row[dim + 2 * b + 1]]));
        for i in 0..32 {
            out[b * 32 + i] = (row[b * 32 + i] as i8) as f32 * s;
        }
    }
}

fn row(id: usize) -> &'static [u8] {
    let st = stride(ix().dim);
    unsafe { &(&*core::ptr::addr_of!(ROWS.0))[id * st..(id + 1) * st] }
}

/// T72: cosine between two entries (their Q8_0 rows; the embedder's vectors are unit length).
pub fn dd(a: usize, b: usize) -> f32 {
    let dim = ix().dim;
    let (mut va, mut vb) = ([0.0f32; DIM], [0.0f32; DIM]);
    dequant(row(a), dim, &mut va);
    dequant(row(b), dim, &mut vb);
    va[..dim].iter().zip(&vb[..dim]).map(|(x, y)| x * y).sum()
}

/// T72: the probe questions' rows (NP rows at the current dim); hub scores are computed again from them.
pub fn set_probes(rows: &[u8]) -> Result<(), &'static str> {
    let st = stride(ix().dim);
    if rows.len() != crate::memrank::NP * st {
        return Err("probe rows do not match the index dim");
    }
    unsafe {
        (&mut *core::ptr::addr_of_mut!(PROBE.0))[..rows.len()].copy_from_slice(rows);
        PROBE_DIM = ix().dim;
        HUB_N = 0;
    }
    Ok(())
}
pub fn probes() -> bool {
    unsafe { PROBE_DIM != 0 && PROBE_DIM == ix().dim }
}

/// T72: hub score of every entry (mean cosine to its HUB_K nearest probes), computed for the entries added since
/// the last call; 0 for all without probes.
pub fn hubs() -> &'static [f32] {
    let x = ix();
    let hub = unsafe { &mut *core::ptr::addr_of_mut!(HUB) };
    if !probes() {
        hub[..x.n].fill(0.0);
        return &hub[..x.n];
    }
    let (np, dim) = (crate::memrank::NP, x.dim);
    let mut v = [0.0f32; DIM];
    let mut c = [0.0f32; crate::memrank::NP];
    unsafe { matmul_set_fp16(1) };
    for d in unsafe { HUB_N }..x.n {
        dequant(row(d), dim, &mut v);
        unsafe { tp_matmul_q8_0(core::ptr::addr_of!(PROBE.0) as *const u8, v.as_ptr(), c.as_mut_ptr(), np as u64, dim as u64) };
        hub[d] = crate::memrank::hub_of(&mut c);
    }
    unsafe {
        matmul_set_fp16(ctx(CTX_FP16_SCALES) & 0xffff_ffff);
        HUB_N = x.n;
    }
    &hub[..x.n]
}

/// v[..dim] -> one row: dim int8, then dim/32 fp16 scales (matmul.asm v2 layout).
fn quantize(v: &[f32], row: &mut [u8]) {
    let dim = v.len();
    for b in 0..dim / 32 {
        let blk = &v[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0.0f32, |a, &x| if x.abs() > a { x.abs() } else { a });
        let h = f16_bits(amax / 127.0);
        let s = f16_f32(h);
        for (i, &x) in blk.iter().enumerate() {
            let q = if s > 0.0 { x / s } else { 0.0 };
            let q = if q >= 0.0 { (q + 0.5) as i32 } else { (q - 0.5) as i32 };
            row[b * 32 + i] = q.clamp(-127, 127) as i8 as u8;
        }
        row[dim + 2 * b..dim + 2 * b + 2].copy_from_slice(&h.to_le_bytes());
    }
}

// ---- BM25 ----

const STOP: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "at", "for", "and", "or", "my", "me", "i",
    "you", "your", "do", "does", "did", "what", "which", "when", "where", "how", "who", "should", "this", "that", "it",
    "with", "from", "have", "has", "will", "can", "il", "lo", "la", "gli", "le", "un", "una", "e", "di", "da", "con", "su",
    "per", "che", "chi", "come", "quando", "dove", "quale", "quanta", "quanto", "ho", "ha", "mio", "mia", "del", "della",
    "dei", "delle", "al", "alla", "col", "sono", "era", "cosa", "tipo",
];

fn wordc(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

/// Lowercased words (\w+ runs) minus stopwords; f gets each word as bytes (ASCII lowercased into a scratch).
fn words(s: &str, mut f: impl FnMut(&[u8])) {
    let b = s.as_bytes();
    let mut i = 0;
    let mut w = [0u8; 64];
    while i < b.len() {
        if !wordc(b[i]) {
            i += 1;
            continue;
        }
        let j = (i..b.len()).find(|&j| !wordc(b[j])).unwrap_or(b.len());
        let k = (j - i).min(64);
        for (d, s) in w[..k].iter_mut().zip(&b[i..i + k]) {
            *d = s.to_ascii_lowercase();
        }
        if !STOP.iter().any(|st| st.as_bytes() == &w[..k]) {
            f(&w[..k]);
        }
        i = j;
    }
}

/// ln without libm: x = m * 2^e, ln m = 2 atanh((m-1)/(m+1)).
fn ln(x: f64) -> f64 {
    let b = x.to_bits();
    let e = ((b >> 52) & 0x7ff) as i64 - 1023;
    let m = f64::from_bits((b & 0x000f_ffff_ffff_ffff) | (1023u64 << 52));
    let z = (m - 1.0) / (m + 1.0);
    let (z2, mut t, mut s) = (z * z, z, 0.0);
    for k in 0..12 {
        s += t / (2 * k + 1) as f64;
        t *= z2;
    }
    e as f64 * core::f64::consts::LN_2 + 2.0 * s
}

/// BM25 of every document into BM; returns (top doc, top score, second score, top doc has every query term).
fn bm25(q: &str) -> (usize, f32, f32, bool) {
    let x = ix();
    let mut qt = [[0u8; 64]; QT];
    let mut ql = [0usize; QT];
    let mut nq = 0;
    words(q, |w| {
        if nq < QT {
            qt[nq][..w.len()].copy_from_slice(w);
            ql[nq] = w.len();
            nq += 1;
        }
    });
    let (tf, dl, bm) = unsafe { (&mut *core::ptr::addr_of_mut!(TF), &mut *core::ptr::addr_of_mut!(DL), &mut *core::ptr::addr_of_mut!(BM)) };
    let mut df = [0u32; QT];
    let mut total = 0u64;
    for d in 0..x.n {
        tf[d][..nq].fill(0);
        let mut n = 0u32;
        words(text(d), |w| {
            n += 1;
            for t in 0..nq {
                if &qt[t][..ql[t]] == w {
                    tf[d][t] = tf[d][t].saturating_add(1);
                }
            }
        });
        dl[d] = n.min(u16::MAX as u32) as u16;
        total += n as u64;
        for t in 0..nq {
            df[t] += (tf[d][t] > 0) as u32;
        }
    }
    let avg = (total as f64 / x.n.max(1) as f64).max(1.0);
    let nd = x.n as f64;
    let idf: [f64; QT] = core::array::from_fn(|t| if t < nq && df[t] > 0 { ln(1.0 + (nd - df[t] as f64 + 0.5) / (df[t] as f64 + 0.5)) } else { 0.0 });
    let (mut top, mut s1, mut s2) = (0usize, 0.0f64, 0.0f64);
    for d in 0..x.n {
        let mut s = 0.0f64;
        for t in 0..nq {
            let f = tf[d][t] as f64;
            if f > 0.0 {
                s += idf[t] * f * (K1 + 1.0) / (f + K1 * (1.0 - B + B * dl[d] as f64 / avg));
            }
        }
        bm[d] = s as f32;
        if s > s1 {
            (s2, s1, top) = (s1, s, d);
        } else if s > s2 {
            s2 = s;
        }
    }
    let all = nq > 0 && s1 > 0.0 && (0..nq).all(|t| tf[top][t] > 0);
    (top, s1 as f32, s2 as f32, all)
}

/// Query looks like an identifier, file name or path: "notes/todo.md", "HL-L2350DW", "casa-wifi", "JLPT".
pub fn identifier_like(q: &str) -> bool {
    q.split_whitespace().any(|w| {
        let w = w.trim_matches(|c: char| matches!(c, '?' | '!' | ',' | ';' | '"' | '\'' | '(' | ')'));
        let b = w.as_bytes();
        let inner = (1..b.len().saturating_sub(1))
            .any(|i| b"/\\._-:@#".contains(&b[i]) && b[i - 1].is_ascii_alphanumeric() && b[i + 1].is_ascii_alphanumeric());
        let alpha = b.iter().any(|c| c.is_ascii_alphabetic());
        let digit = b.iter().any(|c| c.is_ascii_digit());
        let caps = b.len() >= 2 && b.iter().all(|c| c.is_ascii_uppercase());
        inner || (alpha && digit) || caps || b.first() == Some(&b'/')
    })
}

// ---- API ----

pub struct Added {
    pub id: usize,
    pub info: embed::Info,
}

/// Embed a document and store it with date `ts`. Err if the index or its text space is full.
pub fn add(text: &str, ts: u64) -> Result<Added, &'static str> {
    room(text)?;
    let v = unsafe { &mut *core::ptr::addr_of_mut!(VEC) };
    let info = embed::embed(text.trim(), false, ix().dim, v)?;
    store(text, &v[..ix().dim], info, ts)
}

/// Err if `text` would not fit (checked before spending time on its embedding).
pub fn room(text: &str) -> Result<(), &'static str> {
    let x = ix();
    let text = text.trim();
    if text.is_empty() {
        return Err("empty text");
    }
    if x.n == CAP || x.text_n + text.len().min(TEXT_MAX) > TEXT_BYTES {
        return Err("memory index full");
    }
    Ok(())
}

/// Store a text with its embedding v (len = dim()) and its date: one Q8_0 row + the text.
pub fn store(text: &str, v: &[f32], info: embed::Info, ts: u64) -> Result<Added, &'static str> {
    let id = store_row(text, &row_of(&v[..ix().dim]), ts)?;
    let x = ix();
    x.embed_us += info.us;
    x.embed_tok += info.n_tok as u64;
    Ok(Added { id, info })
}

/// T38: v[..dim] as the Q8_0 row `store` would write (the baked manual vectors are made with this).
pub fn row_of(v: &[f32]) -> alloc::vec::Vec<u8> {
    let mut r = alloc::vec![0u8; stride(v.len())];
    quantize(v, &mut r);
    r
}
/// Bytes of one row at the current dim.
pub fn row_len() -> usize {
    stride(ix().dim)
}
/// T38: store a text with a row that is already quantized (row_len() bytes, the current dim), no embedding.
pub fn store_row(text: &str, row: &[u8], ts: u64) -> Result<usize, &'static str> {
    room(text)?;
    let x = ix();
    let st = stride(x.dim);
    if row.len() != st {
        return Err("row size differs from the index dim");
    }
    let text = text.trim();
    let text = &text[..crate::agent::floor_char(text, TEXT_MAX)];
    unsafe { (&mut *core::ptr::addr_of_mut!(ROWS.0))[x.n * st..(x.n + 1) * st].copy_from_slice(row) };
    let t = unsafe { &mut *core::ptr::addr_of_mut!(TXT) };
    t[x.text_n..x.text_n + text.len()].copy_from_slice(text.as_bytes());
    x.off[x.n] = x.text_n as u32;
    x.text_n += text.len();
    x.off[x.n + 1] = x.text_n as u32;
    unsafe { TS[x.n] = ts };
    x.n += 1;
    x.gen += 1;
    Ok(x.n - 1)
}

/// Keep only the entries whose text passes `keep` (file chunks of a rewritten or removed file); returns how
/// many were dropped. Ids after a dropped entry shift down.
pub fn retain(mut keep: impl FnMut(&str) -> bool) -> usize {
    let x = ix();
    let st = stride(x.dim);
    let (rows, t) = unsafe { (&mut *core::ptr::addr_of_mut!(ROWS.0), &mut *core::ptr::addr_of_mut!(TXT)) };
    let (mut w, mut tw) = (0, 0);
    for r in 0..x.n {
        let (a, b) = (x.off[r] as usize, x.off[r + 1] as usize);
        if !keep(core::str::from_utf8(&t[a..b]).unwrap_or("")) {
            continue;
        }
        rows.copy_within(r * st..(r + 1) * st, w * st);
        t.copy_within(a..b, tw);
        unsafe { TS[w] = TS[r] };
        x.off[w] = tw as u32;
        tw += b - a;
        w += 1;
    }
    x.off[w] = tw as u32;
    let dropped = x.n - w;
    (x.n, x.text_n) = (w, tw);
    x.gen += dropped as u64;
    if dropped > 0 {
        unsafe { HUB_N = 0 };
    }
    dropped
}

#[derive(Clone, Copy, PartialEq)]
pub enum Via {
    Emb,
    Bm25,
    Both,
}

impl Via {
    pub fn name(self) -> &'static str {
        match self {
            Via::Emb => "emb",
            Via::Bm25 => "bm25",
            Via::Both => "emb+bm25",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub id: usize,
    /// raw embedding cosine (the floors read this one, T72)
    pub cos: f32,
    /// what the ranking used: cos minus the caller's penalty (= cos without one)
    pub score: f32,
    pub bm25: f32,
    pub via: Via,
}
pub const HIT0: Hit = Hit { id: 0, cos: 0.0, score: 0.0, bm25: 0.0, via: Via::Emb };

pub struct Found {
    pub n: usize,
    /// why BM25 joined the ranking ("" = it did not, embedding order only)
    pub gate: &'static str,
    pub embed: embed::Info,
    pub search_us: u64,
}

/// Top k (at most hits.len()) for a query. `pen` (one value per entry, T72) is subtracted from each cosine before
/// the ranking: the embedding order and the BM25 fusion both use the lowered score.
pub fn search(q: &str, hits: &mut [Hit], pen: Option<&[f32]>) -> Result<Found, &'static str> {
    let x = ix();
    if x.n == 0 {
        return Err("the memory index is empty");
    }
    let v = unsafe { &mut *core::ptr::addr_of_mut!(VEC) };
    let info = embed::embed(q, true, x.dim, v)?;
    let t0 = smp::rdtsc();
    let (cos, ord, rank_e) = unsafe {
        (&mut *core::ptr::addr_of_mut!(COS), &mut *core::ptr::addr_of_mut!(ORD), &mut *core::ptr::addr_of_mut!(RANK_E))
    };
    unsafe {
        matmul_set_fp16(1); // the rows carry fp16 scales; the loaded model's mode goes back right after
        tp_matmul_q8_0(core::ptr::addr_of!(ROWS.0) as *const u8, v.as_ptr(), cos.as_mut_ptr(), x.n as u64, x.dim as u64);
        matmul_set_fp16(ctx(CTX_FP16_SCALES) & 0xffff_ffff);
    }
    let n = x.n;
    let sc = unsafe { &mut *core::ptr::addr_of_mut!(SCORE) };
    for d in 0..n {
        sc[d] = cos[d] - pen.map_or(0.0, |p| p[d]);
    }
    let desc = |a: f32, b: f32| b.partial_cmp(&a).unwrap_or(core::cmp::Ordering::Equal);
    ord[..n].iter_mut().enumerate().for_each(|(i, o)| *o = i as u16);
    ord[..n].sort_unstable_by(|&a, &b| desc(sc[a as usize], sc[b as usize]).then(a.cmp(&b)));
    for (r, &d) in ord[..n].iter().enumerate() {
        rank_e[d as usize] = r as u16;
    }
    let (_top, s1, s2, all) = bm25(q);
    let bm = unsafe { &*core::ptr::addr_of!(BM) };
    let gate = if identifier_like(q) {
        "identifier-like query"
    } else if s1 > 0.0 && all && s1 >= 1.5 * s2 {
        "strong keyword match"
    } else {
        ""
    };
    if !gate.is_empty() {
        // RRF over the embedding ranks and the BM25 hits (docs with a term match only)
        static mut RB: [u16; CAP] = [0; CAP];
        let rb = unsafe { &mut *core::ptr::addr_of_mut!(RB) };
        let mut hb = 0;
        for d in 0..n {
            if bm[d] > 0.0 {
                rb[hb] = d as u16;
                hb += 1;
            }
        }
        rb[..hb].sort_unstable_by(|&a, &b| desc(bm[a as usize], bm[b as usize]).then(a.cmp(&b)));
        static mut RRF: [f32; CAP] = [0.0; CAP];
        let rrf = unsafe { &mut *core::ptr::addr_of_mut!(RRF) };
        for d in 0..n {
            rrf[d] = (1.0 / (RRF_K + rank_e[d] as f64 + 1.0)) as f32;
        }
        for (r, &d) in rb[..hb].iter().enumerate() {
            rrf[d as usize] += (1.0 / (RRF_K + r as f64 + 1.0)) as f32;
        }
        ord[..n].sort_unstable_by(|&a, &b| desc(rrf[a as usize], rrf[b as usize]).then(rank_e[a as usize].cmp(&rank_e[b as usize])));
    }
    let k = hits.len().min(n);
    for (h, &d) in hits[..k].iter_mut().zip(&ord[..k]) {
        let d = d as usize;
        let via = match (gate.is_empty(), bm[d] > 0.0, (rank_e[d] as usize) < k) {
            (true, _, _) | (false, false, _) => Via::Emb,
            (false, true, true) => Via::Both,
            (false, true, false) => Via::Bm25,
        };
        *h = Hit { id: d, cos: cos[d], score: sc[d], bm25: bm[d], via };
    }
    Ok(Found { n: k, gate, embed: info, search_us: smp::tsc_to_us(smp::rdtsc() - t0) })
}

// ---- on disk: header, extra (agentd's file table and store offsets), off[n+1], rows, texts, dates (v2) ----

const MAGIC: &[u8; 4] = b"RZIX";
/// 2 (T72): the entries' dates (u64 Unix seconds) after the texts. A version 1 file still loads (no dates).
const VER: u32 = 2;
/// Row layout id: int8 x dim, then dim/32 fp16 block scales (matmul.asm v2, fp16 mode).
const LAYOUT: u32 = 0x5108;
const HDR: usize = 72; // magic ver emb, 7 x u32, embed_us embed_tok sum, 4 bytes pad

fn payload(ver: u32) -> [&'static [u8]; 4] {
    let x = ix();
    unsafe {
        [
            core::slice::from_raw_parts(x.off.as_ptr() as *const u8, (x.n + 1) * 4),
            &(&*core::ptr::addr_of!(ROWS.0))[..x.n * stride(x.dim)],
            &(&*core::ptr::addr_of!(TXT))[..x.text_n],
            core::slice::from_raw_parts(core::ptr::addr_of!(TS) as *const u8, if ver >= 2 { x.n * 8 } else { 0 }),
        ]
    }
}

/// Write the index to `path`; `emb` = embedder hash, `chunker` = fidx chunker version. Returns bytes.
pub fn save(path: &str, emb: u64, chunker: u32, extra: &[u8]) -> Result<u64, &'static str> {
    let x = ix();
    let p = payload(VER);
    let sum = p.iter().fold(crate::mstore::fnv(crate::mstore::FNV0, extra), |h, b| crate::mstore::fnv(h, b));
    let mut h = [0u8; HDR];
    h[..4].copy_from_slice(MAGIC);
    let mut o = 4;
    let mut put = |b: &[u8]| {
        h[o..o + b.len()].copy_from_slice(b);
        o += b.len();
    };
    put(&VER.to_le_bytes());
    put(&emb.to_le_bytes());
    for v in [x.dim as u32, stride(x.dim) as u32, LAYOUT, chunker, x.n as u32, x.text_n as u32, extra.len() as u32] {
        put(&v.to_le_bytes());
    }
    for v in [x.embed_us, x.embed_tok, sum] {
        put(&v.to_le_bytes());
    }
    crate::fs::write_parts(path, &[&h, extra, p[0], p[1], p[2], p[3]])
}

/// A loaded index: agentd's extra block, the file's version (1 = no dates: the caller dates the entries) and
/// the chunker version it was cut with.
pub struct Loaded {
    pub extra: alloc::vec::Vec<u8>,
    pub ver: u32,
    pub chunker: u32,
}

/// Load the index from `path` if its header matches this embedder, dim and layout, and its chunker version is
/// `chunker` or an older one the caller can migrate (`min_chunker` and up). Err(why) leaves the index empty.
pub fn load(path: &str, emb: u64, chunker: u32, min_chunker: u32) -> Result<Loaded, &'static str> {
    let mut h = [0u8; HDR];
    let (k, size) = crate::fs::read_into(path, 0, &mut h).map_err(|_| "no saved index")?;
    let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap()) as usize;
    let u64_at = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
    let ver = if k >= HDR { u32_at(4) as u32 } else { 0 };
    if k < HDR || &h[..4] != MAGIC || !(1..=VER).contains(&ver) {
        return Err("bad magic or version");
    }
    let (dim, st, n, tn, xl) = (u32_at(16), u32_at(20), u32_at(32), u32_at(36), u32_at(40));
    let ch = u32_at(28) as u32;
    let tsb = if ver >= 2 { n * 8 } else { 0 };
    let why = if u64_at(8) != emb {
        "embedder changed"
    } else if dim != ix().dim {
        "dim differs"
    } else if st != stride(dim) || u32_at(24) != LAYOUT as usize {
        "row layout differs"
    } else if ch > chunker || ch < min_chunker {
        "chunker version differs"
    } else if n > CAP || tn > TEXT_BYTES || (HDR + xl + (n + 1) * 4 + n * st + tn + tsb) as u64 != size {
        "size does not match the header"
    } else {
        ""
    };
    if !why.is_empty() {
        return Err(why);
    }
    clear();
    let x = ix();
    let mut extra = alloc::vec![0u8; xl];
    let mut off = HDR as u64;
    let mut rd = |b: &mut [u8]| -> Result<(), &'static str> {
        let (k, _) = crate::fs::read_into(path, off, b)?;
        off += k as u64;
        if k == b.len() { Ok(()) } else { Err("short read") }
    };
    rd(&mut extra)?;
    unsafe {
        rd(core::slice::from_raw_parts_mut(x.off.as_mut_ptr() as *mut u8, (n + 1) * 4))?;
        rd(&mut (&mut *core::ptr::addr_of_mut!(ROWS.0))[..n * st])?;
        rd(&mut (&mut *core::ptr::addr_of_mut!(TXT))[..tn])?;
        let tsv = &mut *core::ptr::addr_of_mut!(TS);
        tsv[..n].fill(0);
        rd(core::slice::from_raw_parts_mut(tsv.as_mut_ptr() as *mut u8, tsb))?;
    }
    (x.n, x.text_n) = (n, tn);
    let sum = payload(ver).iter().fold(crate::mstore::fnv(crate::mstore::FNV0, &extra), |h, b| crate::mstore::fnv(h, b));
    if sum != u64_at(60) || x.off[n] as usize != tn || x.off[..n].windows(2).any(|w| w[0] > w[1]) {
        clear();
        return Err("checksum mismatch");
    }
    (x.embed_us, x.embed_tok) = (u64_at(44), u64_at(52));
    x.gen += 1;
    Ok(Loaded { extra, ver, chunker: ch })
}

pub fn stats<const N: usize>(o: &mut crate::log::SBuf<N>) {
    use core::fmt::Write;
    let x = ix();
    let st = stride(x.dim);
    let _ = write!(o, "memory index: {} of {} entries, dim {} (Q8_0 rows of {} B, {} KiB), text {} of {} KiB",
        x.n, CAP, x.dim, st, x.n * st / 1024, x.text_n / 1024, TEXT_BYTES / 1024);
    if x.n > 0 {
        let _ = write!(o, ", embed avg {} ms/entry ({} tok/entry, {} ms per 100 tok)", x.embed_us / x.n as u64 / 1000,
            x.embed_tok / x.n as u64, x.embed_us / 10 / x.embed_tok.max(1));
    }
    let _ = write!(o, "; ");
    crate::fidx::stats(o);
}

/// Boot self-check of the pure-Rust parts (fp16, Q8_0 round trip, ln, identifier gate, retain); run on the
/// empty index at boot.
pub fn selftest() -> bool {
    let f16 = [1.0f32, 0.5, 1e-3, 6.1e-5, 65504.0, -2.0].iter().all(|&v| {
        let r = f16_f32(f16_bits(v));
        (r - v).abs() <= v.abs() / 1024.0
    });
    let v: [f32; 32] = core::array::from_fn(|i| (i as f32 - 15.5) / 100.0);
    let mut row = [0u8; 34];
    quantize(&v, &mut row);
    let s = f16_f32(u16::from_le_bytes([row[32], row[33]]));
    let q8 = (0..32).all(|i| ((row[i] as i8) as f32 * s - v[i]).abs() <= s * 0.51 + 1e-6);
    let l = (ln(10.0) - 2.302_585_093).abs() < 1e-9 && (ln(0.25) + 1.386_294_361).abs() < 1e-9;
    let g = identifier_like("notes/todo.md") && identifier_like("HL-L2350DW") && identifier_like("JLPT")
        && identifier_like("casa-wifi") && !identifier_like("how are my files backed up?") && !identifier_like("restic Backblaze");
    let z = [0.0f32; DIM];
    let info = || embed::Info { n_tok: 0, us: 0, raw_norm: 0.0 };
    ["[/a/x] one", "[/b] two", "[/a/y] three"].iter().enumerate().for_each(|(i, t)| { let _ = store(t, &z[..ix().dim], info(), 10 + i as u64); });
    let r = retain(|t| !t.starts_with("[/a/")) == 2 && len() == 1 && text(0) == "[/b] two" && ts(0) == 11;
    clear();
    f16 && q8 && l && g && r
}
