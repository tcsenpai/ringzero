// M4.8a: multi-stream ("multi-mind") batched decode. Decode is bandwidth bound: one step reads every weight once
// for one token. transformer_prefill with a pf_ms row table (engine/prefill.asm) runs rows of different
// conversations in one pass, each at its own position on its own KV cache and DeltaNet state, so the weights are
// read once for all of them. Every row runs the per-token kernels on its own state and the batched matmuls repeat
// the per-token arithmetic (docs/BATCHED_PREFILL_RESULT.md), so each stream's logits are bit-identical to running
// it alone (`/multi` checks it).
//
// Streams: 0 is the agent's context (the g_ctx buffers, 32768 positions); 1..3 are extra slots with a KV cache of
// SLOT_CAP positions and a DeltaNet state each. A slot's memory is lent by the snapshot pool (snap::lend) while
// the slot is open and given back after, so no memory is set aside at boot. Slots are never saved in snapshots or
// the hibernation image: the one user, agentd's consolidation (bg below), is scratch work that starts over.
use crate::engine::*;
use crate::{llm, print, println, smp};
use alloc::vec::Vec;

// engine/constants.inc (QWEN35)
const CTX_HEAD_DIM: usize = 128;
const CTX_N_KV_HEADS: usize = 120;
const CTX_KV_DIM: usize = 144;
const CTX_VOCAB_SIZE: usize = 152;
const CTX_MAX_SEQ: usize = 160;
const CTX_BUF_LOGITS: usize = 280;
const CTX_BUF_K_CACHE: usize = 328;
const CTX_BUF_V_CACHE: usize = 336;
const CTX_BUF_DN_STATE: usize = 384;
const CTX_BUF_DN_CONV: usize = 392;
const CTX_N_FULL: usize = 416;
const CTX_SAMPLE_N: usize = 488;
const CTX_DN_ZERO_BYTES: usize = 496;

/// The agent's context plus 3 extra slots.
pub const MAX_STREAMS: usize = 4;
/// Positions of an extra slot (KV rows): a forked conversation plus the consolidation instruction and output.
pub const SLOT_CAP: u64 = 4096;
/// One row of the engine's pf_ms table (prefill.asm): 6 x u64.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Row {
    pos: u64,
    kc: u64,
    vc: u64,
    dns: u64,
    dnc: u64,
    max_seq: u64,
}

#[derive(Clone, Copy)]
struct Slot {
    kc: u64,
    vc: u64,
    dns: u64,
    cap: u64, // 0 = closed
    mib: usize,
}
static mut SLOTS: [Slot; MAX_STREAMS] = [Slot { kc: 0, vc: 0, dns: 0, cap: 0, mib: 0 }; MAX_STREAMS];
/// Rows of one pass (<= PREFILL_MAX_B) and logits rows (pf_lg_ptr, llm::LG_ROWS).
const ROWS_MAX: usize = llm::PREFILL_MAX_B as usize;

fn slots() -> &'static mut [Slot; MAX_STREAMS] {
    unsafe { &mut *core::ptr::addr_of_mut!(SLOTS) }
}

pub fn cap(s: usize) -> u64 {
    if s == 0 { ctx(CTX_MAX_SEQ) } else { slots()[s].cap }
}

fn al64(x: u64) -> u64 {
    (x + 63) & !63
}
fn kv_bytes(cap: u64) -> u64 {
    ctx(CTX_N_FULL) * cap * ctx(CTX_KV_DIM) * unsafe { kv_elt_bytes }
}

/// MiB of one extra slot (K + V for SLOT_CAP positions, the DeltaNet state and conv history).
pub fn slot_mib() -> usize {
    (2 * al64(kv_bytes(SLOT_CAP)) + al64(ctx(CTX_DN_ZERO_BYTES))).div_ceil(1 << 20) as usize
}

/// Open slot s (1..MAX_STREAMS) with memory lent by the snapshot pool. false = no batched prefill, or the pool
/// cannot lend it now.
pub fn open(s: usize) -> bool {
    if s == 0 || s >= MAX_STREAMS || !llm::spec_ok() {
        return false;
    }
    if slots()[s].cap != 0 {
        return true;
    }
    let mib = slot_mib();
    let Some(p) = crate::snap::lend(mib) else { return false };
    let kc = p as u64;
    let vc = kc + al64(kv_bytes(SLOT_CAP));
    let dns = vc + al64(kv_bytes(SLOT_CAP));
    slots()[s] = Slot { kc, vc, dns, cap: SLOT_CAP, mib };
    true
}

/// Close slot s: its memory goes back to the snapshot pool.
pub fn close(s: usize) {
    let x = slots()[s];
    if s > 0 && x.cap != 0 {
        crate::snap::take_back(x.kc as *mut u8, x.mib);
        slots()[s].cap = 0;
    }
}

fn row(s: usize, pos: u64) -> Row {
    if s == 0 {
        let d = ctx(CTX_BUF_DN_STATE);
        return Row { pos, kc: ctx(CTX_BUF_K_CACHE), vc: ctx(CTX_BUF_V_CACHE), dns: d, dnc: ctx(CTX_BUF_DN_CONV), max_seq: ctx(CTX_MAX_SEQ) };
    }
    let x = slots()[s];
    // the conv history sits after the recurrent state, at the same offset as in the agent's context
    let dnc = x.dns + ctx(CTX_BUF_DN_CONV) - ctx(CTX_BUF_DN_STATE);
    Row { pos, kc: x.kc, vc: x.vc, dns: x.dns, dnc, max_seq: x.cap }
}

/// Start stream s at position 0: clear its DeltaNet state and conv history (transformer_forward does it at pos 0).
pub fn clear(s: usize) {
    unsafe { core::ptr::write_bytes(row(s, 0).dns as *mut u8, 0, ctx(CTX_DN_ZERO_BYTES) as usize) };
}

/// Stream s (>= 1) continues the agent's context: copy its KV rows 0..pos and its DeltaNet state. false = no room.
pub fn fork(s: usize, pos: u64) -> bool {
    if s == 0 || s >= MAX_STREAMS || pos >= cap(s) {
        return false;
    }
    let (from, to) = (row(0, 0), row(s, 0));
    let e = unsafe { kv_elt_bytes };
    let rb = ctx(CTX_HEAD_DIM) * e; // one position of one kv head
    for (a, b) in [(from.kc, to.kc), (from.vc, to.vc)] {
        for r in 0..ctx(CTX_N_FULL) * ctx(CTX_N_KV_HEADS) {
            // planes are [full][kv_h][max_seq][head_dim]: the row stride is each stream's own max_seq
            unsafe { core::ptr::copy_nonoverlapping((a + r * from.max_seq * rb) as *const u8, (b + r * to.max_seq * rb) as *mut u8, (pos * rb) as usize) };
        }
    }
    unsafe { core::ptr::copy_nonoverlapping(from.dns as *const u8, to.dns as *mut u8, ctx(CTX_DN_ZERO_BYTES) as usize) };
    true
}

/// One engine pass over rows (stream, token, position). lg0 = Some(r): logits of rows r.. into the pf_lg rows
/// (row r -> llm::lg_row(0)), None: no logits. Rows of one stream must be consecutive positions in order.
pub fn step(rows: &[(usize, u32, u64)], lg0: Option<usize>) {
    let n = rows.len();
    assert!(n >= 1 && n <= ROWS_MAX && lg0.map_or(true, |r| r < n && n - r <= llm::LG_ROWS));
    let mut tab = [Row::default(); ROWS_MAX];
    let mut toks = [0u32; ROWS_MAX];
    for (j, &(s, t, p)) in rows.iter().enumerate() {
        (tab[j], toks[j]) = (row(s, p), t);
    }
    unsafe {
        pf_ms = tab.as_ptr() as u64;
        pf_lg0 = lg0.unwrap_or(0) as u64;
        transformer_prefill(toks.as_ptr(), n as u64, 0, if lg0.is_some() { 2 } else { 0 });
        pf_ms = 0;
    }
}

/// Greedy pick of logits row r of the last step.
pub fn argmax_row(r: usize) -> u64 {
    unsafe { argmax_f32(llm::lg_row(r), ctx(CTX_SAMPLE_N)) }
}

const FNV0: u64 = 0xcbf2_9ce4_8422_2325;
fn fnv_f32(h: &mut u64, p: *const f32) {
    let l = unsafe { core::slice::from_raw_parts(p as *const u32, ctx(CTX_VOCAB_SIZE) as usize) };
    l.iter().for_each(|&x| *h = (*h ^ x as u64).wrapping_mul(0x100_0000_01b3));
}

// ---- the background stream: agentd's consolidation, one row next to the foreground's rows ----

pub struct Bg {
    pub slot: usize,
    input: Vec<u32>, // instruction tokens not fed yet (i = next)
    i: usize,
    pub pos: u64,
    next: u64,
    pub out: Vec<u32>,
    max: u64,
    pub done: bool,
    /// tokens fed on top of a foreground step (the "almost free" ones) and alone (idle slices)
    pub rode: u64,
    pub alone: u64,
}
static mut BG: Option<Bg> = None;

pub fn bg() -> Option<&'static mut Bg> {
    unsafe { (*core::ptr::addr_of_mut!(BG)).as_mut() }
}

/// Fork the agent's context at pos into slot 1 and queue the instruction; max = tokens to generate at most.
pub fn bg_start(pos: u64, input: Vec<u32>, max: u64) -> Result<(), &'static str> {
    if pos + input.len() as u64 + max + 8 >= SLOT_CAP {
        return Err("the conversation does not fit in an extra stream");
    }
    if !open(1) {
        return Err("the snapshot pool cannot lend an extra stream now");
    }
    if !fork(1, pos) {
        close(1);
        return Err("fork failed");
    }
    unsafe { BG = Some(Bg { slot: 1, input, i: 0, pos, next: 0, out: Vec::new(), max, done: false, rode: 0, alone: 0 }) };
    Ok(())
}

/// The finished (or abandoned) background stream; its slot goes back to the pool.
pub fn bg_take() -> Option<Bg> {
    let b = unsafe { (*core::ptr::addr_of_mut!(BG)).take() };
    if let Some(x) = &b {
        close(x.slot);
    }
    b
}

/// The background row of the next pass: (slot, token, position, wants logits). None when idle or done.
pub fn bg_row() -> Option<(usize, u32, u64, bool)> {
    let b = bg()?;
    if b.done {
        return None;
    }
    if b.i < b.input.len() {
        return Some((b.slot, b.input[b.i], b.pos, b.i + 1 == b.input.len()));
    }
    Some((b.slot, b.next as u32, b.pos, true))
}

/// After a pass that fed bg_row(): advance, with the row's logits when it asked for them.
pub fn bg_advance(lg: Option<usize>, rode: bool) {
    let Some(b) = bg() else { return };
    if rode { b.rode += 1 } else { b.alone += 1 }
    if b.i < b.input.len() {
        b.i += 1;
    } else {
        b.out.push(b.next as u32);
    }
    b.pos += 1;
    if let Some(r) = lg {
        b.next = argmax_row(r);
        if llm::is_eos(b.next) || b.out.len() as u64 >= b.max || b.pos + 3 >= cap(b.slot) {
            b.done = true;
        }
    }
}

/// Idle slices: the background stream alone (instruction in blocks of 32 rows, then one token per pass) until stop().
pub fn bg_run(stop: &mut dyn FnMut() -> bool) {
    while let Some(b) = bg() {
        if b.done || stop() {
            return;
        }
        if b.i < b.input.len() {
            let e = (b.i + 32).min(b.input.len());
            let mut rows = [(0usize, 0u32, 0u64); 32];
            for (j, k) in (b.i..e).enumerate() {
                rows[j] = (b.slot, b.input[k], b.pos + j as u64);
            }
            let m = e - b.i;
            let last = e == b.input.len();
            step(&rows[..m], if last { Some(m - 1) } else { None });
            for j in 0..m {
                bg_advance(if last && j == m - 1 { Some(0) } else { None }, false); // logits row 0 = row lg0
            }
            continue;
        }
        let (s, t, p, _) = bg_row().unwrap();
        step(&[(s, t, p)], Some(0));
        bg_advance(Some(0), false);
    }
}

// ---- `/multi [N] [T]`: the done criterion of M4.8a ----

const PROMPTS: [&[u8]; MAX_STREAMS] = [
    b"Explain how a CPU cache works in 3 concise sentences.\0",
    b"Write a short poem about the sea.\0",
    b"What are the pros and cons of microkernels?\0",
    b"che cos'e un kernel? spiegalo in 5 frasi\0",
];

/// Stream s's prompt (chat template, think off) into out.
fn prompt(s: usize, out: &mut [u32]) -> usize {
    unsafe { tok_build_chat_prompt(PROMPTS[s].as_ptr(), out.as_mut_ptr(), out.len() as u64, 0, 0) as usize }
}

/// `/multi [N] [T]`: N (1..streams) prompts, T (64) greedy tokens each with EOS ignored. First every prompt alone
/// on the agent's context (the old path: transformer_forward per token), then all N together, one row per stream
/// per pass. Prints tok/s of both, and per stream whether the ids and every step's logits are the same (FNV).
/// Overwrites the live state (the shell restores the conversation).
pub fn bench(arg: &str) {
    let mut it = arg.split_whitespace();
    let want = it.next().and_then(|a| a.parse::<usize>().ok()).unwrap_or(MAX_STREAMS).clamp(1, MAX_STREAMS);
    let t = it.next().and_then(|a| a.parse::<usize>().ok()).unwrap_or(64).clamp(1, 512);
    crate::agentd::stop_bg("/multi");
    let mut n = 1;
    while n < want && open(n) {
        n += 1;
    }
    print!("{}", crate::console::sgr(crate::console::DIM));
    println!("[multi] {} streams: the agent's context + {} slots of {} positions lent by the snapshot pool ({} MiB each){}",
        n, n - 1, SLOT_CAP, slot_mib(), if n < want { " (the pool could not lend more)" } else { "" });
    print!("{}", crate::console::sgr(crate::console::RESET));
    let mut p = [[0u32; 64]; MAX_STREAMS];
    let mut np = [0usize; MAX_STREAMS];
    for s in 0..n {
        np[s] = prompt(s, &mut p[s]);
    }
    let mut ids = [[0u32; 512]; 2 * MAX_STREAMS];
    let mut lh = [FNV0; 2 * MAX_STREAMS];
    smp::park_phase(true);
    // 1. each prompt alone, the per-token path
    let mut us1 = 0;
    for s in 0..n {
        llm::forward(&p[s][..np[s]], 0, true);
        let mut next = llm::pick_greedy();
        let t0 = smp::rdtsc();
        for g in 0..t {
            ids[s][g] = next as u32;
            fnv_f32(&mut lh[s], ctx(CTX_BUF_LOGITS) as *const f32);
            unsafe { transformer_forward(next as u32, (np[s] + g) as u64, 1) };
            next = llm::pick_greedy();
        }
        us1 += smp::tsc_to_us(smp::rdtsc() - t0);
    }
    // 2. all together: prefill each stream (rows of one stream), then one row per stream per pass
    let mut next = [0u64; MAX_STREAMS];
    for s in 0..n {
        clear(s);
        let mut rows = [(0usize, 0u32, 0u64); 64];
        for j in 0..np[s] {
            rows[j] = (s, p[s][j], j as u64);
        }
        step(&rows[..np[s]], Some(np[s] - 1));
        // logits rows start at lg0: row 0 is the prompt's last token (hashed as the per-token path hashes it)
        fnv_f32(&mut lh[MAX_STREAMS + s], llm::lg_row(0));
        next[s] = argmax_row(0);
    }
    unsafe { (pf_tsc_all, pf_tsc_mm) = (0, 0) };
    let t0 = smp::rdtsc();
    for g in 0..t {
        let mut rows = [(0usize, 0u32, 0u64); MAX_STREAMS];
        for s in 0..n {
            ids[MAX_STREAMS + s][g] = next[s] as u32;
            rows[s] = (s, next[s] as u32, (np[s] + g) as u64);
        }
        step(&rows[..n], Some(0));
        for s in 0..n {
            next[s] = argmax_row(s);
            if g + 1 < t {
                fnv_f32(&mut lh[MAX_STREAMS + s], llm::lg_row(s));
            }
        }
    }
    let us2 = smp::tsc_to_us(smp::rdtsc() - t0);
    let (all, mm) = unsafe { (smp::tsc_to_us(pf_tsc_all) / 1000, smp::tsc_to_us(pf_tsc_mm) / 1000) };
    smp::park_phase(false);
    (1..n).for_each(close);
    let tot = (n * t) as u64;
    let rate = |k: u64, us: u64| (k * 100_000_000 / us.max(1)) as u64;
    let (a, b) = (rate(t as u64, us1 / n as u64), rate(tot, us2));
    print!("{}", crate::console::sgr(crate::console::DIM));
    println!("[multi] {} streams x {} tokens: alone {}.{:02} tok/s per stream; together {}.{:02} tok/s aggregate ({}.{:02}x), {}.{:02} tok/s per stream; passes {} ms, matmuls {} ms ({})",
        n, t, a / 100, a % 100, b / 100, b % 100, b * 100 / a.max(1) / 100, b * 100 / a.max(1) % 100, b / n as u64 / 100, b / n as u64 % 100,
        all, mm, if unsafe { bmm_vnni } != 0 { "vnni" } else { "avx2" });
    for s in 0..n {
        let same = ids[s][..t] == ids[MAX_STREAMS + s][..t];
        let mut h = FNV0;
        ids[s][..t].iter().for_each(|&x| h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
        println!("[multi] stream {}: prompt {} tok, ids {} (fnv {:#x}), logits {}", s, np[s], if same { "same" } else { "DIFFER" }, h,
            if lh[s] == lh[MAX_STREAMS + s] { "same" } else { "DIFFER" });
        if let Some(g) = (0..t).find(|&g| ids[s][g] != ids[MAX_STREAMS + s][g]) {
            println!("[multi] stream {}: first different id at token {}: alone {}, together {}", s, g, ids[s][g], ids[MAX_STREAMS + s][g]);
        }
    }
    print!("{}", crate::console::sgr(crate::console::RESET));
}
