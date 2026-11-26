// Generation (mirror of kernel.asm run_inference / pick_next_token), the golden boot self-test, the
// bench, and the agent's building blocks: tokenizer with special tokens, prefill, streamed decode with
// stop tokens and Ctrl-C, and the prefix snapshot (KV cache rows + DeltaNet state).
use crate::engine::*;
use crate::smp;
use crate::{print, println};

// engine/constants.inc (QWEN35)
const CTX_ARENA_SIZE: usize = 24;
const CTX_MAX_SEQ: usize = 160;
const CTX_EOS_1: usize = 184;
const CTX_EOS_2: usize = 192;
const CTX_BUF_LOGITS: usize = 280;
const CTX_BUF_PROMPT: usize = 344;
const CTX_BUF_GEN: usize = 352;
const CTX_BUF_PROBS: usize = 360;
const CTX_SAMPLE_N: usize = 488;
const CTX_TOK_IM_START: usize = 512;
const CTX_TOK_IM_END: usize = 520;
const CTX_TOK_THINK_START: usize = 528;
const CTX_TOK_THINK_END: usize = 536;
const CTX_TOK_NL: usize = 560;
const MAX_PROMPT_TOKENS: u64 = 8192;
const MAX_GEN_TOKENS: u64 = 8192;
pub const CTX_LEN: u64 = 32768; // M2.5 (M1-M3: 2048, kernel.asm opt_ctx_len default)

// docs/qwen35-2b/golden_2b.json
const GOLDEN_PROMPT: &[u8] = b"Say hi in 5 words.\0";
const GOLDEN_PROMPT_IDS: [u32; 19] =
    [248045, 846, 198, 44240, 15131, 303, 220, 20, 4105, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271];
const GOLDEN_GEN_IDS: [u32; 3] = [9419, 1017, 0];
const BENCH_PROMPT: &[u8] = b"Explain how a CPU cache works in 3 concise sentences.\0";

pub static mut THINK: u64 = 0;
pub static mut TEMP_BITS: u32 = 0; // 0.0f = greedy argmax
static mut RNG: u64 = 42;
const TOPP_BITS: u32 = 0x3F73_3333; // 0.95f

struct Opts {
    max_tok: u64,
    print_toks: bool,
    /// /gen debug: print prompt size + FNV, generated ids and the top1 - top2 logit margin per token
    margins: bool,
}

/// top1 - top2 logit of the current logits (ordered-int compare, no float compares in the scan).
fn margin() -> f32 {
    let l = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_LOGITS) as *const u32, ctx(CTX_SAMPLE_N) as usize) };
    let key = |b: u32| if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 };
    let (mut b1, mut b2) = (0u32, 0u32); // raw bits of the best two
    let (mut k1, mut k2) = (0u32, 0u32);
    for &b in l {
        let k = key(b);
        if k > k1 {
            (k2, b2, k1, b1) = (k1, b1, k, b);
        } else if k > k2 {
            (k2, b2) = (k, b);
        }
    }
    f32::from_bits(b1) - f32::from_bits(b2)
}

static mut MARGINS: [f32; 512] = [0.0; 512];

pub struct Stats {
    pub n_prompt: u64,
    pub n_gen: u64,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub pos: u64,
}

fn rate(n: u64, us: u64) -> (u64, u64) {
    let h = n * 100_000_000 / us.max(1); // tok/s * 100
    (h / 100, h % 100)
}

fn pick_next() -> u64 {
    let logits = ctx(CTX_BUF_LOGITS) as *const f32;
    let n = ctx(CTX_SAMPLE_N);
    unsafe {
        if TEMP_BITS == 0 {
            argmax_f32(logits, n)
        } else {
            sample_top_p_bits(logits, ctx(CTX_BUF_PROBS) as *mut f32, n, core::ptr::addr_of_mut!(RNG), TEMP_BITS, TOPP_BITS)
        }
    }
}

fn print_csv(label: &str, t: &[u32]) {
    print!("{}", label);
    for (i, x) in t.iter().enumerate() {
        print!("{}{}", if i > 0 { "," } else { "" }, x);
    }
    println!("");
}

/// prompt must be NUL-terminated.
fn run(prompt: &[u8], o: &Opts) -> Stats {
    let max_seq = ctx(CTX_MAX_SEQ);
    let ptoks = ctx(CTX_BUF_PROMPT) as *mut u32;
    let gtoks = ctx(CTX_BUF_GEN) as *mut u32;
    let mut n = unsafe { tok_build_chat_prompt(prompt.as_ptr(), ptoks, MAX_PROMPT_TOKENS, 0, THINK) };
    let mut pos = 0;
    n = n.min(max_seq);
    if o.print_toks {
        print_csv("PROMPT_TOKENS ", unsafe { core::slice::from_raw_parts(ptoks, n as usize) });
    }
    smp::park_phase(true);

    let t0 = smp::rdtsc();
    for i in 0..n {
        unsafe { transformer_forward(*ptoks.add(i as usize), pos, (i + 1 == n) as i32) };
        pos += 1;
    }
    let t1 = smp::rdtsc();
    let mut next = pick_next();

    let t2 = smp::rdtsc();
    OUT_TSC.store(0, core::sync::atomic::Ordering::Relaxed);
    let (eos1, eos2) = (ctx(CTX_EOS_1), ctx(CTX_EOS_2));
    let max_tok = o.max_tok.min(MAX_GEN_TOKENS);
    let mut n_gen = 0u64;
    while n_gen < max_tok && next != eos1 && next != eos2 {
        if o.margins && n_gen < 512 {
            unsafe { MARGINS[n_gen as usize] = margin() };
        }
        unsafe { *gtoks.add(n_gen as usize) = next as u32 };
        n_gen += 1;
        unsafe { tok_write_stdout(next) };
        if n_gen >= max_tok {
            break;
        }
        if pos >= max_seq {
            break;
        }
        unsafe { transformer_forward(next as u32, pos, 1) };
        pos += 1;
        next = pick_next();
    }
    let t3 = smp::rdtsc();
    println!("");

    smp::park_phase(false);

    let s = Stats {
        n_prompt: n,
        n_gen,
        prefill_us: smp::tsc_to_us(t1 - t0),
        decode_us: smp::tsc_to_us(t3 - t2),
        pos,
    };
    if o.print_toks || o.margins {
        print_csv("GEN_TOKENS ", unsafe { core::slice::from_raw_parts(gtoks, n_gen as usize) });
    }
    if o.margins {
        let p = unsafe { core::slice::from_raw_parts(ptoks, n as usize) };
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        p.iter().for_each(|&x| h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
        println!("PROMPT {} tokens fnv {:#x}, KV {}", n, h, kv_type());
        print!("MARGINS ");
        for i in 0..(n_gen as usize).min(512) {
            print!("{}{:.3}", if i > 0 { "," } else { "" }, unsafe { MARGINS[i] });
        }
        println!("");
    }
    print_speed(&s);
    s
}

pub fn print_speed(s: &Stats) {
    let (pa, pb) = rate(s.n_prompt, s.prefill_us);
    let (da, db) = rate(s.n_gen.saturating_sub(1).max(1), s.decode_us);
    println!(
        "[prefill] {} tok in {}.{:02} ms ({}.{:02} tok/s)  [decode] {} tok in {}.{:02} ms ({}.{:02} tok/s, output {} ms)  cores {}  ctx {}/{} ({}%)",
        s.n_prompt, s.prefill_us / 1000, s.prefill_us % 1000 / 10, pa, pb,
        s.n_gen, s.decode_us / 1000, s.decode_us % 1000 / 10, da, db,
        smp::tsc_to_us(OUT_TSC.load(core::sync::atomic::Ordering::Relaxed)) / 1000,
        smp::threads(), s.pos, ctx(CTX_MAX_SEQ), s.pos * 100 / ctx(CTX_MAX_SEQ).max(1)
    );
}

/// KV cache element type of this build.
pub fn kv_type() -> &'static str {
    if unsafe { kv_elt_bytes } == 2 { "fp16" } else { "fp32" }
}

pub fn ctx_max_seq() -> u64 {
    ctx(CTX_MAX_SEQ)
}

pub fn init(base: *const u8, size: u64) -> bool {
    let t0 = smp::rdtsc();
    let r = unsafe { model_init(base, size, CTX_LEN) };
    if r < 0 {
        crate::klog!(b'E', "model: model_init failed ({})", r);
        return false;
    }
    crate::klog!(b'I',
        "model: model_init ok in {} ms, max_seq {}, KV cache {}, arena {} MiB, vocab sample_n {}",
        smp::tsc_to_us(smp::rdtsc() - t0) / 1000,
        ctx(CTX_MAX_SEQ),
        kv_type(),
        ctx(CTX_ARENA_SIZE) >> 20,
        ctx(CTX_SAMPLE_N)
    );
    true
}

/// Golden boot self-test (greedy, single turn). Prints SELFTEST PASS/FAIL.
pub fn selftest() -> bool {
    crate::klog!(b'I', "selftest: prompt \"Say hi in 5 words.\" (golden_2b.json), greedy, {} threads", smp::threads());
    let s = run(GOLDEN_PROMPT, &Opts { max_tok: 16, print_toks: true, margins: false });
    let p = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_PROMPT) as *const u32, s.n_prompt as usize) };
    let g = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_GEN) as *const u32, s.n_gen as usize) };
    let ok = p == GOLDEN_PROMPT_IDS && g == GOLDEN_GEN_IDS;
    crate::klog!(b'I',
        "SELFTEST {} (prompt ids {}, generated ids {} vs golden 9419,1017,0 + <|im_end|>)",
        if ok { "PASS" } else { "FAIL" },
        if p == GOLDEN_PROMPT_IDS { "match" } else { "DIFFER" },
        if g == GOLDEN_GEN_IDS { "match" } else { "DIFFER" }
    );
    ok
}

/// 64-token decode bench (single turn from pos 0: overwrites the agent prefix, restore it afterwards).
pub fn bench() -> Stats {
    run(BENCH_PROMPT, &Opts { max_tok: 64, print_toks: false, margins: false })
}

/// `/gen N text`: greedy single turn from position 0 (engine chat template), N tokens at most, prints the
/// generated ids and per-token logit margins (to compare the FP16 and FP32 KV builds). Overwrites the prefix.
pub fn gen_debug(max_tok: u64, text: &str) {
    static mut P: [u8; 16384] = [0; 16384];
    let p = unsafe { &mut *core::ptr::addr_of_mut!(P) };
    let k = text.len().min(p.len() - 1);
    p[..k].copy_from_slice(&text.as_bytes()[..k]);
    p[k] = 0;
    let temp = unsafe { core::ptr::addr_of_mut!(TEMP_BITS).replace(0) };
    run(&p[..=k], &Opts { max_tok: max_tok.clamp(1, 512), print_toks: false, margins: true });
    unsafe { TEMP_BITS = temp };
}

const FILL_TEXT: &[u8] = b"The history of operating systems starts with machines that ran one program at a time, loaded by hand from \
punched cards. Batch monitors came next, then time sharing, which let many users share one computer through terminals. \
Virtual memory, file systems and processes became the common vocabulary, and small kernels grew into large ones. \
Personal computers brought simple systems back, until networking and multitasking made them as complex as the old ones. ";

/// `/fill N`: prefill an N-token user turn from position 0 (a repeated paragraph), then decode 64 tokens with
/// EOS ignored, to measure prefill and decode speed at a given context fill. Overwrites the agent prefix.
pub fn bench_fill(n: u64) {
    static mut T: [u32; 32768] = [0; 32768];
    let t = unsafe { &mut *core::ptr::addr_of_mut!(T) };
    let max_seq = ctx(CTX_MAX_SEQ);
    let n = (n as usize).clamp(64, (max_seq - 80) as usize);
    let mut tail = [0u32; 32];
    let nt = encode(b"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", &mut tail);
    let mut k = encode(b"<|im_start|>user\nRead this text, then continue it.\n\n", t);
    let np = encode(FILL_TEXT, &mut t[k..]);
    let p0 = k;
    k += np;
    while k < n - nt {
        t[k] = t[p0 + (k - p0) % np];
        k += 1;
    }
    t[k..k + nt].copy_from_slice(&tail[..nt]);
    let n = k + nt;
    println!("[fill] {} tokens (KV {}), progress every 4096", n, kv_type());
    smp::park_phase(true);
    let t0 = smp::rdtsc();
    for i in 0..n {
        if i % 64 == 0 && crate::shell::poll_intr() {
            println!("^C [fill] interrupted at {}", i);
            smp::park_phase(false);
            return;
        }
        if i > 0 && i % 4096 == 0 {
            let (a, b) = rate(i as u64, smp::tsc_to_us(smp::rdtsc() - t0));
            println!("[fill] {}/{} tok, {}.{:02} tok/s so far", i, n, a, b);
        }
        unsafe { transformer_forward(t[i], i as u64, (i + 1 == n) as i32) };
    }
    let t1 = smp::rdtsc();
    OUT_TSC.store(0, core::sync::atomic::Ordering::Relaxed);
    let mut pos = n as u64;
    let mut next = pick_next();
    for g in 0..65 {
        unsafe { tok_write_stdout(next) };
        if g == 64 {
            break;
        }
        unsafe { transformer_forward(next as u32, pos, 1) };
        pos += 1;
        next = pick_next();
    }
    let t2 = smp::rdtsc();
    println!("");
    smp::park_phase(false);
    print_speed(&Stats { n_prompt: n as u64, n_gen: 65, prefill_us: smp::tsc_to_us(t1 - t0), decode_us: smp::tsc_to_us(t2 - t1), pos });
}


// ---- tokenizer: special tokens + the Qwen pre-tokenizer split, then the engine's BPE per chunk ----

pub const TOK_TOOL_CALL: u64 = 248058;
pub const TOK_TOOL_CALL_END: u64 = 248059;

fn special_at(s: &[u8]) -> Option<(usize, u32)> {
    let table: [(&[u8], u64); 8] = [
        (b"<|im_start|>", ctx(CTX_TOK_IM_START)),
        (b"<|im_end|>", ctx(CTX_TOK_IM_END)),
        (b"<think>", ctx(CTX_TOK_THINK_START)),
        (b"</think>", ctx(CTX_TOK_THINK_END)),
        (b"<tool_call>", TOK_TOOL_CALL),
        (b"</tool_call>", TOK_TOOL_CALL_END),
        (b"<tool_response>", 248066),
        (b"</tool_response>", 248067),
    ];
    table.iter().find(|(t, _)| s.starts_with(t)).map(|(t, id)| (t.len(), *id as u32))
}

// shortcut: every non-ASCII byte counts as a letter (fine for Latin scripts), so \p{L}/\p{N} on
// other scripts and Unicode punctuation can split differently from the HF tokenizer.
fn letter(b: u8) -> bool {
    b.is_ascii_alphabetic() || b >= 0x80
}
fn space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | 0x0B | 0x0C)
}
fn punct(b: u8) -> bool {
    !letter(b) && !b.is_ascii_digit() && !space(b)
}

/// End of the pre-token starting at i (the Qwen2/3 split regex, alternatives in order).
fn chunk_end(s: &[u8], i: usize) -> usize {
    let n = s.len();
    let c = s[i];
    let lc = |k: usize| s.get(k).map(|b| b.to_ascii_lowercase());
    if c == b'\'' {
        match (lc(i + 1), lc(i + 2)) {
            (Some(b's' | b't' | b'm' | b'd'), _) => return i + 2,
            (Some(b'r'), Some(b'e')) | (Some(b'v'), Some(b'e')) | (Some(b'l'), Some(b'l')) => return i + 3,
            _ => {}
        }
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    let mut j = i;
    if !letter(c) && !c.is_ascii_digit() && c != b'\r' && c != b'\n' && i + 1 < n && letter(s[i + 1]) {
        j = i + 1;
    }
    if letter(s[j]) {
        while j < n && letter(s[j]) {
            j += 1;
        }
        return j;
    }
    if c.is_ascii_digit() {
        return i + 1;
    }
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    let mut j = i + (c == b' ') as usize;
    if j < n && punct(s[j]) {
        while j < n && punct(s[j]) {
            j += 1;
        }
        while j < n && (s[j] == b'\r' || s[j] == b'\n') {
            j += 1;
        }
        return j;
    }
    // whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
    let mut j = i;
    let mut last_nl = None;
    while j < n && space(s[j]) {
        if s[j] == b'\n' || s[j] == b'\r' {
            last_nl = Some(j);
        }
        j += 1;
    }
    if let Some(k) = last_nl {
        return k + 1;
    }
    if j == n || j - i == 1 { j } else { j - 1 }
}

/// Tokenize text with special tokens into out. Returns the token count.
pub fn encode(text: &[u8], out: &mut [u32]) -> usize {
    fn plain(seg: &[u8], out: &mut [u32], n: &mut usize) {
        let mut i = 0;
        while i < seg.len() {
            let j = chunk_end(seg, i);
            let room = (out.len() - *n) as u64;
            *n += unsafe { tok_encode_bpe(seg[i..].as_ptr(), (j - i) as u64, out[*n..].as_mut_ptr(), room) } as usize;
            i = j;
        }
    }
    let (mut n, mut i, mut seg) = (0, 0, 0);
    while i < text.len() {
        if let Some((len, id)) = if text[i] == b'<' { special_at(&text[i..]) } else { None } {
            plain(&text[seg..i], out, &mut n);
            if n < out.len() {
                out[n] = id;
                n += 1;
            }
            i += len;
            seg = i;
        } else {
            i += 1;
        }
    }
    plain(&text[seg..], out, &mut n);
    n
}

// ---- prefill and streamed decode for agent turns ----

/// Feed tokens at *pos. Logits only for the last one. false = interrupted by Ctrl-C (state is partial).
pub fn feed(toks: &[u32], pos: &mut u64) -> bool {
    for (i, &t) in toks.iter().enumerate() {
        if i % 4 == 0 && crate::shell::poll_intr() {
            return false;
        }
        unsafe { transformer_forward(t, *pos, (i + 1 == toks.len()) as i32) };
        *pos += 1;
    }
    true
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Stop {
    Eos,
    ToolEnd,
    MaxTok,
    Ctx,
    Intr,
}

/// Decode after a feed(): stream tokens (hidden from <tool_call> on), stop at EOS, </tool_call>, max_tok,
/// a full context or Ctrl-C, then close the assistant turn in the KV cache (<|im_end|>\n) so the next
/// turn lines up. Returns the stop reason and the number of generated tokens.
pub fn generate(pos: &mut u64, max_tok: u64) -> (Stop, u64) {
    let max_seq = ctx(CTX_MAX_SEQ);
    let (eos1, eos2) = (ctx(CTX_EOS_1), ctx(CTX_EOS_2));
    let mut next = pick_next();
    let mut n = 0;
    let stop = loop {
        if next == eos1 || next == eos2 {
            break Stop::Eos;
        }
        if crate::shell::poll_intr() {
            break Stop::Intr;
        }
        if next == TOK_TOOL_CALL {
            unsafe { ECHO = false };
        }
        unsafe { tok_write_stdout(next) };
        n += 1;
        if next == TOK_TOOL_CALL_END {
            break Stop::ToolEnd;
        }
        if n >= max_tok {
            break Stop::MaxTok;
        }
        if *pos + 3 >= max_seq {
            break Stop::Ctx;
        }
        unsafe { transformer_forward(next as u32, *pos, 1) };
        *pos += 1;
        next = pick_next();
    };
    unsafe {
        ECHO = true;
        if matches!(stop, Stop::ToolEnd | Stop::MaxTok | Stop::Ctx) {
            transformer_forward(next as u32, *pos, 0); // emitted but not fed yet
            *pos += 1;
        }
        if *pos + 2 <= max_seq {
            transformer_forward(ctx(CTX_TOK_IM_END) as u32, *pos, 0);
            transformer_forward(ctx(CTX_TOK_NL) as u32, *pos + 1, 0);
            *pos += 2;
        }
    }
    (stop, n)
}

// ---- prefix snapshot: KV rows 0..pos of every full-attention layer + the DeltaNet S and conv state ----

const CTX_N_KV_HEADS: usize = 120;
const CTX_HEAD_DIM: usize = 128;
const CTX_BUF_K_CACHE: usize = 328;
const CTX_BUF_V_CACHE: usize = 336;
const CTX_BUF_DN_STATE: usize = 384;
const CTX_N_FULL: usize = 416;
const CTX_DN_ZERO_BYTES: usize = 496;

static mut SNAP_BUF: *mut u8 = core::ptr::null_mut();
static mut SNAP_POS: u64 = 0;

/// Calls f(cache_ptr, bytes) for every contiguous KV run of positions 0..pos (head-major layout
/// cache[l][kv_h][pos][head_dim] of FP16 or FP32 elements, transformer.asm FAST_Q8), K then V, then
/// the DeltaNet block.
fn snap_regions(pos: u64, mut f: impl FnMut(*mut u8, usize)) {
    let hb = ctx(CTX_HEAD_DIM) * unsafe { kv_elt_bytes };
    let stride = ctx(CTX_MAX_SEQ) * hb;
    let heads = ctx(CTX_N_FULL) * ctx(CTX_N_KV_HEADS);
    for cache in [ctx(CTX_BUF_K_CACHE), ctx(CTX_BUF_V_CACHE)] {
        for h in 0..heads {
            f((cache + h * stride) as *mut u8, (pos * hb) as usize);
        }
    }
    f(ctx(CTX_BUF_DN_STATE) as *mut u8, ctx(CTX_DN_ZERO_BYTES) as usize);
}

/// Save the engine state after a prefill of `pos` tokens. Returns the snapshot size in bytes.
pub fn snapshot_save(pos: u64) -> u64 {
    let mut size = 0;
    snap_regions(pos, |_, n| size += n as u64);
    unsafe {
        if SNAP_BUF.is_null() {
            SNAP_BUF = crate::mem::sys_mmap_anon(size);
            assert!((SNAP_BUF as i64) > 0, "snapshot: arena full");
        }
        let mut o = 0;
        snap_regions(pos, |p, n| {
            core::ptr::copy_nonoverlapping(p, SNAP_BUF.add(o), n);
            o += n;
        });
        SNAP_POS = pos;
    }
    size
}

pub fn snapshot_pos() -> u64 {
    unsafe { SNAP_POS }
}

/// Restore the saved prefix state. Returns the position to continue from.
pub fn snapshot_restore() -> u64 {
    unsafe {
        let pos = SNAP_POS;
        let mut o = 0;
        snap_regions(pos, |p, n| {
            core::ptr::copy_nonoverlapping(SNAP_BUF.add(o), p, n);
            o += n;
        });
        pos
    }
}
