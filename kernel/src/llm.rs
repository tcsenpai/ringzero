// Generation (mirror of kernel.asm run_inference / pick_next_token), the golden boot self-test, the
// bench, and the agent's building blocks: tokenizer with special tokens, prefill, streamed decode with
// stop tokens and Ctrl-C, and the live token array the snapshots store (snap.rs).
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
const CTX_VOCAB_SIZE: usize = 152;
const CTX_TOK_IM_START: usize = 512;
const CTX_TOK_IM_END: usize = 520;
const CTX_TOK_THINK_START: usize = 528;
const CTX_TOK_THINK_END: usize = 536;
const CTX_TOK_NL: usize = 560;
const CTX_N_KV_HEADS: usize = 120;
const CTX_HEAD_DIM: usize = 128;
const CTX_BUF_K_CACHE: usize = 328;
const CTX_BUF_V_CACHE: usize = 336;
const CTX_BUF_DN_STATE: usize = 384;
const CTX_N_FULL: usize = 416;
const CTX_DN_ZERO_BYTES: usize = 496;
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
pub static mut RNG: u64 = 42;
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

/// Next token from the logits of the last forward call (greedy, or top-p at TEMP_BITS).
pub fn pick_next() -> u64 {
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

// ---- M4.8 batched prefill ----

/// Prompt tokens per engine pass (engine/prefill.asm); 1 = one transformer_forward per token (the old path).
static mut PREFILL_B: u64 = 1;
pub const PREFILL_MAX_B: u64 = 64;
const PREFILL_DEFAULT: u64 = 32;
static mut PREFILL_OK: bool = false;
static mut VNNI_CPU: u32 = 0; // what prefill_init found

pub fn prefill_block() -> u64 {
    unsafe { PREFILL_B }
}

/// Set the prefill block (clamped to 1..=64; 1 when the batched buffers are missing). Returns the value set.
pub fn set_prefill_block(b: u64) -> u64 {
    unsafe {
        PREFILL_B = if PREFILL_OK { b.clamp(1, PREFILL_MAX_B) } else { 1 };
        PREFILL_B
    }
}

/// Forward prompt tokens at pos0.. in blocks of PREFILL_B (logits for the last token when `logits`).
/// Every block size leaves the same state, bit for bit (prefill.asm header).
pub fn forward(t: &[u32], pos0: u64, logits: bool) {
    let b = prefill_block().max(1) as usize;
    let mut p = pos0;
    for (k, c) in t.chunks(b).enumerate() {
        let last = (logits && k * b + c.len() == t.len()) as i32;
        unsafe {
            if c.len() == 1 {
                transformer_forward(c[0], p, last);
            } else {
                transformer_prefill(c.as_ptr(), c.len() as u64, p, last);
            }
        }
        p += c.len() as u64;
    }
}

/// FNV-1a (64-bit words) over the sequence state: K and V rows of positions < pos in every full-attention
/// layer, the DeltaNet recurrent state and conv history; and over the logits. The exactness check.
pub fn state_hash(pos: u64) -> (u64, u64) {
    let fnv = |h: &mut u64, p: u64, bytes: u64| {
        let w = unsafe { core::slice::from_raw_parts(p as *const u64, (bytes / 8) as usize) };
        w.iter().for_each(|&x| *h = (*h ^ x).wrapping_mul(0x100_0000_01b3));
    };
    let e = unsafe { kv_elt_bytes };
    let (rows, row) = (ctx(CTX_N_FULL) * ctx(CTX_N_KV_HEADS), ctx(CTX_HEAD_DIM) * e); // [full][kv_h][max_seq][hd]
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for base in [ctx(CTX_BUF_K_CACHE), ctx(CTX_BUF_V_CACHE)] {
        for r in 0..rows {
            fnv(&mut h, base + r * ctx(CTX_MAX_SEQ) * row, pos * row);
        }
    }
    fnv(&mut h, ctx(CTX_BUF_DN_STATE), ctx(CTX_DN_ZERO_BYTES));
    let mut hl = 0xcbf2_9ce4_8422_2325u64;
    fnv(&mut hl, ctx(CTX_BUF_LOGITS), ctx(CTX_SAMPLE_N) * 4 & !7);
    (h, hl)
}

fn print_state(pos: u64) {
    let (h, hl) = state_hash(pos);
    println!("STATE pos {} prefill block {} kv+deltanet fnv {:#018x} logits fnv {:#018x}", pos, prefill_block(), h, hl);
}

/// `/prefill [B] [avx2|vnni]`: set the prompt block (1 = per token) and the batched matmul kernel, then
/// prefill the given tokens (the boot prefix) from position 0 and print the time and the state hash.
/// Overwrites the live state.
pub fn prefill_bench(arg: &str, toks: &[u32]) {
    for a in arg.split_whitespace() {
        match a {
            "avx2" => unsafe { bmm_vnni = 0 },
            "vnni" => unsafe { bmm_vnni = VNNI_CPU },
            _ => {
                if let Ok(b) = a.parse::<u64>() {
                    set_prefill_block(b);
                }
            }
        }
    }
    smp::park_phase(true);
    unsafe { (pf_tsc_all, pf_tsc_mm) = (0, 0) };
    let t0 = smp::rdtsc();
    forward(toks, 0, true);
    let us = smp::tsc_to_us(smp::rdtsc() - t0);
    smp::park_phase(false);
    let (a, f) = rate(toks.len() as u64, us);
    let (all, mm) = unsafe { (smp::tsc_to_us(pf_tsc_all) / 1000, smp::tsc_to_us(pf_tsc_mm) / 1000) };
    println!("[prefill] block {} ({}): {} tok in {}.{:03} s ({}.{:02} tok/s), next token {}, batched passes {} ms (matmuls {} ms)",
        prefill_block(), if unsafe { bmm_vnni } != 0 { "vnni" } else { "avx2" }, toks.len(), us / 1_000_000, us / 1000 % 1000, a, f, pick_next(), all, mm);
    print_state(toks.len() as u64);
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
    forward(unsafe { core::slice::from_raw_parts(ptoks, n as usize) }, 0, true);
    pos += n;
    let t1 = smp::rdtsc();
    if o.margins {
        print_state(pos);
    }
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
    print!(
        "[prefill] {} tok in {}.{:02} ms ({}.{:02} tok/s)  [decode] {} tok in {}.{:02} ms ({}.{:02} tok/s, output {} ms)  cores {}  ctx {}/{} ({}%)",
        s.n_prompt, s.prefill_us / 1000, s.prefill_us % 1000 / 10, pa, pb,
        s.n_gen, s.decode_us / 1000, s.decode_us % 1000 / 10, da, db,
        smp::tsc_to_us(OUT_TSC.load(core::sync::atomic::Ordering::Relaxed)) / 1000,
        smp::threads(), s.pos, ctx(CTX_MAX_SEQ), s.pos * 100 / ctx(CTX_MAX_SEQ).max(1)
    );
    // M4.8b: the speculative counters of the generate() call this line reports (taken: printed once)
    let x = unsafe { core::ptr::addr_of_mut!(SPEC_ST).replace(SPEC0) };
    if x.used && x.on {
        println!("  spec {}/{} accepted ({}%) in {} verify, {} rollback, verify {} ms, grammar {}/{}, ids {:#x}", x.accepted, x.drafted,
            x.accepted as u64 * 100 / (x.drafted as u64).max(1), x.verifies, x.rollbacks, x.verify_us / 1000, x.gaccepted, x.gdrafted, x.ids);
    } else if x.used {
        println!("  spec off, ids {:#x}", x.ids);
    } else {
        println!("");
    }
}

/// KV cache element type of this build.
pub fn kv_type() -> &'static str {
    if unsafe { kv_elt_bytes } == 2 { "fp16" } else { "fp32" }
}

/// End of turn / end of text.
pub fn is_eos(t: u64) -> bool {
    t == ctx(CTX_EOS_1) || t == ctx(CTX_EOS_2)
}

/// Sample hash of the model file (mstore::sample_hash), part of the boot snapshot cache key.
pub static mut MODEL_HASH: u64 = 0;

pub fn ctx_max_seq() -> u64 {
    ctx(CTX_MAX_SEQ)
}

pub fn init(base: *const u8, size: u64) -> bool {
    let t0 = smp::rdtsc();
    unsafe { MODEL_HASH = crate::mstore::sample_hash(base, size) };
    let r = unsafe { model_init(base, size, CTX_LEN) };
    if r < 0 {
        crate::klog!(b'E', "model: model_init failed ({})", r);
        return false;
    }
    // batched prefill buffers now, before the snapshot pool takes the rest of the arena
    let pr = unsafe { prefill_init(PREFILL_MAX_B) };
    unsafe { (PREFILL_OK, VNNI_CPU) = (pr == 0, bmm_vnni) };
    set_prefill_block(PREFILL_DEFAULT);
    if pr != 0 {
        crate::klog!(b'W', "model: batched prefill off ({}), one token per pass", pr);
    }
    spec_init();
    crate::klog!(b'I',
        "model: model_init ok in {} ms, max_seq {}, KV cache {}, arena {} MiB, vocab sample_n {}, prefill block {}",
        smp::tsc_to_us(smp::rdtsc() - t0) / 1000,
        ctx(CTX_MAX_SEQ),
        kv_type(),
        ctx(CTX_ARENA_SIZE) >> 20,
        ctx(CTX_SAMPLE_N),
        prefill_block()
    );
    crate::klog!(b'I', "model: batched prefill matmul {}", if unsafe { bmm_vnni } != 0 { "AVX-VNNI" } else { "AVX2" });
    true
}

/// Golden boot self-test (greedy, single turn). Prints SELFTEST PASS/FAIL.
pub fn selftest() -> bool {
    crate::klog!(b'I', "selftest: prompt \"Say hi in 5 words.\" (golden_2b.json), greedy, {} threads", smp::threads());
    let s = run(GOLDEN_PROMPT, &Opts { max_tok: 16, print_toks: true, margins: false });
    let p = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_PROMPT) as *const u32, s.n_prompt as usize) };
    let g = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_GEN) as *const u32, s.n_gen as usize) };
    let ok = p == GOLDEN_PROMPT_IDS && g == GOLDEN_GEN_IDS;
    // M7: golden_2b.json is the 2B Q8_0 (.asmbin v3, dim 2048) reference; other models print their ids only
    if unsafe { *((ctx(0) + 4) as *const u32) } != 3 || ctx(88) != 2048 {
        crate::klog!(b'I', "SELFTEST n/a (golden is the 2B Q8_0 reference, not this model; ids above)");
        return true;
    }
    crate::klog!(b'I',
        "SELFTEST {} (prompt ids {}, generated ids {} vs golden 9419,1017,0 + <|im_end|>)",
        if ok { "PASS" } else { "FAIL" },
        if p == GOLDEN_PROMPT_IDS { "match" } else { "DIFFER" },
        if g == GOLDEN_GEN_IDS { "match" } else { "DIFFER" }
    );
    ok
}

/// 64-token decode bench (single turn from pos 0: overwrites the agent context, snap.rs restores it).
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
    let mut i = 0;
    while i < n {
        if crate::shell::poll_intr() {
            println!("^C [fill] interrupted at {}", i);
            smp::park_phase(false);
            return;
        }
        if i > 0 && i % 4096 == 0 {
            let (a, b) = rate(i as u64, smp::tsc_to_us(smp::rdtsc() - t0));
            println!("[fill] {}/{} tok, {}.{:02} tok/s so far", i, n, a, b);
        }
        let e = (i + 256).min(n);
        forward(&t[i..e], i as u64, e == n);
        i = e;
    }
    let t1 = smp::rdtsc();
    print_state(n as u64);
    let t1b = smp::rdtsc();
    OUT_TSC.store(0, core::sync::atomic::Ordering::Relaxed);
    let mut pos = n as u64;
    let mut next = pick_next();
    let mut ids = [0u32; 65];
    for g in 0..65 {
        ids[g] = next as u32;
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
    print_csv("GEN_TOKENS ", &ids);
    print_speed(&Stats { n_prompt: n as u64, n_gen: 65, prefill_us: smp::tsc_to_us(t1 - t0), decode_us: smp::tsc_to_us(t2 - t1b), pos });
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

fn plain(seg: &[u8], out: &mut [u32], n: &mut usize) {
    let mut i = 0;
    while i < seg.len() {
        let j = chunk_end(seg, i);
        let room = (out.len() - *n) as u64;
        *n += unsafe { tok_encode_bpe(seg[i..].as_ptr(), (j - i) as u64, out[*n..].as_mut_ptr(), room) } as usize;
        i = j;
    }
}

/// Tokenize text without special tokens (pre-split + the loaded model's BPE): the embedder's input (embed.rs).
pub fn encode_plain(text: &[u8], out: &mut [u32]) -> usize {
    let mut n = 0;
    plain(text, out, &mut n);
    n
}

/// Tokenize text with special tokens into out. Returns the token count.
pub fn encode(text: &[u8], out: &mut [u32]) -> usize {
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

/// Token id at every live position of the agent context (snapshots store and restore it with the KV
/// rows, a thinned snapshot is rebuilt by replaying it).
static mut REC: [u32; CTX_LEN as usize] = [0; CTX_LEN as usize];

pub fn rec() -> &'static mut [u32; CTX_LEN as usize] {
    unsafe { &mut *core::ptr::addr_of_mut!(REC) }
}

pub fn set_rng(r: u64) {
    unsafe { RNG = r };
}

/// One forward pass of the agent context, recorded in REC.
pub fn fwd(t: u32, pos: u64, logits: i32) {
    if let Some(r) = rec().get_mut(pos as usize) {
        *r = t;
    }
    unsafe { transformer_forward(t, pos, logits) };
}

/// fwd() for a block of prompt tokens: recorded in REC, then forward() (batched).
pub fn fwd_block(t: &[u32], pos: u64, logits: bool) {
    for (j, &x) in t.iter().enumerate() {
        if let Some(r) = rec().get_mut(pos as usize + j) {
            *r = x;
        }
    }
    forward(t, pos, logits);
}

/// Feed tokens at *pos. Logits only for the last one. false = interrupted by Ctrl-C (state is partial).
pub fn feed(toks: &[u32], pos: &mut u64) -> bool {
    let step = prefill_block().max(4) as usize; // Ctrl-C checked every 4 tokens or every block
    let mut i = 0;
    while i < toks.len() {
        if crate::shell::poll_intr() {
            return false;
        }
        let e = (i + step).min(toks.len());
        fwd_block(&toks[i..e], *pos, e == toks.len());
        *pos += (e - i) as u64;
        i = e;
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

// ---- M4.8b: speculative decoding by prompt lookup ----
//
// Draft: the earlier place in the context (REC + the emitted tokens not fed yet) that agrees longest (3 to 32 tokens)
// with the end of the context now; the tokens that followed it are the draft, at most match length - 1. Verify: the pending tokens plus
// the draft in one transformer_prefill pass with logits for every row (compute_logits = 2, one batched LM-head
// matmul), then each row is picked exactly as the per-token loop picks (sampler, grammar fix, RNG order) and the
// draft is accepted while it matches. Every row's logits are bit-identical to a transformer_forward at that
// position (the batched matmuls are, `/spec` checks it), so the output is the same token for token, greedy or
// sampled. Rollback: the KV rows of rejected tokens are simply overwritten later (attention reads 0..pos); the
// DeltaNet recurrent state + conv history cannot be cut, so it is copied before the verify and copied back on a
// partial accept, and the accepted tokens are fed again at the front of the next pass.

/// Logits rows of one verify (draft + 1).
pub const SPEC_ROWS: usize = 16;
/// Emitted tokens not fed yet, and so the largest verify pass (<= PREFILL_MAX_B).
const TODO_MAX: usize = 48;
pub static mut SPEC_ON: bool = true;
/// Longest draft (1..SPEC_ROWS-1).
pub static mut SPEC_K: usize = 8;
/// T39: inside a tool call, draft the text the grammar implies (gram::draft) before the prompt lookup.
pub static mut GRAM_DRAFT: bool = true;
static mut SPEC_OK: bool = false;
static mut DN_BAK: *mut u8 = core::ptr::null_mut();

#[derive(Clone, Copy)]
struct SpecStats {
    used: bool,
    on: bool,
    verifies: u32,
    drafted: u32,
    accepted: u32,
    rollbacks: u32,
    verify_us: u64,
    gdrafted: u32, // T39: of drafted / accepted, the grammar's
    gaccepted: u32,
    ids: u64, // FNV-1a of the emitted ids (the equality check of spec on vs off)
}
const SPEC0: SpecStats = SpecStats { used: false, on: false, verifies: 0, drafted: 0, accepted: 0, rollbacks: 0, verify_us: 0, gdrafted: 0, gaccepted: 0, ids: 0 };
static mut SPEC_ST: SpecStats = SPEC0;
const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

fn st() -> &'static mut SpecStats {
    unsafe { &mut *core::ptr::addr_of_mut!(SPEC_ST) }
}

/// Boot, after prefill_init: SPEC_ROWS logits rows and a copy of the DeltaNet state, from the engine arena.
fn spec_init() {
    if !unsafe { PREFILL_OK } {
        return;
    }
    let stride = (ctx(CTX_VOCAB_SIZE) * 4 + 63) & !63;
    let lg = crate::mem::sys_mmap_anon(stride * SPEC_ROWS as u64);
    let bak = crate::mem::sys_mmap_anon(ctx(CTX_DN_ZERO_BYTES));
    if (lg as i64) <= 0 || (bak as i64) <= 0 {
        crate::klog!(b'W', "model: speculative decoding off (arena full)");
        return;
    }
    unsafe { (pf_lg_ptr, pf_lg_stride, DN_BAK, SPEC_OK) = (lg as u64, stride, bak, true) };
    crate::klog!(b'I', "model: speculative decoding (prompt lookup): {} logits rows {} KiB, DeltaNet state copy {} KiB",
        SPEC_ROWS, stride * SPEC_ROWS as u64 >> 10, ctx(CTX_DN_ZERO_BYTES) >> 10);
}

pub fn spec_ok() -> bool {
    unsafe { SPEC_OK }
}

fn dn_copy(save: bool) {
    let (live, bak) = (ctx(CTX_BUF_DN_STATE) as *mut u8, unsafe { DN_BAK });
    let n = ctx(CTX_DN_ZERO_BYTES) as usize;
    unsafe {
        if save { core::ptr::copy_nonoverlapping(live, bak, n) } else { core::ptr::copy_nonoverlapping(bak, live, n) }
    }
}

/// Logits row r of the last verify -> BUF_LOGITS (what pick_next and the grammar read).
fn logits_row(r: usize) {
    let v = ctx(CTX_VOCAB_SIZE) as usize;
    unsafe {
        core::ptr::copy_nonoverlapping((pf_lg_ptr + r as u64 * pf_lg_stride) as *const f32, ctx(CTX_BUF_LOGITS) as *mut f32, v)
    };
}

/// fwd_block() with logits for the rows from lg0 on (one pass, t.len() <= PREFILL_MAX_B).
fn fwd_verify(t: &[u32], pos: u64, lg0: usize) {
    for (j, &x) in t.iter().enumerate() {
        if let Some(r) = rec().get_mut(pos as usize + j) {
            *r = x;
        }
    }
    unsafe {
        pf_lg0 = lg0 as u64;
        transformer_prefill(t.as_ptr(), t.len() as u64, pos, 2);
    }
}

/// Shortest context match that drafts (tokens), and the longest one measured.
const MATCH_MIN: usize = 3;
const MATCH_MAX: usize = 32;

/// Prompt lookup over REC[lo..pos] ++ todo: the earlier place where the context agrees longest with the end of the
/// context now (MATCH_MIN..MATCH_MAX tokens, the latest one on a tie), and the tokens that followed it: at most
/// out.len() and at most the match length - 1, so a short (chance) match drafts little. Returns how many were written.
fn lookup(lo: usize, pos: usize, todo: &[u32], out: &mut [u32]) -> usize {
    let rec = rec();
    let len = pos + todo.len();
    if len < lo + MATCH_MIN || out.is_empty() {
        return 0;
    }
    let at = |i: usize| if i < pos { rec[i] } else { todo[i - pos] };
    let (l1, l2) = (at(len - 1), at(len - 2));
    let (mut best, mut best_l) = (0, 0);
    let mut e = len - 1; // a match ends at e (exclusive)
    while e >= lo + 2 {
        if at(e - 1) == l1 && at(e - 2) == l2 {
            let mut l = 2;
            while l < MATCH_MAX && l < e - lo && at(e - 1 - l) == at(len - 1 - l) {
                l += 1;
            }
            if l > best_l {
                (best, best_l) = (e, l);
                if l == MATCH_MAX {
                    break;
                }
            }
        }
        e -= 1;
    }
    if best_l < MATCH_MIN {
        return 0;
    }
    let k = out.len().min(len - best).min(best_l - 1);
    for (j, o) in out[..k].iter_mut().enumerate() {
        *o = at(best + j);
    }
    k
}

/// The sampler's pick, through the tool-call grammar inside a call (M4.7a).
fn pick_fixed() -> u64 {
    let t = pick_next();
    if crate::gram::active() { crate::gram::fix(t) } else { t }
}

/// generate()'s per-token step before token t is fed, in the order of the per-token loop: stop on EOS or Ctrl-C,
/// stream t (hidden from <tool_call> on), stop after </tool_call>, at max_tok or a full context. lpos = t's position.
fn emit(t: u64, lpos: u64, n: &mut u64, max_tok: u64) -> Option<Stop> {
    if is_eos(t) {
        return Some(Stop::Eos);
    }
    if crate::shell::poll_intr() {
        return Some(Stop::Intr);
    }
    if t == TOK_TOOL_CALL {
        unsafe { ECHO = false };
        crate::gram::begin();
    }
    unsafe { tok_write_stdout(t) };
    *n += 1;
    st().ids = (st().ids ^ t).wrapping_mul(0x100_0000_01b3);
    if t == TOK_TOOL_CALL_END {
        return Some(Stop::ToolEnd);
    }
    if *n >= max_tok {
        return Some(Stop::MaxTok);
    }
    if lpos + 3 >= ctx(CTX_MAX_SEQ) {
        return Some(Stop::Ctx);
    }
    None
}

/// Decode after a feed(): stream tokens (hidden from <tool_call> on), stop at EOS, </tool_call>, max_tok,
/// a full context or Ctrl-C, then close the assistant turn in the KV cache (<|im_end|>\n) so the next
/// turn lines up. Returns the stop reason and the number of generated tokens. With speculation on, tokens come
/// in verified runs (see above); the emitted ids and the final state are the same as one token per pass.
pub fn generate(pos: &mut u64, max_tok: u64) -> (Stop, u64) {
    let max_seq = ctx(CTX_MAX_SEQ);
    let spec = unsafe { SPEC_ON && SPEC_OK };
    let kmax = unsafe { SPEC_K }.clamp(1, SPEC_ROWS - 1);
    *st() = SpecStats { used: true, on: spec, ids: FNV0, ..SPEC0 };
    let mut next = pick_next();
    let mut n = 0;
    let mut todo = [0u32; TODO_MAX]; // emitted tokens before `next` that are not fed yet (after a rollback)
    let mut nt = 0;
    let mut blk = [0u32; TODO_MAX];
    // shortcut: draft length and back-off are simple heuristics (grow by 2 on a full accept, shrink by 1 otherwise;
    // after a verify with nothing accepted no drafting for 2, 4, .. 32 tokens), tuned on docs/M48B_RESULT.md
    // (a stricter gate, 4-token matches and 8..64, was not better there)
    let (mut k_cur, mut cool, mut fails) = (kmax, 0u32, 0u32);
    // the conversation only: drafts from the boot prefix (tool JSON, the example call of the template) were mostly
    // rejected inside tool calls (docs/M48B_RESULT.md)
    let lo = (crate::snap::root_pos() as usize).min(*pos as usize);
    let stop = 'gen: loop {
        if let Some(s) = emit(next, *pos + nt as u64, &mut n, max_tok) {
            if nt > 0 {
                fwd_block(&todo[..nt], *pos, false);
                *pos += nt as u64;
            }
            break s;
        }
        todo[nt] = next as u32;
        nt += 1;
        let mut k = 0;
        // T39: inside a call the grammar's own continuation first (it does not touch the lookup's gate)
        if spec && unsafe { GRAM_DRAFT } && crate::gram::active() && *pos + (nt + SPEC_ROWS) as u64 + 4 < max_seq {
            let room = (SPEC_ROWS - 1).min(TODO_MAX - nt);
            k = crate::gram::draft(&mut blk[nt..nt + room]);
        }
        let gd = k > 0;
        if k == 0 && spec && cool == 0 && *pos + (nt + k_cur) as u64 + 4 < max_seq {
            let room = k_cur.min(TODO_MAX - nt);
            k = lookup(lo, *pos as usize, &todo[..nt], &mut blk[nt..nt + room]);
        }
        cool = cool.saturating_sub(1);
        if k == 0 {
            if nt == 1 { fwd(todo[0], *pos, 1) } else { fwd_block(&todo[..nt], *pos, true) }
            *pos += nt as u64;
            nt = 0;
            next = pick_fixed();
            continue;
        }
        let t0 = smp::rdtsc();
        blk[..nt].copy_from_slice(&todo[..nt]);
        dn_copy(true);
        fwd_verify(&blk[..nt + k], *pos, nt - 1);
        let x = st();
        (x.verifies, x.drafted, x.verify_us) = (x.verifies + 1, x.drafted + k as u32, x.verify_us + smp::tsc_to_us(smp::rdtsc() - t0));
        let mut r = 0; // drafts accepted
        next = loop {
            logits_row(r);
            let p = pick_fixed();
            if r < k && p == blk[nt + r] as u64 {
                if let Some(s) = emit(p, *pos + (nt + r) as u64, &mut n, max_tok) {
                    // stop on an accepted draft token: keep the state of the tokens before it only
                    dn_copy(false);
                    fwd_block(&blk[..nt + r], *pos, false);
                    *pos += (nt + r) as u64;
                    next = p;
                    st().accepted += r as u32;
                    break 'gen s;
                }
                r += 1;
                continue;
            }
            break p; // the correction (r < k) or the token after a fully accepted draft
        };
        st().accepted += r as u32;
        if gd {
            (st().gdrafted, st().gaccepted) = (st().gdrafted + k as u32, st().gaccepted + r as u32);
        }
        if r == k {
            *pos += (nt + k) as u64;
            nt = 0;
            if !gd {
                k_cur = (k_cur + 2).min(kmax);
            }
        } else {
            let t1 = smp::rdtsc();
            dn_copy(false);
            st().verify_us += smp::tsc_to_us(smp::rdtsc() - t1);
            st().rollbacks += 1;
            todo[nt..nt + r].copy_from_slice(&blk[nt..nt + r]); // fed again in front of the next pass
            nt += r;
            if !gd {
                k_cur = (k_cur - 1).max(1);
            }
        }
        if gd {
            continue;
        }
        if r == 0 {
            fails += 1;
            cool = 1 << fails.min(5);
        } else {
            fails = 0;
        }
    };
    unsafe { ECHO = true };
    crate::gram::finish(stop == Stop::ToolEnd);
    if matches!(stop, Stop::ToolEnd | Stop::MaxTok | Stop::Ctx) {
        fwd(next as u32, *pos, 0); // emitted but not fed yet
        *pos += 1;
    }
    if *pos + 2 <= max_seq {
        fwd(ctx(CTX_TOK_IM_END) as u32, *pos, 0);
        fwd(ctx(CTX_TOK_NL) as u32, *pos + 1, 0);
        *pos += 2;
    }
    (stop, n)
}

/// `/spec [N]`: the verifier's logits against the per-token path: prefill `toks[..m]`, then N (16) tokens one
/// transformer_forward each (logits hashed per position), the DeltaNet copy back, the same tokens as one
/// compute_logits = 2 pass; prints how many rows are bit-identical and the state hashes. Overwrites the live state.
pub fn spec_check(toks: &[u32], rows: usize) {
    if !spec_ok() {
        println!("[spec] not available (no batched prefill buffers)");
        return;
    }
    let n = rows.clamp(1, SPEC_ROWS).min(toks.len() / 2);
    let m = toks.len() - n;
    let lh = || {
        let mut h = FNV0;
        let l = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_LOGITS) as *const u32, ctx(CTX_VOCAB_SIZE) as usize) };
        l.iter().for_each(|&x| h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
        h
    };
    smp::park_phase(true);
    forward(&toks[..m], 0, false);
    let ts = smp::rdtsc();
    dn_copy(true);
    let ts = smp::rdtsc() - ts;
    let mut want = [0u64; SPEC_ROWS];
    let mut am = [0u64; SPEC_ROWS];
    let t0 = smp::rdtsc();
    for i in 0..n {
        unsafe { transformer_forward(toks[m + i], (m + i) as u64, 1) };
        (want[i], am[i]) = (lh(), unsafe { argmax_f32(ctx(CTX_BUF_LOGITS) as *const f32, ctx(CTX_SAMPLE_N)) });
    }
    let t1 = smp::rdtsc();
    let (h1, _) = state_hash((m + n) as u64);
    let tc = smp::rdtsc();
    dn_copy(false);
    let t2 = smp::rdtsc();
    unsafe {
        pf_lg0 = 0;
        transformer_prefill(toks[m..].as_ptr(), n as u64, m as u64, 2);
    }
    let t3 = smp::rdtsc();
    let (h2, _) = state_hash((m + n) as u64);
    let mut same = 0;
    let mut same_am = 0;
    for i in 0..n {
        logits_row(i);
        same += (lh() == want[i]) as usize;
        same_am += (unsafe { argmax_f32(ctx(CTX_BUF_LOGITS) as *const f32, ctx(CTX_SAMPLE_N)) } == am[i]) as usize;
    }
    smp::park_phase(false);
    println!("[spec] {} positions after {} tokens: logits bit-identical {}/{}, argmax {}/{}, kv+deltanet {:#018x} vs {:#018x} ({}); {} x transformer_forward {} ms, one verify pass {} ms; DeltaNet state copy {} KiB: save {} us, restore {} us",
        n, m, same, n, same_am, n, h1, h2, if h1 == h2 { "same" } else { "DIFFER" },
        n, smp::tsc_to_us(t1 - t0) / 1000, smp::tsc_to_us(t3 - t2) / 1000,
        ctx(CTX_DN_ZERO_BYTES) >> 10, smp::tsc_to_us(ts), smp::tsc_to_us(t2 - tc));
}
