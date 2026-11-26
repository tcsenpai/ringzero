// Qwen3-Embedding-0.6B next to Qwen3.5-2B on the same asm engine. All engine model state is one global
// (g_ctx, 640 B) plus matmul.asm's Q8_0 scale mode, so a second model is a second saved g_ctx: with_embedder()
// swaps it in for a whole embedding call and puts the 2B's back before it returns. Rules (docs/spikes/EMBEDDER.md):
// swap only between whole forward calls, on the BSP; outside with_embedder the 2B's context is always loaded, so
// snap.rs, llm.rs and the agent never see the embedder's buffers, and the embedder never touches the 2B's KV,
// DeltaNet state or llm::REC (its forward calls go to its own arena; pos 0 there clears no 2B state).
use crate::engine::{ctx, g_ctx, matmul_set_fp16, model_init, transformer_forward};
use crate::{klog, llm, smp};

// engine/constants.inc
const CTX_ARENA_SIZE: usize = 24;
const CTX_FINAL_NORM_PTR: usize = 80;
const CTX_DIM: usize = 88;
const CTX_MAX_SEQ: usize = 160;
const CTX_BUF_X: usize = 208;
const CTX_FP16_SCALES: usize = 368; // u32 (low half of the u64 read by ctx)

/// Embedder context length: KV cache for 1024 tokens (~115 MiB of arena); longer texts are cut.
pub const MAX_TOK: usize = 1024;
pub const DIM: usize = 1024;
/// <|endoftext|>: appended to every input (Qwen3-Embedding pools the last token; without it quality drops silently)
const TOK_EOT: u32 = 151643;
/// English task line of the query instruction (the one used by the spike's retrieval test)
pub const TASK: &str = "Given a question from the user of an operating system agent, retrieve the stored memory that answers it";

type Ctx = [u64; 80];
static mut CTX_2B: Ctx = [0; 80];
static mut CTX_EMB: Ctx = [0; 80];
static mut READY: bool = false;
static mut TOKS: [u32; MAX_TOK] = [0; MAX_TOK];
static mut TEXT: crate::log::SBuf<8192> = crate::log::SBuf::new();

fn save(to: *mut Ctx) {
    unsafe { core::ptr::copy_nonoverlapping(core::ptr::addr_of!(g_ctx), to, 1) };
}
fn load(from: *const Ctx) {
    unsafe {
        core::ptr::copy_nonoverlapping(from, core::ptr::addr_of_mut!(g_ctx), 1);
        matmul_set_fp16(ctx(CTX_FP16_SCALES) & 0xffff_ffff);
    }
}

pub fn ready() -> bool {
    unsafe { READY }
}

/// Sample hash of the embedder file: saved in the memory index header (a new export means a rebuild).
static mut HASH: u64 = 0;
pub fn hash() -> u64 {
    unsafe { HASH }
}

/// Boot, after llm::init (the 2B's context is loaded): model_init the embedder into a second context.
pub fn init(base: *const u8, size: u64) -> bool {
    let t0 = smp::rdtsc();
    unsafe { HASH = crate::mstore::sample_hash(base, size) };
    save(core::ptr::addr_of_mut!(CTX_2B));
    let r = unsafe { model_init(base, size, MAX_TOK as u64) };
    let ok = r >= 0 && ctx(CTX_DIM) as usize == DIM;
    if ok {
        save(core::ptr::addr_of_mut!(CTX_EMB));
        klog!(b'I', "embed: model_init ok in {} ms, dim {}, max_seq {}, arena {} MiB (own g_ctx, swapped per call)",
            smp::tsc_to_us(smp::rdtsc() - t0) / 1000, ctx(CTX_DIM), ctx(CTX_MAX_SEQ), ctx(CTX_ARENA_SIZE) >> 20);
    } else {
        klog!(b'E', "embed: model_init failed ({}, dim {}), no embedder", r, ctx(CTX_DIM));
    }
    load(core::ptr::addr_of!(CTX_2B));
    unsafe { READY = ok };
    ok
}

/// Run f with the embedder's context loaded; the 2B's is back when it returns.
fn with_embedder<R>(f: impl FnOnce() -> R) -> R {
    save(core::ptr::addr_of_mut!(CTX_2B));
    load(core::ptr::addr_of!(CTX_EMB));
    let r = f();
    save(core::ptr::addr_of_mut!(CTX_EMB));
    load(core::ptr::addr_of!(CTX_2B));
    r
}

pub struct Info {
    pub n_tok: usize,
    pub us: u64,
    /// L2 norm of the pooled x * w_final (first `dim` values) before normalizing
    pub raw_norm: f32,
}

/// sqrt without libm (soft-float target): Newton on f64 from an exponent-halved guess.
pub fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51)); // ~sqrt within 2x
    for _ in 0..6 {
        g = 0.5 * (g + x / g);
    }
    g
}

/// Embed text into out[..dim] (Matryoshka: first dim of 1024 values, L2-normalized). Queries get the
/// "Instruct: {TASK}\nQuery:" prefix, documents none; every input ends with <|endoftext|>. The pooled vector is
/// the last token's hidden state after the final RMSNorm: the norm's per-vector scale cancels in the L2
/// normalization, so x * w_final is enough and the forward skips the lm_head (compute_logits 0).
/// BSP only, between whole 2B forward calls (never inside a generation step).
pub fn embed(text: &str, is_query: bool, dim: usize, out: &mut [f32]) -> Result<Info, &'static str> {
    let job = begin(text, is_query)?;
    resume(job, dim, out, || false).map(|r| r.expect("not stopped"))
}

// One embedding in flight at a time; the idle indexer (fidx.rs) runs it in slices. The embedder's KV rows
// 0..NEXT stay valid between slices as long as no other embed() starts (EPOCH changes, the job restarts).
static mut EPOCH: u64 = 0;
static mut N: usize = 0;
static mut NEXT: usize = 0;
static mut BUSY_US: u64 = 0;

/// Tokenize (with the embedder's vocabulary) and start a job; returns its id for resume().
pub fn begin(text: &str, is_query: bool) -> Result<u64, &'static str> {
    if !ready() {
        return Err("no embedder loaded");
    }
    let t0 = smp::rdtsc();
    let t = unsafe { &mut *core::ptr::addr_of_mut!(TEXT) };
    t.clear();
    if is_query {
        use core::fmt::Write;
        let _ = write!(t, "Instruct: {}\nQuery:", TASK);
    }
    t.push(text.as_bytes()); // shortcut: texts past 8 KiB are cut here (and past MAX_TOK tokens below)
    let toks = unsafe { &mut *core::ptr::addr_of_mut!(TOKS) };
    let n = with_embedder(|| llm::encode_plain(&t.b[..t.n], &mut toks[..MAX_TOK - 1]));
    toks[n] = TOK_EOT;
    unsafe {
        EPOCH += 1;
        (N, NEXT, BUSY_US) = (n + 1, 0, smp::tsc_to_us(smp::rdtsc() - t0));
        Ok(EPOCH)
    }
}

/// Run the job's forward calls until done (Some) or until stop() says so between two of them (None, call
/// again later). Err if another embedding started in between (begin again).
pub fn resume(job: u64, dim: usize, out: &mut [f32], mut stop: impl FnMut() -> bool) -> Result<Option<Info>, &'static str> {
    if dim == 0 || dim > DIM || dim % 32 != 0 || out.len() < dim {
        return Err("bad embedding dim");
    }
    if job != unsafe { EPOCH } {
        return Err("preempted by another embedding");
    }
    let t0 = smp::rdtsc();
    let toks = unsafe { &*core::ptr::addr_of!(TOKS) };
    let n = unsafe { N };
    let done = with_embedder(|| {
        let mut i = unsafe { NEXT };
        while i < n {
            if stop() {
                break;
            }
            unsafe { transformer_forward(toks[i], i as u64, 0) };
            i += 1;
        }
        unsafe { NEXT = i };
        if i < n {
            return None;
        }
        let x = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_X) as *const f32, DIM) };
        let w = unsafe { core::slice::from_raw_parts(ctx(CTX_FINAL_NORM_PTR) as *const f32, DIM) };
        let mut ss = 0.0f64;
        for i in 0..dim {
            out[i] = x[i] * w[i];
            ss += out[i] as f64 * out[i] as f64;
        }
        Some(sqrt(ss))
    });
    unsafe { BUSY_US += smp::tsc_to_us(smp::rdtsc() - t0) };
    let Some(raw_norm) = done else { return Ok(None) };
    let inv = if raw_norm > 0.0 { 1.0 / raw_norm } else { 0.0 };
    out[..dim].iter_mut().for_each(|v| *v = (*v as f64 * inv) as f32);
    unsafe { EPOCH += 1 }; // done: the job id is spent
    Ok(Some(Info { n_tok: n, us: unsafe { BUSY_US }, raw_norm: raw_norm as f32 }))
}

/// Boot: one embedding through the context swap (the 2B's golden self-test runs after it).
pub fn boot_check() {
    let mut v = [0.0f32; DIM];
    smp::park_phase(true);
    let r = embed("ringzero boots", false, DIM, &mut v);
    smp::park_phase(false);
    match r {
        Ok(i) => klog!(b'I', "embed: boot check \"ringzero boots\": {} tok in {}.{:03} ms, v[0..4] = {:.4} {:.4} {:.4} {:.4}",
            i.n_tok, i.us / 1000, i.us % 1000, v[0], v[1], v[2], v[3]),
        Err(e) => klog!(b'E', "embed: boot check failed: {}", e),
    }
}
