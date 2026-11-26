// FFI to the NASM engine: its entry points, the symbols it imports, and the M0 ABI self-test.
//
// ABI note: this target is soft-float, so Rust never passes f32/f64 in xmm registers. Engine
// entry points that take float args in xmm (rmsnorm_avx2 eps, sample_top_p temp/topp) need an
// asm shim before Rust can call them: sample_top_p_bits in engine/cpu.asm.

use crate::println;

unsafe extern "C" {
    pub fn enable_avx_fpu();
    fn silu_mul_avx2(gate_inout: *mut f32, up: *const f32, n: u64);
    pub fn argmax_f32(x: *const f32, n: u64) -> u64;
    pub fn model_init(base: *const u8, size: u64, max_seq_override: u64) -> i64;
    pub fn transformer_forward(token: u32, pos: u64, compute_logits: i32);
    pub fn tok_encode_bpe(text: *const u8, len: u64, out: *mut u32, max: u64) -> u64;
    pub fn tok_build_chat_prompt(prompt: *const u8, out: *mut u32, max: u64, raw: u64, think: u64) -> u64;
    pub fn tok_write_stdout(token: u64);
    /// cpu.asm shim: floats as raw bits (soft-float Rust cannot pass xmm args).
    pub fn sample_top_p_bits(logits: *const f32, probs: *mut f32, n: u64, rng: *mut u64, temp: u32, topp: u32) -> u64;
    static g_ctx: [u64; 80];
}

/// g_ctx field at byte offset `off` (engine/constants.inc CTX_*).
pub fn ctx(off: usize) -> u64 {
    unsafe { (core::ptr::addr_of!(g_ctx) as *const u64).add(off / 8).read_volatile() }
}

// ---- imports of transformer.asm / tokenizer.asm (sys_mmap_anon/sys_munmap live in mem.rs,
// tp_* in engine/smp.asm) ----

#[unsafe(no_mangle)]
extern "C" fn sys_write_all(_fd: u64, buf: *const u8, len: u64) -> i64 {
    let t0 = crate::smp::rdtsc();
    let b = unsafe { core::slice::from_raw_parts(buf, len as usize) };
    // BSP only (the engine prints from transformer_forward's caller), so plain statics are fine
    unsafe {
        if CAPTURE {
            (*core::ptr::addr_of_mut!(CAP)).push(b);
        }
        if !ECHO {
            return 0;
        }
    }
    crate::console::write_bytes(b);
    OUT_TSC.fetch_add(crate::smp::rdtsc() - t0, core::sync::atomic::Ordering::Relaxed);
    0
}

/// Model output capture (agent turns) and echo switch (off hides tool-call markup).
pub static mut CAPTURE: bool = false;
pub static mut ECHO: bool = true;
pub static mut CAP: crate::log::SBuf<16384> = crate::log::SBuf::new();

/// TSC cycles spent printing model output (serial + framebuffer).
pub static OUT_TSC: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

#[unsafe(no_mangle)]
extern "C" fn str_len(s: *const u8) -> u64 {
    let mut n = 0;
    while unsafe { *s.add(n) } != 0 {
        n += 1;
    }
    n as u64
}

// ---- self-test ----

/// exp(x) in f64 without libm: Taylor series on x/1024, then square 10 times.
fn exp_ref(x: f64) -> f64 {
    let r = x / 1024.0;
    let (mut term, mut sum) = (1.0f64, 1.0f64);
    for k in 1..12 {
        term *= r / k as f64;
        sum += term;
    }
    for _ in 0..10 {
        sum *= sum;
    }
    sum
}

#[repr(C, align(32))]
struct Buf([f32; 64]);

/// Calls silu_mul_avx2 and argmax_f32 and checks them against Rust. Returns true on PASS.
pub fn selftest() -> bool {
    let mut gate = Buf([0.0; 64]);
    let mut up = Buf([0.0; 64]);
    for i in 0..64 {
        gate.0[i] = (i as f32 - 31.5) * 0.37; // -11.7 .. +11.7, both signs, past the exp knee
        up.0[i] = 1.0 + (i % 7) as f32 * 0.25;
    }
    let want: [f64; 64] = core::array::from_fn(|i| {
        let x = gate.0[i] as f64;
        x / (1.0 + exp_ref(-x)) * up.0[i] as f64
    });
    unsafe { silu_mul_avx2(gate.0.as_mut_ptr(), up.0.as_ptr(), 64) };

    let mut max_err = 0.0f64;
    for i in 0..64 {
        let err = (gate.0[i] as f64 - want[i]).abs() / (want[i].abs() + 1e-6);
        if err > max_err {
            max_err = err;
        }
    }
    let am = unsafe { argmax_f32(gate.0.as_ptr(), 64) } as usize;
    let am_ref = (0..64).fold(0, |b, i| if gate.0[i] > gate.0[b] { i } else { b });

    // fast_expf is ~2e-7 relative; 1e-5 leaves room for the divide and multiply.
    let silu_ok = max_err < 1e-5;
    println!(
        "engine: silu_mul_avx2 n=64 max_rel_err={}e-9 -> {}",
        (max_err * 1e9) as u64,
        if silu_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "engine: argmax_f32 = {} (rust {}) -> {}",
        am,
        am_ref,
        if am == am_ref { "PASS" } else { "FAIL" }
    );
    silu_ok && am == am_ref
}
