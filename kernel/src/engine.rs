// FFI to the NASM engine: the symbols it imports (provided here) and an M0 link/ABI self-test.
//
// ABI note: this target is soft-float, so Rust never passes f32/f64 in xmm registers. Engine
// entry points that take float args in xmm (rmsnorm_avx2 eps, sample_top_p temp/topp) need an
// asm shim before Rust can call them. Everything called here takes only pointers and integers.

use crate::println;

unsafe extern "C" {
    pub fn enable_avx_fpu();
    fn silu_mul_avx2(gate_inout: *mut f32, up: *const f32, n: u64);
    fn argmax_f32(x: *const f32, n: u64) -> u64;
    // Not called in M0. Referenced only so --gc-sections keeps the whole engine and the
    // linker must resolve every symbol it imports (M1 calls these for real).
    fn model_init();
    fn model_free();
    fn transformer_forward();
    fn tok_build_chat_prompt();
    fn tok_encode_bpe();
    fn tok_write_stdout();
    fn sample_top_p();
    fn matmul_q8_0_range();
    fn quantize_x_q8_0();
}

#[used]
pub static ENTRY_POINTS: [(&str, unsafe extern "C" fn()); 9] = [
    ("model_init", model_init),
    ("model_free", model_free),
    ("transformer_forward", transformer_forward),
    ("tok_build_chat_prompt", tok_build_chat_prompt),
    ("tok_encode_bpe", tok_encode_bpe),
    ("tok_write_stdout", tok_write_stdout),
    ("sample_top_p", sample_top_p),
    ("matmul_q8_0_range", matmul_q8_0_range),
    ("quantize_x_q8_0", quantize_x_q8_0),
];

// ---- imports of transformer.asm / tokenizer.asm ----

#[unsafe(no_mangle)]
extern "C" fn sys_mmap_anon(size: u64) -> *mut u8 {
    panic!("sys_mmap_anon({size}): no model arena yet (M1)");
}

#[unsafe(no_mangle)]
extern "C" fn sys_munmap(_ptr: *mut u8, _size: u64) -> i64 {
    0 // bump arena: freeing is a no-op, same as the asm OS
}

#[unsafe(no_mangle)]
extern "C" fn sys_write_all(_fd: u64, buf: *const u8, len: u64) -> i64 {
    crate::console::write_bytes(unsafe { core::slice::from_raw_parts(buf, len as usize) });
    0
}

#[unsafe(no_mangle)]
extern "C" fn str_len(s: *const u8) -> u64 {
    let mut n = 0;
    while unsafe { *s.add(n) } != 0 {
        n += 1;
    }
    n as u64
}

#[unsafe(no_mangle)]
extern "C" fn tp_matmul_q8_0() {
    panic!("tp_matmul_q8_0: SMP dispatch not linked yet (M1)");
}

#[unsafe(no_mangle)]
extern "C" fn tp_gqa_attn() {
    panic!("tp_gqa_attn: SMP dispatch not linked yet (M1)");
}

#[unsafe(no_mangle)]
extern "C" fn tp_deltanet() {
    panic!("tp_deltanet: SMP dispatch not linked yet (M1)");
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
    println!("engine: {} entry points linked, e.g. {} @ {:p}, {} @ {:p}", ENTRY_POINTS.len(),
        ENTRY_POINTS[0].0, ENTRY_POINTS[0].1 as *const u8, ENTRY_POINTS[2].0, ENTRY_POINTS[2].1 as *const u8);
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
