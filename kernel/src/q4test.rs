// M7 boot self-test of the Q4_0 decode kernel: 64 rows of layer 0's W_gateup (K = dim) and W_down (K = hidden)
// through matmul_q4_0_range (tagged pointer), and the same rows unpacked here to Q8_0 (int8 q - 8, the same fp16
// scales) through the Q8_0 kernel. engine/matmul.asm promises bit-identical outputs; this checks it on the real model.
use crate::engine::{ctx, tp_matmul_q8_0};
use crate::mem::sys_mmap_anon;

const CTX_MMAP_BASE: usize = 0;
const CTX_DIM: usize = 88;
const CTX_HIDDEN_DIM: usize = 96;
const CTX_LAYER_TABLE: usize = 168;
const QTYPE_Q4_0: u32 = 2;
const ROWS: usize = 64;

/// None when layer 0 has no Q4_0 MLP tensors (Q8 models), else PASS/FAIL.
pub fn selftest() -> Option<bool> {
    let d = ctx(CTX_LAYER_TABLE) as *const u8;
    let qt = unsafe { (d.add(4) as *const u32).read_unaligned() };
    let mut ok = true;
    for (slot, k) in [(4usize, ctx(CTX_DIM) as usize), (5, ctx(CTX_HIDDEN_DIM) as usize)] {
        if qt >> (2 * slot) & 3 != QTYPE_Q4_0 {
            return None;
        }
        let w = ctx(CTX_MMAP_BASE) + unsafe { (d.add(8 + 8 * slot) as *const u64).read_unaligned() };
        let (s4, s8) = (k / 2 + k / 16, k + k / 16);
        let q8 = sys_mmap_anon((ROWS * s8) as u64);
        let x = sys_mmap_anon((k * 4) as u64) as *mut f32;
        let out = sys_mmap_anon((2 * ROWS * 4) as u64) as *mut u32;
        if (q8 as i64) < 0 || (x as i64) < 0 || (out as i64) < 0 {
            return Some(false);
        }
        unsafe {
            let src = core::slice::from_raw_parts(w as *const u8, ROWS * s4);
            let dst = core::slice::from_raw_parts_mut(q8, ROWS * s8);
            for r in 0..ROWS {
                let (a, b) = (&src[r * s4..(r + 1) * s4], &mut dst[r * s8..(r + 1) * s8]);
                for blk in 0..k / 32 {
                    for j in 0..16 {
                        let v = a[blk * 16 + j];
                        b[blk * 32 + j] = ((v & 15) as i8 - 8) as u8;
                        b[blk * 32 + j + 16] = ((v >> 4) as i8 - 8) as u8;
                    }
                }
                b[k..].copy_from_slice(&a[k / 2..]);
            }
            for i in 0..k {
                *x.add(i) = ((i * 7919 % 2001) as f32) * 0.001 - 1.0; // -1.0 .. 1.0, both signs
            }
            tp_matmul_q8_0((w | 1) as *const u8, x, out as *mut f32, ROWS as u64, k as u64);
            tp_matmul_q8_0(q8, x, out.add(ROWS) as *mut f32, ROWS as u64, k as u64);
            let o = core::slice::from_raw_parts(out, 2 * ROWS);
            let same = o[..ROWS] == o[ROWS..];
            crate::klog!(b'I', "q4: slot {} K {}: {} rows Q4_0 kernel vs Q8_0 kernel on the unpacked rows: {} (row 0 {:#010x} / {:#010x})",
                slot, k, ROWS, if same { "bit-identical" } else { "DIFFER" }, o[0], o[ROWS]);
            ok &= same;
        }
    }
    Some(ok)
}
