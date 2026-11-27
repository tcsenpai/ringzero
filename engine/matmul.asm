; VENDORED from llm.asm by Alexey Fateev (alesha-pro), https://github.com/alesha-pro/llm.asm (MIT); taken via our fork github.com/tcsenpai/llm.asm, branch qwen35-kernel, commit 1c73255 (adds Qwen3.5) (shared/matmul.asm).
; Do not edit here: change upstream and re-vendor. ringzero changes (M4.8 batched prefill), all marked "ringzero":
; quantize_x_q8_0_to (caller buffers), matmul_q8_0_batch_range, quantize_batch_q8_0 and the bmm_* job.
; ringzero (M7): Q4_0 weights (W4A8). A weight pointer with bit 0 set (transformer.asm desc_tag_q4) is a Q4_0 split-row
; tensor [K/2 nibble bytes][K/32 fp16 d]: matmul_q8_0_range jumps to matmul_q4_0_range, and the batch kernels unpack
; each Q4_0 row once into a per-thread Q8_0 row (q - 8 as int8, same fp16 scales) and run unchanged on it. The integer
; block dots and the float sequence are the Q8 kernel's, so a Q4_0 row gives bit for bit what the Q8 kernel gives on
; its int8 unpacking, in decode and in both batch kernels.
; ============================================================================
; matmul.asm — High-throughput AVX2 + FMA + F16C Q8_0 matmul kernels (zero libc)
;
; Weight layout (per-row split Q8_0, block size = 32, int8 in [-127, 127]):
;   v2 (FP16 scales, HDR_FLAG_FP16_SCALES):
;     row_stride = K + (K / 32) * 2 = K + (K >> 4) bytes
;     [row + 0 .. row + K - 1]         : K int8 weights
;     [row + K .. row + row_stride - 1]: K/32 IEEE-754 fp16 block scales
;   v1 (FP32 scales, legacy):
;     row_stride = K + (K / 32) * 4 = K + (K >> 3) bytes
;     [row + 0 .. row + K - 1]         : K int8 weights
;     [row + K .. row + row_stride - 1]: K/32 fp32 block scales
;
; Compile-time modes:
;   FAST_Q8=1 (default): Q8_0 x Q8_0 INT8xINT8 AVX2 (vpabsb + vpsignb +
;                        vpmaddubsw + vpmaddwd + vcvtdq2ps + vfmadd231ps),
;                        1-row sequential 4-block (128B) unrolled with
;                        next-row software prefetch for max DRAM streaming BW.
;                        Auto-detects v1 (fp32) and v2 (fp16) scales.
;   FAST_Q8=0 (legacy) : Q8_0 x FP32 AVX2+FMA exact FP32-activation matmul,
;                        auto-detects v1 (fp32) and v2 (fp16) scales.
;
; Functions:
;   matmul_set_fp16   : (uint64 is_fp16)
;   quantize_x_q8_0   : (const float *x, uint64 K)  [FAST_Q8=1]
;   dequant_row_q8_0  : (const void *row_ptr, float *out, uint64 K)
;   matmul_q8_0_range : (const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
;   dot_f32_avx2      : (const float *a, const float *b, uint64 n) -> xmm0
;   saxpy_f32_avx2    : (float *dst, const float *src, uint64 n, float alpha)
; ============================================================================

bits 64
default rel

%ifndef FAST_Q8
%define FAST_Q8 1
%endif

%ifndef Q8_PREFETCH
%define Q8_PREFETCH prefetcht0
%endif

%define Q8_MAX_K 16384
%define Q4_ROWBUF (Q8_MAX_K + Q8_MAX_K / 16)  ; ringzero (M7)

section .text

global matmul_set_fp16
%if FAST_Q8
global quantize_x_q8_0
global quantize_x_q8_0_to                ; ringzero
global matmul_q8_0_batch_range           ; ringzero
global quantize_batch_q8_0               ; ringzero
global bmm_w
global bmm_out
global bmm_k
global bmm_b
global bmm_xq
global bmm_xs
global bmm_xstride
global bmm_ostride
global bmm_vnni
%endif
global dequant_row_q8_0
global matmul_q8_0_range
global matmul_q4_0_range                 ; ringzero (M7)
global dot_f32_avx2
global saxpy_f32_avx2

; ----------------------------------------------------------------------------
; void matmul_set_fp16(uint64 is_fp16)
; rdi = 0 for v1 FP32 scales, non-zero for v2 FP16 scales
; ----------------------------------------------------------------------------
matmul_set_fp16:
    xor     eax, eax
    test    rdi, rdi
    setnz   al
    mov     [rel g_fp16_scales], eax
    ret

; ----------------------------------------------------------------------------
; void dequant_row_q8_0(const void *row_ptr, float *out, uint64 K)
; rdi = row_ptr, rsi = out, rdx = K (multiple of 32)
; Auto-detects v1 FP32 vs v2 FP16 scales via g_fp16_scales.
; ----------------------------------------------------------------------------
dequant_row_q8_0:
    lea     r8, [rdi + rdx]             ; scales_ptr = row_ptr + K
    mov     rcx, rdx
    shr     rcx, 5                      ; nb = K / 32
    mov     r11d, [rel g_fp16_scales]
    xor     eax, eax                    ; b = 0
.lp:
    cmp     rax, rcx
    jae     .done
    test    r11d, r11d
    jnz     .sc_f16
    vbroadcastss ymm4, [r8 + rax*4]     ; v1 fp32 scale[b]
    jmp     .sc_ok
.sc_f16:
    movzx   r10d, word [r8 + rax*2]     ; v2 fp16 scale[b]
    vmovd   xmm4, r10d
    vcvtph2ps xmm4, xmm4
    vbroadcastss ymm4, xmm4
.sc_ok:
    mov     r9, rax
    shl     r9, 5                       ; b * 32 (q offset)
    mov     r10, rax
    shl     r10, 7                      ; b * 128 (out offset)

    vpmovsxbd ymm0, qword [rdi + r9]
    vpmovsxbd ymm1, qword [rdi + r9 + 8]
    vpmovsxbd ymm2, qword [rdi + r9 + 16]
    vpmovsxbd ymm3, qword [rdi + r9 + 24]
    vcvtdq2ps ymm0, ymm0
    vcvtdq2ps ymm1, ymm1
    vcvtdq2ps ymm2, ymm2
    vcvtdq2ps ymm3, ymm3
    vmulps  ymm0, ymm0, ymm4
    vmulps  ymm1, ymm1, ymm4
    vmulps  ymm2, ymm2, ymm4
    vmulps  ymm3, ymm3, ymm4
    vmovups [rsi + r10], ymm0
    vmovups [rsi + r10 + 32], ymm1
    vmovups [rsi + r10 + 64], ymm2
    vmovups [rsi + r10 + 96], ymm3

    inc     rax
    jmp     .lp
.done:
    ret

%if FAST_Q8
; ----------------------------------------------------------------------------
; void quantize_x_q8_0(const float *x, uint64 K)
; rdi = x (f32), rsi = K (multiple of 32, <= Q8_MAX_K)
; Quantizes x into Q8_0 blocks of 32 ONCE before matmul dispatch:
;   q8_xq[0..K-1]  : int8 in [-127, 127] (32-byte aligned, fits in L1D!)
;   q8_xs[0..K/32] : float32 scale = max_abs / 127.0f
; ----------------------------------------------------------------------------
quantize_x_q8_0:
    lea     rdx, [rel q8_xq]
    lea     rcx, [rel q8_xs]
; ringzero (batched prefill): void quantize_x_q8_0_to(const float *x, uint64 K, int8 *xq /*32B aligned*/, float *xs)
; the same quantizer with caller buffers (one call per prompt token, identical bytes)
quantize_x_q8_0_to:
    mov     r9, rdx
    mov     r10, rcx
    mov     rcx, rsi
    shr     rcx, 5                      ; nb = K / 32

    vpbroadcastd ymm12, [rel q8_abs_mask]
    vmovss  xmm13, [rel q8_inv_127]
    vmovss  xmm14, [rel q8_f127]
    vxorps  xmm15, xmm15, xmm15
    vmovdqa ymm11, [rel q8_pack_perm]
    vpbroadcastb ymm10, [rel q8_min_i8]

    xor     eax, eax                    ; b = 0
    xor     r8d, r8d                    ; x byte offset (b * 128)
    xor     edx, edx                    ; xq byte offset (b * 32)
.qlp:
    cmp     rax, rcx
    jae     .qdone

    vmovups ymm0, [rdi + r8]
    vmovups ymm1, [rdi + r8 + 32]
    vmovups ymm2, [rdi + r8 + 64]
    vmovups ymm3, [rdi + r8 + 96]

    vandps  ymm4, ymm0, ymm12
    vandps  ymm5, ymm1, ymm12
    vandps  ymm6, ymm2, ymm12
    vandps  ymm7, ymm3, ymm12
    vmaxps  ymm4, ymm4, ymm5
    vmaxps  ymm6, ymm6, ymm7
    vmaxps  ymm4, ymm4, ymm6

    vextractf128 xmm5, ymm4, 1
    vmaxps  xmm4, xmm4, xmm5
    vshufps xmm5, xmm4, xmm4, 0x0E
    vmaxps  xmm4, xmm4, xmm5
    vshufps xmm5, xmm4, xmm4, 0x01
    vmaxss  xmm4, xmm4, xmm5            ; xmm4 = max_abs

    vmulss  xmm5, xmm4, xmm13           ; scale_x = max_abs / 127.0f
    vmovss  [r10 + rax*4], xmm5

    vxorps  xmm6, xmm6, xmm6            ; inv_scale = 0.0f if max_abs == 0
    vucomiss xmm4, xmm15
    jbe     .qinv_ok
    vdivss  xmm6, xmm14, xmm4           ; inv_scale = 127.0f / max_abs
.qinv_ok:
    vbroadcastss ymm6, xmm6

    vmulps  ymm0, ymm0, ymm6
    vmulps  ymm1, ymm1, ymm6
    vmulps  ymm2, ymm2, ymm6
    vmulps  ymm3, ymm3, ymm6

    vcvtps2dq ymm0, ymm0
    vcvtps2dq ymm1, ymm1
    vcvtps2dq ymm2, ymm2
    vcvtps2dq ymm3, ymm3

    vpackssdw ymm0, ymm0, ymm1
    vpackssdw ymm2, ymm2, ymm3
    vpacksswb ymm0, ymm0, ymm2
    vpermd  ymm0, ymm11, ymm0           ; restore exact linear [0..31] byte order
    vpmaxsb ymm0, ymm0, ymm10           ; clamp >= -127 (prevents -128 overflow)

    vmovdqa [r9 + rdx], ymm0            ; q8_xq[b*32 .. b*32+31]

    add     r8, 128
    add     rdx, 32
    inc     rax
    jmp     .qlp
.qdone:
    ret

; Macro: 1-row x 4-block (128 bytes = 2 cache lines) INT8xINT8 AVX2 body.
; Preconditions:
;   rax = w (current row), r8 = s_cur = w + K (scales of current row)
;   r11 = q8_xq, rbx = byte offset into weights (multiple of 128)
;   ymm0..ymm3 = 4 independent f32 accumulators (hides FMA latency!)
;   ymm12 = [s0*xs0, s1*xs1, s2*xs2, s3*xs3] duplicated in both 128b lanes
;   ymm15 = 16x int16(1)
; Prefetching [r8 + rbx] (= [w + K + rbx]) streams monotonically across both
; the current row's scales and the entire next row without jumping pages or
; resetting the CPU's L1/L2 hardware stream prefetcher.
%macro INT8_1ROW_4BLK 0
    Q8_PREFETCH [r8 + rbx]
    Q8_PREFETCH [r8 + rbx + 64]

    ; Blocks 0 & 1 (+0, +32)
    vmovdqa    ymm4,  [r11 + rbx]
    vmovdqa    ymm10, [r11 + rbx + 32]
    vpabsb     ymm5,  ymm4
    vpabsb     ymm11, ymm10
    vmovdqu    ymm6,  [rax + rbx]
    vmovdqu    ymm7,  [rax + rbx + 32]
    vpsignb    ymm6,  ymm6, ymm4
    vpsignb    ymm7,  ymm7, ymm10
    vpmaddubsw ymm6,  ymm5, ymm6
    vpmaddubsw ymm7,  ymm11, ymm7
    vpmaddwd   ymm6,  ymm6, ymm15
    vpmaddwd   ymm7,  ymm7, ymm15
    vcvtdq2ps  ymm6,  ymm6
    vcvtdq2ps  ymm7,  ymm7
    vpermilps  ymm8,  ymm12, 0x00
    vpermilps  ymm9,  ymm12, 0x55
    vfmadd231ps ymm0, ymm6, ymm8
    vfmadd231ps ymm1, ymm7, ymm9

    ; Blocks 2 & 3 (+64, +96)
    vmovdqa    ymm4,  [r11 + rbx + 64]
    vmovdqa    ymm10, [r11 + rbx + 96]
    vpabsb     ymm5,  ymm4
    vpabsb     ymm11, ymm10
    vmovdqu    ymm6,  [rax + rbx + 64]
    vmovdqu    ymm7,  [rax + rbx + 96]
    vpsignb    ymm6,  ymm6, ymm4
    vpsignb    ymm7,  ymm7, ymm10
    vpmaddubsw ymm6,  ymm5, ymm6
    vpmaddubsw ymm7,  ymm11, ymm7
    vpmaddwd   ymm6,  ymm6, ymm15
    vpmaddwd   ymm7,  ymm7, ymm15
    vcvtdq2ps  ymm6,  ymm6
    vcvtdq2ps  ymm7,  ymm7
    vpermilps  ymm8,  ymm12, 0xAA
    vpermilps  ymm9,  ymm12, 0xFF
    vfmadd231ps ymm2, ymm6, ymm8
    vfmadd231ps ymm3, ymm7, ymm9
%endmacro

; ----------------------------------------------------------------------------
; void matmul_q8_0_range(const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
; FAST_Q8=1: 1-row sequential 4-block (128B) unrolled AVX2 + FMA + F16C
; INT8xINT8 kernel with monotonic next-row software prefetch.
;   - Linear row-by-row memory traversal maximizes DRAM/L2 streaming BW
;   - 4 independent FP32 accumulators saturate Zen2/Haswell FMA & int pipes
;   - Zero stack accesses inside row/block loops
;   - 100% bit-for-bit identical per row for ANY partitioning (1T..64T)
; ----------------------------------------------------------------------------
matmul_q8_0_range:
    test    dil, 1                      ; ringzero (M7): Q4_0-tagged weight pointer
    jnz     matmul_q4_0_range
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rsi, r9
    sub     rsi, r8                     ; rsi = rows_left = r1 - r0
    jle     .exit

    lea     rdx, [rdx + r8 * 4]         ; rdx = out_ptr = out + r0 * 4
    vpbroadcastw ymm15, [rel q8_one16]  ; 16x int16(1)
    lea     r11, [rel q8_xq]
    lea     rbp, [rel q8_xs]
    mov     r15, rcx                    ; r15 = K
    mov     r13, rcx
    and     r13, -128                   ; r13 = K_128 = K & ~127

    cmp     dword [rel g_fp16_scales], 0
    jne     .fast_fp16

    ; ========================================================================
    ; FAST_Q8=1, v1 FP32 scales: row_stride = K + (K >> 3)
    ; ========================================================================
    mov     r14, rcx
    shr     r14, 3
    add     r14, rcx                    ; r14 = row_stride
    mov     rax, r8
    imul    rax, r14
    add     rax, rdi                    ; rax = w_cur = W + r0 * row_stride

align 16
.row_f32:
    lea     r12, [rax + r14]            ; r12 = w_next
    lea     r8,  [rax + r15]            ; r8  = s_cur (fp32)

    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     ebx, ebx                    ; b_q: 0, 128, ...
    xor     r9d, r9d                    ; b_s: 0, 16, ... (fp32 scale & xs offset)
    test    r13, r13
    jz      .tail_f32

align 16
.blk4_f32:
    vmovups    xmm12, [r8 + r9]         ; 4x fp32 scale_w
    vmulps     xmm12, xmm12, [rbp + r9] ; * 4x fp32 scale_x
    vinsertf128 ymm12, ymm12, xmm12, 1

    INT8_1ROW_4BLK

    add     rbx, 128
    add     r9, 16
    cmp     rbx, r13
    jb      .blk4_f32

.tail_f32:
    cmp     rbx, r15
    jae     .red_f32
    vmovdqa    ymm4, [r11 + rbx]
    vpabsb     ymm5, ymm4
    vmovdqu    ymm6, [rax + rbx]
    vpsignb    ymm6, ymm6, ymm4
    vpmaddubsw ymm6, ymm5, ymm6
    vpmaddwd   ymm6, ymm6, ymm15
    vcvtdq2ps  ymm6, ymm6
    vmovss     xmm8, [r8 + r9]
    vmulss     xmm8, xmm8, [rbp + r9]
    vbroadcastss ymm8, xmm8
    vfmadd231ps ymm0, ymm6, ymm8
    add     rbx, 32
    add     r9, 4
    jmp     .tail_f32

.red_f32:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    vmovss  [rdx], xmm0
    add     rdx, 4
    mov     rax, r12
    dec     rsi
    jnz     .row_f32
    jmp     .exit

    ; ========================================================================
    ; FAST_Q8=1, v2 FP16 scales: row_stride = K + (K >> 4)
    ; ========================================================================
.fast_fp16:
    mov     r14, rcx
    shr     r14, 4
    add     r14, rcx                    ; r14 = row_stride
    mov     rax, r8
    imul    rax, r14
    add     rax, rdi                    ; rax = w_cur = W + r0 * row_stride

align 16
.row_f16:
    lea     r12, [rax + r14]            ; r12 = w_next
    lea     r8,  [rax + r15]            ; r8  = s_cur (fp16)

    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     ebx, ebx                    ; b_q: 0, 128, ...
    xor     r9d, r9d                    ; b_s: 0, 8, ... (fp16 scale offset; xs = r9*2)
    test    r13, r13
    jz      .tail_f16

align 16
.blk4_f16:
    vmovq      xmm12, qword [r8 + r9]   ; 4x fp16 scale_w
    vcvtph2ps  xmm12, xmm12             ; -> 4x fp32
    vmulps     xmm12, xmm12, [rbp + r9*2] ; * 4x fp32 scale_x
    vinsertf128 ymm12, ymm12, xmm12, 1

    INT8_1ROW_4BLK

    add     rbx, 128
    add     r9, 8
    cmp     rbx, r13
    jb      .blk4_f16

.tail_f16:
    cmp     rbx, r15
    jae     .red_f16
    vmovdqa    ymm4, [r11 + rbx]
    vpabsb     ymm5, ymm4
    vmovdqu    ymm6, [rax + rbx]
    vpsignb    ymm6, ymm6, ymm4
    vpmaddubsw ymm6, ymm5, ymm6
    vpmaddwd   ymm6, ymm6, ymm15
    vcvtdq2ps  ymm6, ymm6
    movzx      ecx, word [r8 + r9]
    vmovd      xmm8, ecx
    vcvtph2ps  xmm8, xmm8
    vmulss     xmm8, xmm8, [rbp + r9*2]
    vbroadcastss ymm8, xmm8
    vfmadd231ps ymm0, ymm6, ymm8
    add     rbx, 32
    add     r9, 2
    jmp     .tail_f16

.red_f16:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    vmovss  [rdx], xmm0
    add     rdx, 4
    mov     rax, r12
    dec     rsi
    jnz     .row_f16

.exit:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; Macro (ringzero M7): 1-row x 4-block Q4_0 x INT8 body, the INT8_1ROW_4BLK sequence with the weight loads replaced by
; a nibble unpack (llama.cpp bytes_from_nibbles_32 order: low nibbles = elements 0-15, high = 16-31, minus 8).
;   rax = w (nibbles of the current row), r8 = s_cur = w + K/2, r10 = nibble offset (= rbx / 2), rbx = xq offset
;   ymm13 = 32 x 0x0F, ymm14 = 32 x 8, ymm12 / ymm15 / ymm0..3 as in INT8_1ROW_4BLK
%macro Q4_1ROW_4BLK 0
    Q8_PREFETCH [r8 + r10]              ; one line per 4 blocks, ~one row ahead (as the Q8 kernel)

    vmovdqu    xmm6,  [rax + r10]
    vmovdqu    xmm7,  [rax + r10 + 16]
    vpsrlw     xmm8,  xmm6, 4
    vpsrlw     xmm9,  xmm7, 4
    vinserti128 ymm6, ymm6, xmm8, 1
    vinserti128 ymm7, ymm7, xmm9, 1
    vpand      ymm6,  ymm6, ymm13
    vpand      ymm7,  ymm7, ymm13
    vpsubb     ymm6,  ymm6, ymm14
    vpsubb     ymm7,  ymm7, ymm14
    vmovdqa    ymm4,  [r11 + rbx]
    vmovdqa    ymm10, [r11 + rbx + 32]
    vpabsb     ymm5,  ymm4
    vpabsb     ymm11, ymm10
    vpsignb    ymm6,  ymm6, ymm4
    vpsignb    ymm7,  ymm7, ymm10
    vpmaddubsw ymm6,  ymm5, ymm6
    vpmaddubsw ymm7,  ymm11, ymm7
    vpmaddwd   ymm6,  ymm6, ymm15
    vpmaddwd   ymm7,  ymm7, ymm15
    vcvtdq2ps  ymm6,  ymm6
    vcvtdq2ps  ymm7,  ymm7
    vpermilps  ymm8,  ymm12, 0x00
    vpermilps  ymm9,  ymm12, 0x55
    vfmadd231ps ymm0, ymm6, ymm8
    vfmadd231ps ymm1, ymm7, ymm9

    vmovdqu    xmm6,  [rax + r10 + 32]
    vmovdqu    xmm7,  [rax + r10 + 48]
    vpsrlw     xmm8,  xmm6, 4
    vpsrlw     xmm9,  xmm7, 4
    vinserti128 ymm6, ymm6, xmm8, 1
    vinserti128 ymm7, ymm7, xmm9, 1
    vpand      ymm6,  ymm6, ymm13
    vpand      ymm7,  ymm7, ymm13
    vpsubb     ymm6,  ymm6, ymm14
    vpsubb     ymm7,  ymm7, ymm14
    vmovdqa    ymm4,  [r11 + rbx + 64]
    vmovdqa    ymm10, [r11 + rbx + 96]
    vpabsb     ymm5,  ymm4
    vpabsb     ymm11, ymm10
    vpsignb    ymm6,  ymm6, ymm4
    vpsignb    ymm7,  ymm7, ymm10
    vpmaddubsw ymm6,  ymm5, ymm6
    vpmaddubsw ymm7,  ymm11, ymm7
    vpmaddwd   ymm6,  ymm6, ymm15
    vpmaddwd   ymm7,  ymm7, ymm15
    vcvtdq2ps  ymm6,  ymm6
    vcvtdq2ps  ymm7,  ymm7
    vpermilps  ymm8,  ymm12, 0xAA
    vpermilps  ymm9,  ymm12, 0xFF
    vfmadd231ps ymm2, ymm6, ymm8
    vfmadd231ps ymm3, ymm7, ymm9
%endmacro

; ----------------------------------------------------------------------------
; ringzero (M7): void matmul_q4_0_range(const void *W /*bit 0 set*/, const float *x, float *out, uint64 K, uint64 r0, uint64 r1)
; Q4_0 split rows, row_stride = K/2 + K/16, K % 128 == 0 (the exporter writes Q4_0 only for K % 512 == 0), fp16 d.
; Same activations (q8_xq / q8_xs), accumulators, scale products and reduction as .row_f16 of matmul_q8_0_range.
; ----------------------------------------------------------------------------
matmul_q4_0_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    and     rdi, -2                     ; drop the Q4_0 tag
    mov     rsi, r9
    sub     rsi, r8                     ; rows_left
    jle     .exit
    lea     rdx, [rdx + r8 * 4]
    vpbroadcastw ymm15, [rel q8_one16]
    vpbroadcastb ymm13, [rel q4_m0f]
    vpbroadcastb ymm14, [rel q4_b8]
    lea     r11, [rel q8_xq]
    lea     rbp, [rel q8_xs]
    mov     r15, rcx                    ; K
    mov     r13, rcx
    and     r13, -128                   ; K_128 (== K)
    mov     r14, rcx
    shr     r14, 4
    mov     rax, rcx
    shr     rax, 1
    add     r14, rax                    ; row_stride = K/2 + K/16
    mov     rax, r8
    imul    rax, r14
    add     rax, rdi                    ; w_cur = W + r0 * row_stride
    mov     rdi, r15
    shr     rdi, 1                      ; K/2: the scales follow the nibbles

align 16
.row:
    lea     r12, [rax + r14]            ; w_next
    lea     r8,  [rax + rdi]            ; s_cur (fp16)
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     ebx, ebx                    ; xq offset: 0, 128, ...
    xor     r9d, r9d                    ; fp16 scale offset: 0, 8, ... (xs = r9*2)
    xor     r10d, r10d                  ; nibble offset: 0, 64, ...

align 16
.blk4:
    vmovq      xmm12, qword [r8 + r9]
    vcvtph2ps  xmm12, xmm12
    vmulps     xmm12, xmm12, [rbp + r9*2]
    vinsertf128 ymm12, ymm12, xmm12, 1

    Q4_1ROW_4BLK

    add     rbx, 128
    add     r10, 64
    add     r9, 8
    cmp     rbx, r13
    jb      .blk4

    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    vmovss  [rdx], xmm0
    add     rdx, 4
    mov     rax, r12
    dec     rsi
    jnz     .row
.exit:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; Macro (ringzero M7): unpack the Q4_0 row at rax into the Q8_0 row layout at %1 (K int8 = q - 8, then the K/32 fp16
; scales), so the Q8 batch kernels run on it unchanged. K % 512 == 0. Clobbers rbx, r9, rcx, ymm4..ymm9; r15 = K.
%macro Q4_UNPACK_ROW 1
    vpbroadcastb ymm8, [rel q4_m0f]
    vpbroadcastb ymm9, [rel q4_b8]
    xor     ebx, ebx                    ; nibble offset
    xor     r9d, r9d                    ; int8 offset
%%blk:
    vmovdqu    xmm4, [rax + rbx]
    vmovdqu    xmm5, [rax + rbx + 16]
    vpsrlw     xmm6, xmm4, 4
    vpsrlw     xmm7, xmm5, 4
    vinserti128 ymm4, ymm4, xmm6, 1
    vinserti128 ymm5, ymm5, xmm7, 1
    vpand      ymm4, ymm4, ymm8
    vpand      ymm5, ymm5, ymm8
    vpsubb     ymm4, ymm4, ymm9
    vpsubb     ymm5, ymm5, ymm9
    vmovdqa    [%1 + r9], ymm4
    vmovdqa    [%1 + r9 + 32], ymm5
    add     rbx, 32
    add     r9, 64
    cmp     r9, r15
    jb      %%blk
    mov     r9, r15
    shr     r9, 4                       ; K/16 scale bytes (rbx = K/2 = their offset in the Q4 row)
    xor     ecx, ecx
%%sc:
    vmovdqu    ymm4, [rax + rbx]
    vmovdqu    [%1 + r15], ymm4
    add     rbx, 32
    add     %1, 32
    add     ecx, 32
    cmp     rcx, r9
    jb      %%sc
    sub     %1, rcx                     ; back to the row start
%endmacro

; row stride of the batch job's W into r14 (Q8_0: K + K/16, Q4_0: K/2 + K/16), w_cur = W + r8 * stride into rax
%macro BMM_ROW0 0
    mov     r14, r15
    shr     r14, 4
    mov     rax, r15
    test    byte [rel bmm_w], 1
    jz      %%q8
    shr     rax, 1
%%q8:
    add     r14, rax
    mov     rax, r8
    imul    rax, r14
    mov     rcx, [rel bmm_w]
    and     rcx, -2
    add     rax, rcx
%endmacro

; ----------------------------------------------------------------------------
; ringzero (batched prefill): void matmul_q8_0_batch_range(uint64 r0, uint64 r1, uint64 tid)
; Rows [r0, r1) of W against bmm_b quantized activation rows (bmm_xq: stride K bytes, bmm_xs: stride
; K/8 bytes), out[b * bmm_ostride + r * 4]. FP16 scales only (prefill_init refuses v1 files).
; Each weight row is read from DRAM once and stays in L1 for the B tokens; per (row, token) the
; instruction sequence is exactly .row_f16 above, so every output is bit-identical to the
; per-token matmul_q8_0_range result. ringzero (M7): a Q4_0 W (bit 0) is unpacked row by row into
; q4_rowbuf[tid] first (tid = SMP worker id, 0 on the BSP).
; ----------------------------------------------------------------------------
matmul_q8_0_batch_range:
    cmp     dword [rel bmm_vnni], 0
    jne     matmul_q8_0_batch_vnni
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    imul    rdx, rdx, Q4_ROWBUF
    lea     rcx, [rel q4_rowbuf]
    add     rdx, rcx
    push    rdx                         ; [rsp] = this thread's Q4 row buffer
    mov     r8, rdi                     ; r0
    sub     rsi, rdi                    ; rows_left
    jle     .exit
    vpbroadcastw ymm15, [rel q8_one16]
    mov     r15, [rel bmm_k]            ; K
    mov     r13, r15
    and     r13, -128                   ; K_128
    BMM_ROW0                            ; r14 = row_stride, rax = w_cur = W + r0 * row_stride
    mov     rdi, [rel bmm_out]
    lea     rdi, [rdi + r8*4]           ; &out[token 0][r0]

align 16
.row:
    lea     r12, [rax + r14]            ; w_next
    test    byte [rel bmm_w], 1
    jz      .row8
    mov     rdx, [rsp]
    Q4_UNPACK_ROW rdx
    mov     rax, rdx
.row8:
    lea     r8,  [rax + r15]            ; s_cur (fp16)
    mov     rdx, rdi
    mov     r11, [rel bmm_xq]
    mov     rbp, [rel bmm_xs]
    mov     r10, [rel bmm_b]
.tok:
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     ebx, ebx
    xor     r9d, r9d
    test    r13, r13
    jz      .tail

align 16
.blk4:
    vmovq      xmm12, qword [r8 + r9]
    vcvtph2ps  xmm12, xmm12
    vmulps     xmm12, xmm12, [rbp + r9*2]
    vinsertf128 ymm12, ymm12, xmm12, 1

    INT8_1ROW_4BLK

    add     rbx, 128
    add     r9, 8
    cmp     rbx, r13
    jb      .blk4

.tail:
    cmp     rbx, r15
    jae     .red
    vmovdqa    ymm4, [r11 + rbx]
    vpabsb     ymm5, ymm4
    vmovdqu    ymm6, [rax + rbx]
    vpsignb    ymm6, ymm6, ymm4
    vpmaddubsw ymm6, ymm5, ymm6
    vpmaddwd   ymm6, ymm6, ymm15
    vcvtdq2ps  ymm6, ymm6
    movzx      ecx, word [r8 + r9]
    vmovd      xmm8, ecx
    vcvtph2ps  xmm8, xmm8
    vmulss     xmm8, xmm8, [rbp + r9*2]
    vbroadcastss ymm8, xmm8
    vfmadd231ps ymm0, ymm6, ymm8
    add     rbx, 32
    add     r9, 2
    jmp     .tail

.red:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    vmovss  [rdx], xmm0
    add     rdx, [rel bmm_ostride]
    add     r11, r15                    ; next token's int8 row
    mov     rcx, r15
    shr     rcx, 3
    add     rbp, rcx                    ; next token's scales (K/32 * 4 B)
    dec     r10
    jnz     .tok

    add     rdi, 4
    mov     rax, r12
    dec     rsi
    jnz     .row
.exit:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; ringzero (batched prefill): AVX-VNNI variant of matmul_q8_0_batch_range (bmm_vnni != 0, set by prefill_init
; from CPUID). Activations are stored as x + 128 (u8, quantize_batch_q8_0 flips the sign bit), and per
; weight row C = -128 * sum(w) over each 4-byte group is computed once into a stack buffer; then
; vpdpbusd(C, x + 128, w) = sum(x * w) per int32 lane, the exact integer the AVX2 path gets from
; vpabsb/vpsignb/vpmaddubsw/vpmaddwd (no int16 saturation is possible there: |pair| <= 2*127*127).
; The float part (scales, 4 accumulators, reduction) is the same instruction sequence, so the outputs
; stay bit-identical. 4 vector ALU ops per 32 MACs per token instead of 7.
; ----------------------------------------------------------------------------
matmul_q8_0_batch_vnni:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    mov     r8, rdi                     ; r0
    sub     rsi, rdi                    ; rows_left
    jle     .exit_nf
    lea     r9, [rel q4_rowbuf]         ; ringzero (M7): this thread's Q4 row buffer
    imul    rdx, rdx, Q4_ROWBUF
    add     r9, rdx
    mov     r15, [rel bmm_k]            ; K
    mov     rcx, rsp
    sub     rsp, r15
    sub     rsp, 64
    and     rsp, -32
    mov     [rsp], rcx                  ; caller rsp; C buffer at rsp + 32 (K bytes)
    mov     [rsp + 8], r9               ; Q4 row buffer; [rsp + 16] = current (packed) row
    lea     r12, [rsp + 32]
    vpbroadcastb ymm14, [rel q8_b80]    ; 32 x 0x80 (u8 128)
    vpxor   ymm13, ymm13, ymm13
    mov     r13, r15
    and     r13, -128                   ; K_128
    BMM_ROW0                            ; r14 = row_stride, rax = w_cur
    mov     rdi, [rel bmm_out]
    lea     rdi, [rdi + r8*4]

align 16
.row:
    mov     [rsp + 16], rax
    test    byte [rel bmm_w], 1
    jz      .row8
    mov     rdx, [rsp + 8]
    Q4_UNPACK_ROW rdx
    mov     rax, rdx
.row8:
    lea     r8, [rax + r15]             ; s_cur (fp16)
    xor     ebx, ebx
.corr:                                  ; C[j] = -(128 * sum of each 4 weights), streams the row from DRAM once
    Q8_PREFETCH [r8 + rbx]
    vpxor   ymm6, ymm6, ymm6
    {vex} vpdpbusd ymm6, ymm14, [rax + rbx]
    vpsubd  ymm6, ymm13, ymm6
    vmovdqa [r12 + rbx], ymm6
    add     rbx, 32
    cmp     rbx, r15
    jb      .corr
    mov     rdx, rdi
    mov     r11, [rel bmm_xq]
    mov     rbp, [rel bmm_xs]
    mov     r10, [rel bmm_b]
.tok:
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     ebx, ebx
    xor     r9d, r9d
    test    r13, r13
    jz      .tail

align 16
.blk4:
    vmovq      xmm12, qword [r8 + r9]
    vcvtph2ps  xmm12, xmm12
    vmulps     xmm12, xmm12, [rbp + r9*2]
    vinsertf128 ymm12, ymm12, xmm12, 1
    vmovdqa    ymm4, [r11 + rbx + 0]
    vmovdqa    ymm6, [r12 + rbx + 0]
    {vex} vpdpbusd ymm6, ymm4, [rax + rbx + 0]
    vcvtdq2ps  ymm6, ymm6
    vpermilps  ymm8, ymm12, 0x00
    vfmadd231ps ymm0, ymm6, ymm8
    vmovdqa    ymm4, [r11 + rbx + 32]
    vmovdqa    ymm6, [r12 + rbx + 32]
    {vex} vpdpbusd ymm6, ymm4, [rax + rbx + 32]
    vcvtdq2ps  ymm6, ymm6
    vpermilps  ymm8, ymm12, 0x55
    vfmadd231ps ymm1, ymm6, ymm8
    vmovdqa    ymm4, [r11 + rbx + 64]
    vmovdqa    ymm6, [r12 + rbx + 64]
    {vex} vpdpbusd ymm6, ymm4, [rax + rbx + 64]
    vcvtdq2ps  ymm6, ymm6
    vpermilps  ymm8, ymm12, 0xAA
    vfmadd231ps ymm2, ymm6, ymm8
    vmovdqa    ymm4, [r11 + rbx + 96]
    vmovdqa    ymm6, [r12 + rbx + 96]
    {vex} vpdpbusd ymm6, ymm4, [rax + rbx + 96]
    vcvtdq2ps  ymm6, ymm6
    vpermilps  ymm8, ymm12, 0xFF
    vfmadd231ps ymm3, ymm6, ymm8
    add     rbx, 128
    add     r9, 8
    cmp     rbx, r13
    jb      .blk4

.tail:
    cmp     rbx, r15
    jae     .red
    vmovdqa    ymm4, [r11 + rbx]
    vmovdqa    ymm6, [r12 + rbx]
    {vex} vpdpbusd ymm6, ymm4, [rax + rbx]
    vcvtdq2ps  ymm6, ymm6
    movzx      ecx, word [r8 + r9]
    vmovd      xmm8, ecx
    vcvtph2ps  xmm8, xmm8
    vmulss     xmm8, xmm8, [rbp + r9*2]
    vbroadcastss ymm8, xmm8
    vfmadd231ps ymm0, ymm6, ymm8
    add     rbx, 32
    add     r9, 2
    jmp     .tail

.red:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    vmovss  [rdx], xmm0
    add     rdx, [rel bmm_ostride]
    add     r11, r15
    mov     rcx, r15
    shr     rcx, 3
    add     rbp, rcx
    dec     r10
    jnz     .tok

    add     rdi, 4
    mov     rax, [rsp + 16]
    add     rax, r14
    dec     rsi
    jnz     .row
    mov     rsp, [rsp]
.exit_nf:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ringzero (batched prefill): void quantize_batch_q8_0(const float *x /*rdi*/)
; quantize bmm_b rows of K floats (row b at x + b * bmm_xstride) into bmm_xq / bmm_xs
quantize_batch_q8_0:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdi
    mov     r12, [rel bmm_xq]
    mov     r13, [rel bmm_xs]
    mov     r14, [rel bmm_b]
    mov     r15, [rel bmm_k]
.lp:
    mov     rdi, rbx
    mov     rsi, r15
    mov     rdx, r12
    mov     rcx, r13
    call    quantize_x_q8_0_to
    cmp     dword [rel bmm_vnni], 0
    je      .no_flip
    vpbroadcastb ymm0, [rel q8_b80]     ; VNNI kernel: store x + 128 (u8)
    xor     eax, eax
.flip:
    vpxor   ymm1, ymm0, [r12 + rax]
    vmovdqa [r12 + rax], ymm1
    add     rax, 32
    cmp     rax, r15
    jb      .flip
.no_flip:
    add     rbx, [rel bmm_xstride]
    add     r12, r15
    mov     rax, r15
    shr     rax, 3
    add     r13, rax
    dec     r14
    jnz     .lp
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

%else
; ----------------------------------------------------------------------------
; void matmul_q8_0_range(const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
; FAST_Q8=0 (legacy): 4-row unrolled AVX2 + FMA Q8_0 x FP32 matmul.
; Full FP32 activations (zero activation quantization noise).
; Auto-detects v1 FP32 scales vs v2 FP16 scales via g_fp16_scales.
; ----------------------------------------------------------------------------
matmul_q8_0_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 48
    ; [rsp + 0]  = out_ptr
    ; [rsp + 8]  = rows_left (r1 - r0)
    ; [rsp + 16] = row_stride
    ; [rsp + 24] = stride4
    ; [rsp + 32] = base_w_cur
    ; [rsp + 40] = is_fp16 (0 or 1)

    mov     rax, r9
    sub     rax, r8                     ; rows_left = r1 - r0
    jle     .exit
    mov     [rsp + 8], rax

    mov     ebp, [rel g_fp16_scales]
    mov     [rsp + 40], rbp

    ; row_stride = K + (K >> (is_fp16 ? 4 : 3))
    mov     rbx, rcx
    test    ebp, ebp
    jnz     .stride_f16
    shr     rbx, 3
    jmp     .stride_done
.stride_f16:
    shr     rbx, 4
.stride_done:
    add     rbx, rcx
    mov     [rsp + 16], rbx
    lea     rax, [rbx * 4]
    mov     [rsp + 24], rax

    lea     rdx, [rdx + r8 * 4]
    mov     [rsp + 0], rdx

    mov     rax, r8
    imul    rax, rbx
    add     rdi, rax
    mov     [rsp + 32], rdi

    mov     r11, rsi                    ; x_ptr (float*)
    mov     r15, rcx                    ; K

.row4_lp:
    cmp     qword [rsp + 8], 4
    jb      .row1_check

    mov     rax, [rsp + 32]             ; w0
    mov     rbx, [rsp + 16]             ; row_stride
    lea     rcx, [rax + rbx]            ; w1
    lea     rdx, [rcx + rbx]            ; w2
    lea     rsi, [rdx + rbx]            ; w3

    lea     rdi, [rax + r15]            ; s0
    lea     r8,  [rcx + r15]            ; s1
    lea     r9,  [rdx + r15]            ; s2
    lea     r10, [rsi + r15]            ; s3

    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3

    xor     r12d, r12d                  ; b_x = 0, 128, ...
    xor     r13d, r13d                  ; b_q = 0, 32,  ...
    xor     r14d, r14d                  ; b_idx = 0, 1, 2, ... (block index)
    mov     ebp, [rsp + 40]             ; is_fp16

.blk4_lp:
    vmovups ymm4, [r11 + r12]
    vmovups ymm5, [r11 + r12 + 32]
    vmovups ymm6, [r11 + r12 + 64]
    vmovups ymm7, [r11 + r12 + 96]

    ; --- Interleaved Rows 0 & 1 ---
    vpmovsxbd ymm8,  qword [rax + r13]
    vpmovsxbd ymm12, qword [rcx + r13]
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm12, ymm12
    vmulps    ymm8,  ymm8,  ymm4
    vmulps    ymm12, ymm12, ymm4

    vpmovsxbd ymm9,  qword [rax + r13 + 8]
    vpmovsxbd ymm13, qword [rcx + r13 + 8]
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm13, ymm13
    vfmadd231ps ymm8,  ymm9,  ymm5
    vfmadd231ps ymm12, ymm13, ymm5

    vpmovsxbd ymm10, qword [rax + r13 + 16]
    vpmovsxbd ymm14, qword [rcx + r13 + 16]
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm14, ymm14
    vfmadd231ps ymm8,  ymm10, ymm6
    vfmadd231ps ymm12, ymm14, ymm6

    vpmovsxbd ymm11, qword [rax + r13 + 24]
    vpmovsxbd ymm15, qword [rcx + r13 + 24]
    vcvtdq2ps ymm11, ymm11
    vcvtdq2ps ymm15, ymm15
    vfmadd231ps ymm8,  ymm11, ymm7
    vfmadd231ps ymm12, ymm15, ymm7

    test    ebp, ebp
    jnz     .s01_f16
    vbroadcastss ymm9,  dword [rdi + r14*4]
    vbroadcastss ymm13, dword [r8 + r14*4]
    jmp     .s01_ok
.s01_f16:
    movzx   ebx, word [rdi + r14*2]
    vmovd   xmm9, ebx
    vcvtph2ps xmm9, xmm9
    vbroadcastss ymm9, xmm9
    movzx   ebx, word [r8 + r14*2]
    vmovd   xmm13, ebx
    vcvtph2ps xmm13, xmm13
    vbroadcastss ymm13, xmm13
.s01_ok:
    vfmadd231ps ymm0, ymm8,  ymm9
    vfmadd231ps ymm1, ymm12, ymm13

    ; --- Interleaved Rows 2 & 3 ---
    vpmovsxbd ymm8,  qword [rdx + r13]
    vpmovsxbd ymm12, qword [rsi + r13]
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm12, ymm12
    vmulps    ymm8,  ymm8,  ymm4
    vmulps    ymm12, ymm12, ymm4

    vpmovsxbd ymm9,  qword [rdx + r13 + 8]
    vpmovsxbd ymm13, qword [rsi + r13 + 8]
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm13, ymm13
    vfmadd231ps ymm8,  ymm9,  ymm5
    vfmadd231ps ymm12, ymm13, ymm5

    vpmovsxbd ymm10, qword [rdx + r13 + 16]
    vpmovsxbd ymm14, qword [rsi + r13 + 16]
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm14, ymm14
    vfmadd231ps ymm8,  ymm10, ymm6
    vfmadd231ps ymm12, ymm14, ymm6

    vpmovsxbd ymm11, qword [rdx + r13 + 24]
    vpmovsxbd ymm15, qword [rsi + r13 + 24]
    vcvtdq2ps ymm11, ymm11
    vcvtdq2ps ymm15, ymm15
    vfmadd231ps ymm8,  ymm11, ymm7
    vfmadd231ps ymm12, ymm15, ymm7

    test    ebp, ebp
    jnz     .s23_f16
    vbroadcastss ymm9,  dword [r9 + r14*4]
    vbroadcastss ymm13, dword [r10 + r14*4]
    jmp     .s23_ok
.s23_f16:
    movzx   ebx, word [r9 + r14*2]
    vmovd   xmm9, ebx
    vcvtph2ps xmm9, xmm9
    vbroadcastss ymm9, xmm9
    movzx   ebx, word [r10 + r14*2]
    vmovd   xmm13, ebx
    vcvtph2ps xmm13, xmm13
    vbroadcastss ymm13, xmm13
.s23_ok:
    vfmadd231ps ymm2, ymm8,  ymm9
    vfmadd231ps ymm3, ymm12, ymm13

    add     r12, 128
    add     r13, 32
    inc     r14
    cmp     r13, r15
    jb      .blk4_lp

    vhaddps ymm0, ymm0, ymm1
    vhaddps ymm2, ymm2, ymm3
    vhaddps ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1

    mov     rax, [rsp + 0]
    vmovups [rax], xmm0
    add     rax, 16
    mov     [rsp + 0], rax

    mov     rax, [rsp + 24]
    add     [rsp + 32], rax
    sub     qword [rsp + 8], 4
    jmp     .row4_lp

.row1_check:
    cmp     qword [rsp + 8], 0
    je      .exit
.row1_lp:
    mov     rax, [rsp + 32]
    lea     rdi, [rax + r15]
    vxorps  ymm0, ymm0, ymm0
    xor     r12d, r12d
    xor     r13d, r13d
    xor     r14d, r14d
    mov     ebp, [rsp + 40]
.blk1_lp:
    vpmovsxbd ymm8,  qword [rax + r13]
    vpmovsxbd ymm9,  qword [rax + r13 + 8]
    vpmovsxbd ymm10, qword [rax + r13 + 16]
    vpmovsxbd ymm11, qword [rax + r13 + 24]
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm11, ymm11
    vmulps    ymm8,  ymm8,  [r11 + r12]
    vfmadd231ps ymm8, ymm9,  [r11 + r12 + 32]
    vfmadd231ps ymm8, ymm10, [r11 + r12 + 64]
    vfmadd231ps ymm8, ymm11, [r11 + r12 + 96]
    test    ebp, ebp
    jnz     .s1_f16
    vbroadcastss ymm12, dword [rdi + r14*4]
    jmp     .s1_ok
.s1_f16:
    movzx   ebx, word [rdi + r14*2]
    vmovd   xmm12, ebx
    vcvtph2ps xmm12, xmm12
    vbroadcastss ymm12, xmm12
.s1_ok:
    vfmadd231ps ymm0, ymm8, ymm12

    add     r12, 128
    add     r13, 32
    inc     r14
    cmp     r13, r15
    jb      .blk1_lp

    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    mov     rax, [rsp + 0]
    vmovss  [rax], xmm0
    add     rax, 4
    mov     [rsp + 0], rax
    mov     rax, [rsp + 16]
    add     [rsp + 32], rax
    dec     qword [rsp + 8]
    jnz     .row1_lp

.exit:
    add     rsp, 48
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
%endif

; ----------------------------------------------------------------------------
; float dot_f32_avx2(const float *a, const float *b, uint64 n) -> xmm0
; rdi = a, rsi = b, rdx = n (multiple of 32, e.g. head_dim = 128)
; ----------------------------------------------------------------------------
dot_f32_avx2:
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .red
    vmovups ymm4, [rdi + rax]
    vmovups ymm5, [rdi + rax + 32]
    vmovups ymm6, [rdi + rax + 64]
    vmovups ymm7, [rdi + rax + 96]
    vfmadd231ps ymm0, ymm4, [rsi + rax]
    vfmadd231ps ymm1, ymm5, [rsi + rax + 32]
    vfmadd231ps ymm2, ymm6, [rsi + rax + 64]
    vfmadd231ps ymm3, ymm7, [rsi + rax + 96]
    add     rax, 128
    jmp     .lp
.red:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    ret

; ----------------------------------------------------------------------------
; void saxpy_f32_avx2(float *dst, const float *src, uint64 n, float alpha)
; rdi = dst, rsi = src, rdx = n (multiple of 32, e.g. head_dim = 128), xmm0 = alpha
; dst[i] += alpha * src[i]
; ----------------------------------------------------------------------------
saxpy_f32_avx2:
    vbroadcastss ymm0, xmm0
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .done
    vmovups ymm1, [rdi + rax]
    vmovups ymm2, [rdi + rax + 32]
    vmovups ymm3, [rdi + rax + 64]
    vmovups ymm4, [rdi + rax + 96]
    vfmadd231ps ymm1, ymm0, [rsi + rax]
    vfmadd231ps ymm2, ymm0, [rsi + rax + 32]
    vfmadd231ps ymm3, ymm0, [rsi + rax + 64]
    vfmadd231ps ymm4, ymm0, [rsi + rax + 96]
    vmovups [rdi + rax], ymm1
    vmovups [rdi + rax + 32], ymm2
    vmovups [rdi + rax + 64], ymm3
    vmovups [rdi + rax + 96], ymm4
    add     rax, 128
    jmp     .lp
.done:
    ret

%if FAST_Q8
section .rodata
align 32
q8_pack_perm: dd 0, 4, 1, 5, 2, 6, 3, 7
q8_abs_mask:  dd 0x7FFFFFFF
q8_f127:      dd 127.0
q8_inv_127:   dd 0.007874015748031496       ; 1.0f / 127.0f
q8_one16:     dw 1
q8_min_i8:    db -127
q8_b80:       db 0x80                   ; ringzero
q4_m0f:       db 0x0F                   ; ringzero (M7)
q4_b8:        db 8                      ; ringzero (M7)
%endif

section .bss
alignb 64
g_fp16_scales: resd 1
%if FAST_Q8
alignb 64
q8_xq:        resb Q8_MAX_K
alignb 64
q8_xs:        resb (Q8_MAX_K / 32) * 4
; ringzero: batched matmul job (written by the BSP before dispatch, read by every worker)
alignb 64
bmm_w:        resq 1
bmm_out:      resq 1
bmm_k:        resq 1
bmm_b:        resq 1
bmm_xq:       resq 1                    ; B x K int8 (prefill_init buffer, 64B aligned)
bmm_xs:       resq 1                    ; B x K/32 f32
bmm_xstride:  resq 1                    ; input row stride (bytes) for quantize_batch_q8_0
bmm_ostride:  resq 1                    ; output row stride (bytes)
bmm_vnni:     resd 1                    ; 1: AVX-VNNI batch kernel (prefill_init, CPUID 7.1:EAX[4])
; ringzero (M7): one Q8_0-layout row per SMP worker for the batch kernels' Q4_0 unpack (K + K/16 bytes, K <= Q8_MAX_K)
alignb 64
q4_rowbuf:    resb 64 * Q4_ROWBUF
%endif
