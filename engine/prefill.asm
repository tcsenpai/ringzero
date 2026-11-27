; prefill.asm: batched prompt prefill for the Qwen3.5 engine (ringzero M4.8, own code, not vendored).
;
; transformer_forward reads every weight matrix once per token. transformer_prefill runs a block of n
; prompt tokens per pass: the matmuls see all n activation rows at once (tp_matmul_q8_0_batch: each
; weight row is read from DRAM once and reused from L1 for the n tokens), everything else runs per
; token in prompt order with the same functions and the same inputs as transformer_forward:
;   - full attention: QK-norm, RoPE, K/V store at pos, attention over 0..pos (so causal inside the
;     block and over the cached past), output gate
;   - Gated DeltaNet: a/b dots, causal conv step, SiLU, the head recurrence (tp_deltanet), silu(z) gate
; The per-token parts point g_ctx BUF_Q / BUF_ATT_OUT at the token's row (restored on return), so the
; vendored per-token kernels are reused unchanged. Every quantization and accumulation is the same
; instruction sequence on the same values as the per-token path: the state after a prefill is
; bit-identical to n transformer_forward calls (verified by llm.rs `/prefill`, docs/BATCHED_PREFILL_RESULT.md).
;
;   int64 prefill_init(uint64 max_b)     allocate the block buffers (once, before the snapshot pool), pick the
;                                        AVX-VNNI batch matmul when the CPU has it (matmul.asm, bmm_vnni)
;   void  transformer_prefill(const uint32 *toks, uint64 n /*1..max_b*/, uint64 pos0, int compute_logits)
;                                        compute_logits: 0 none, 1 the last token -> BUF_LOGITS, 2 (M4.8b, the
;                                        speculative verifier) every token from row pf_lg0 on -> pf_lg_ptr rows
;                                        (stride pf_lg_stride bytes), with one batched LM-head matmul

bits 64
default rel

%include "constants.inc"

%ifdef QWEN35
%ifdef KV_F16
%define KV_SHIFT 1
%else
%define KV_SHIFT 2
%endif

extern g_ctx
extern g_dn_job
extern sys_mmap_anon
extern dequant_row_q8_0
extern rmsnorm_avx2
extern vec_add_avx2
extern silu_mul_avx2
extern sigmoid_mul_avx2
extern conv4_step_avx2
extern compute_rope_cs
extern rope_heads
extern dot_f32_avx2
extern tp_matmul_q8_0
extern tp_matmul_q8_0_batch
extern tp_gqa_attn
extern tp_deltanet
extern desc_tag_q4
extern bmm_xq
extern bmm_xs
extern bmm_xstride
extern bmm_ostride
extern bmm_vnni

global prefill_init
global transformer_prefill
global pf_tsc_mm
global pf_tsc_all
global pf_lg_ptr
global pf_lg_stride
global pf_lg0

section .text

; ----------------------------------------------------------------------------
; int64 prefill_init(uint64 max_b): 0, -1 (v1 fp32-scale file: unsupported), -2 (arena full)
; ----------------------------------------------------------------------------
prefill_init:
    push    rbx
    push    r12
    push    r13
    lea     rbx, [rel g_ctx]
    mov     rax, -1
    cmp     dword [rbx + CTX_FP16_SCALES], 0
    je      .ret
    mov     r12, rdi                    ; max_b
    mov     eax, 7                      ; AVX-VNNI (VEX vpdpbusd): CPUID.(7,1):EAX bit 4
    mov     ecx, 1
    cpuid
    shr     eax, 4
    and     eax, 1
    mov     [rel bmm_vnni], eax
    lea     rbx, [rel g_ctx]

    ; per-token row strides (bytes, 64B multiples)
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    call    .al64
    mov     [rel pf_sx], rax
    ; proj region: max(q + 2 kv + gate, conv + val) floats (model_init's BUF_Q sizing)
    mov     rax, [rbx + CTX_Q_DIM]
    add     rax, [rbx + CTX_KV_DIM]
    add     rax, [rbx + CTX_KV_DIM]
    add     rax, [rbx + CTX_GATE_DIM]
    mov     rcx, [rbx + CTX_CONV_DIM]
    add     rcx, [rbx + CTX_VAL_DIM]
    cmp     rax, rcx
    jae     .p_ok
    mov     rax, rcx
.p_ok:
    shl     rax, 2
    call    .al64
    mov     [rel pf_sproj], rax
    mov     rax, [rbx + CTX_Q_DIM]
    cmp     rax, [rbx + CTX_VAL_DIM]
    jae     .a_ok
    mov     rax, [rbx + CTX_VAL_DIM]
.a_ok:
    shl     rax, 2
    call    .al64
    mov     [rel pf_satt], rax
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    shl     rax, 3                      ; gate | up
    call    .al64
    mov     [rel pf_sgu], rax
    ; max matmul K: dim, q_dim, val_dim, hidden
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    mov     rcx, [rbx + CTX_DIM]
    cmp     rax, rcx
    cmovb   rax, rcx
    mov     rcx, [rbx + CTX_Q_DIM]
    cmp     rax, rcx
    cmovb   rax, rcx
    mov     rcx, [rbx + CTX_VAL_DIM]
    cmp     rax, rcx
    cmovb   rax, rcx
    mov     r13, rax                    ; kmax

    ; one allocation: x | xb | proj | att | gu | xq | xs, each max_b rows
    mov     rax, [rel pf_sx]
    shl     rax, 1
    add     rax, [rel pf_sproj]
    add     rax, [rel pf_satt]
    add     rax, [rel pf_sgu]
    add     rax, r13
    mov     rcx, r13
    shr     rcx, 3
    add     rcx, 63
    and     rcx, -64
    add     rax, rcx
    imul    rax, r12
    add     rax, 64 * 8
    mov     rdi, rax
    call    sys_mmap_anon
    test    rax, rax
    jle     .oom
    mov     rcx, r12
    mov     [rel pf_x], rax
    mov     rdx, [rel pf_sx]
    imul    rdx, rcx
    add     rax, rdx
    mov     [rel pf_xb], rax
    add     rax, rdx
    mov     [rel pf_proj], rax
    mov     rdx, [rel pf_sproj]
    imul    rdx, rcx
    add     rax, rdx
    mov     [rel pf_att], rax
    mov     rdx, [rel pf_satt]
    imul    rdx, rcx
    add     rax, rdx
    mov     [rel pf_gu], rax
    mov     rdx, [rel pf_sgu]
    imul    rdx, rcx
    add     rax, rdx
    mov     [rel bmm_xq], rax           ; 64B aligned, row b at b*K (K multiple of 32)
    mov     rdx, r13
    imul    rdx, rcx
    add     rax, rdx
    add     rax, 63
    and     rax, -64
    mov     [rel bmm_xs], rax
    mov     [rel pf_maxb], r12
    xor     eax, eax
    jmp     .ret
.oom:
    mov     rax, -2
.ret:
    pop     r13
    pop     r12
    pop     rbx
    ret
.al64:
    add     rax, 63
    and     rax, -64
    ret

; row pointer: %1 = base_sym + %2 * stride_sym   (%1 must not be %2)
%macro ROW 4
    mov     %1, %2
    imul    %1, [rel %4]
    add     %1, [rel %3]
%endmacro

; batched matmul: W, x base sym, x stride sym, out base, out stride sym, M (reg), K (reg)
; x / out are register or memory operands that hold addresses; M and K must not be rax/rdi/rsi/rdx/rcx/r8/r9
%macro BMM 7
    mov     rax, [rel %3]
    mov     [rel bmm_xstride], rax
    mov     rax, [rel %5]
    mov     [rel bmm_ostride], rax
    mov     rdi, %1
    mov     rsi, %2
    mov     rdx, %4
    mov     rcx, %6
    mov     r8,  %7
    mov     r9,  [rsp + L_N]
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    sub     [rel pf_tsc_mm], rax
    mov     rdx, %4
    call    tp_matmul_q8_0_batch
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    add     [rel pf_tsc_mm], rax
%endmacro

; stack locals of transformer_prefill
%define L_TOKS   0
%define L_N      8
%define L_POS0   16
%define L_LOGITS 24
%define L_L      32
%define L_FULL   40
%define L_LIN    48
%define L_TYPE   56
%define L_SAVEQ  64
%define L_SAVEA  72
%define L_KVBASE 80
%define L_PACK   88                     ; (gqa_ratio << 32) | att_scale bits
%define L_QSCALE 96
%define L_POS    104
%define SLOT(i)  (112 + 8*(i))          ; 12 descriptor slots -> 112..207

; ----------------------------------------------------------------------------
; void transformer_prefill(const uint32 *toks, uint64 n, uint64 pos0, int compute_logits)
; ----------------------------------------------------------------------------
transformer_prefill:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 232                    ; 48 + 232 + 8 = 288 -> 16B aligned
    push    rdx
    push    rdi
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    sub     [rel pf_tsc_all], rax
    pop     rdi
    pop     rdx
    mov     [rsp + L_TOKS], rdi
    mov     [rsp + L_N], rsi
    mov     [rsp + L_POS0], rdx
    mov     [rsp + L_LOGITS], rcx
    lea     rbx, [rel g_ctx]
    mov     rax, [rbx + CTX_BUF_Q]
    mov     [rsp + L_SAVEQ], rax
    mov     rax, [rbx + CTX_BUF_ATT_OUT]
    mov     [rsp + L_SAVEA], rax

    ; start of a sequence: clear the DeltaNet state + conv history (as transformer_forward)
    test    rdx, rdx
    jnz     .no_zero
    mov     rcx, [rbx + CTX_DN_ZERO_BYTES]
    shr     rcx, 3
    jz      .no_zero
    cld
    mov     rdi, [rbx + CTX_BUF_DN_STATE]
    xor     eax, eax
    rep stosq
.no_zero:

    ; att_scale = 1/sqrt(head_dim), gqa_ratio = n_heads / n_kv_heads, packed for tp_gqa_attn
    mov     rax, [rbx + CTX_HEAD_DIM]
    vcvtsi2ss xmm0, xmm0, rax
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel p_one]
    vdivss  xmm1, xmm1, xmm0
    vmovd   ecx, xmm1
    mov     rax, [rbx + CTX_N_HEADS]
    xor     edx, edx
    div     qword [rbx + CTX_N_KV_HEADS]
    shl     rax, 32
    or      rax, rcx
    mov     [rsp + L_PACK], rax
    ; DeltaNet q scale = (float)(1.0 / sqrt(Dk)) (f64, as deltanet_layer)
    vcvtsi2sd xmm0, xmm0, qword [rbx + CTX_LIN_DIM]
    vsqrtsd xmm0, xmm0, xmm0
    vmovsd  xmm1, [rel p_d_one]
    vdivsd  xmm1, xmm1, xmm0
    vcvtsd2ss xmm1, xmm1, xmm1
    vmovss  [rsp + L_QSCALE], xmm1

    ; 1. embeddings: x_b = dequant(embed[tok_b])
    xor     r12d, r12d
.emb_lp:
    cmp     r12, [rsp + L_N]
    jae     .emb_done
    mov     rax, [rsp + L_TOKS]
    mov     edi, [rax + r12*4]
    mov     rax, [rbx + CTX_DIM]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     rdx, rax
    shr     rdx, cl
    add     rdx, rax                    ; emb_row_stride
    imul    rdi, rdx
    add     rdi, [rbx + CTX_EMB_PTR]
    ROW     rsi, r12, pf_x, pf_sx
    mov     rdx, [rbx + CTX_DIM]
    call    dequant_row_q8_0
    inc     r12
    jmp     .emb_lp
.emb_done:

    mov     qword [rsp + L_L], 0
    mov     qword [rsp + L_FULL], 0
    mov     qword [rsp + L_LIN], 0

.layer_lp:
    mov     rax, [rsp + L_L]
    cmp     rax, [rbx + CTX_N_LAYERS]
    jae     .layers_done
    imul    rax, rax, DESC_STRIDE
    add     rax, [rbx + CTX_LAYER_TABLE]
    mov     rdx, [rbx + CTX_MMAP_BASE]
    mov     ecx, [rax + DESC_TYPE]
    mov     [rsp + L_TYPE], rcx
%assign i 0
%rep N_DESC_SLOTS
    mov     rcx, [rax + DESC_OFFS + 8*i]
    add     rcx, rdx
    mov     [rsp + SLOT(i)], rcx
%assign i i+1
%endrep
    mov     rdi, rax                    ; M7: Q4_0 slots get bit 0 set (the batch kernels unpack them)
    lea     rsi, [rsp + SLOT(0)]
    call    desc_tag_q4

    ; input RMSNorm: xb_b = rmsnorm(x_b, in_norm)
    xor     r12d, r12d
.innorm_lp:
    cmp     r12, [rsp + L_N]
    jae     .innorm_done
    ROW     rdi, r12, pf_xb, pf_sx
    ROW     rsi, r12, pf_x, pf_sx
    mov     rdx, [rsp + SLOT(0)]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    inc     r12
    jmp     .innorm_lp
.innorm_done:

    cmp     qword [rsp + L_TYPE], LAYER_TYPE_LINEAR_ATTN
    jne     .full_attn

    ; ===================== Gated DeltaNet mixer =====================
    ; [qkv | z]_b = W_proj @ xb_b
    mov     r10, [rbx + CTX_CONV_DIM]
    add     r10, [rbx + CTX_VAL_DIM]
    mov     r11, [rbx + CTX_DIM]
    BMM     [rsp + SLOT(2)], [rel pf_xb], pf_sx, [rel pf_proj], pf_sproj, r10, r11

    ; head job parameters (constant over the block)
    lea     rax, [rel g_dn_job]
    mov     rcx, [rsp + L_LIN]
    mov     [rax + 0], rcx
    mov     rcx, [rsp + SLOT(7)]
    mov     [rax + 8], rcx              ; negExpA
    mov     rcx, [rsp + SLOT(8)]
    mov     [rax + 16], rcx             ; dt_bias
    mov     rcx, [rsp + SLOT(9)]
    mov     [rax + 24], rcx             ; gnorm_w
    mov     ecx, [rsp + L_QSCALE]
    mov     [rax + 32], ecx

    xor     r12d, r12d
.dn_tok_lp:
    cmp     r12, [rsp + L_N]
    jae     .dn_tok_done
    ; a[h] = W_a[h] . xb_b -> MISC[0..], b[h] = W_b[h] . xb_b -> MISC[128..]
    mov     r14, [rsp + SLOT(11)]
    mov     r15, [rsp + SLOT(10)]
    ROW     rbp, r12, pf_xb, pf_sx
    xor     r13d, r13d
.ab_lp:
    cmp     r13, [rbx + CTX_LIN_HEADS]
    jae     .ab_done
    mov     rdi, r14
    mov     rsi, rbp
    mov     rdx, [rbx + CTX_DIM]
    call    dot_f32_avx2
    mov     rax, [rbx + CTX_BUF_MISC]
    vmovss  [rax + r13*4], xmm0
    mov     rdi, r15
    mov     rsi, rbp
    mov     rdx, [rbx + CTX_DIM]
    call    dot_f32_avx2
    mov     rax, [rbx + CTX_BUF_MISC]
    vmovss  [rax + 128 + r13*4], xmm0
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    add     r14, rax
    add     r15, rax
    inc     r13
    jmp     .ab_lp
.ab_done:
    ; causal depthwise conv (k=4) over the qkv channels, then SiLU
    ROW     r13, r12, pf_proj, pf_sproj
    mov     rsi, [rbx + CTX_CONV_DIM]
    imul    rsi, [rsp + L_LIN]
    imul    rsi, 12
    add     rsi, [rbx + CTX_BUF_DN_CONV]
    mov     rdi, r13
    mov     rdx, [rsp + SLOT(6)]
    mov     rcx, [rbx + CTX_CONV_DIM]
    call    conv4_step_avx2
    mov     rdi, r13
    mov     rsi, [rbx + CTX_BUF_ONES]
    mov     rdx, [rbx + CTX_CONV_DIM]
    call    silu_mul_avx2
    ; heads (SMP): reads BUF_Q / BUF_MISC, writes BUF_ATT_OUT
    ROW     r14, r12, pf_att, pf_satt
    mov     [rbx + CTX_BUF_Q], r13
    mov     [rbx + CTX_BUF_ATT_OUT], r14
    mov     rdi, [rbx + CTX_LIN_HEADS]
    call    tp_deltanet
    ; z = silu(z) * o
    mov     rdi, [rbx + CTX_CONV_DIM]
    lea     rdi, [r13 + rdi*4]
    mov     rsi, r14
    mov     rdx, [rbx + CTX_VAL_DIM]
    call    silu_mul_avx2
    inc     r12
    jmp     .dn_tok_lp
.dn_tok_done:
    ; xb_b = out_proj @ z_b
    mov     r10, [rbx + CTX_CONV_DIM]
    shl     r10, 2
    add     r10, [rel pf_proj]
    mov     r11, [rbx + CTX_DIM]
    mov     r13, [rbx + CTX_VAL_DIM]
    BMM     [rsp + SLOT(3)], r10, pf_sproj, [rel pf_xb], pf_sx, r11, r13
    inc     qword [rsp + L_LIN]
    jmp     .attn_residual

    ; ===================== full attention =====================
.full_attn:
    mov     r10, [rbx + CTX_Q_DIM]
    add     r10, [rbx + CTX_KV_DIM]
    add     r10, [rbx + CTX_KV_DIM]
    add     r10, [rbx + CTX_GATE_DIM]
    mov     r11, [rbx + CTX_DIM]
    BMM     [rsp + SLOT(2)], [rel pf_xb], pf_sx, [rel pf_proj], pf_sproj, r10, r11

    ; layer_kv_base = full_ord * max_seq * kv_dim * elt
    mov     rax, [rsp + L_FULL]
    inc     qword [rsp + L_FULL]
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, KV_SHIFT
    mov     [rsp + L_KVBASE], rax

    xor     r12d, r12d
.fa_tok_lp:
    cmp     r12, [rsp + L_N]
    jae     .fa_tok_done
    mov     rax, [rsp + L_POS0]
    add     rax, r12
    mov     [rsp + L_POS], rax
    ROW     rbp, r12, pf_proj, pf_sproj ; q_b (k, v, gate follow)

    ; QK-Norm per head (in place)
    mov     r13, rbp
    mov     r14, [rbx + CTX_N_HEADS]
.qn_lp:
    mov     rdi, r13
    mov     rsi, r13
    mov     rdx, [rsp + SLOT(6)]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rax, [rbx + CTX_HEAD_DIM]
    lea     r13, [r13 + rax*4]
    dec     r14
    jnz     .qn_lp
    mov     r13, [rbx + CTX_Q_DIM]
    lea     r13, [rbp + r13*4]          ; k_b
    mov     r15, r13
    mov     r14, [rbx + CTX_N_KV_HEADS]
.kn_lp:
    mov     rdi, r15
    mov     rsi, r15
    mov     rdx, [rsp + SLOT(7)]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rax, [rbx + CTX_HEAD_DIM]
    lea     r15, [r15 + rax*4]
    dec     r14
    jnz     .kn_lp

    ; RoPE at pos on the rot_dim dims of every Q and K head
    mov     rdi, [rbx + CTX_BUF_COS]
    mov     rsi, [rbx + CTX_BUF_SIN]
    mov     rdx, [rbx + CTX_BUF_INV_FREQ]
    mov     rcx, [rbx + CTX_ROT_DIM]
    shr     rcx, 1
    mov     r8,  [rsp + L_POS]
    call    compute_rope_cs
    mov     rdi, rbp
    mov     rcx, [rbx + CTX_N_HEADS]
    call    rope_heads
    mov     rdi, r13
    mov     rcx, [rbx + CTX_N_KV_HEADS]
    call    rope_heads

    ; store K, V at cache[full_ord, kv_h, pos, head_dim]
    mov     rax, [rsp + L_KVBASE]
    mov     rsi, r13                    ; k_b
    mov     r8, [rbx + CTX_KV_DIM]
    lea     r8, [r13 + r8*4]            ; v_b
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2                      ; f32 head bytes
    mov     r15, [rbx + CTX_MAX_SEQ]
    imul    r15, [rbx + CTX_HEAD_DIM]
    shl     r15, KV_SHIFT               ; kv_head_stride
    mov     rcx, [rsp + L_POS]
    imul    rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, KV_SHIFT
    add     rax, rcx
    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rax
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rax
    mov     r10, [rbx + CTX_N_KV_HEADS]
.kvh_lp:
    xor     r9d, r9d
.kvh_el:
    cmp     r9, [rbx + CTX_HEAD_DIM]
    jae     .kvh_next
    vmovups ymm0, [rsi + r9*4]
    vmovups ymm1, [r8 + r9*4]
%ifdef KV_F16
    vcvtps2ph [rdi + r9*2], ymm0, 0
    vcvtps2ph [rdx + r9*2], ymm1, 0
%else
    vmovups [rdi + r9*4], ymm0
    vmovups [rdx + r9*4], ymm1
%endif
    add     r9, 8
    jmp     .kvh_el
.kvh_next:
    add     rsi, r14
    add     r8,  r14
    add     rdi, r15
    add     rdx, r15
    dec     r10
    jnz     .kvh_lp

    ; attention over 0..pos (SMP over heads): reads BUF_Q, writes BUF_ATT_OUT
    ROW     r14, r12, pf_att, pf_satt
    mov     [rbx + CTX_BUF_Q], rbp
    mov     [rbx + CTX_BUF_ATT_OUT], r14
    mov     rdi, [rsp + L_KVBASE]
    mov     rsi, [rsp + L_POS]
    mov     rdx, [rsp + L_PACK]
    mov     rcx, [rbx + CTX_N_HEADS]
    call    tp_gqa_attn
    ; output gate: att *= sigmoid(gate), gate rows follow v
    cmp     qword [rbx + CTX_GATE_DIM], 0
    je      .fa_next
    mov     rdi, r14
    mov     rsi, [rbx + CTX_KV_DIM]
    lea     rsi, [r13 + rsi*8]          ; k_b + 2 * kv_dim floats
    mov     rdx, [rbx + CTX_Q_DIM]
    call    sigmoid_mul_avx2
.fa_next:
    inc     r12
    jmp     .fa_tok_lp
.fa_tok_done:
    ; xb_b = o_proj @ att_b
    mov     r11, [rbx + CTX_DIM]
    mov     r13, [rbx + CTX_Q_DIM]
    BMM     [rsp + SLOT(3)], [rel pf_att], pf_satt, [rel pf_xb], pf_sx, r11, r13

.attn_residual:
    ; x_b += xb_b ; xb_b = rmsnorm(x_b, post_norm)
    xor     r12d, r12d
.res1_lp:
    cmp     r12, [rsp + L_N]
    jae     .res1_done
    ROW     r13, r12, pf_x, pf_sx
    ROW     r14, r12, pf_xb, pf_sx
    mov     rdi, r13
    mov     rsi, r14
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2
    mov     rdi, r14
    mov     rsi, r13
    mov     rdx, [rsp + SLOT(1)]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    inc     r12
    jmp     .res1_lp
.res1_done:

    ; SwiGLU MLP: [gate | up]_b = W_gateup @ xb_b ; gate = silu(gate) * up ; xb_b = W_down @ gate
    mov     r10, [rbx + CTX_HIDDEN_DIM]
    add     r10, r10
    mov     r11, [rbx + CTX_DIM]
    BMM     [rsp + SLOT(4)], [rel pf_xb], pf_sx, [rel pf_gu], pf_sgu, r10, r11
    xor     r12d, r12d
.swi_lp:
    cmp     r12, [rsp + L_N]
    jae     .swi_done
    ROW     rdi, r12, pf_gu, pf_sgu
    mov     rsi, [rbx + CTX_HIDDEN_DIM]
    lea     rsi, [rdi + rsi*4]
    mov     rdx, [rbx + CTX_HIDDEN_DIM]
    call    silu_mul_avx2
    inc     r12
    jmp     .swi_lp
.swi_done:
    mov     r11, [rbx + CTX_DIM]
    mov     r13, [rbx + CTX_HIDDEN_DIM]
    BMM     [rsp + SLOT(5)], [rel pf_gu], pf_sgu, [rel pf_xb], pf_sx, r11, r13

    xor     r12d, r12d
.res2_lp:
    cmp     r12, [rsp + L_N]
    jae     .res2_done
    ROW     rdi, r12, pf_x, pf_sx
    ROW     rsi, r12, pf_xb, pf_sx
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2
    inc     r12
    jmp     .res2_lp
.res2_done:
    inc     qword [rsp + L_L]
    jmp     .layer_lp

.layers_done:
    mov     rax, [rsp + L_SAVEQ]
    mov     [rbx + CTX_BUF_Q], rax
    mov     rax, [rsp + L_SAVEA]
    mov     [rbx + CTX_BUF_ATT_OUT], rax
    cmp     qword [rsp + L_LOGITS], 2
    je      .all_logits
    cmp     qword [rsp + L_LOGITS], 0
    je      .done

    ; last token: BUF_X = x_{n-1}, final RMSNorm + LM head -> BUF_LOGITS (as transformer_forward)
    mov     r12, [rsp + L_N]
    dec     r12
    ROW     rsi, r12, pf_x, pf_sx
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    cld
    rep movsb
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_FINAL_NORM_PTR]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rdi, [rbx + CTX_LM_HEAD_PTR] ; M7: tied (= embed) or untied lm_head
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_BUF_LOGITS]
    mov     rcx, [rbx + CTX_VOCAB_SIZE]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0
    jmp     .done

    ; M4.8b: rows pf_lg0..n-1: final RMSNorm in place (as transformer_forward on BUF_X), then the LM head for
    ; all of them in one batched matmul (bit-identical to tp_matmul_q8_0 per row, like the layer matmuls)
.all_logits:
    mov     r12, [rel pf_lg0]
.fn_lp:
    cmp     r12, [rsp + L_N]
    jae     .fn_done
    ROW     rdi, r12, pf_x, pf_sx
    mov     rsi, rdi
    mov     rdx, [rbx + CTX_FINAL_NORM_PTR]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    inc     r12
    jmp     .fn_lp
.fn_done:
    mov     rax, [rel pf_sx]
    mov     [rel bmm_xstride], rax
    mov     rax, [rel pf_lg_stride]
    mov     [rel bmm_ostride], rax
    mov     rdi, [rbx + CTX_LM_HEAD_PTR]
    ROW     rsi, [rel pf_lg0], pf_x, pf_sx
    mov     rdx, [rel pf_lg_ptr]
    mov     rcx, [rbx + CTX_VOCAB_SIZE]
    mov     r8,  [rbx + CTX_DIM]
    mov     r9,  [rsp + L_N]
    sub     r9,  [rel pf_lg0]
    call    tp_matmul_q8_0_batch

.done:
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    add     [rel pf_tsc_all], rax
    add     rsp, 232
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

section .rodata
align 16
p_one:   dd 1.0
align 8
p_d_one: dq 1.0

section .bss
alignb 64
pf_x:     resq 1
pf_xb:    resq 1
pf_proj:  resq 1
pf_att:   resq 1
pf_gu:    resq 1
pf_sx:    resq 1
pf_sproj: resq 1
pf_satt:  resq 1
pf_sgu:   resq 1
pf_maxb:  resq 1
pf_tsc_mm:  resq 1                      ; TSC in the batched matmuls (quantize + dispatch + wait)
pf_tsc_all: resq 1                      ; TSC in transformer_prefill
pf_lg_ptr:    resq 1                    ; M4.8b: logits rows of compute_logits = 2 (llm.rs allocates them)
pf_lg_stride: resq 1                    ; bytes between two rows
pf_lg0:       resq 1                    ; first block row that gets logits
%endif
