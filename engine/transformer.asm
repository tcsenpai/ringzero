; VENDORED from llm.asm by Alexey Fateev (alesha-pro), https://github.com/alesha-pro/llm.asm (MIT); taken via our fork github.com/tcsenpai/llm.asm, branch qwen35-kernel, commit 1c73255 (adds Qwen3.5) (shared/transformer.asm).
; Do not edit here: change upstream and re-vendor. AIOS changes (M2.5), all marked "AIOS": optional
; FP16 KV cache (-DKV_F16): vcvtps2ph on store, dot_f16_avx2 / saxpy_f16_avx2 in gqa_attn_range; and
; model_init lets max_seq_override exceed the header's max_seq_len.
; ============================================================================
%ifdef QWEN35
; transformer.asm — Qwen3 / Qwen3.5 Transformer Forward Pass & Model Loader
%else
; transformer.asm — Full Qwen3-0.6B Transformer Forward Pass & Model Loader
%endif
;
%ifdef QWEN35
; Qwen3.5 (hybrid) additions, all selected by the .asmbin v3 header / per-layer descriptor:
;   - layer type 1 = Gated DeltaNet (deltanet_layer), full-attn layers with output gate and partial RoPE
;   - KV cache only for full-attention layers; DeltaNet S + conv state zeroed when pos == 0
;   - optional per-layer debug taps (CTX_TAP_FN)
;
; Qwen3 architecture (all-full-attention path, unchanged math):
%else
; Implements exact Qwen3 architecture:
%endif
;   - Q8_0 tied embeddings + LM head
;   - 28 layers: RMSNorm -> Q/K/V (shared Q8_0 activation quant) -> per-head
;     QK-Norm -> split-half NeoX RoPE -> KV-cache -> GQA Attention (16Q/8KV)
;     -> O-proj + residual -> RMSNorm -> SwiGLU MLP + residual
;   - Final RMSNorm -> tied LM head logits
; ============================================================================

bits 64
default rel

%include "constants.inc"

; AIOS: KV cache element type (FP16 with -DKV_F16, else FP32 as upstream)
%ifdef KV_F16
%define KV_SHIFT 1
%define KV_DOT   dot_f16_avx2
%define KV_SAXPY saxpy_f16_avx2
%else
%define KV_SHIFT 2
%define KV_DOT   dot_f32_avx2
%define KV_SAXPY saxpy_f32_avx2
%endif
global kv_elt_bytes

%ifndef FAST_Q8
%define FAST_Q8 1
%endif

%ifndef BARE_METAL
extern sys_openat_ro
extern sys_close
extern sys_fstat
extern sys_mmap_file
%endif
extern sys_mmap_anon
extern sys_munmap

extern init_rope_freqs
extern compute_rope_cs
extern rope_apply_avx2
extern rmsnorm_avx2
extern softmax_avx2
extern silu_mul_avx2
extern vec_add_avx2
%ifdef QWEN35
extern sigmoid_mul_avx2
extern scale_f32_avx2
extern conv4_step_avx2
extern deltanet_head_avx2
extern exp_f64
extern log_f64
%endif

extern matmul_set_fp16
extern dequant_row_q8_0
extern dot_f32_avx2
extern saxpy_f32_avx2

extern tp_matmul_q8_0
%ifdef QWEN35
%ifdef BARE_METAL
extern tp_deltanet
%endif
%endif
%if FAST_Q8
extern tp_gqa_attn
%endif

section .text

global g_ctx
global model_init
global model_free
global transformer_forward
%ifdef QWEN35
global deltanet_heads_range
%endif
%if FAST_Q8
global gqa_attn_range
%endif

; Helper macro: align reg up to 64 bytes
%macro ALIGN64 1
    add     %1, 63
    and     %1, -64
%endmacro

%ifdef QWEN35
; Debug tap: if CTX_TAP_FN != 0 call it as tap(layer=%1 (-1 embed, -2 final norm), pos, buf_x, dim, token)
%macro TAP_CALL 1
    cmp     qword [rbx + CTX_TAP_FN], 0
    je      %%skip
    mov     rdi, %1
    mov     rsi, [rsp + 0]
    mov     rdx, [rbx + CTX_BUF_X]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rsp + 128]
    call    [rbx + CTX_TAP_FN]
%%skip:
%endmacro

; ctx[%2] = header[%1] if non-zero else the legacy Qwen3 constant %3
%macro LOAD_TOK 3
    mov     eax, [r15 + %1]
    test    eax, eax
    jnz     %%ok
    mov     eax, %3
%%ok:
    mov     [rbx + %2], rax
%endmacro

%endif
; ----------------------------------------------------------------------------
; Linux:      int64 model_init(const char *path, uint64 max_seq_override)
; Bare-Metal: int64 model_init(const void *base, uint64 size, uint64 max_seq_override)
; Returns 0 on success, negative error code on failure.
; ----------------------------------------------------------------------------
model_init:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, STAT_SIZE + 8          ; 6 pushes (48) + 152 + 8(ret) = 208 (16B aligned)

    lea     rbx, [rel g_ctx]
%ifdef BARE_METAL
    mov     r15, rdi                    ; base ptr in physical RAM
    mov     r14, rsi                    ; size
    mov     r12, rdx                    ; max_seq_override
%else
    mov     r12, rsi                    ; max_seq_override

    ; Open .asmbin read-only
    call    sys_openat_ro
    test    rax, rax
    js      .err
    mov     r13, rax                    ; fd

    ; fstat to get file size
    mov     rdi, r13
    mov     rsi, rsp
    call    sys_fstat
    test    rax, rax
    js      .err_close
    mov     r14, [rsp + STAT_ST_SIZE]   ; file size

    ; mmap entire .asmbin read-only
    mov     rdi, r14
    mov     rsi, r13
    call    sys_mmap_file
    mov     r15, rax                    ; mmap_base

    ; Close fd (mmap keeps mapping alive)
    mov     rdi, r13
    call    sys_close

    test    r15, r15
    js      .err_mmap
%endif

%ifdef QWEN35
    ; Verify header magic and version (v3 only; fp16 vs fp32 scales from HDR_FLAGS)
%else
    ; Verify header magic and version (v1 fp32 scales or v2 fp16 scales)
%endif
    cmp     dword [r15 + HDR_MAGIC], ASMBIN_MAGIC
    jne     .err_bad_hdr
%ifdef QWEN35
    cmp     dword [r15 + HDR_VERSION], ASMBIN_VERSION_V3
%else
    mov     eax, [r15 + HDR_VERSION]
    cmp     eax, ASMBIN_VERSION
    je      .ver_ok
    cmp     eax, ASMBIN_VERSION_V2
%endif
    jne     .err_bad_hdr
%ifndef QWEN35
.ver_ok:
%endif
    xor     ecx, ecx
    test    dword [r15 + HDR_FLAGS], HDR_FLAG_FP16_SCALES
    setnz   cl
%ifndef QWEN35
    cmp     eax, ASMBIN_VERSION_V2
    jb      .store_scale_mode
    mov     ecx, 1
.store_scale_mode:
%endif
    mov     [rbx + CTX_FP16_SCALES], ecx
    mov     edx, 3
    add     edx, ecx                    ; 3 for fp32 (K>>3), 4 for fp16 (K>>4)
    mov     [rbx + CTX_SCALE_SHIFT], edx
    mov     edi, ecx
    call    matmul_set_fp16

    mov     [rbx + CTX_MMAP_BASE], r15
    mov     [rbx + CTX_MMAP_SIZE], r14

    ; Load scalar dimensions from header
    mov     eax, [r15 + HDR_DIM]
    mov     [rbx + CTX_DIM], rax
    mov     eax, [r15 + HDR_HIDDEN_DIM]
    mov     [rbx + CTX_HIDDEN_DIM], rax
    mov     eax, [r15 + HDR_N_LAYERS]
    mov     [rbx + CTX_N_LAYERS], rax
    mov     eax, [r15 + HDR_N_HEADS]
    mov     [rbx + CTX_N_HEADS], rax
    mov     eax, [r15 + HDR_N_KV_HEADS]
    mov     [rbx + CTX_N_KV_HEADS], rax
    mov     eax, [r15 + HDR_HEAD_DIM]
    mov     [rbx + CTX_HEAD_DIM], rax

    ; q_dim = n_heads * head_dim; kv_dim = n_kv_heads * head_dim
    mov     rax, [rbx + CTX_N_HEADS]
    imul    rax, [rbx + CTX_HEAD_DIM]
    mov     [rbx + CTX_Q_DIM], rax
    mov     rax, [rbx + CTX_N_KV_HEADS]
    imul    rax, [rbx + CTX_HEAD_DIM]
    mov     [rbx + CTX_KV_DIM], rax

    mov     eax, [r15 + HDR_VOCAB_SIZE]
    mov     [rbx + CTX_VOCAB_SIZE], rax
    mov     eax, [r15 + HDR_MAX_SEQ_LEN]
    test    r12, r12
    jz      .use_hdr_seq
    ; AIOS (M2.5): a non-zero override wins even above the header value (the v3 .asmbin
    ; stores the converter default 4096; Qwen3.5-2B config.json: max_position_embeddings 262144)
    mov     rax, r12
.use_hdr_seq:
    mov     [rbx + CTX_MAX_SEQ], rax

%ifndef QWEN35
    mov     rax, [r15 + HDR_LAYER_STRIDE]
    mov     [rbx + CTX_LAYER_STRIDE], rax
%endif
    mov     eax, [r15 + HDR_HT_MASK]
    mov     [rbx + CTX_HT_MASK], rax
    mov     eax, [r15 + HDR_EOS_1]
    mov     [rbx + CTX_EOS_1], rax
    mov     eax, [r15 + HDR_EOS_2]
    mov     [rbx + CTX_EOS_2], rax
    mov     eax, [r15 + HDR_RMS_EPS]
    mov     [rbx + CTX_RMS_EPS], eax
    mov     eax, [r15 + HDR_ROPE_THETA]
    mov     [rbx + CTX_ROPE_THETA], eax
%ifdef QWEN35

    ; Qwen3.5 header extras (0 in Qwen3 files -> legacy defaults)
    mov     eax, [r15 + HDR_ROT_DIM]
    test    eax, eax
    jnz     .rot_ok
    mov     eax, [r15 + HDR_HEAD_DIM]
.rot_ok:
    mov     [rbx + CTX_ROT_DIM], rax
    mov     eax, [r15 + HDR_FLAGS2]
    mov     [rbx + CTX_FLAGS2], rax
    xor     eax, eax
    test    dword [r15 + HDR_FLAGS2], HDR_FLAG2_ATTN_OUT_GATE
    jz      .no_gate_dim
    mov     rax, [rbx + CTX_Q_DIM]
.no_gate_dim:
    mov     [rbx + CTX_GATE_DIM], rax
    mov     eax, [r15 + HDR_N_LIN_V_HEADS]
    mov     [rbx + CTX_LIN_HEADS], rax
    mov     ecx, [r15 + HDR_LIN_V_DIM]
    mov     [rbx + CTX_LIN_DIM], rcx
    imul    rax, rcx
    mov     [rbx + CTX_VAL_DIM], rax            ; Hv * Dv
    mov     edx, [r15 + HDR_N_LIN_K_HEADS]
    imul    edx, [r15 + HDR_LIN_K_DIM]          ; Hk * Dk
    lea     rdx, [rax + rdx*2]
    mov     [rbx + CTX_CONV_DIM], rdx           ; 2*Hk*Dk + Hv*Dv
    mov     eax, [r15 + HDR_ARGMAX_LIMIT]
    test    eax, eax
    jnz     .samp_ok
    mov     eax, [r15 + HDR_VOCAB_SIZE]
.samp_ok:
    mov     [rbx + CTX_SAMPLE_N], rax
    xor     eax, eax
    cmp     dword [r15 + HDR_TOK_IM_START], 0
    setnz   al
    mov     [rbx + CTX_TPL_THINK_NL], rax
    LOAD_TOK HDR_TOK_IM_START,    CTX_TOK_IM_START,    TOK_IM_START
    LOAD_TOK HDR_TOK_IM_END,      CTX_TOK_IM_END,      TOK_IM_END
    LOAD_TOK HDR_TOK_THINK_START, CTX_TOK_THINK_START, TOK_THINK_START
    LOAD_TOK HDR_TOK_THINK_END,   CTX_TOK_THINK_END,   TOK_THINK_END
    LOAD_TOK HDR_TOK_USER,        CTX_TOK_USER,        TOK_USER
    LOAD_TOK HDR_TOK_ASSISTANT,   CTX_TOK_ASSISTANT,   TOK_ASSISTANT
    LOAD_TOK HDR_TOK_NL,          CTX_TOK_NL,          TOK_NL
    LOAD_TOK HDR_TOK_NL2,         CTX_TOK_NL2,         TOK_NL2
%endif

    ; Resolve pointers into mmap'd .asmbin
    mov     rax, [r15 + HDR_OFF_TOKENIZER]
    add     rax, r15                    ; tok_base
    mov     rcx, [rax + TOKH_OFF_B2T]
    add     rcx, rax
    mov     [rbx + CTX_B2T_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_OFFSETS]
    add     rcx, rax
    mov     [rbx + CTX_TOK_OFFS_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_BYTES]
    add     rcx, rax
    mov     [rbx + CTX_TOK_BYTES_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_HT]
    add     rcx, rax
    mov     [rbx + CTX_HT_PTR], rcx

    mov     rax, [r15 + HDR_OFF_EMBED]
    add     rax, r15
    mov     [rbx + CTX_EMB_PTR], rax
%ifdef QWEN35
    mov     rax, [r15 + HDR_OFF_LAYER_TABLE]
%else
    mov     rax, [r15 + HDR_OFF_LAYERS]
%endif
    add     rax, r15
%ifdef QWEN35
    mov     [rbx + CTX_LAYER_TABLE], rax
%else
    mov     [rbx + CTX_LAYERS_PTR], rax
%endif
    mov     rax, [r15 + HDR_OFF_FINAL_NORM]
    add     rax, r15
    mov     [rbx + CTX_FINAL_NORM_PTR], rax

%ifdef QWEN35
    ; Count full-attention / DeltaNet layers; refuse unknown types and unsupported DeltaNet shapes
    mov     rcx, [rbx + CTX_N_LAYERS]
    mov     rdx, [rbx + CTX_LAYER_TABLE]
    xor     r12d, r12d                  ; n_full
    xor     r13d, r13d                  ; n_lin
.type_chk_lp:
    test    rcx, rcx
    jz      .type_chk_done
    mov     eax, [rdx + DESC_TYPE]
    cmp     eax, LAYER_TYPE_FULL_ATTN
    jne     .t_not_full
    inc     r12
    jmp     .t_next
.t_not_full:
    cmp     eax, LAYER_TYPE_LINEAR_ATTN
    jne     .err_layer_type
    inc     r13
.t_next:
    add     rdx, DESC_STRIDE
    dec     rcx
    jmp     .type_chk_lp
.type_chk_done:
    mov     [rbx + CTX_N_FULL], r12
    mov     [rbx + CTX_N_LIN], r13
    test    r13, r13
    jz      .lin_cfg_ok
    ; DeltaNet kernel assumptions: Hk == Hv <= 32, Dk == Dv == 128, conv kernel 4
    mov     eax, [r15 + HDR_N_LIN_K_HEADS]
    cmp     eax, [r15 + HDR_N_LIN_V_HEADS]
    jne     .err_layer_type
    cmp     eax, 32
    ja      .err_layer_type
    test    eax, eax
    jz      .err_layer_type
    cmp     dword [r15 + HDR_LIN_K_DIM], 128
    jne     .err_layer_type
    cmp     dword [r15 + HDR_LIN_V_DIM], 128
    jne     .err_layer_type
    cmp     dword [r15 + HDR_CONV_K], 4
    jne     .err_layer_type
.lin_cfg_ok:

%endif
    ; Compute single anonymous mmap arena size for all activation buffers + KV cache
    ; First pass: compute exact aligned offsets (stored temporarily as relative offsets!)
    xor     rbp, rbp                    ; running offset = 0

    %macro ALLOC_BUF 2
        ALIGN64 rbp
        mov     [rbx + %1], rbp
        add     rbp, %2
    %endmacro

    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2                      ; dim * 4
    ALLOC_BUF CTX_BUF_X, rax
    ALLOC_BUF CTX_BUF_XB, rax

%ifdef QWEN35
    ; One projection region, contiguous so each layer's projections run as ONE fused matmul:
    ;   full attn: [q | k | v | gate]    DeltaNet: [qkv | z]   (sized for the larger)
%else
    ; Place buf_q, buf_k, buf_v CONTIGUOUSLY so Q+K+V execute in ONE fused matmul!
%endif
    mov     rax, [rbx + CTX_Q_DIM]
%ifdef QWEN35
    add     rax, [rbx + CTX_KV_DIM]
    add     rax, [rbx + CTX_KV_DIM]
    add     rax, [rbx + CTX_GATE_DIM]
    mov     rcx, [rbx + CTX_CONV_DIM]
    add     rcx, [rbx + CTX_VAL_DIM]
    cmp     rax, rcx
    jae     .proj_sz
    mov     rax, rcx
.proj_sz:
    shl     rax, 2
%else
    shl     rax, 2                      ; q_dim * 4
%endif
    ALLOC_BUF CTX_BUF_Q, rax
%ifdef QWEN35
    mov     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2
    add     rax, [rbx + CTX_BUF_Q]
    mov     [rbx + CTX_BUF_K], rax      ; (still an arena-relative offset here)
%endif
    mov     rax, [rbx + CTX_KV_DIM]
%ifdef QWEN35
    shl     rax, 2
    add     rax, [rbx + CTX_BUF_K]
    mov     [rbx + CTX_BUF_V], rax
%else
    shl     rax, 2                      ; kv_dim * 4
    ALLOC_BUF CTX_BUF_K, rax
    ALLOC_BUF CTX_BUF_V, rax
%endif

    mov     rax, [rbx + CTX_Q_DIM]
%ifdef QWEN35
    cmp     rax, [rbx + CTX_VAL_DIM]
    jae     .att_out_sz
    mov     rax, [rbx + CTX_VAL_DIM]
.att_out_sz:
    shl     rax, 2                      ; max(q_dim, Hv*Dv) * 4
%else
    shl     rax, 2                      ; q_dim * 4
%endif
    ALLOC_BUF CTX_BUF_ATT_OUT, rax

    mov     rax, [rbx + CTX_N_HEADS]
    imul    rax, [rbx + CTX_MAX_SEQ]
    shl     rax, 2                      ; n_heads * max_seq * 4 (private per-head scores)
    ALLOC_BUF CTX_BUF_ATT_SCR, rax

    ; Place buf_gate and buf_up CONTIGUOUSLY so gate+up execute in ONE fused matmul!
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    shl     rax, 2                      ; hidden_dim * 4
    ALLOC_BUF CTX_BUF_GATE, rax
    ALLOC_BUF CTX_BUF_UP, rax

    mov     rax, [rbx + CTX_VOCAB_SIZE]
    shl     rax, 2                      ; vocab_size * 4
    ALLOC_BUF CTX_BUF_LOGITS, rax
    ALLOC_BUF CTX_BUF_PROBS, rax

    ; Activation Q8_0 buffers (max K = max(dim, q_dim, hidden_dim) <= hidden_dim + 4096)
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    cmp     rax, [rbx + CTX_Q_DIM]
    jae     .kmax_ok
    mov     rax, [rbx + CTX_Q_DIM]
.kmax_ok:
    ALLOC_BUF CTX_BUF_XQ, rax           ; max_K int8
    mov     rcx, rax
    shr     rcx, 3                      ; (max_K / 32) * 4
    ALLOC_BUF CTX_BUF_XS, rcx

    ; RoPE buffers: inv_freq (head_dim/2 * 8), cos (head_dim/2 * 4), sin (head_dim/2 * 4)
    mov     rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2                      ; (head_dim/2) * 8 = head_dim * 4
    ALLOC_BUF CTX_BUF_INV_FREQ, rax
    shr     rax, 1                      ; (head_dim/2) * 4
    ALLOC_BUF CTX_BUF_COS, rax
    ALLOC_BUF CTX_BUF_SIN, rax

    ; Prompt & gen token buffers
    ALLOC_BUF CTX_BUF_PROMPT, MAX_PROMPT_TOKENS * 4
    ALLOC_BUF CTX_BUF_GEN,    MAX_GEN_TOKENS * 4

%ifdef QWEN35
    ; KV-cache (full-attention layers only): each is n_full * max_seq * kv_dim * 4 bytes
    mov     rax, [rbx + CTX_N_FULL]
%else
    ; KV-cache: each is n_layers * max_seq * kv_dim * 4 bytes
    mov     rax, [rbx + CTX_N_LAYERS]
%endif
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, KV_SHIFT               ; AIOS: 2 (f32) or 1 (KV_F16)
    ALLOC_BUF CTX_BUF_K_CACHE, rax
    ALLOC_BUF CTX_BUF_V_CACHE, rax

%ifdef QWEN35
    ; DeltaNet recurrent state S [n_lin][Hv][128][128] f32 + conv state [n_lin][3][conv_dim] f32,
    ; adjacent so ONE memset at pos == 0 clears both
    mov     rax, [rbx + CTX_LIN_HEADS]
    shl     rax, 16                     ; Hv * 128*128*4
    imul    rax, [rbx + CTX_N_LIN]
    ALLOC_BUF CTX_BUF_DN_STATE, rax
    mov     rax, [rbx + CTX_CONV_DIM]
    imul    rax, [rbx + CTX_N_LIN]
    imul    rax, 12                     ; 3 rows * 4 B
    ALLOC_BUF CTX_BUF_DN_CONV, rax
%endif
    ALIGN64 rbp
%ifdef QWEN35
    mov     rax, rbp
    sub     rax, [rbx + CTX_BUF_DN_STATE]
    mov     [rbx + CTX_DN_ZERO_BYTES], rax
    mov     eax, 1024
    ALLOC_BUF CTX_BUF_MISC, rax
    mov     rax, [rbx + CTX_CONV_DIM]
    shl     rax, 2
    ALLOC_BUF CTX_BUF_ONES, rax

    ALIGN64 rbp
%endif
    mov     [rbx + CTX_ARENA_SIZE], rbp

    ; Allocate arena via anonymous mmap
    mov     rdi, rbp
    call    sys_mmap_anon
    test    rax, rax
    js      .err_arena
    mov     [rbx + CTX_ARENA_BASE], rax

    ; Relocate all 20 CTX_BUF_* offsets -> absolute pointers
    lea     rcx, [rbx + CTX_BUF_X]
    mov     edx, 20
.reloc_lp:
    add     [rcx], rax
    add     rcx, 8
    dec     edx
    jnz     .reloc_lp
%ifdef QWEN35
    lea     rcx, [rbx + CTX_BUF_DN_STATE]
    mov     edx, CTX_N_RELOC2
.reloc2_lp:
    add     [rcx], rax
    add     rcx, 8
    dec     edx
    jnz     .reloc2_lp
%endif

%ifdef QWEN35
    ; ones[conv_dim] = 1.0f (silu(x) == silu_mul(x, ones))
    cld
    mov     rdi, [rbx + CTX_BUF_ONES]
    mov     rcx, [rbx + CTX_CONV_DIM]
    mov     eax, 0x3F800000
    rep stosd

    ; Precompute exact FP64 RoPE inverse frequencies (over the rotary dims only)
%else
    ; Precompute exact FP64 RoPE inverse frequencies
%endif
    mov     rdi, [rbx + CTX_BUF_INV_FREQ]
%ifdef QWEN35
    mov     rsi, [rbx + CTX_ROT_DIM]
%else
    mov     rsi, [rbx + CTX_HEAD_DIM]
%endif
    vmovss  xmm0, [rbx + CTX_ROPE_THETA]
    call    init_rope_freqs

    xor     eax, eax                    ; success!
    jmp     .ret

.err_arena:
    mov     rdi, r15
    mov     rsi, r14
    call    sys_munmap
    mov     rax, -4
    jmp     .ret
.err_bad_hdr:
    mov     rdi, r15
    mov     rsi, r14
    call    sys_munmap
    mov     rax, -3
    jmp     .ret
%ifdef QWEN35
.err_layer_type:
    mov     rdi, r15
    mov     rsi, r14
    call    sys_munmap
    mov     rax, -5                     ; unsupported layer type (callers print a message)
    jmp     .ret
%endif
%ifndef BARE_METAL
.err_mmap:
    mov     rax, -2
    jmp     .ret
.err_close:
    mov     rdi, r13
    call    sys_close
.err:
    mov     rax, -1
%endif
.ret:
    add     rsp, STAT_SIZE + 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void model_free(void)
; Unmaps arena and .asmbin file cleanly.
; ----------------------------------------------------------------------------
model_free:
    push    rbx
    lea     rbx, [rel g_ctx]
    mov     rdi, [rbx + CTX_ARENA_BASE]
    test    rdi, rdi
    jz      .unmap_model
    mov     rsi, [rbx + CTX_ARENA_SIZE]
    call    sys_munmap
.unmap_model:
    mov     rdi, [rbx + CTX_MMAP_BASE]
    test    rdi, rdi
    jz      .done
    mov     rsi, [rbx + CTX_MMAP_SIZE]
    call    sys_munmap
.done:
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void transformer_forward(uint32 token_id, uint64 pos, int compute_logits)
; rdi = token_id, rsi = pos, rdx = compute_logits (0 during non-final prefill)
; ----------------------------------------------------------------------------
transformer_forward:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
%ifdef QWEN35
    ; 6 pushes (48) + 264 + 8 (ret) = 320 -> 16B aligned
    sub     rsp, 264
%else
    ; Reserve 128 bytes of stack locals (6 pushes=48 + 128 + 8(ret)=184 -> use 136 for 16B align!)
    sub     rsp, 136
%endif
    ; Stack locals:
    ;   [rsp + 0]  = pos
    ;   [rsp + 8]  = compute_logits
    ;   [rsp + 16] = layer_idx l
%ifdef QWEN35
    ;   [rsp + 24] = full-attention layer ordinal (KV cache index)
    ;   [rsp + 32] = DeltaNet layer ordinal (state index)
    ;   [rsp + 40] = current layer type
%else
    ;   [rsp + 24] = p_in_norm
    ;   [rsp + 32] = p_q_norm
    ;   [rsp + 40] = p_k_norm
    ;   [rsp + 48] = p_post_norm
    ;   [rsp + 56] = p_q_proj
    ;   [rsp + 64] = p_k_proj
    ;   [rsp + 72] = p_v_proj
    ;   [rsp + 80] = p_o_proj
    ;   [rsp + 88] = p_gate_proj
    ;   [rsp + 96] = p_up_proj
    ;   [rsp + 104]= p_down_proj
%endif
    ;   [rsp + 112]= att_scale (float32)
    ;   [rsp + 116]= gqa_ratio (uint32)
%ifdef QWEN35
    ;   [rsp + 120]= layer_kv_base_bytes (full_ord * max_seq * kv_dim * 4)
    ;   [rsp + 128]= token id (for taps)
    ;   [rsp + SLOT(i)] = absolute pointer of descriptor slot i (0..11), see constants.inc DSLOT_*
%define SLOT(i) (136 + 8*(i))
%else
    ;   [rsp + 120]= layer_kv_base_bytes (l * max_seq * kv_dim * 4)
%endif

    mov     [rsp + 0], rsi              ; pos
    mov     [rsp + 8], rdx              ; compute_logits
%ifdef QWEN35
    mov     [rsp + 128], rdi            ; token
%endif
    lea     rbx, [rel g_ctx]

%ifdef QWEN35
    ; Start of a sequence: clear the DeltaNet state + conv history (covers /clear, ctx-full, bench)
    test    rsi, rsi
    jnz     .no_state_zero
    mov     rcx, [rbx + CTX_DN_ZERO_BYTES]
    shr     rcx, 3
    jz      .no_state_zero
    cld
    mov     rdi, [rbx + CTX_BUF_DN_STATE]
    xor     eax, eax
    rep stosq
    mov     rdi, [rsp + 128]
.no_state_zero:

%endif
    ; Precompute att_scale = 1.0f / sqrt(head_dim) and gqa_ratio = n_heads / n_kv_heads
    mov     rax, [rbx + CTX_HEAD_DIM]
    vcvtsi2ss xmm0, xmm0, rax
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel t_one]
    vdivss  xmm1, xmm1, xmm0
    vmovss  [rsp + 112], xmm1

    mov     rax, [rbx + CTX_N_HEADS]
    xor     edx, edx
    div     qword [rbx + CTX_N_KV_HEADS]
    mov     [rsp + 116], eax

    ; 1. Embedding lookup: dequantize row token_id of embed_tokens into buf_x
    mov     rax, [rbx + CTX_DIM]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     rdx, rax
    shr     rdx, cl
    add     rdx, rax                    ; emb_row_stride = dim + (dim >> scale_shift)
    imul    rdi, rdx
    add     rdi, [rbx + CTX_EMB_PTR]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, rax                    ; dim
    call    dequant_row_q8_0
%ifdef QWEN35
    TAP_CALL -1
%endif

%ifdef QWEN35
    ; 2. Compute RoPE cos/sin tables ONCE for this position (reused by all layers!)
%else
    ; 2. Compute RoPE cos/sin tables ONCE for this position (reused by all 28 layers!)
%endif
    mov     rdi, [rbx + CTX_BUF_COS]
    mov     rsi, [rbx + CTX_BUF_SIN]
    mov     rdx, [rbx + CTX_BUF_INV_FREQ]
%ifdef QWEN35
    mov     rcx, [rbx + CTX_ROT_DIM]
    shr     rcx, 1                      ; half_dim of the rotary part
%else
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shr     rcx, 1                      ; half_dim
%endif
    mov     r8,  [rsp + 0]              ; pos
    call    compute_rope_cs

    ; 3. Transformer Layer Loop (l = 0 .. n_layers - 1)
    mov     qword [rsp + 16], 0
%ifdef QWEN35
    mov     qword [rsp + 24], 0
    mov     qword [rsp + 32], 0
%endif

.layer_lp:
    mov     rax, [rsp + 16]
    cmp     rax, [rbx + CTX_N_LAYERS]
    jae     .layers_done

%ifdef QWEN35
    ; Load the layer's tensor pointers from its v3 descriptor (absolute = file base + offset)
    imul    rax, rax, DESC_STRIDE
    add     rax, [rbx + CTX_LAYER_TABLE]
    mov     rdx, [rbx + CTX_MMAP_BASE]
    mov     ecx, [rax + DESC_TYPE]
    mov     [rsp + 40], rcx
%assign i 0
%rep N_DESC_SLOTS
    mov     rcx, [rax + DESC_OFFS + 8*i]
    add     rcx, rdx
    mov     [rsp + SLOT(i)], rcx
%assign i i+1
%endrep
%else
    ; Compute all 11 tensor pointers for layer l (matching export_weights.py layout)
    imul    rax, [rbx + CTX_LAYER_STRIDE]
    add     rax, [rbx + CTX_LAYERS_PTR]
%endif

%ifndef QWEN35
    mov     [rsp + 24], rax             ; in_norm
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + 32], rax             ; q_norm
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + 40], rax             ; k_norm
    add     rax, rcx
    mov     [rsp + 48], rax             ; post_norm
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    add     rax, rcx

    ; Helper: row_stride(K) = K + (K >> scale_shift) [3 for v1 fp32, 4 for v2 fp16]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     r12, [rbx + CTX_DIM]
    mov     r13, r12
    shr     r13, cl
    add     r12, r13                    ; r12 = dim_stride (1152 v1 / 1088 v2)

    ALIGN64 rax
    mov     [rsp + 56], rax             ; q_proj [q_dim, dim]
    mov     rcx, [rbx + CTX_Q_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 64], rax             ; k_proj [kv_dim, dim]
    mov     rcx, [rbx + CTX_KV_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 72], rax             ; v_proj [kv_dim, dim]
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 80], rax             ; o_proj [dim, q_dim]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     r13, [rbx + CTX_Q_DIM]
    mov     rdx, r13
    shr     rdx, cl
    add     r13, rdx                    ; q_dim_stride (2304 v1 / 2176 v2)
    imul    r13, [rbx + CTX_DIM]
    add     rax, r13

    ALIGN64 rax
    mov     [rsp + 88], rax             ; gate_proj [hidden_dim, dim]
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 96], rax             ; up_proj [hidden_dim, dim]
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 104], rax            ; down_proj [dim, hidden_dim]

%endif
    ; --- 3a. Input RMSNorm: buf_xb = rmsnorm(buf_x, in_norm) ---
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_X]
%ifdef QWEN35
    mov     rdx, [rsp + SLOT(0)]
%else
    mov     rdx, [rsp + 24]
%endif
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

%ifdef QWEN35
    cmp     qword [rsp + 40], LAYER_TYPE_LINEAR_ATTN
    jne     .full_attn
    ; --- Gated DeltaNet mixer: BUF_XB (normed) -> BUF_XB (out_proj result) ---
    lea     rdi, [rsp + SLOT(0)]
    mov     rsi, [rsp + 32]
    call    deltanet_layer
    inc     qword [rsp + 32]
    jmp     .attn_residual

.full_attn:
    ; --- 3b. Fused Q + K + V (+ output gate) projection in ONE parallel matmul ---
    mov     rdi, [rsp + SLOT(2)]        ; W_proj rows [q | k | v | gate?]
%else
    ; --- 3b. Fused Q + K + V projection in ONE parallel matmul (zero padding between q/k/v!) ---
    mov     rdi, [rsp + 56]             ; q_proj (contiguous with k_proj, v_proj)
%endif
    mov     rsi, [rbx + CTX_BUF_XB]
%ifdef QWEN35
    mov     rdx, [rbx + CTX_BUF_Q]      ; buf_q (contiguous with buf_k, buf_v, gate)
%else
    mov     rdx, [rbx + CTX_BUF_Q]      ; buf_q (contiguous with buf_k, buf_v)
%endif
    mov     rcx, [rbx + CTX_Q_DIM]
    add     rcx, [rbx + CTX_KV_DIM]
%ifdef QWEN35
    add     rcx, [rbx + CTX_KV_DIM]
    add     rcx, [rbx + CTX_GATE_DIM]   ; M = q_dim + 2*kv_dim (+ q_dim if output gate)
%else
    add     rcx, [rbx + CTX_KV_DIM]     ; M = q_dim + 2*kv_dim (4096)
%endif
    mov     r8,  [rbx + CTX_DIM]        ; K = dim (1024)
    call    tp_matmul_q8_0

    ; --- 3c. Qwen3 QK-Norm (RMSNorm per head over head_dim, in-place) ---
    mov     r12, [rbx + CTX_BUF_Q]
    mov     r13, [rbx + CTX_N_HEADS]
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2                      ; head_bytes
.qnorm_lp:
    mov     rdi, r12
    mov     rsi, r12
%ifdef QWEN35
    mov     rdx, [rsp + SLOT(6)]        ; q_norm
%else
    mov     rdx, [rsp + 32]             ; q_norm
%endif
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    add     r12, r14
    dec     r13
    jnz     .qnorm_lp

    mov     r12, [rbx + CTX_BUF_K]
    mov     r13, [rbx + CTX_N_KV_HEADS]
.knorm_lp:
    mov     rdi, r12
    mov     rsi, r12
%ifdef QWEN35
    mov     rdx, [rsp + SLOT(7)]        ; k_norm
%else
    mov     rdx, [rsp + 40]             ; k_norm
%endif
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    add     r12, r14
    dec     r13
    jnz     .knorm_lp

%ifdef QWEN35
    ; --- 3d. Split-half RoPE on the first rot_dim dims of every Q and K head ---
%else
    ; --- 3d. Split-half RoPE on Q and K ---
%endif
    mov     rdi, [rbx + CTX_BUF_Q]
%ifndef QWEN35
    mov     rsi, [rbx + CTX_BUF_COS]
    mov     rdx, [rbx + CTX_BUF_SIN]
%endif
    mov     rcx, [rbx + CTX_N_HEADS]
%ifdef QWEN35
    call    rope_heads
%else
    mov     r8,  [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2

%endif
    mov     rdi, [rbx + CTX_BUF_K]
%ifndef QWEN35
    mov     rsi, [rbx + CTX_BUF_COS]
    mov     rdx, [rbx + CTX_BUF_SIN]
%endif
    mov     rcx, [rbx + CTX_N_KV_HEADS]
%ifdef QWEN35
    call    rope_heads
%else
    mov     r8,  [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2
%endif

%ifdef QWEN35
    ; --- 3e. Store K, V at contiguous cache[full_ord, kv_h, pos, head_dim] ---
    ; layer_kv_base_bytes = full_ord * max_seq * kv_dim * 4
    mov     rax, [rsp + 24]             ; full-attention layer ordinal
    inc     qword [rsp + 24]
%else
    ; --- 3e. Store K, V at contiguous cache[l, kv_h, pos, head_dim] ---
    ; layer_kv_base_bytes = l * max_seq * kv_dim * 4
    mov     rax, [rsp + 16]             ; l
%endif
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, KV_SHIFT
    mov     [rsp + 120], rax

%if FAST_Q8
    ; Head-major contiguous KV layout: cache[l, kv_h, pos, head_dim]
    ; Eliminates 4KB L1D set aliasing & TLB thrashing across timesteps!
%ifdef KV_F16
    ; AIOS: K/V rows are stored as FP16 (vcvtps2ph, round to nearest even)
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2                      ; f32 head_bytes in buf_k / buf_v
    mov     r15, [rbx + CTX_MAX_SEQ]
    imul    r15, [rbx + CTX_HEAD_DIM]
    shl     r15, 1                      ; kv_head_stride = max_seq * head_dim * 2
    mov     rcx, [rsp + 0]              ; pos
    imul    rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 1                      ; pos_off = pos * head_dim * 2
    add     rax, rcx
    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rax
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rax
    mov     rsi, [rbx + CTX_BUF_K]
    mov     r8,  [rbx + CTX_BUF_V]
    mov     r12, [rbx + CTX_N_KV_HEADS]
.kvh_lp:
    xor     r9d, r9d                    ; element index, 8 per step
.kvh_cvt:
    cmp     r9, [rbx + CTX_HEAD_DIM]
    jae     .kvh_next
    vmovups ymm0, [rsi + r9*4]
    vmovups ymm1, [r8 + r9*4]
    vcvtps2ph [rdi + r9*2], ymm0, 0
    vcvtps2ph [rdx + r9*2], ymm1, 0
    add     r9, 8
    jmp     .kvh_cvt
.kvh_next:
    add     rsi, r14
    add     r8,  r14
    add     rdi, r15
    add     rdx, r15
    dec     r12
    jnz     .kvh_lp
    jmp     .kv_stored
%endif
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2                      ; head_bytes (512)
    mov     r15, [rbx + CTX_MAX_SEQ]
    imul    r15, r14                    ; kv_head_stride = max_seq * head_bytes
    mov     rcx, [rsp + 0]              ; pos
    imul    rcx, r14                    ; pos_off = pos * head_bytes
    add     rax, rcx                    ; layer_kv_base + pos_off

    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rax
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rax
    mov     rsi, [rbx + CTX_BUF_K]
    mov     r8,  [rbx + CTX_BUF_V]
    mov     r12, [rbx + CTX_N_KV_HEADS]
.kv_head_lp:
    xor     r9d, r9d
.kv_copy_lp:
    cmp     r9, r14
    jae     .kv_head_next
    vmovups ymm0, [rsi + r9]
    vmovups ymm1, [r8 + r9]
    vmovups [rdi + r9], ymm0
    vmovups [rdx + r9], ymm1
    add     r9, 32
    jmp     .kv_copy_lp
.kv_head_next:
    add     rsi, r14
    add     r8,  r14
    add     rdi, r15
    add     rdx, r15
    dec     r12
    jnz     .kv_head_lp
.kv_stored:

    ; --- 3f. Parallel GQA Attention over heads h = 0 .. n_heads - 1 ---
    mov     rdi, [rsp + 120]            ; layer_kv_base
    mov     rsi, [rsp + 0]              ; pos
    mov     edx, [rsp + 116]            ; gqa_ratio (upper 32 bits)
    shl     rdx, 32
    mov     eax, [rsp + 112]            ; att_scale float32 bits (lower 32 bits)
    or      rdx, rax
    mov     rcx, [rbx + CTX_N_HEADS]
    call    tp_gqa_attn
%else
    ; Legacy layout: cache[l, pos, kv_h, head_dim]
    mov     rcx, [rsp + 0]              ; pos
    imul    rcx, [rbx + CTX_KV_DIM]
    shl     rcx, 2
    add     rax, rcx                    ; total byte offset into k_cache / v_cache

    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rax
    mov     rsi, [rbx + CTX_BUF_K]
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rax
    mov     r8,  [rbx + CTX_BUF_V]
    mov     rcx, [rbx + CTX_KV_DIM]
    shl     rcx, 2                      ; kv_bytes
    xor     r9d, r9d
.kv_copy_lp:
    cmp     r9, rcx
    jae     .kv_copy_done
    vmovups ymm0, [rsi + r9]
    vmovups ymm1, [r8 + r9]
    vmovups [rdi + r9], ymm0
    vmovups [rdx + r9], ymm1
    add     r9, 32
    jmp     .kv_copy_lp
.kv_copy_done:

    ; --- 3f. GQA Attention over all heads h = 0 .. n_heads - 1 ---
    xor     r12d, r12d                  ; h = 0
.head_lp:
    cmp     r12, [rbx + CTX_N_HEADS]
    jae     .att_done

    ; kv_h = h / gqa_ratio
    mov     eax, r12d
    xor     edx, edx
    div     dword [rsp + 116]           ; eax = kv_h
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2                      ; kv_h_bytes = kv_h * head_dim * 4
    mov     r13, rax                    ; r13 = kv_h_bytes

    ; q_head = buf_q + h * head_dim * 4
    mov     rax, r12
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    mov     r14, [rbx + CTX_BUF_Q]
    add     r14, rax                    ; r14 = q_head

    ; Compute attention scores[0..pos]: dot(q_head, k_cache[l, t, kv_h]) * att_scale
    mov     r15, [rbx + CTX_BUF_K_CACHE]
    add     r15, [rsp + 120]            ; layer_k_base
    add     r15, r13                    ; k_ptr for t = 0
    mov     rbp, [rbx + CTX_KV_DIM]
    shl     rbp, 2                      ; step per timestep t = kv_dim * 4

    xor     ecx, ecx                    ; t = 0
    mov     r11, [rbx + CTX_BUF_ATT_SCR]
.score_lp:
    cmp     rcx, [rsp + 0]
    ja      .scores_done
    push    rcx
    push    r11                         ; 2 pushes -> 16B aligned
    mov     rdi, r14                    ; q_head
    mov     rsi, r15                    ; k_vec
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    dot_f32_avx2                ; xmm0 = dot
    pop     r11
    pop     rcx
    vmulss  xmm0, xmm0, [rsp + 112]     ; * att_scale
    vmovss  [r11 + rcx*4], xmm0
    add     r15, rbp
    inc     rcx
    jmp     .score_lp

.scores_done:
    ; Softmax scores[0..pos] (length = pos + 1)
    mov     rdi, [rbx + CTX_BUF_ATT_SCR]
    mov     rsi, [rsp + 0]
    inc     rsi
    call    softmax_avx2

    ; Zero att_head = buf_att_out + h * head_dim * 4
    mov     rax, r12
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    mov     r14, [rbx + CTX_BUF_ATT_OUT]
    add     r14, rax                    ; r14 = att_head

    mov     rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.zero_att:
    cmp     rax, rcx
    jae     .val_accum
    vmovups [r14 + rax], ymm0
    add     rax, 32
    jmp     .zero_att

.val_accum:
    ; Weighted sum of V vectors: att_head += probs[t] * v_cache[l, t, kv_h]
    mov     r15, [rbx + CTX_BUF_V_CACHE]
    add     r15, [rsp + 120]
    add     r15, r13                    ; v_ptr for t = 0
    xor     ecx, ecx
    mov     r11, [rbx + CTX_BUF_ATT_SCR]
.val_lp:
    cmp     rcx, [rsp + 0]
    ja      .head_next
    vmovss  xmm0, [r11 + rcx*4]         ; prob[t]
    push    rcx
    push    r11
    mov     rdi, r14                    ; att_head
    mov     rsi, r15                    ; v_vec
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    saxpy_f32_avx2
    pop     r11
    pop     rcx
    add     r15, rbp
    inc     rcx
    jmp     .val_lp

.head_next:
    inc     r12
    jmp     .head_lp
%endif

.att_done:
%ifdef QWEN35
    ; --- 3f'. Output gate (Qwen3.5): att_out *= sigmoid(gate), gate rows follow v in the proj region ---
    cmp     qword [rbx + CTX_GATE_DIM], 0
    je      .no_out_gate
    mov     rdi, [rbx + CTX_BUF_ATT_OUT]
    mov     rsi, [rbx + CTX_KV_DIM]
    shl     rsi, 2
    add     rsi, [rbx + CTX_BUF_V]
    mov     rdx, [rbx + CTX_Q_DIM]
    call    sigmoid_mul_avx2
.no_out_gate:
%endif
    ; --- 3g. Output projection + residual: buf_x += o_proj @ buf_att_out ---
%ifdef QWEN35
    mov     rdi, [rsp + SLOT(3)]        ; o_proj
%else
    mov     rdi, [rsp + 80]             ; o_proj
%endif
    mov     rsi, [rbx + CTX_BUF_ATT_OUT]
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_Q_DIM]
    call    tp_matmul_q8_0

%ifdef QWEN35
.attn_residual:
%endif
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2

    ; --- 3h. Post-attention RMSNorm + SwiGLU MLP + residual ---
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_X]
%ifdef QWEN35
    mov     rdx, [rsp + SLOT(1)]        ; post_norm
%else
    mov     rdx, [rsp + 48]             ; post_norm
%endif
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

    ; Fused gate_proj + up_proj in ONE parallel matmul (contiguous in weights & arena!)
%ifdef QWEN35
    mov     rdi, [rsp + SLOT(4)]        ; gate_proj (contiguous with up_proj)
%else
    mov     rdi, [rsp + 88]             ; gate_proj (contiguous with up_proj)
%endif
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_GATE]   ; buf_gate (contiguous with buf_up)
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    add     rcx, rcx                    ; M = 2 * hidden_dim (6144)
    mov     r8,  [rbx + CTX_DIM]        ; K = dim (1024)
    call    tp_matmul_q8_0

    ; SwiGLU: buf_gate = silu(buf_gate) * buf_up
    mov     rdi, [rbx + CTX_BUF_GATE]
    mov     rsi, [rbx + CTX_BUF_UP]
    mov     rdx, [rbx + CTX_HIDDEN_DIM]
    call    silu_mul_avx2

    ; down_proj + residual
%ifdef QWEN35
    mov     rdi, [rsp + SLOT(5)]        ; down_proj
%else
    mov     rdi, [rsp + 104]            ; down_proj
%endif
    mov     rsi, [rbx + CTX_BUF_GATE]
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_HIDDEN_DIM]
    call    tp_matmul_q8_0

    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2

%ifdef QWEN35
    TAP_CALL [rsp + 16]
%endif
    inc     qword [rsp + 16]
    jmp     .layer_lp

.layers_done:
    cmp     qword [rsp + 8], 0
    je      .done

    ; 4. Final RMSNorm + tied LM-head projection -> buf_logits
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_FINAL_NORM_PTR]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
%ifdef QWEN35
    TAP_CALL -2
%endif

    mov     rdi, [rbx + CTX_EMB_PTR]    ; tied embeddings!
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_BUF_LOGITS]
    mov     rcx, [rbx + CTX_VOCAB_SIZE]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0

.done:
%ifdef QWEN35
    add     rsp, 264
%else
    add     rsp, 136
%endif
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
%ifdef QWEN35
    pop     rbx
    ret


; ----------------------------------------------------------------------------
; void rope_heads(float *vec /*rdi*/, uint64 n_heads /*rcx*/)
; Split-half RoPE on the first rot_dim dims of each head (stride head_dim); cos/sin from g_ctx.
; ----------------------------------------------------------------------------
rope_heads:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15                         ; 5 pushes + ret = 48 -> 16B aligned
    lea     rbx, [rel g_ctx]
    mov     r12, rdi
    mov     r13, rcx
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2
.lp:
    test    r13, r13
    jz      .done
    mov     rdi, r12
    mov     rsi, [rbx + CTX_BUF_COS]
    mov     rdx, [rbx + CTX_BUF_SIN]
    mov     ecx, 1
    mov     r8,  [rbx + CTX_ROT_DIM]
    call    rope_apply_avx2
    add     r12, r14
    dec     r13
    jmp     .lp
.done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
%endif
    pop     rbx
    ret

%ifdef QWEN35
; ----------------------------------------------------------------------------
; void l2norm128(float *x /*rdi*/):  x *= 1/sqrt(sum(x^2) + 1e-6)   (n = 128, spec C.3)
; ----------------------------------------------------------------------------
l2norm128:
    push    rbx
    mov     rbx, rdi
    mov     rsi, rdi
    mov     edx, 128
    call    dot_f32_avx2                ; xmm0 = sum(x^2)
    vaddss  xmm0, xmm0, [rel t_l2eps]
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel t_one]
    vdivss  xmm0, xmm1, xmm0
    mov     rdi, rbx
    mov     esi, 128
    call    scale_f32_avx2
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void deltanet_layer(const u64 *slots /*rdi: 12 absolute descriptor-slot pointers*/, u64 lin_ord /*rsi*/)
; Gated DeltaNet mixer (spec C), BSP only. In: BUF_XB = normed input. Out: BUF_XB = out_proj result
; (residual not added). Slots: 2 W_proj[qkv|z] 3 W_out 6 conv_w 7 negExpA 8 dt_bias 9 gnorm_w 10 W_b 11 W_a.
; ----------------------------------------------------------------------------
deltanet_layer:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 56                     ; 6 pushes + 56 + ret = 112 -> 16B aligned
    ; [rsp + 24] dec f32, [rsp + 28] beta f32, [rsp + 32] sp f64, [rsp + 40] qscale f32, [rsp + 48] row bytes
    lea     rbx, [rel g_ctx]
    mov     rbp, rdi
    mov     r13, rsi                    ; lin_ord

    ; q scale = (float)(1.0 / sqrt(Dk))   (f64 math like the mirror)
    vcvtsi2sd xmm0, xmm0, qword [rbx + CTX_LIN_DIM]
    vsqrtsd xmm0, xmm0, xmm0
    vmovsd  xmm1, [rel d_one]
    vdivsd  xmm1, xmm1, xmm0
    vcvtsd2ss xmm1, xmm1, xmm1
    vmovss  [rsp + 40], xmm1

    ; [qkv | z] = W_proj @ xb  (one fused matmul)
    mov     rdi, [rbp + 8*2]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_Q]
    mov     rcx, [rbx + CTX_CONV_DIM]
    add     rcx, [rbx + CTX_VAL_DIM]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0

    ; a[h] = W_a[h] . xb, b[h] = W_b[h] . xb  (fp32 weights, fp32 dots) -> MISC[0..], MISC[128..]
    mov     r14, [rbp + 8*11]           ; W_a row ptr
    mov     r15, [rbp + 8*10]           ; W_b row ptr
    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2
    mov     [rsp + 48], rax             ; row bytes
    xor     r12d, r12d
.ab_lp:
    cmp     r12, [rbx + CTX_LIN_HEADS]
    jae     .ab_done
    mov     rdi, r14
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    dot_f32_avx2
    mov     rax, [rbx + CTX_BUF_MISC]
    vmovss  [rax + r12*4], xmm0
    mov     rdi, r15
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    dot_f32_avx2
    mov     rax, [rbx + CTX_BUF_MISC]
    vmovss  [rax + 128 + r12*4], xmm0
    add     r14, [rsp + 48]
    add     r15, [rsp + 48]
    inc     r12
    jmp     .ab_lp
.ab_done:

    ; causal depthwise conv (k=4) over the 6144 qkv channels, then SiLU
    mov     rsi, [rbx + CTX_CONV_DIM]
    imul    rsi, r13
    imul    rsi, 12                     ; lin_ord * conv_dim * 3 rows * 4 B
    add     rsi, [rbx + CTX_BUF_DN_CONV]
    mov     rdi, [rbx + CTX_BUF_Q]
    mov     rdx, [rbp + 8*6]
    mov     rcx, [rbx + CTX_CONV_DIM]
    call    conv4_step_avx2
    mov     rdi, [rbx + CTX_BUF_Q]
    mov     rsi, [rbx + CTX_BUF_ONES]
    mov     rdx, [rbx + CTX_CONV_DIM]
    call    silu_mul_avx2

    ; heads are independent: l2norm, g/beta, recurrence, gated rmsnorm -> BUF_ATT_OUT (SMP job kind K=1)
    lea     rax, [rel g_dn_job]
    mov     [rax + 0], r13              ; lin_ord
    mov     rcx, [rbp + 8*7]
    mov     [rax + 8], rcx              ; negExpA
    mov     rcx, [rbp + 8*8]
    mov     [rax + 16], rcx             ; dt_bias
    mov     rcx, [rbp + 8*9]
    mov     [rax + 24], rcx             ; gnorm_w
    mov     ecx, [rsp + 40]
    mov     [rax + 32], ecx             ; q scale
    mov     rdi, [rbx + CTX_LIN_HEADS]
%ifdef BARE_METAL
    call    tp_deltanet
%else
    mov     rsi, rdi
    xor     edi, edi
    call    deltanet_heads_range
%endif
    mov     r14, [rbx + CTX_CONV_DIM]
    shl     r14, 2
    add     r14, [rbx + CTX_BUF_Q]      ; z region
    mov     rdi, r14
    mov     rsi, [rbx + CTX_BUF_ATT_OUT]
    mov     rdx, [rbx + CTX_VAL_DIM]
    call    silu_mul_avx2               ; z = silu(z) * on

    mov     rdi, [rbp + 8*3]            ; out_proj
    mov     rsi, r14
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_VAL_DIM]
    call    tp_matmul_q8_0

    add     rsp, 56
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret


; ----------------------------------------------------------------------------
; void deltanet_heads_range(uint64 h0 /*rdi*/, uint64 h1 /*rsi*/)   (thread-safe, heads are independent)
; Per head h in [h0,h1): l2norm q,k ; g, beta ; state recurrence ; gated-norm rmsnorm -> BUF_ATT_OUT[h*128].
; Job parameters from g_dn_job (written by deltanet_layer), the rest from g_ctx.
; ----------------------------------------------------------------------------
deltanet_heads_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 584                    ; 6 pushes + 584 + ret = 640 -> 16B aligned
    ; [rsp + 0] dec f32, [rsp + 4] beta f32, [rsp + 64..575] delta scratch
    lea     rbx, [rel g_ctx]
    lea     rbp, [rel g_dn_job]
    mov     r12, rdi                    ; h
    mov     [rsp + 8], rsi              ; h1
    mov     r13, [rbp + 0]              ; lin_ord
.hd_lp:
    cmp     r12, [rsp + 8]
    jae     .hd_done
    mov     r14, r12
    shl     r14, 9                      ; h * 128 * 4
    add     r14, [rbx + CTX_BUF_Q]      ; q_h
    mov     r15, [rbx + CTX_VAL_DIM]    ; key_dim == value_dim
    shl     r15, 2
    add     r15, r14                    ; k_h
    mov     rdi, r14
    call    l2norm128
    mov     rdi, r14
    mov     esi, 128
    vmovss  xmm0, [rbp + 32]
    call    scale_f32_avx2
    mov     rdi, r15
    call    l2norm128

    ; sp = a + dt_bias ; softplus = sp > 20 ? sp : log(1 + exp(sp))  (f64)
    mov     rax, [rbx + CTX_BUF_MISC]
    vmovss  xmm0, [rax + r12*4]
    vcvtss2sd xmm0, xmm0, xmm0
    mov     rax, [rbp + 16]
    vmovss  xmm1, [rax + r12*4]
    vcvtss2sd xmm1, xmm1, xmm1
    vaddsd  xmm0, xmm0, xmm1
    vucomisd xmm0, [rel d_twenty]
    ja      .sp_done                    ; softplus(sp) = sp
    call    exp_f64
    vaddsd  xmm0, xmm0, [rel d_one]
    call    log_f64
.sp_done:
    mov     rax, [rbp + 8]
    vmovss  xmm1, [rax + r12*4]
    vcvtss2sd xmm1, xmm1, xmm1          ; negExpA[h] = -exp(A_log)
    vmulsd  xmm0, xmm0, xmm1            ; g
    call    exp_f64
    vcvtsd2ss xmm0, xmm0, xmm0
    vmovss  [rsp + 0], xmm0             ; dec = exp(g)
    mov     rax, [rbx + CTX_BUF_MISC]
    vmovss  xmm0, [rax + 128 + r12*4]
    vcvtss2sd xmm0, xmm0, xmm0
    vxorpd  xmm0, xmm0, [rel d_signmask]
    call    exp_f64                     ; exp(-b)
    vaddsd  xmm0, xmm0, [rel d_one]
    vmovsd  xmm1, [rel d_one]
    vdivsd  xmm1, xmm1, xmm0
    vcvtsd2ss xmm1, xmm1, xmm1
    vmovss  [rsp + 4], xmm1             ; beta = sigmoid(b)

    mov     rdi, r13
    imul    rdi, [rbx + CTX_LIN_HEADS]
    add     rdi, r12
    shl     rdi, 16                     ; (lin_ord*H + h) * 64 KiB
    add     rdi, [rbx + CTX_BUF_DN_STATE]
    mov     rsi, r14                    ; q
    mov     rdx, r15                    ; k
    mov     rcx, [rbx + CTX_VAL_DIM]
    shl     rcx, 2
    add     rcx, r15                    ; v_h
    mov     r8, r12
    shl     r8, 9
    add     r8, [rbx + CTX_BUF_ATT_OUT]
    mov     r14, r8                     ; o_h
    lea     r9, [rsp + 64]
    vmovss  xmm0, [rsp + 0]
    vmovss  xmm1, [rsp + 4]
    call    deltanet_head_avx2

    ; gated norm part 1: o_h = rmsnorm(o_h) * w  (the silu(z) factor is applied after the join)
    mov     rdi, r14
    mov     rsi, r14
    mov     rdx, [rbp + 24]
    mov     ecx, 128
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    inc     r12
    jmp     .hd_lp
.hd_done:
    add     rsp, 584
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

%endif
%ifdef KV_F16
; ----------------------------------------------------------------------------
; AIOS: FP16 KV cache helpers (F16C). Same contract as dot_f32_avx2 / saxpy_f32_avx2,
; but b / src are FP16. n multiple of 32.
; float dot_f16_avx2(const float *a, const uint16 *b, uint64 n) -> xmm0
; ----------------------------------------------------------------------------
dot_f16_avx2:
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .red
    vcvtph2ps ymm4, [rsi + rax*2]
    vcvtph2ps ymm5, [rsi + rax*2 + 16]
    vcvtph2ps ymm6, [rsi + rax*2 + 32]
    vcvtph2ps ymm7, [rsi + rax*2 + 48]
    vfmadd231ps ymm0, ymm4, [rdi + rax*4]
    vfmadd231ps ymm1, ymm5, [rdi + rax*4 + 32]
    vfmadd231ps ymm2, ymm6, [rdi + rax*4 + 64]
    vfmadd231ps ymm3, ymm7, [rdi + rax*4 + 96]
    add     rax, 32
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

; void saxpy_f16_avx2(float *dst, const uint16 *src, uint64 n, float alpha): dst[i] += alpha * src[i]
saxpy_f16_avx2:
    vbroadcastss ymm0, xmm0
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .done
    vcvtph2ps ymm5, [rsi + rax*2]
    vcvtph2ps ymm6, [rsi + rax*2 + 16]
    vcvtph2ps ymm7, [rsi + rax*2 + 32]
    vcvtph2ps ymm8, [rsi + rax*2 + 48]
    vmovups ymm1, [rdi + rax*4]
    vmovups ymm2, [rdi + rax*4 + 32]
    vmovups ymm3, [rdi + rax*4 + 64]
    vmovups ymm4, [rdi + rax*4 + 96]
    vfmadd231ps ymm1, ymm0, ymm5
    vfmadd231ps ymm2, ymm0, ymm6
    vfmadd231ps ymm3, ymm0, ymm7
    vfmadd231ps ymm4, ymm0, ymm8
    vmovups [rdi + rax*4], ymm1
    vmovups [rdi + rax*4 + 32], ymm2
    vmovups [rdi + rax*4 + 64], ymm3
    vmovups [rdi + rax*4 + 96], ymm4
    add     rax, 32
    jmp     .lp
.done:
    ret
%endif
%if FAST_Q8
; ----------------------------------------------------------------------------
; void gqa_attn_range(uint64 layer_kv_base, uint64 pos, uint64 scale_gqa_packed,
;                     uint64 h0, uint64 h1)
; rdi = layer_kv_base, rsi = pos, rdx = (gqa_ratio << 32) | att_scale_bits,
; rcx = h0, r8 = h1
; Thread-safe parallel GQA attention over contiguous cache[l, kv_h, pos, hd].
; ----------------------------------------------------------------------------
gqa_attn_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 56                     ; 6 pushes(48) + 56 + ret(8) = 112 (16B aligned)
    ; [rsp + 0]  = layer_kv_base
    ; [rsp + 8]  = pos
    ; [rsp + 16] = att_scale (f32)
    ; [rsp + 20] = gqa_ratio (u32)
    ; [rsp + 24] = h1
    ; [rsp + 32] = kv_head_stride (max_seq * head_dim * KV elt bytes)
    ; [rsp + 40] = KV step per t (head_dim * KV elt bytes)

    mov     [rsp + 0], rdi
    mov     [rsp + 8], rsi
    mov     [rsp + 16], edx             ; att_scale bits
    shr     rdx, 32
    mov     [rsp + 20], edx             ; gqa_ratio
    mov     [rsp + 24], r8              ; h1

    lea     rbx, [rel g_ctx]
    mov     rbp, [rbx + CTX_HEAD_DIM]
    shl     rbp, 2                      ; rbp = head_bytes (step per t = 512B contiguous!)
    mov     rax, rbp
    shr     rax, 2 - KV_SHIFT           ; AIOS: head_dim * KV elt bytes
    mov     [rsp + 40], rax
    imul    rax, [rbx + CTX_MAX_SEQ]
    mov     [rsp + 32], rax             ; kv_head_stride

    mov     r12, rcx                    ; h = h0
.h_lp:
    cmp     r12, [rsp + 24]
    jae     .h_done

    ; kv_h = h / gqa_ratio; r13 = layer_kv_base + kv_h * kv_head_stride
    mov     eax, r12d
    xor     edx, edx
    div     dword [rsp + 20]
    imul    rax, [rsp + 32]
    add     rax, [rsp + 0]
    mov     r13, rax

    ; q_head = buf_q + h * head_bytes
    mov     rax, r12
    imul    rax, rbp
    mov     r14, [rbx + CTX_BUF_Q]
    add     r14, rax

    ; Private score buffer scr_h = buf_att_scr + h * max_seq * 4
    mov     r11, r12
    imul    r11, [rbx + CTX_MAX_SEQ]
    shl     r11, 2
    add     r11, [rbx + CTX_BUF_ATT_SCR]

    ; Scores[0..pos]: dot(q_head, k_cache[l, kv_h, t]) * att_scale
    mov     r15, [rbx + CTX_BUF_K_CACHE]
    add     r15, r13
    xor     ecx, ecx
.sc_lp:
    cmp     rcx, [rsp + 8]
    ja      .sc_done
    push    rcx
    push    r11
    mov     rdi, r14
    mov     rsi, r15
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    KV_DOT
    pop     r11
    pop     rcx
    vmulss  xmm0, xmm0, [rsp + 16]
    vmovss  [r11 + rcx*4], xmm0
    add     r15, [rsp + 40]             ; contiguous KV step
    inc     rcx
    jmp     .sc_lp

.sc_done:
    push    r11
    sub     rsp, 8                      ; 16B align
    mov     rdi, r11
    mov     rsi, [rsp + 24]             ; pos (rsp+8 + 16 = rsp+24)
    inc     rsi
    call    softmax_avx2
    add     rsp, 8
    pop     r11

    ; Zero att_head = buf_att_out + h * head_bytes
    mov     rax, r12
    imul    rax, rbp
    mov     r14, [rbx + CTX_BUF_ATT_OUT]
    add     r14, rax
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.z_lp:
    cmp     rax, rbp
    jae     .v_acc
    vmovups [r14 + rax], ymm0
    add     rax, 32
    jmp     .z_lp

.v_acc:
    mov     r15, [rbx + CTX_BUF_V_CACHE]
    add     r15, r13
    xor     ecx, ecx
.v_lp:
    cmp     rcx, [rsp + 8]
    ja      .h_next
    vmovss  xmm0, [r11 + rcx*4]
    push    rcx
    push    r11
    mov     rdi, r14
    mov     rsi, r15
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    KV_SAXPY
    pop     r11
    pop     rcx
    add     r15, [rsp + 40]             ; contiguous KV step
    inc     rcx
    jmp     .v_lp

.h_next:
    inc     r12
    jmp     .h_lp

.h_done:
    add     rsp, 56
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
%endif

section .rodata
align 8
kv_elt_bytes: dq 1 << KV_SHIFT     ; AIOS: read by the kernel's prefix snapshot
%ifdef QWEN35
align 16
t_one:      dd 1.0
t_l2eps:    dd 1.0e-6
align 16
d_one:      dq 1.0
d_twenty:   dq 20.0
d_signmask: dq 0x8000000000000000, 0
%else
align 4
t_one: dd 1.0
%endif

section .bss
alignb 64
g_ctx: resb CTX_SIZE
%ifdef QWEN35
alignb 64
g_dn_job: resq 8                  ; deltanet_layer -> deltanet_heads_range job parameters
%endif
