; VENDORED from github.com/tcsenpai/llm.asm, branch qwen35-kernel, commit 1c73255 (shared/constants.inc).
; Do not edit here: change upstream and re-vendor. Unmodified below this header.
; ============================================================================
; constants.inc — Linux x86-64 syscalls, .asmbin header layout, Runtime struct
; Pure x86-64 NASM assembly (zero libc, zero external dependencies)
; ============================================================================

; --- Linux x86-64 syscall numbers ---
%define SYS_read            0
%define SYS_write           1
%define SYS_close           3
%define SYS_fstat           5
%define SYS_mmap            9
%define SYS_munmap          11
%define SYS_clone           56
%define SYS_exit            60
%define SYS_futex           202
%define SYS_clock_gettime   228
%define SYS_exit_group      231
%define SYS_openat          257

; --- Linux file / mmap / clone / futex constants ---
%define AT_FDCWD            -100
%define O_RDONLY            0
%define O_WRONLY_CREAT_TR   0x241       ; O_WRONLY | O_CREAT | O_TRUNC
%define MODE_0644           0o644
%define PROT_READ           1
%define PROT_RW             3           ; PROT_READ | PROT_WRITE
%define MAP_PRIVATE         2
%define MAP_PRIV_ANON       0x22        ; MAP_PRIVATE | MAP_ANONYMOUS
%define CLOCK_MONOTONIC     1

; CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD
%define CLONE_THREAD_FLAGS  0x00010F00
%define FUTEX_WAIT_PRIVATE  128
%define FUTEX_WAKE_PRIVATE  129
%define FUTEX_WAKE_ALL      0x7FFFFFFF

; struct stat (x86-64 Linux): st_size at offset 48, total size 144
%define STAT_SIZE           144
%define STAT_ST_SIZE        48

%ifdef QWEN35
; --- .asmbin v3 Header Offsets (256 bytes; v1/v2 are no longer loadable) ---
%else
; --- .asmbin Header Offsets (128 bytes) ---
%endif
%define ASMBIN_MAGIC        0x4C4D5341  ; "ASML"
%ifdef QWEN35
%define ASMBIN_VERSION_V3   3
%else
%define ASMBIN_VERSION      1
%define ASMBIN_VERSION_V2   2
%endif
%define HDR_FLAG_TIED_EMB   1
%define HDR_FLAG_QK_NORM    2
%define HDR_FLAG_FP16_SCALES 4

%define HDR_MAGIC           0
%define HDR_VERSION         4
%define HDR_DIM             8
%define HDR_HIDDEN_DIM      12
%define HDR_N_LAYERS        16
%define HDR_N_HEADS         20
%define HDR_N_KV_HEADS      24
%define HDR_HEAD_DIM        28
%define HDR_VOCAB_SIZE      32
%define HDR_MAX_SEQ_LEN     36
%define HDR_RMS_EPS         40
%define HDR_ROPE_THETA      44
%define HDR_EOS_1           48
%define HDR_EOS_2           52
%define HDR_HT_MASK         56
%define HDR_FLAGS           60
%define HDR_OFF_TOKENIZER   64
%define HDR_OFF_EMBED       72
%define HDR_OFF_LAYERS      80
%define HDR_LAYER_STRIDE    88
%define HDR_OFF_FINAL_NORM  96
%define HDR_FILE_SIZE       104
%ifdef QWEN35
; 112..127 reserved. v3 extension:
%define HDR_OFF_LAYER_TABLE 128         ; u64 file offset of the per-layer descriptor table
%define HDR_LAYER_TYPE_MASK 136         ; u64 bit l = 1 -> layer l is not type 0 (info only)
%define HDR_N_LIN_K_HEADS   144         ; u32 linear-attention dims (0 for pure full-attn models)
%define HDR_N_LIN_V_HEADS   148
%define HDR_LIN_K_DIM       152
%define HDR_LIN_V_DIM       156
%define HDR_CONV_K          160         ; u32 linear-attn conv kernel size
%define HDR_ROT_DIM         164         ; u32 rotary dims per head (== head_dim for Qwen3)
%define HDR_FLAGS2          168         ; u32, see HDR_FLAG2_*
%define HDR_FLAG2_ATTN_OUT_GATE 1       ; full-attn q_proj also carries a sigmoid output gate
%define HDR_FLAG2_NORMS_1PW_BAKED 2     ; exporter already stored (1+w) for zero-centered norms
; 172 reserved. Chat-template token ids / sampling limit (all 0 for Qwen3 files -> asm falls back to TOK_* below):
%define HDR_TOK_ENDOFTEXT   176
%define HDR_TOK_IM_START    180
%define HDR_TOK_IM_END      184
%define HDR_TOK_THINK_START 188
%define HDR_TOK_THINK_END   192
%define HDR_TOK_USER        196
%define HDR_TOK_ASSISTANT   200
%define HDR_TOK_NL          204
%define HDR_TOK_NL2         208
%define HDR_ARGMAX_LIMIT    212         ; valid token ids are < this (0 -> vocab_size)
; 216..255 reserved (zero)
%endif

%ifdef QWEN35
; --- Per-layer descriptor table (HDR_OFF_LAYER_TABLE, n_layers entries) ---
%define DESC_STRIDE         136         ; u32 type, u32 reserved, 16 x u64 absolute file offsets
%define DESC_TYPE           0
%define DESC_OFFS           8
%define LAYER_TYPE_FULL_ATTN 0          ; Qwen3 full attention (only type the kernel runs)
%define LAYER_TYPE_LINEAR_ATTN 1        ; Qwen3.5 gated DeltaNet
; Slot i lives at DESC_OFFS + 8*i. Common: 0 in_norm, 1 post_norm, 2 W_proj (full: q|k|v rows),
; 3 W_out, 4 W_gateup (gate rows then up rows), 5 W_down. Full-attn only: 6 q_norm, 7 k_norm.
%define DSLOT_IN_NORM       (DESC_OFFS + 0*8)
%define DSLOT_POST_NORM     (DESC_OFFS + 1*8)
%define DSLOT_W_PROJ        (DESC_OFFS + 2*8)
%define DSLOT_W_OUT         (DESC_OFFS + 3*8)
%define DSLOT_W_GATEUP      (DESC_OFFS + 4*8)
%define DSLOT_W_DOWN        (DESC_OFFS + 5*8)
%define DSLOT_Q_NORM        (DESC_OFFS + 6*8)
%define DSLOT_K_NORM        (DESC_OFFS + 7*8)
; type 1 (DeltaNet) slots: 2 W_proj = [qkv|z] rows, 6 conv_w f32[4][C], 7 negExpA f32[H], 8 dt_bias f32[H],
; 9 gnorm_w f32[Dv], 10 W_b f32[H][dim], 11 W_a f32[H][dim]   (see docs/qwen35-2b/FORMAT_V3.md)
%define N_DESC_SLOTS        12

%endif
; --- Tokenizer Section Header Offsets (relative to base + off_tokenizer) ---
%define TOKH_OFF_B2T        0
%define TOKH_OFF_OFFSETS    8
%define TOKH_OFF_BYTES      16
%define TOKH_OFF_HT         24

; Merge hash table entry (16 bytes): { uint64 key, uint32 merged_id, uint32 rank }
%define HT_KEY              0
%define HT_MERGED           8
%define HT_RANK             12
%define HT_EMPTY_RANK       0xFFFFFFFF
%define HT_SHIFT            45          ; 64 - 19 (524,288 slots)
%define FIB_MULT_64         0x9E3779B97F4A7C15

; --- Qwen3 Chat Template Special Token IDs ---
%define TOK_ENDOFTEXT       151643
%define TOK_IM_START        151644
%define TOK_IM_END          151645
%define TOK_THINK_START     151667
%define TOK_THINK_END       151668
%define TOK_USER            872
%define TOK_ASSISTANT       77091
%define TOK_NL              198
%define TOK_NL2             271

; --- Limits ---
%define MAX_THREADS         64
%define WORKER_STACK_SIZE   65536
%define MAX_PROMPT_TOKENS   8192
%define MAX_GEN_TOKENS      8192

; --- Runtime Context Struct (g_ctx, 256 bytes in .bss) ---
%define CTX_MMAP_BASE       0
%define CTX_MMAP_SIZE       8
%define CTX_ARENA_BASE      16
%define CTX_ARENA_SIZE      24

%define CTX_B2T_PTR         32
%define CTX_TOK_OFFS_PTR    40
%define CTX_TOK_BYTES_PTR   48
%define CTX_HT_PTR          56
%define CTX_EMB_PTR         64
%ifndef QWEN35
%define CTX_LAYERS_PTR      72
%endif
%define CTX_FINAL_NORM_PTR  80

%define CTX_DIM             88
%define CTX_HIDDEN_DIM      96
%define CTX_N_LAYERS        104
%define CTX_N_HEADS         112
%define CTX_N_KV_HEADS      120
%define CTX_HEAD_DIM        128
%define CTX_Q_DIM           136
%define CTX_KV_DIM          144
%define CTX_VOCAB_SIZE      152
%define CTX_MAX_SEQ         160
%ifdef QWEN35
%define CTX_LAYER_TABLE     168         ; absolute ptr to the layer descriptor table
%else
%define CTX_LAYER_STRIDE    168
%endif
%define CTX_HT_MASK         176
%define CTX_EOS_1           184
%define CTX_EOS_2           192
%define CTX_RMS_EPS         200         ; float32 (4B)
%define CTX_ROPE_THETA      204         ; float32 (4B)

; Scratch & KV-cache pointers (inside mmap arena)
%define CTX_BUF_X           208
%define CTX_BUF_XB          216
%define CTX_BUF_Q           224
%define CTX_BUF_K           232
%define CTX_BUF_V           240
%define CTX_BUF_ATT_OUT     248
%define CTX_BUF_ATT_SCR     256
%define CTX_BUF_GATE        264
%define CTX_BUF_UP          272
%define CTX_BUF_LOGITS      280
%define CTX_BUF_XQ          288
%define CTX_BUF_XS          296
%define CTX_BUF_INV_FREQ    304
%define CTX_BUF_COS         312
%define CTX_BUF_SIN         320
%define CTX_BUF_K_CACHE     328
%define CTX_BUF_V_CACHE     336
%define CTX_BUF_PROMPT      344
%define CTX_BUF_GEN         352
%define CTX_BUF_PROBS       360
%define CTX_FP16_SCALES     368         ; uint32: 0 = v1 fp32 scales, 1 = v2 fp16 scales
%define CTX_SCALE_SHIFT     372         ; uint32: 3 = fp32 (K>>3), 4 = fp16 (K>>4)
%ifdef QWEN35
; --- Qwen3.5 additions (relocated arena pointers first, 4 entries) ---
%define CTX_BUF_DN_STATE    384         ; DeltaNet S [n_lin][H][128][128] f32 (zeroed at pos 0)
%define CTX_BUF_DN_CONV     392         ; conv state [n_lin][3][conv_dim] f32 (adjacent to DN_STATE, zeroed with it)
%define CTX_BUF_MISC        400         ; 1 KiB scratch: +0 a[H], +128 b[H], +512 delta[128]
%define CTX_BUF_ONES        408         ; conv_dim x 1.0f (silu(x) = silu_mul(x, ones))
%define CTX_N_RELOC2        4
%define CTX_N_FULL          416         ; number of full-attention layers (KV cache layers)
%define CTX_N_LIN           424         ; number of DeltaNet layers
%define CTX_ROT_DIM         432
%define CTX_FLAGS2          440
%define CTX_LIN_HEADS       448         ; Hk == Hv
%define CTX_LIN_DIM         456         ; Dk == Dv == 128 (asserted)
%define CTX_CONV_DIM        464         ; 2*Hk*Dk + Hv*Dv
%define CTX_VAL_DIM         472         ; Hv*Dv
%define CTX_GATE_DIM        480         ; q_dim when ATTN_OUT_GATE else 0
%define CTX_SAMPLE_N        488         ; argmax/sampling limit
%define CTX_DN_ZERO_BYTES   496         ; DN_STATE + DN_CONV bytes
%define CTX_TAP_FN          504         ; 0 or tap_emit(layer=rdi, pos=rsi, vec=rdx, n=rcx, token=r8)
%define CTX_TOK_IM_START    512
%define CTX_TOK_IM_END      520
%define CTX_TOK_THINK_START 528
%define CTX_TOK_THINK_END   536
%define CTX_TOK_USER        544
%define CTX_TOK_ASSISTANT   552
%define CTX_TOK_NL          560
%define CTX_TOK_NL2         568
%define CTX_TPL_THINK_NL    576         ; 1: think=1 appends "<think>\n" (Qwen3.5 template); 0: nothing (Qwen3)
%define CTX_SIZE            640
%else
%define CTX_SIZE            384
%endif
