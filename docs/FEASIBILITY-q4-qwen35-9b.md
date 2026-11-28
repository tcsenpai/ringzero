# Feasibility: Q4-class quantization in the ringzero NASM engine, and Qwen3.5-9B in ring 0

Date 2026-11-26. Research, code reading and small measurements only. No engine, kernel or repo changes.
Legend: **[M]** measured here, **[S]** cited source, **[C]** read in our code, **[E]** estimate (method stated), **[U]** unverified.

## Verdict

1. **Yes, the engine can take a Q4 format cheaply.** A Q4_0-style block (32 weights, fp16 scale, 4.5 bpw) in our split-row layout,
   kept W4A8 with the existing Q8_0 g32 activations, is one new matvec kernel, a per-tensor type in the descriptor, and an exporter
   change. Quantization is not the hard part of the 9B. Memory/boot (a 5.6-6 GB module, an 8-10 GiB guest) and the 9B's
   Gated DeltaNet shape (16 K heads, 32 V heads; the engine assumes equal counts) are.
2. **The speed gain is smaller than the byte ratio.** On this host, an AVX2 Q4 matvec at 8 threads is about as fast in compute
   as DRAM can feed it. Measured Q4/Q8 weight throughput is 1.2x-2.3x, depending on host load. 9B Q4 decode, estimated:
   **~10-17 tok/s at 8 vCPU, central ~12**. 9B Q8 would run at ~8.5 tok/s, but it needs a 12-14 GiB guest that the host can't spare today.
3. **Cost: ~23 h for Q4 on the 2B (kernel, exporter, mirror, KLD harness), then ~28 h for the 9B. Total ~50-55 h, range 45-75.**
   Do Q4 on the 2B first, as a stepping stone. As a product default for the 2B it is not worth it.

## 1. Format choice for an asm engine on AVX2

### What our engine does today [C]

- `engine/matmul.asm` `matmul_q8_0_range` (FAST_Q8=1): per-row split layout `[K int8][K/32 fp16]` (row stride `K + K>>4`).
  Inner block `INT8_1ROW_4BLK`: `vpabsb`/`vpsignb` sign trick, `vpmaddubsw`, `vpmaddwd`, `vcvtdq2ps`, `vfmadd231ps` with
  `scale_w*scale_x`. 4 blocks (128 B) per step, software prefetch of the next row.
- Activations: `quantize_x_q8_0` makes Q8_0 g32 (`q8_xq`, `q8_xs`) once per matmul in `tp_matmul_q8_0` (`engine/smp.asm`),
  then every core runs a contiguous row range. `Q8_MAX_K 16384` covers the 9B (largest K = 12288).
- In the v3 (QWEN35) path every tensor pointer comes from the per-layer descriptor (`DESC_STRIDE 136`: u32 type, **u32 reserved**,
  16 u64 offsets). The row stride matters only inside the kernel and in the embedding lookup (`CTX_SCALE_SHIFT`,
  `transformer.asm:737`), so a new format does not touch the layer loop.
- DeltaNet small tensors are already full precision: `W_a`, `W_b` f32 (`dot_f32_avx2`), conv_w, negExpA, dt_bias, gnorm_w,
  and all norms f32 (`tools/export_weights.py` slot table).

### How llama.cpp does it on AVX2 [S]

File `ggml/src/ggml-cpu/arch/x86/quants.c` (llama.cpp master, fetched 2026-11-26):

- `ggml_vec_dot_q4_0_q8_0`: `bytes_from_nibbles_32` (one 16-byte load, `_mm_srli_epi16` by 4, `_mm_and` with 0x0F, both halves joined
  into one ymm: low nibbles = elements 0-15, high nibbles = 16-31), `_mm256_sub_epi8(qx, 8)`, then `mul_sum_i8_pairs_float`
  (the same `sign`/`maddubs`/`madd`/`cvt` sequence our Q8 kernel uses), one `fmadd` with `d_x*d_y` per block.
- `ggml_vec_dot_q4_K_q8_K`: 256-weight super-blocks, 6-bit scales and mins unpacked with masks (`kmask1..3`). The unsigned nibbles go
  straight into `_mm256_maddubs_epi16(q4, q8)` with no sign trick. The scales go in by `_mm256_madd_epi16` with a
  `vpshufb` (`get_scale_shuffle_k4`). The mins are subtracted once per super-block using `bsums` (per-16 sums of the Q8_K activations).
  The integer accumulation runs over 256 weights, with one float conversion per super-block.
- `ggml_vec_dot_q5_K_q8_K` adds the 5th bit plane (`qh`). `ggml_vec_dot_q6_K_q8_K` merges 4 low bits and 2 high bits, applies a -32 offset
  and int8 scales per 16 weights.
- `ggml_vec_dot_iq4_nl_q8_0` / `ggml_vec_dot_iq4_xs_q8_K`: a nibble is an index into the 16-entry non-linear table `kvalues_iq4nl`,
  looked up with `_mm_shuffle_epi8`, then a signed int8 dot.
- For batched work llama.cpp repacks Q4_0 at load time into 8-row interleaved tiles (`ggml/src/ggml-cpu/arch/x86/repack.cpp`:
  `ggml_gemv_q4_0_8x8_q8_0`, `ggml_gemm_q4_0_8x8_q8_0`, also `q4_K_8x8`, `iq4_nl_8x8`). The bartowski 9B card says Q4_0
  "supports online repacking for ARM and AVX CPUs".

### Comparison

Rel. error = `||W - Q(W)|| / ||W||`, RTN (no calibration), on real Qwen3.5-2B BF16 tensors (12 tensors across DeltaNet, full attention,
MLP and embed), **[M]** with `~/coding/q4study/wq_err.py` on the Fedora box. The value is the same within ±0.01 across all tensor types.

| Format | bpw | AVX2 ops per 32 weights (dot, beyond the load) | Rel. weight error, 2B RTN [M] | Published accuracy [S] | Pure-asm complexity |
|---|---:|---|---:|---|---|
| Q8_0 (today) | 8.5 | sign x2, maddubs, madd, cvt, fma (~6) | 0.0056 | Qwen3 8-bit GPTQ/AWQ g128: Wiki2 PPL equal to FP16 at 0.6B-8B (Qwen3 study, Table 4) | done |
| Q6_K | 6.5625 | ql/qh merge, -32, sign trick, int8 scale per 16 (~11) | n/m | LLaMA-7B PPL 5.9110 vs F16 5.9066 (+0.07%) (PR #1684); Qwen3.5-35B-A3B Q6_K_S mean KLD 0.0065 (Unsloth) | high: bit planes, 16-weight scales, Q8_K activations |
| Q5_K | 5.5 | Q4_K path + high-bit plane (~9 + per-256 overhead) | n/m | LLaMA-7B Q5_K_M +0.24% (PR #1684); Qwen3.5-35B-A3B Q5_K_M mean KLD 0.0058 (AesSedai) / 0.0106 (bartowski) | high |
| Q4_K | 4.5 | and, shift, maddubs x1, madd with shuffled scale, + mins via bsums once per 256 (~6 + overhead) | n/m (RTN not comparable: its quality comes from llama.cpp's scale search) | LLaMA-7B Q4_K_M 5.9601 (+0.9%) (PR #1684); Qwen3.5-35B-A3B Q4_K_M mean KLD 0.0096-0.0192 (Unsloth table) | medium-high: 6-bit packed scales/mins, 256-weight Q8_K activations with bsums (a second activation quantizer) |
| IQ4_NL | 4.5 | and, shift, 2x pshufb (128-bit), insert, sign trick (~11) | 0.080 (simple scale; llama.cpp searches better) | Llama-3.1-8B 4.68 bpw recipe (quantize README); no Qwen KLD for IQ4_NL found | medium: one lookup table, otherwise Q4_0 |
| IQ4_XS | 4.25 | IQ4_NL + 6-bit sub-block scales in 256 super-blocks | n/m | Qwen3.5-35B-A3B mean KLD 0.0234-0.0235 | medium-high |
| **Q4_0** | 4.5 | llama.cpp form: srl, insert, and, sub, then the Q8 ops (~10). "u-form": unsigned nibbles straight into maddubs + per-row correction (~8) | **0.090** | LLaMA-v1-7B 6.1162 (no imatrix) / 6.0276 (imatrix) vs F16 5.9066: +3.5% / +2.0% (PR #4969, PR #1684, different builds); Qwen3.5-35B-A3B Q4_0 mean KLD 0.0142 (Ubergarm, imatrix) | **low**: one new kernel shaped like the current one |
| Q4_1 | 5.0 | u-form + `m_b * sum(x_b)` term (~8) | 0.080 | PR #4969: LLaMA-v2-7B Q4_1 6.0008 / 5.8605 (imatrix) vs Q4_0 5.9635 / 5.9107 | low: Q4_0 u-form plus a second fp16 per block |
| int4 asym g64 (GPTQ/AWQ-like RTN) | 4.25 | u-form, one scale per 2 activation blocks | 0.094 | - | low |
| int4 asym g128 | 4.125 | same | 0.107 | Qwen3-8B GPTQ 4-bit g128 Wiki2 9.96 vs FP16 9.71 (+2.6%), AWQ 11.3 (+16%); Qwen3-1.7B GPTQ 19.5 vs 16.7, MMLU 60.0 -> 55.7 (Qwen3 study, Table 4) | low, but needs GPTQ/AWQ calibration to be good |
| int4 sym g128 | 4.125 | same | 0.111 | - | low |

Notes on the table:

- 4-bit RTN error is ~16x the Q8_0 error (0.09 vs 0.0056), as expected from 4 fewer bits. Smaller groups help more than
  asymmetry at equal size. Q4_1 g32 (5.0 bpw) beats asym g64 (4.25) by 15% error. IQ4_NL gets Q4_1's error at Q4_0's size.
- The IQ4_NL value for L0 `in_proj_qkv` is excluded. My simplified quantizer mishandles blocks whose fp16 scale underflows to 0.
  Such tiny blocks exist in the 2B, so the real exporter must flush them to zero, as it does today for Q8.
- W4A8 is not the W4A8 that the Qwen3 study condemns. That paper's W4A8 uses per-channel weights and SmoothQuant activations
  (8B Wiki2 9.71 -> 12.5). Ours is g32 weights x g32 activations, which is what llama.cpp's Q4_0 x Q8_0 does, and the g32
  activation error (~0.5%) is small next to the 9% weight error.
- KLD numbers come from Qwen3.5-35B-A3B (MoE), not the dense 9B. I found no published dense-9B KLD per quant **[U]**.

**Format recommendation:** **Q4_0 g32, split row `[K/2 nibbles][K/32 fp16 d]`, u-form kernel**. All 2B/9B row strides stay
64-byte multiples: K=2048 -> 1152 B, 4096 -> 2304, 6144 -> 3456, 12288 -> 6912. Q4_1 is the cheap quality upgrade: the u-form already
computes the correction term, so Q4_1 costs +11% bytes and no extra compute. IQ4_NL is the other one (+1 table lookup, same
bytes). Skip the K-quants and Q6_K in asm: twice the code for 1.3x fewer bytes than Q8, or for quality that comes mostly from
llama.cpp's calibration tooling, not the layout.

**Mixed precision** (keep in Q8_0 or f32):
- f32 as today: norms, conv1d, A (negExpA), dt_bias, gated-norm w, `in_proj_a/b`. llama.cpp agrees: its Q4_0 9B GGUF keeps `ssm_a`,
  `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm` in F32 (header parsed below). `llama-quant.cpp` excludes `ssm_conv1d` and
  every tensor with fewer than 2 dims from quantization.
- lm_head: Q8_0 (llama.cpp uses Q6_K for `output.weight` in Q4 recipes, `llama-quant.cpp` "new_type = GGML_TYPE_Q6_K"; bartowski's
  `*_L` files keep embed/output at Q8_0). It is 11% of the 9B parameters and 22% of the bytes read per token at Q4.
- Q4 candidates, in order of risk: MLP (54% of the 9B), then attention/DeltaNet projections. Unsloth's Qwen3.5 page calls `attn_*`
  "especially sensitive" in the hybrid models and says quantizing `ssm_out` low "is not a good idea". bartowski's 9B Q4_0 recipe
  (below) does put `attn_qkv`/`attn_gate` (= DeltaNet `in_proj_qkv`/`in_proj_z`) at Q4_0, and half the `ssm_out` (= `out_proj`) at Q8_0.

## 2. Qwen3.5-9B specifics

### Config (Qwen/Qwen3.5-9B `config.json`, re-fetched 2026-11-26) [S]

hidden 4096, 32 layers (`full_attention_interval` 4: layers 3,7,..,31 full, 24 linear), intermediate 12288 (SwiGLU),
full attention 16 Q / 4 KV heads, head_dim 256, `attn_output_gate` true, partial rotary 0.25 (64 dims), rope_theta 1e7,
mrope_section [11,11,10]. DeltaNet: conv kernel 4, **16 key heads x 128, 32 value heads x 128**, `mamba_ssm_dtype` float32. vocab 248320,
`tie_word_embeddings` **false**, rms_eps 1e-6, max_position 262144, 1 MTP layer (dropped). Matches `docs/qwen35-9b-architecture.md`.

### Parameters per group [E: from the shapes, total matches the safetensors count 8,952,741,888 in 2-D matrices]

| Group | Params | Share |
|---|---:|---:|
| MLP (32 x 3 x 4096 x 12288) | 4,831,838,208 | 54.0% |
| DeltaNet projections (24 x [qkv 8192 + z 4096 + out 4096 + a,b 64] x 4096) | 1,616,904,192 | 18.1% |
| Full attention (8 x [q+gate 8192, k 1024, v 1024, o 4096] x 4096) | 469,762,048 | 5.2% |
| embed_tokens (248320 x 4096) | 1,017,118,720 | 11.4% |
| lm_head (untied) | 1,017,118,720 | 11.4% |
| vectors (norms, conv, A, dt) | ~1.06 M | - |

### Model size per format (text only) [E]

| Plan | Weights file | Bytes read per decoded token (all but the embed) |
|---|---:|---:|
| BF16 | 17.9 GB | 15.9 GB |
| Q8_0 everywhere (today's format) | 9.52 GB | 8.43 GB |
| Q6_K everywhere | 7.34 GB | 6.51 GB |
| Q5_K everywhere | 6.16 GB | 5.46 GB |
| Q4_0 conservative: MLP Q4, attention + DeltaNet + embed + lm_head Q8 | 7.10 GB | 6.02 GB |
| **Q4_0 mixed: everything Q4 except lm_head + embed Q8** | **6.05 GB** | **4.97 GB** |
| Q4_0 mixed, embed also Q4 (lm_head Q8) | 5.55 GB | 4.97 GB |
| Q4_0 everywhere | 5.04 GB | 4.46 GB |
| bartowski Q4_0 GGUF (as published) [S] | 5.74 GB | - |

What llama.cpp actually wrote into `bartowski/Qwen_Qwen3.5-9B-GGUF` `Qwen_Qwen3.5-9B-Q4_0.gguf` [M: GGUF header parsed from the first
16 MB with `q4study/gguf_types.py`; imatrix from `calibration_datav5.txt`, 802 chunks]: `token_embd` Q4_0, `output` Q6_K,
`attn_qkv`/`attn_gate` (DeltaNet) Q4_0, `ssm_out` 12 x Q8_0 + 12 x Q4_0, `attn_q` 4 x Q6_K + 4 x Q4_0, `ffn_down` 4 x Q4_1 + 28 x Q4_0,
MTP layer Q8_0, all `ssm_*` small tensors F32. Totals: Q4_0 4.16 GB, Q6_K 0.94, Q8_0 0.47, Q4_1 0.13, F32 0.03.

### RAM need [E, per-token sizes from the config; the 2B check matches M25: 2B FP16 KV at 32k = 384 MiB vs arena 406 MiB]

| Item | 2B (today) | 9B |
|---|---:|---:|
| FP16 KV per token (full layers only) | 12 KiB (6 x 2 KV x 256 x 2 x 2 B) | 32 KiB (8 x 4 x 256 x 2 x 2 B) |
| KV at 4k / 32k | 48 MiB / 384 MiB | 128 MiB / 1 GiB |
| DeltaNet state (S f32 + conv) | 18 x 1 MiB + 0.4 MiB | 24 x 2 MiB + 2.3 MiB = 50 MiB |
| Prefix snapshot (743-token boot prefix) | 27 MiB (M25) | ~75 MiB |
| Engine arena scratch (logits 1 MiB, buffers) | tens of MiB | tens of MiB |
| Embedder (Qwen3-Embedding-0.6B) + snapshot pool | 613 MiB + 1 GiB | same |

Guest totals for the 9B: Q4 mixed (6.05 GB = 5.6 GiB), 32k ctx, embedder, 1 GiB pool: **~8.5 GiB -> `GUEST_MEM=10G`**. With 4k ctx and a 512 MiB pool: ~7 GiB -> 8G.
Q8 9B (8.9 GiB of weights): ~11.7 GiB -> 12-14G.

- 16 GB machine: Q4 9B at 4k-8k ctx fits (8 GiB guest), 32k is tight. Q8 does not fit.
- 32 GB machine: both fit with one guest.
- Our Fedora box (31 GB total) had **7.2-15.3 GB available** during this study (`free -m`, the user's guest running), and the Makefile
  refuses to start below 8192 MiB available. Q4 9B fits only after freeing memory or stopping the other guest. Q8 9B does not fit beside it.

### Quantization sensitivity of Gated DeltaNet

- No paper found on quantizing Gated DeltaNet specifically **[U]**. The closest literature covers Mamba/SSMs. Quamba (arXiv 2410.13229)
  is 8-bit W8A8 with activation clamping plus a Hadamard transform for output outliers. MambaQuant (arXiv 2501.13484) reports W8A8 under 1% loss,
  4-bit weights only for vision tasks, and outliers "amplified by PScan". Mamba-PTQ (arXiv 2407.12397) reports outlier channels.
  These are about activations inside the scan, which we keep in f32. They say little about the projection weights we would quantize.
- Practice for Qwen3.5 [S]: llama.cpp keeps every 1-D/small DeltaNet tensor in F32. Unsloth says `attn_*` and `ssm_out` are sensitive,
  and MXFP4 is much worse than Q4_K on `ssm_alpha`/`ssm_beta`/`attn_gate`/`attn_q`.
- Weight-level, on the 2B [M]: DeltaNet projections quantize like every other matrix (Q4_0 0.090-0.092 vs MLP 0.089). `in_proj_b`
  (beta) shows the largest output error (0.118 for Q4_0). It stays f32, as today.
- Structural concern [E, hypothesis]: errors in k, v and beta enter the recurrent state S and persist across tokens, while an
  attention error stays local to one KV row. So test quality at long context too, not only at 512 tokens.
- Keep at higher precision: A_log/negExpA, dt_bias, conv1d, all norms, gated-norm w and in_proj_a/b in f32 (as today); lm_head in Q8;
  `out_proj` and the full-attention projections in Q8 if KLD shows they matter (the "conservative" plan above, +1 GB).

## 3. Speed

### Micro-benchmark [M]

`~/coding/q4study/qbench.c` on the build box (Ryzen AI 9 HX 370, 4 Zen5 + 8 Zen5c cores; AVX-512, AVX-512-VNNI and AVX-VNNI present).
gcc 16 `-O2 -mavx2 -mfma -mf16c -mavxvnni -fopenmp`, `OMP_PLACES=cores OMP_PROC_BIND=close`, 2 MiB THP buffers, K=4096.
Kernels have the same shape as `INT8_1ROW_4BLK`, and a scalar self-check passes. "L2" = 128 rows per thread repeated (compute ceiling).
"DRAM" = 24576 rows (Q8 107 MB, Q4 57 MB, larger than L3). Gw/s = 1e9 weights/s. Host load average 8-10 during all runs (other guests and
agents), so results vary up to 2x between runs. Ranges are over 5 runs. The 8-thread "best" column is the one quiet run.

| Kernel | 1 thr L2 Gw/s | 1 thr DRAM Gw/s | 8 thr L2 Gw/s | 8 thr DRAM Gw/s (typical / best) |
|---|---:|---:|---:|---:|
| Q8_0 (engine shape, AVX2) | 46.6-49.3 | 22-40 | 93-218 | 43-49 / 85 |
| Q8_0, AVX-VNNI `vpdpbusd` | 49-58 | 15-40 | 98-290 | 42-62 / 84 |
| Q4_0 llama.cpp form (AVX2) | 31.5-36.2 | 20-30 | 74-196 | 55-60 / 185 |
| Q4_0 u-form (AVX2) | 37.1-39.6 | 21-35 | 80-213 | 56-62 / 195 |
| Q4_0 u-form, AVX-VNNI | 39.3-43.5 | 22-37 | 86-221 | 60-95 / 205 |
| plain streaming read, 8 thr | | | | 52-63 GB/s / 99 GB/s |

What this shows:
- Per weight, Q4 costs 1.2-1.4x more compute than Q8 on one core (34-40 vs 47-49 Gw/s in L2). One Q4 core is compute bound:
  its DRAM rate is the same as its L2 rate.
- At 8 threads, Q8 runs at the DRAM limit (Q8 DRAM GB/s ~ plain read GB/s). Q4 reaches 1.2-1.4x Q8's weights/s under load, and
  2.3x in the quiet run, where it hit its compute ceiling and the memory limit at the same time (~110 GB/s of Q4 bytes vs 99 GB/s plain read).
  **Q4 on AVX2 at 8 cores is close to compute bound on this box**, so the byte saving (1.89x) is not fully realized.
- The u-form saves 2 `vpsignb` and a `vpsubb` per block: +6-10% over the llama.cpp form. VNNI adds +5-10% to Q4 in the quiet runs and is inconsistent
  under load. An AVX2-only engine loses little. VNNI is an optional later path (`-cpu host` passes it).
- The engine itself reaches ~70 GB/s with the 2B Q8 (1.99 GB read per token x 36 tok/s), better than my C Q8 under load, so the
  absolute numbers below are scaled from the engine, and only the Q4/Q8 ratio r = 1.2 / 1.5 / 2.3 comes from the bench.

### Decode estimates [E]

Model: `t = W_q8/R8 + W_q4/(r*R8)`, where R8 = 67.7 Gw/s is calibrated from the measured 2B Q8 36 tok/s (M25, 8 vCPU, short ctx; includes all
non-matmul time), and r = Q4/Q8 throughput ratio from the bench. Short context. At 32k add the KV read (9B: 1 GiB/token, ~15 ms; 2B measured: 36 -> 24 tok/s).

| Model / plan | Bytes per token | tok/s (r = 1.2 / 1.5 / 2.3) |
|---|---:|---:|
| 2B Q8 (today) | 2.0 GB | **36 measured** (34.4-36.6, M25) |
| 2B Q4, tied embed/lm_head Q8 | 1.31 GB | 41 / 48 / 61 |
| 2B Q4 everywhere | 1.06 GB | 43 / 54 / 83 |
| 9B Q8 | 8.43 GB | 8.5 (bandwidth bound, no r) |
| 9B Q6_K | 6.51 GB | <= 11 (bandwidth bound; compute not measured) |
| 9B Q5 | 5.46 GB | <= 13 (likely compute bound, ~10-12) |
| 9B Q4 conservative (MLP only) | 6.02 GB | 9.5 / 10.7 / 13.0 |
| **9B Q4 mixed (lm_head Q8)** | 4.97 GB | **10.0 / 12.0 / 16.8** |
| 9B Q4 everywhere | 4.46 GB | 10.2 / 12.8 / 19.6 |

Prefill is token by token today, so prefill runs at decode speed: a 743-token boot prefix takes ~60 s on the 9B. The boot snapshot cache avoids
repeating it. The batched prefill being written by the other agent changes this. A Q4 GEMM can unpack a weight block once and reuse it for N tokens,
so the unpack cost is amortized and Q4 prefill should run about as fast as Q8 prefill **[E]**.

## 4. Engine changes required

### Exporter (`tools/export_weights.py`, vendored; changes go upstream first per the file header)
- Q4_0 row-split writer (`quantize_matrix_q4_0_row_split`): llama.cpp `quantize_row_q4_0_ref` rule (`d = signed absmax / -8`, q = clamp(round(x/d)+8, 0, 15)),
  nibble order lo = element j, hi = element j+16, fp16 d; flush fp16-underflow blocks to 0. Optional Q4_1 (`d=(max-min)/15`, `m=min`).
- A quant plan per tensor role (`--quant mlp=q4_0,dn=q4_0,attn=q8_0,lm_head=q8_0,embed=q8_0`).
- 9B: 4 shards, `model.language_model.*` names, untied `lm_head.weight`, drop `model.visual.*` and `mtp.*`, tensor-by-tensor streaming
  (peak RAM ~ one BF16 tensor; the largest is lm_head, 2 GB BF16 / 4 GB f32, so quantize it in row chunks).
- **RTN needs no calibration and no GPU.** GPTQ/AWQ need a calibration set and a forward pass of the full-precision model: for the 9B on
  CPU, 18-36 GB of RAM and hours. Tool support for the qwen3_5 hybrid in GPTQ/AWQ toolkits **[U]**. The cheap calibration-aware route is llama.cpp's
  imatrix, already applied to bartowski's GGUF.
- **GGUF -> asmbin import is feasible and nearly lossless** for Q4_0/Q8_0 tensors: a GGUF Q4_0 block (fp16 d + 16 bytes) holds exactly our block
  content, so it is a byte repack into split rows. What the importer must handle [S: `conversion/qwen.py` `Qwen3_5TextModel`,
  `_LinearAttentionVReorderBase`, `Qwen3NextModel.modify_tensors`]:
  - V heads reordered from grouped to tiled order in `in_proj_qkv` (V part), `in_proj_z`, `in_proj_a/b`, `out_proj` columns. All permutations are at
    128-element (head) granularity, so on 32-blocks they are exact. Either undo them, or map V head j to K head `j % 16` instead of `j / 2`.
  - Norms are stored as w+1, and `ssm_a = -exp(A_log)`. These match what our exporter already bakes (`FLAG2_NORMS_1PW_BAKED`, negExpA).
  - Q6_K (`output`, 4 x `attn_q`) and Q4_1 (4 x `ffn_down`) blocks need a kernel or a requantize to Q8_0 (dequant Q6_K -> Q8_0 adds a small error).
  - Estimated 4 h for an importer once Q4 runs. The benefit is imatrix quality without any calibration work.

### `.asmbin` v4
- `HDR_VERSION` 4 (v3 files keep loading; the descriptor walk is the same).
- Per-slot quant type in the **existing reserved u32 of each layer descriptor** (offset 4): 2 bits x 16 slots (0 = f32/none, 1 = Q8_0 fp16,
  2 = Q4_0, 3 = Q4_1). `DESC_STRIDE` stays 136.
- Header fields in the reserved 216..255 area: `HDR_EMB_QTYPE`, `HDR_OFF_LM_HEAD` (untied, 0 = tied), `HDR_LM_QTYPE`.
- Scales stay at the end of each row as today (fp16 d; for Q4_1, d array then m array), so row stride = `K/2 + K/16` (Q4_0) or `K/2 + K/8` (Q4_1).

### Kernels
- `matmul_q4_0_range` (decode), same ABI as `matmul_q8_0_range`, u-form:
  `vmovdqu xmm` 16 B nibbles, `vpsrlw`, `vinserti128`, `vpand`, `vpmaddubsw(w_u, x_s)`, `vpmaddwd`, `vcvtdq2ps`, `vfmadd231ps` with `d_w*d_x`.
  Plus a per-row `sum_b d_w[b] * c[b]` (c[b] = 8*d_x[b]*sum(xq_b); for Q4_1 m_w[b]*d_x[b]*sum(xq_b)), done with `vcvtph2ps` 8 scales at a time.
  The `vpmaddubsw` u4 x s8 pair sum is at most 2*15*127 = 3810, so it never saturates.
- `quantize_x_q8_0` also writes the per-block xq sums (`q8_xsum`, K/32 int32 or a precomputed float), ~5 instructions per block.
- Dispatch: `tp_matmul_q8_0` and the AP worker loop (`smp.asm` calls `matmul_q8_0_range` directly) need a kernel selector
  (a global function pointer or a qtype argument). The `K==0` (attention) / `K==1` (DeltaNet) job encoding stays.
- `dequant_row_q4_0` for an embed row if the embed goes Q4.
- Batched prefill: a Q4 GEMM variant of whatever the batched Q8 kernel becomes. Unpack once into registers and dot with 4-8 token columns.
  Ask the batched-prefill agent to keep the weight-loading step a macro so the Q4 variant is a second instantiation.
- Not quantization but needed for the 9B [C]: `model_init` sets `CTX_LIN_HEADS` = V heads and builds `CONV_DIM` from Hk and Hv, but
  `deltanet_heads_range` treats K and V heads as one count (the 2B has 16/16). The 9B needs a K-head index per V head (`j / 2` in HF order).
  Also an untied lm_head pointer, and `GUEST_MEM` plus the Makefile's 8192 MiB check scaled to the model.
- Boot risk [U]: the model is one ISO file loaded by Limine as a module. `xorriso -as mkisofs` without `-iso-level 3` cannot store files over 4 GiB,
  and Limine's ISO9660 multi-extent support is unverified. Fallback: split the model into 2 modules and map them virtually contiguous in the
  lower half (2 MiB pages, as the arena and models are mapped today).

### Verification
1. **Exactness (asm vs mirror):** extend `docs/qwen35-2b/mirror/mirror_q8.py` to read the exported Q4 bytes (not requantize), with an exact integer
   W4A8 block dot. Criterion as today: greedy token-exact vs the mirror on the golden prompts, plus per-layer tap fingerprints within the FP32 ordering
   noise seen for Q8. This also checks the exporter's layout.
2. **Quality (quantized vs fp32):** the mirror runs 5.7 s/token (MIRROR_2B.md), too slow for thousands of tokens. Use HF transformers on CPU with
   fake-quantized weights (dequantized Q4 loaded into the model), batched over a fixed corpus (e.g. 50 chunks x 512 tokens of wiki + code + chat,
   plus 4 chunks at 8k-16k for the DeltaNet long-context check). Compare logits with the unquantized model: mean KLD, 99th-percentile KLD, top-1 agreement,
   PPL delta. Run Q8_0 the same way first as the baseline. Activation Q8 is not simulated there; step 1 covers it.
3. **Proposed acceptance [E, to tune after the Q8 baseline]:**
   - 2B Q4 mixed: mean KLD <= 0.05 and top-1 >= 90% vs fp32.
   - 9B Q4 mixed: mean KLD <= 0.03 and top-1 >= 93% (reference: Q4 KLD on Qwen3.5-35B-A3B is 0.010-0.019).
   - Long-context chunks no worse than 1.5x the short-chunk KLD.
   - Selftest, golden and the M25 FP16/FP32 style divergence check unchanged in spirit.
   - Reference cost for the 9B: fp32 needs ~36 GB and bf16 ~18 GB of RAM. Not available on Fedora beside the user's guest. Either stream layers
     from safetensors or use another machine **[U]**.

## 5. Effort, risks, path

### Effort [E: same style as the earlier 2B estimate, 1 person + AI, debug on KVM]

| Part | Hours |
|---|---:|
| Exporter: Q4_0 (+Q4_1) writer, quant plan, v4 descriptor qtype bits | 4 |
| HF fake-quant KLD harness (2B), Q8 baseline + Q4 runs | 4 |
| Mirror Q4 (exact W4A8) + parity | 3 |
| `matmul_q4_0_range` + `q8_xsum` + dispatch in `tp_matmul` and AP loop + `dequant_row_q4_0` | 8 |
| Selftest, bench, docs (RESULT.md, ARCHITECTURE.md diagram, README per project rules) | 4 |
| **Subtotal: Q4 on 2B** | **~23 (18-30)** |
| Q4 GEMM for batched prefill (after the Q8 batched kernel lands) | 4-6 |
| 9B exporter (shards, untied lm_head, streaming, drop vision/MTP) | 4 |
| 9B engine: Hk != Hv DeltaNet, untied lm_head, dims | 5 |
| 9B memory/boot: >4 GiB module or split modules, GUEST_MEM 10G, Makefile check, snapshot sizes | 6 (high variance) |
| 9B reference + golden + KLD (needs a large-RAM machine or streaming) | 5 |
| 9B debug, perf, docs | 8 |
| **Subtotal: 9B on top of Q4** | **~28 (20-40)** |
| Optional: GGUF importer (imatrix Q4_0), Q4_1 or IQ4_NL variant | 4 + 3-4 each |

### Top risks
1. **Speed:** measured Q4/Q8 gain at 8 cores is 1.2x-2.3x, not 1.9x, because AVX2 Q4 is near compute bound here and the host is shared.
   9B Q4 ~10-17 tok/s is the honest range. A 9B is 2.6-4x slower than today's 2B in decode either way.
2. **Memory and boot:** 8.5 GiB guest at 32k vs 7-15 GB available on the host today. A >4 GiB ISO module is unverified (ISO9660/Limine).
3. **Quality:** RTN Q4 on a Gated DeltaNet hybrid has no published dense-model numbers. Recurrent-state error accumulation is a hypothesis
   to test at long context. The 2B will lose more than the 9B (Qwen3 study: 1.7B GPTQ 4-bit MMLU -4.3, AWQ -7.5; 8B GPTQ -1.3). A 9B fp
   reference needs RAM we do not have on Fedora.

### Recommended path
1. **2B, Q4_0 RTN mixed** (lm_head/embed Q8, f32 small tensors as today): exporter + mirror + KLD harness first, then the kernel.
   Gate: KLD/top-1 thresholds above, plus measured tok/s vs 36. Keep Q8 the default build. (~23 h)
2. If the KLD fails on DeltaNet/attention tensors: move them to Q8 (conservative plan), or try Q4_1/IQ4_NL (+3-4 h), or import bartowski's
   imatrix Q4_0 GGUF (+4 h).
3. Q4 GEMM for batched prefill once the other agent's Q8 GEMM lands (4-6 h).
4. **9B:** Hk != Hv DeltaNet, untied lm_head, memory/boot, reference. Q4 mixed (6.05 GB) is the only plan that fits a 10G guest next to the user's
   instance; Q8 would not. (~28 h)

**Is Q4 on the 2B worth it by itself?** As a product, no: estimated +15-70% decode (41-61 tok/s vs 36) for a measurable quality loss on a
2B model, and the 2B already fits. As a stepping stone, yes. The whole loop (export, mirror, KVM test) runs in seconds on the 2B, the kernel and
format carry over unchanged to the 9B, and it gives the first real KLD numbers for Q4 on our DeltaNet layers.

## Sources
- Qwen/Qwen3.5-9B config.json: https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json
- llama.cpp `ggml/src/ggml-cpu/arch/x86/quants.c`, `arch/x86/repack.cpp`, `src/llama-quant.cpp`, `conversion/qwen.py`, `tools/quantize/README.md`
  (master, fetched 2026-11-26): https://github.com/ggml-org/llama.cpp
- k-quants PR #1684 (PPL table, Q4_K/Q5_K/Q6_K block layouts): https://github.com/ggml-org/llama.cpp/pull/1684
- imatrix for legacy quants PR #4969 (Q4_0/Q4_1/Q5_0 PPL with/without imatrix): https://github.com/ggml-org/llama.cpp/pull/4969
- Unsloth Qwen3.5 GGUF benchmarks (KLD per quant, tensor sensitivity): https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
- bartowski Qwen3.5-9B GGUF (sizes, recipe): https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF
- An Empirical Study of Qwen3 Quantization, arXiv 2505.02214: https://arxiv.org/abs/2505.02214
- Quamba arXiv 2410.13229, MambaQuant arXiv 2501.13484, Mamba-PTQ arXiv 2407.12397
- Our code: `ringzero/engine/matmul.asm`, `smp.asm`, `transformer.asm`, `constants.inc`, `tools/export_weights.py`, `docs/M25_RESULT.md`,
  `docs/qwen35-2b/mirror/MIRROR_2B.md`
- Measurement scripts (Fedora `~/coding/q4study/`): `qbench.c` (matvec bench), `wq_err.py` (RTN weight error), `gguf_types.py` (GGUF header parse).
