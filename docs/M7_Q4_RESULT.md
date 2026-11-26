# M7: Q4_0 weights and Qwen3.5-4B

Branch `m7-q4-9b` (on the batched-prefill base 170d20d/4405eb8). Plan: docs/FEASIBILITY-q4-qwen35-9b.md (qwenbaremetal
repo). Mid-milestone the target changed from the 9B to **Qwen/Qwen3.5-4B** (user decision 2026-11-26); the 9B is a
documented future option, not built. The 2B Q8 default build is unchanged (golden `9419,1017,0` PASS, same state hashes
as M4.8).

## What changed

| Part | Change |
|---|---|
| `.asmbin v4` | v3 + a 2-bit quant type per descriptor slot in the reserved u32 (0 f32, 1 Q8_0, 2 Q4_0) + `HDR_OFF_LM_HEAD` (u64 at 216, untied lm_head, 0 = tied). A file is v4 only when it has a Q4 tensor or an untied lm_head, so Q8 exports stay v3 byte for byte (checked: the 2B Q8 export is `cmp`-identical to the shipped file). |
| Q4_0 format | llama.cpp block_q4_0 content in our split-row layout: `[K/2 nibble bytes][K/32 fp16 d]` per row (low nibble = element j, high = j + 16, stored q + 8), RTN `d = signed absmax / -8`, `q = clamp(floor(x/d + 8.5), 0, 15)`, the effective fp16 d is used for 1/d, fp16-underflow blocks flushed to 0. 4.5 bpw. W4A8: activations stay the engine's Q8_0 g32. |
| `tools/export_weights.py` | `--q4 ROLES` (attn, attn_out, dn, dn_out, mlp, mlp_down or all), multi-shard safetensors, Hv = r * Hk DeltaNet, untied lm_head writer (Q8_0, row chunks). Embeddings and lm_head always Q8_0; norms, conv1d, A_log, dt_bias, in_proj_a/b f32 as before. |
| `engine/transformer.asm` | v3 or v4; `desc_tag_q4` sets bit 0 of a slot pointer whose type is Q4_0 (tensors are 64-byte aligned); `CTX_LM_HEAD_PTR`; DeltaNet with Hv = r * Hk: V head h reads K head h / r (HF `repeat_interleave`), q/k of that K head are copied to the stack before the in-place l2norm (two V heads share it, possibly on two CPUs). |
| `engine/matmul.asm` | `matmul_q8_0_range` jumps to `matmul_q4_0_range` on a tagged pointer: the Q8 kernel's block sequence with the 32-byte weight load replaced by a nibble unpack (`vpsrlw`, `vinserti128`, `vpand`, `vpsubb 8`), so the integer lane sums and the float order are the Q8 kernel's. Batch kernels (AVX2 and AVX-VNNI): a Q4 row is unpacked once into a per-worker Q8_0 row (`q4_rowbuf[tid]`, 64 x 17 KiB .bss) and the unchanged Q8 inner loops run on it for the B tokens. |
| `engine/smp.asm`, `prefill.asm` | the batch job passes the worker tid; prefill tags slots like the decode loop and uses `CTX_LM_HEAD_PTR`. |
| `kernel/src/q4test.rs` (new) | boot check on v4 models: 64 rows of layer 0 W_gateup and W_down through the Q4 kernel vs the same rows unpacked to Q8_0 through the Q8 kernel, bitwise. |
| `kernel/src/main.rs`, `llm.rs`, `build.rs` | banner from `RINGZERO_EDITION` (Makefile); the 2B golden check only on the 2B Q8 (v3, dim 2048), other models print `SELFTEST n/a` and their ids. 3 small hunks. |
| `Makefile` | `MODEL=2b|4b`, `QUANT=q8|q4|q4all`, model file, export args, edition and guest RAM per choice; `make model` fetches/exports only the selected one; ISO with `-iso-level 3` (the 4B Q8 module is 4.19 GiB: Limine loads it fine). |
| `tools/mirror_q4.py`, `q4_kld.py`, `hf_agree.py` (new) | numpy mirror for Q4 / any Qwen3.5 config, HF fake-quant KLD harness, top-1 agreement of a boot log with HF. |

The quant choices (`QUANT`): `q8` every matrix Q8_0; `q4` the MLP (gate, up, down) Q4_0, attention and DeltaNet
projections Q8_0; `q4all` every projection Q4_0. Why `q4` is not "everything Q4": see Quality.

## Qwen3.5-4B (HF config.json)

hidden 2560, 32 layers (8 full attention: every 4th; 24 Gated DeltaNet), intermediate 9216, attention 16 Q / 4 KV heads
x 256 (output gate, partial rotary 64, theta 1e7), DeltaNet **16 K heads / 32 V heads** x 128 (the generalization the
9B also needed), conv 4, vocab 248320, **tied** embeddings, max_position 262144 (we run 32768). Two safetensors shards.

| File | Size | Module |
|---|---:|---|
| `qwen35-4b-q8_0-v3.asmbin` (`QUANT=q8`) | 4,494,217,152 B | 4.19 GiB, guest 8G |
| `qwen35-4b-q4_0mlp-v4.asmbin` (`QUANT=q4`) | 3,361,755,072 B | 3.13 GiB, guest 6G |
| `qwen35-4b-q4_0-v4.asmbin` (`QUANT=q4all`) | 2,711,637,952 B | 2.53 GiB, guest 6G |
| (2B) `qwen35-2b-q8_0-v3.asmbin` / `q4_0-v4` (all) | 2.0 GB / 1.33 GB | |

Export times on Fedora: 4B q4all 153 s, q4 84 s, q8 61 s (the 2B q4all 86 s).

## Kernel exactness

| Check | Result |
|---|---|
| Q4_0 kernel vs Q8_0 kernel on the unpacked rows (q4test.rs, every v4 boot) | 2B: slot 4 K 2048 and slot 5 K 6144 bit-identical; 4B: K 2560 and K 9216 bit-identical (e.g. row 0 `0x3f3392e0 / 0x3f3392e0`). PASS on every boot. |
| Batched prefill vs per token (state hash of KV + DeltaNet state + logits after the 1140-token boot prefix) | 2B q4all: block 1 / 32 VNNI / 32 AVX2 / 48 VNNI all `de1b5fe771186564 / 404157943a2f3ac2`; 4B q4all: block 1 / 32 VNNI / 32 AVX2 `b94ff09e7b835604 / 1162800c91af8ce7`; 4B q8: `ac7252473ad95206 / 641ef280e30f6e35` for all three. max abs diff 0. |
| 2B Q8 regression | golden `9419,1017,0` PASS, boot prefix hash `d84dd4574357f4aa / cb1c89f9334b3d11` = M4.8 (BATCHED_PREFILL_RESULT.md) for blocks 1, 32 VNNI, 32 AVX2, 48. |
| Engine vs numpy mirror (`tools/mirror_q4.py`, the exporter's Q4 rule, W4A8 lane sums), golden prompt, greedy | **4B q4all**: 4/4 tokens identical `9419,11,4099,0`, including a step with a top1-top2 margin of 0.0058 logits. **2B q4all**: first 6 tokens identical `9419,11,1204,628,353,1438`; step 7 differs (engine 30 = "!", mirror 488) where the mirror's margin is 0.062: a near-tie, the expected effect of the mirror's documented float drift (BLAS order in the DeltaNet recurrence, MIRROR_2B.md), not a format error (a layout error breaks step 1). |

So the Q4 kernels are exact against the validated Q8 kernel by construction and by test, and the format / exporter /
Hv = 2 Hk head mapping agree with an independent numpy implementation token for token up to near-ties.

## Quality vs HF (4B)

`tools/q4_kld.py`, HF Qwen3.5-4B in **bf16** on CPU (fp32 would need 18 GB; bf16 used 9 GB with 14 GB available):
reference log-probs on 16 x 512 tokens (8 prose: repo docs, 8 code: Python stdlib) plus 2 x 4096-token chunks scored
on their last 512 positions (DeltaNet over a long context), 9198 scored positions. Then every weight the engine
quantizes is replaced by its dequantized value with the exporter's rules (embed Q8_0 in every plan). Activation Q8 is
not simulated here (the mirror check covers it). KLD(ref || quant) in nats.

| Plan (`--q4` roles) | file | KLD mean | KLD p99 | top-1 | PPL (ref 7.690) | long chunks KLD |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0 everywhere (`QUANT=q8`) | 4.49 GB | **0.0013** | 0.0058 | **98.4%** | 7.690 | 0.0006 / 0.0015 |
| Q4: gate + up only | ~3.7 GB | 0.0162 | 0.122 | 94.2% | 7.776 | 0.0090 / 0.0259 |
| **Q4: MLP (gate, up, down)** (`QUANT=q4`) | 3.36 GB | **0.0275** | 0.197 | **92.7%** | 7.809 (+1.5%) | 0.0142 / 0.0328 |
| Q4: all but attn_out, dn_out | ~2.9 GB | 0.0417 | 0.330 | 91.5% | 7.886 | |
| Q4: all projections (`QUANT=q4all`) | 2.71 GB | **0.0610** | 0.468 | **89.8%** | 8.008 (+4.1%) | 0.0296 / 0.0855 |

(long chunks: code-long / prose-long.) Prose loses more than code at every plan (q4all: prose KLD 0.075, code 0.048).
The output projections (attention `o_proj`, DeltaNet `out_proj`) cost 0.019 of the 0.061, the attention and DeltaNet
input projections 0.014, `down_proj` 0.011: the Unsloth note that `attn_*` / `ssm_out` are sensitive holds for RTN on
the hybrid. Long-context chunks stay within 1.5x of the short-chunk KLD for q4 (0.033 vs 0.032 on prose), so no sign of
DeltaNet state error build-up at 4k.

Gate (the study's proposal, interpolated for a 4B: mean KLD <= 0.04, top-1 >= 92%): `q8` passes easily, **`q4` (MLP Q4)
passes** (0.0275, 92.7%), `q4all` fails (0.061, 89.8%). Hence `QUANT=q4` = MLP Q4, and `q4all` is offered as the
smallest / fastest option with the quality cost stated. Next step if more is wanted from Q4: llama.cpp's imatrix
(importing a calibrated GGUF Q4_0 is a byte repack, study section 4) or Q4_1 / IQ4_NL, not tried here.

The 2B (the Q4 test bench), same corpus and harness with an **fp32** HF reference (8 GB): Q8_0 KLD 0.0011, top-1 98.1%,
PPL 9.116 vs 9.119; MLP Q4 0.0335, 90.7%, PPL +3.1%; all Q4 0.0779, 87.0%, PPL +7.5%. The 2B loses more than the 4B at
every plan (as the study expected from the Qwen3 quantization study), and the Q8 rows agree between the bf16 4B and the
fp32 2B reference (0.0013 / 0.0011), so the bf16 reference does not hide much.

End-to-end, engine greedy output vs HF bf16 argmax on the engine's own trajectory (`tools/hf_agree.py`, the golden
prompt + 5 `/gen 48` prompts: CPU cache, Fibonacci in Python, Canberra, Italian translation, TCP vs UDP):

| Engine build | top-1 agreement | |
|---|---:|---|
| 4B q8 | 210 / 211 = **99.5%** | one flip, at position 28 of the CPU-cache answer |
| 4B q4 (MLP) | 196 / 211 = **92.9%** | first flips at positions 4 (golden), 0, 6, 16-17 |
| 4B q4all | 198 / 213 = **93.0%** | first flips at positions 1 (golden: "Hello!" vs "Hello,"), 6, 8, 23-25 |

On 211 greedy tokens q4 and q4all agree with HF equally often; the 9198-position KLD above is the finer measure
(q4 has half the KLD of q4all). Q8 is practically the HF model (one flip).

Sample (4B q4all, `/gen 48 What is the capital of Australia, and why is it not Sydney?`): "...While Sydney is the
country's largest city and its most famous economic hub, it was never designated as the capital. The decision to choose
Canberra instead of Sydney (or Melbourne)..." (4B q8: "...financial and cultural hub, it was never chosen as the capital
due to a specific historical dispute between the two most power...").

## Speed

8 vCPU KVM guest (`-cpu host`, Ryzen AI 9 HX 370), FP16 KV, two interleaved rounds of the five builds (one boot per
build and round, `tools/boottest.py`). The box ran another agent's 8-vCPU guest the whole time (load 2-11). Decode:
the boot bench plus `/bench` x 3 per boot (64 tokens, short context), 8 samples per build. Prefill: `/prefill 32`
(1140-token boot prefix, batched, AVX-VNNI kernel) and the agent prefix prefill at boot.

| Build | weights | decode tok/s median (range) | vs Q8 of the same model | prefill tok/s, `/prefill 32` (r1 / r2) | boot prefix prefill tok/s (r1 / r2) |
|---|---:|---:|---:|---:|---:|
| 2B q8 (default) | 2.0 GB | **33.9** (32.3-36.7) | 1 | 231 / 161 | 207 / 178 |
| 2B q4all | 1.33 GB | **52.1** (34.7-54.4) | 1.54x | 205 / 99 | 209 / 197 |
| 4B q8 | 4.49 GB | **14.0** (12.5-15.9) | 1 | 67 / 83 | 70 / 76 |
| 4B q4 (MLP) | 3.36 GB | **19.2** (18.5-20.6) | 1.37x | 66 / 64 | 92 / 74 |
| 4B q4all | 2.71 GB | **24.0** (22.7-25.7) | 1.71x | 83 / 47 | 77 / 85 |

Decode follows the bytes read per token (decode is DRAM bound at 8 cores, study section 3): 4B q8 -> q4all reads
1.66x fewer bytes and runs 1.71x faster; the 2B gets 1.54x for 1.50x fewer bytes. Earlier single boots under heavier
load (load 6-17) gave 4B q4all 22.4-23.9 and 4B q8 15.7-17.0 tok/s, the same ratio. Prefill is compute bound
(BATCHED_PREFILL_RESULT.md): Q4 there costs one row unpack per weight row and block, so Q4 and Q8 prefill are within
the noise of this box (the r2 dips are host load, they hit Q8 and Q4 alike). The 2B Q8 numbers match M4.8
(36 tok/s decode, 228 tok/s prefill) on the quieter round.

## RAM and boot

| Build | guest | module(s) | largest usable run (arena) | engine arena | snapshot pool | boot to prompt |
|---|---|---|---:|---:|---:|---:|
| 2B q8 | 6G | 2.0 GB + embedder 0.6 GB | 3482 MiB | 406 MiB | 1024 MiB | 20.1 / 20.7 s |
| 2B q4all | 6G | 1.33 GB + 0.6 GB | 4096 MiB | 406 MiB | 1024 MiB | 18.7 / 14.0 s |
| 4B q4 | 6G | 3.36 GB + 0.6 GB | 1406 MiB | 1078 MiB | 128 MiB | 30.7 / 33.4 s |
| 4B q4all | 6G | 2.71 GB + 0.6 GB | 1509 MiB | 1078 MiB | 256 MiB | 30.4 / 28.5 s |
| 4B q8 | 8G | 4.49 GB + 0.6 GB | 1857 MiB | 1078 MiB | 512 MiB | 41.6 / 38.1 s |

Boot to prompt = guest start to the shell prompt (Limine loading the modules from the ISO is 3.5-16 s of it, then
model_init, self-tests, the 1140-token agent prefix prefill and the first snapshot).

The 4B KV cache at 32768 tokens is 1 GiB (8 full-attention layers x 4 KV heads x 256 x 2 (K, V) x FP16 = 32 KiB/token,
vs 12 KiB/token on the 2B); the DeltaNet state is 24 x 32 heads x 64 KiB = 48 MiB. The snapshot pool takes what is left
of the largest usable run (it halves from 1024 MiB until it fits), so the 4B on a 6G guest boots with a smaller pool;
`GUEST_MEM=8G` gives the 4B q4 at least the 512 MiB pool the 4B q8 gets at 8G (not measured for q4). The module is loaded contiguously by Limine: a >4 GiB file
(4B q8) works with `xorriso -iso-level 3`, no split needed.

## Limits

- RTN only; no calibration. `q4all` misses the quality gate.
- The Fedora box was shared (another 8-vCPU guest, load 2-17) during every run; single numbers move 10-40%. The speed
  table is two interleaved rounds of the five builds.
- The HF reference is bf16 (fp32 does not fit next to the guests); KLD for Q8 (0.0013) is the noise floor of that
  reference plus Q8 itself.
- 9B: not built. Everything it needs except memory/boot is in this branch (Hv = 2 Hk, multi-shard, untied lm_head,
  >4 GiB module). See docs/FEASIBILITY-q4-qwen35-9b.md.
