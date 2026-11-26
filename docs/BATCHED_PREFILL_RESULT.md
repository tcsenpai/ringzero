# M4.8 base: batched prefill in the engine

Before: one `transformer_forward` per prompt token, so the ~1.4 GB of layer weights stream from DRAM once
per token and prefill runs at decode-like speed (~46 tok/s at 8 cores). Now `transformer_prefill`
(engine/prefill.asm) takes a block of B prompt tokens per pass and each weight matrix is read once per
block. Decode is untouched (`transformer_forward`, one token per call).

## What changed

| File | Change |
|---|---|
| engine/prefill.asm (new, AIOS) | `prefill_init(max_b)` (7 MiB of block buffers for B <= 64, AVX-VNNI detection), `transformer_prefill(toks, n, pos0, logits)`, TSC profile counters |
| engine/matmul.asm (vendored, AIOS-marked) | `quantize_x_q8_0_to` (the existing quantizer with caller buffers; `quantize_x_q8_0` falls into it), `quantize_batch_q8_0`, `matmul_q8_0_batch_range` (AVX2), `matmul_q8_0_batch_vnni` (AVX-VNNI), `bmm_*` job block |
| engine/smp.asm | job kind `tp_K = 2` (batch matmul) in the worker and BSP slice, `tp_matmul_q8_0_batch` |
| engine/transformer.asm | `rope_heads` and `g_dn_job` made global (2 lines); no code change |
| kernel/build.rs, engine.rs | assemble prefill.asm, FFI |
| kernel/src/llm.rs | `forward(tokens, pos0, logits)` (blocks of B, default 32) used by `run` (self-test, bench, `/gen`), `/fill` and `feed` (boot prefix, agent turns); `state_hash`; `/prefill` bench; STATE line in `/gen` and `/fill` |
| kernel/src/shell.rs, agent.rs | `/prefill [B] [avx2\|vnni]`; `agent::prefix_tokens()` |

Per layer and block: the input rmsnorm runs per token, the matmuls (`W_proj`, `W_out`, `W_gateup`,
`W_down`) get all n rows at once, and everything with sequence state runs per token in prompt order with
the vendored kernels: full attention = QK-norm, RoPE(pos), K/V store at pos, `tp_gqa_attn` over 0..pos
(causal inside the block and over the cached past, FP16 KV), output gate; Gated DeltaNet = a/b dots,
`conv4_step_avx2`, SiLU, `tp_deltanet` (recurrence), silu(z) gate. Those parts point `g_ctx`
BUF_Q / BUF_ATT_OUT at the token's row and restore them on return.

Batch matmul (one SMP job over the M weight rows, same row split as `tp_matmul_q8_0`): for each weight row
and each of the n tokens, the AVX2 kernel runs the exact instruction sequence of the decode kernel's
`.row_f16` loop on that token's quantized row, so the weight row is read from DRAM once and from L1 n
times. The AVX-VNNI kernel stores the activations as x + 128 (u8), computes C = -128 * sum(w) per 4-byte
group once per weight row, and gets each block's int32 lanes as `vpdpbusd(C, x + 128, w)` = sum(x * w),
the same integers that `vpabsb / vpsignb / vpmaddubsw / vpmaddwd` produce (no int16 saturation is possible
there, |pair| <= 2 * 127 * 127). The float part (scales, 4 accumulators, reduction) is unchanged. VNNI is
picked by CPUID (7.1:EAX[4]) at `prefill_init`; `/prefill 32 avx2` switches back for comparison.

## Correctness gate: bit-exact

The state hash is FNV-1a over 64-bit words of the K and V cache rows of positions < pos in all 6
full-attention layers, the whole DeltaNet recurrent state and conv history of the 18 DeltaNet layers, and
separately over the last-token logits. Block 1 is the old per-token path (`transformer_forward` per
token). One boot, `tools/boottest.py` driving `/prefill B` then `/gen 32 <lighthouse log prompt>`,
`/fill 1000`, `/fill 4000` for each block:

| Prompt | tokens | block 1 (per token) | block 32 (VNNI) | block 48 (AVX2) |
|---|---|---|---|---|
| boot prefix (system prompt + 14 tools) | 1140 | `d84dd4574357f4aa` / logits `cb1c89f9334b3d11` | same | same |
| `/gen 32` lighthouse log | 178 | `63edf9dbcae8a872` / `b021f3cb0753bd5c` | same | same |
| `/fill 1000` | 1000 | `45f4e169524b8d0d` / `d851d1a396eb2933` | same | same |
| `/fill 4000` | 4000 | `3d399e10399dd875` / `d50bee902a030da8` | same | same |

Greedy continuations are identical too: the 32 `/gen` ids and the 65 `/fill` ids match in every column
(e.g. `/gen`: 13962,20858,314,279,2743,271,760,1433,...). The boot prefix hash is also identical for
blocks 8, 16, 32, 64 (VNNI) and 32 (AVX2) in the other runs, block 48 exercises a last block of 36 and
the 19-token golden prompt a single short block. Golden self-test `SELFTEST PASS` (9419,1017,0) on every
boot, now through the batched path. No deviation to report: max |diff| = 0.

## Speed

8 vCPU guest (`-cpu host`, Ryzen AI 9 HX 370, KVM), FP16 KV. The Fedora box was shared with another
8-vCPU test VM during all runs (load average 5-12), so single numbers move by 10-30%; the sweep below is
one boot, interleaved, with the per-token path in the same boot.

Prefill tok/s (`/fill N`, user turn with a repeated paragraph, from position 0):

| block | kernel | 512 tok | 4096 tok | boot prefix 1140 (`/prefill`) |
|---|---|---|---|---|
| 1 (per token) | decode path | 46.7 | 43.5 | 46.1 |
| 8 | VNNI | 191.2 | 191.1 | 194.2 |
| 16 | VNNI | 222.6 | 201.9 | 220.2 |
| 32 (default) | VNNI | 232.8 | 205.2 | 228.1 |
| 64 | VNNI | 220.7 | 220.3 | 230.3 |
| 32 | AVX2 | 194.6 | 177.9 | 187.8 |

So 4.4-5x at B >= 16; past B = 16 the matmuls are compute-bound (B = 16/32/64 within noise). Profile of
the boot prefix at block 32 (VNNI): 4997 ms in `transformer_prefill`, 4252 ms of it in the batch matmuls
(quantize + dispatch + wait), the rest is the per-token attention / DeltaNet / norms. AVX2 -> VNNI cuts the
matmul time 5305 -> 4252 ms.

Boot (same boots as the decode A/B below): the agent prefix prefill (1140 tokens) takes
23.9 s (47 tok/s) on the base build and 5.0-5.2 s (218-228 tok/s) on this branch; banner-to-prompt boot
work drops by ~19 s accordingly. The golden prompt (19 tok) prefills in 107-133 ms instead of 470 ms.

Decode (boot `/bench`, 64 tokens, same prompt): see the A/B table below. The decode code path is the
same instructions (`transformer_forward`, `tp_matmul_q8_0`; the only touched lines are the entry of
`quantize_x_q8_0`, which now loads its two buffer addresses into rdx/rcx before falling into
`quantize_x_q8_0_to`, and one extra `cmp/je` on the worker's job kind).

Decode tok/s, `/bench` x4 + the boot bench per boot, boots interleaved base / branch on the shared box
(base = commit 0d0abf6; "no buffers" = this branch with `prefill_init` skipped, i.e. block 1 and the
base arena layout):

| round | base (10 samples) | this branch (10) | no buffers (10) |
|---|---|---|---|
| 2 (2 boots each) | median 36.7, max 38.1 | median 34.4, max 36.8 | - |
| 3 (2 boots each) | median 35.1, max 36.8 | median 35.4, max 38.2 | median 35.4, max 37.5 |

Round 2 suggested a 6% drop, round 3 reverses it with the same builds; the spread inside one build
(20-38 tok/s under the other VM's load) is larger than any difference. Decode is unchanged within the
noise of this box; a quiet-machine A/B is still worth doing before quoting a sub-5% number.

## Limits and next steps

- Attention and the DeltaNet head jobs still run per token (one SMP dispatch each); they are 15% of the batched
  prefill at 1140 tokens and grow with the prompt (4096: 205 vs 233 tok/s at 512). Batched attention over the block (one score matrix per head) is the
  next step for long prompts.
- The batch matmul needs FP16-scale (v2/v3) files; `prefill_init` returns -1 on v1 files and llm.rs falls
  back to block 1.
- snap.rs replay (thinned snapshots) and the embedder still use `transformer_forward` per token.

## Merge onto main (M4.5 agentd, M5 MicroPython, M4.7 grammar + hibernation)

The branch forked at 0d0abf6. Only docs/ARCHITECTURE.md conflicted textually; llm.rs, agent.rs and shell.rs
merged clean. On top of the merge every prompt prefill now goes through `llm::forward`: `feed` (boot prefix on a
cache miss, agent turns with the agentd memory slot, tool results, compaction summary and re-prefill), agentd
consolidation (new `llm::fwd_block`, one prefill block per idle step instead of one token, still stoppable
between engine calls) and the snapshot replay of a thinned point (snap.rs, blocks of B with Ctrl-C between
blocks; also the path a hibernation resume takes when a thinned point is visited). Per token remain: decode,
the 2 tokens that close the system turn, the embedder. The boot snapshot cache key includes the engine source
hash, so the merged build misses once (key `0xa28d6a5e0aa05cf2`, main `0xebd64e0555ffbf3e`), and a hibernation
image from an older build is not resumed (key mismatch, fresh conversation).

Fedora, 8 vCPU guest, box shared with another agent's job (load average 8-17), prefix now 1315 tokens (17 tools):

| Check | Result |
|---|---|
| golden self-test | PASS on every boot (19-token prompt 107-200 ms vs 675 ms on main) |
| `/prefill 1` vs `/prefill 32` (vnni) vs `/prefill 32 avx2`, boot prefix | all `0xd437b61790338fa6` / logits `0x10ff9908437b0e73`; 33.9 s (38.7 tok/s) vs 6.1-6.4 s (205-216 tok/s) vs 8.2 s (161 tok/s) |
| boot prefix prefill on a cache miss | 5.8 s (226 tok/s); main 40.3 s (32 tok/s), M4.7 deploy 36.3 s |
| cold boot (fresh disk), QEMU start to prompt | 28 s; main 71 s |
| cached boot | hit in 0.24 s, `What is my cat called?` after `/clear`: same memory slot, same 71-token prefill, same answer (`Miso`) as on the miss boot |
| hibernation | `!poweroff` saved #5, hash `0xed3f2f4bbc91a670`; next boot resumed 6 points, same hash; `/switch #3` replayed 345 tokens (3.6 s), `/switch #1` 166 tokens (1.8 s), R hash matches in both |
| agentd | `remember` x2 (valid constrained calls), `/consolidate` 1.8 s (2 facts, 1 episode), `/clear`, recall through the memory slot |
| snapshots | `/retry` at temp 0: same tool calls, same answer; `/undo`, `/redo` ok |
| `!py`, file tools | `!py print(sum(range(10)), 2**10)` -> `45 1024`; `write_file` + `cat` on /notes/merge.txt |
| decode `/bench` (64 tok), boots interleaved | main: 23.1-27.5 tok/s (median 24.3, 10 samples); merge: 24.5-30.0 (median 28.5, 11 samples) |

Decode is not slower; the higher merge median comes from the first merge boot running while the box was quieter
(the interleaved pair mC / mM2 gave 24.5-29.7 vs 24.1-27.5), so read it as unchanged within the noise. The replay of a
thinned point runs at about 95 tok/s including the restore copies, so short replays are dominated by fixed
costs; it was about 38 tok/s per token.
