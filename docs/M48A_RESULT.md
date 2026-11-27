# M4.8a: multi-stream batched decode

Decode reads every weight matrix once per token. Rows of different conversations can share that read: one engine
pass feeds one token of each stream, each at its own position on its own KV cache and DeltaNet state. This
milestone adds that pass, extra stream slots, and one user of it: agentd's idle consolidation runs as a second
stream next to the user's chat.

## Method

Engine (engine/prefill.asm): `transformer_prefill` takes an optional row table, `pf_ms` (6 x u64 per row:
position, K cache, V cache, DeltaNet state, conv history, max_seq). Before a row's attention step or DeltaNet step
the five g_ctx stream fields are pointed at that row's stream (restored on return), the row's position comes from
the table, and nothing is cleared at position 0 (the caller clears a new stream). The batched matmuls are the
M4.8 kernels: each weight row is read once and used for every row of the pass. Attention, the DeltaNet recurrence
and the conv step run per row with the vendored per-token kernels on that row's own state. Rows of one stream at
consecutive positions behave as the batched prefill (the per-row layer base now uses the row's max_seq, so a
stream may have a smaller cache than the agent's context).

Streams (kernel/src/multi.rs): stream 0 is the agent's context (the g_ctx buffers, 32768 positions, saved in
snapshots and the hibernation image). Up to 3 extra slots of 4096 positions are opened only while they are used.
Their memory (K and V for 4096 positions plus the DeltaNet state: 67 MiB on the 2B q8, 179 MiB on the 4B q4) is
lent by the snapshot pool (snap.rs `lend`: the lowest run of free 1 MiB chunks, only while 256 chunks stay free)
and given back when the slot closes. The first build allocated the slots at boot from the engine arena; on the 4B
q4 in an 8G guest the arena had 1319 MiB left after the model, which the embedder and the 1024 MiB snapshot pool
need, so it got no slot (boot line `0 slots`, `/multi` ran one stream). Lending from the pool costs nothing while
no slot is open. `step(rows)` builds the table for rows (stream, token, position) and runs one pass with logits
from a given row on, into the M4.8b logits rows (now 19: a 16-row verify plus one per other stream).
`fork(s, pos)` copies the agent's KV rows 0..pos of every plane (row stride = each stream's own max_seq) and its
DeltaNet state into slot s. Extra slots are scratch: never saved, gone after a reboot.

Why the output is the same: every row runs the same instruction sequence on the same values as when its stream
decodes alone. The batched matmul repeats the per-token kernel's integer and float operations per row
(docs/BATCHED_PREFILL_RESULT.md), the LM head of a pass is the M4.8b batched LM head (bit-identical to the per-token
one, docs/M48B_RESULT.md), and the per-row parts only read and write their own stream's state.

Background stream (multi.rs `Bg`, agentd.rs): an idle consolidation (after `!set consolidate` seconds at the prompt)
forks the conversation at the current point into slot 1 and queues the fixed instruction. It advances alone in
idle slices (`bg_run`: the instruction in blocks of 32 rows, then one row per pass) and, while the user's turn
decodes, as one extra row of every foreground decode pass (`llm::pass_bg`: a plain step or a speculative verify;
prefill passes of a user turn do not carry it). It is greedy (argmax of its own logits row), so the foreground's
sampler and RNG are not touched. Input at the prompt does not stop it; `/multi` does (retried later). When it is
done (EOS, 220 tokens or a full slot), the next idle slice turns its ids into text through the same capture as
before and stores the episode and facts. `/consolidate`, and a conversation that does not fit in 4096 positions,
keep the M4.5 path (on top of the current snapshot point, restored after). The session text and turns added while
it ran stay for the next consolidation. The finish line of both paths ends with `ids 0x...`, the FNV-1a of the
output ids.

Check (`/multi [N] [T]`, multi.rs `bench`): N prompts (4 fixed ones: three in English, one in
Italian), T greedy tokens each (64), EOS ignored. First each prompt alone on the agent's context with the old path
(batched prefill, then one `transformer_forward` per token), then all N together: each prompt prefilled on its own
stream, then one row per stream per pass. It prints the decode tok/s of both and, per stream, whether the T ids
are the same and whether the FNV-1a over every step's logits (the prompt's and T - 1 decode steps, every vocabulary
float) is the same. Stream 0 runs on the agent's context, so the shell restores the conversation after.

## Speed and exactness (`/multi`, 8 vCPU guests, 64 tokens per stream)

Build: this branch before its last change (`/consolidate &`, a shell command that these runs do not use). Fedora
box (Ryzen AI 9 HX 370, KVM), 8 vCPU guests (4B q4: 8G, 2B q8: 6G), `/multi N` twice for each N
in one boot per model (1, 2, 3, 4, then again). "alone" is every prompt decoded on its own with the per-token path
(tok/s per stream), "together" all N streams in one pass per token (aggregate tok/s), "ratio" = together / alone.

4B q4:

| streams | run 1: alone / together (ratio) | run 2: alone / together (ratio) | per stream together |
|---|---|---|---|
| 1 | 20.18 / 20.71 (1.02x) | 21.02 / 21.14 (1.00x) | 20.71, 21.14 |
| 2 | 21.16 / 37.39 (**1.76x**) | 20.59 / 38.23 (**1.85x**) | 18.69, 19.11 |
| 3 | 21.49 / 49.45 (2.30x) | 20.67 / 52.22 (2.52x) | 16.48, 17.40 |
| 4 | 19.90 / 51.39 (2.58x) | 21.14 / 59.58 (2.81x) | 12.84, 14.89 |

2B q8:

| streams | run 1: alone / together (ratio) | run 2: alone / together (ratio) | per stream together |
|---|---|---|---|
| 1 | 36.74 / 37.70 (1.02x) | 37.34 / 37.47 (1.00x) | 37.70, 37.47 |
| 2 | 36.87 / 71.38 (**1.93x**) | 35.53 / 67.26 (**1.89x**) | 35.69, 33.63 |
| 3 | 37.35 / 95.40 (2.55x) | 36.18 / 91.20 (2.52x) | 31.80, 30.40 |
| 4 | 36.98 / 120.74 (3.26x) | 36.95 / 109.58 (2.96x) | 30.18, 27.39 |

Two streams give 1.76-1.93x the tok/s of one on both models (an earlier boot of the same 4B kernel: 1.93x, 1.79x,
1.89x; an earlier 2B boot: 1.90x, 1.85x). The done criterion is 1.6x.

Exactness: in every `/multi` line of these boots (20 stream results on each model) the 64 ids and the FNV over every
step's logits were the same alone and together. The ids per prompt (FNV of the 64 ids) were the same in every run:
4B 0x6fc4711793c59d73, 0x91bffb787eeee682, 0x704fa6b55e2723ea, 0x669a189cb03b85f3; 2B 0x4f00238116eb9acb,
0x7d6d75227b577581, 0x57d98d456512060c, 0x558198287edd85d3.

The pass profile (`passes` and `matmuls` in the `/multi` line, the TSC counters of transformer_prefill) for the
64 steps: 4B 2 streams 3305-3376 ms in passes, of them 2517-2569 ms in matmuls; 4 streams 4204-4903 ms, 3151-3679
ms. The rest (attention, the DeltaNet recurrence and conv per row, norms) grows with the rows, the matmuls much less.

### The Q4 kernel for 1-4 rows (4B q4)

The first build used the M4.8 batch kernels for every pass. On the 4B q4 (MLP weights Q4_0) two streams gave 1.64x
and 1.49x in two runs of `/multi 2`, 3 streams 2.21x / 1.95x, 4 streams 2.32x / 2.55x, and one stream through the
batched pass ran at 0.91-0.97x of the per-token path. The AVX-VNNI batch kernel unpacks every Q4_0 row into a Q8_0
buffer and then computes a -128 * sum(w) correction per row: at 1-2 rows that work costs more than the dot
products. `matmul_q4v_range` (engine/matmul.asm) is used instead for Q4_0 weights at 1 to 4 rows on AVX-VNNI: the
nibbles are expanded to bytes q = 0..15 once per token pair and `vpdpbusd(-8 * sum(x), q, x)` gives per int32 lane
the same integer as the AVX2 path, sum((q - 8) * x) over the same 4-byte groups; the activations stay signed for
these calls and the quantizer writes -8 * sum(x) per group. The float part is the per-token Q4 kernel's sequence.
Same build otherwise, one run (`/multi 2` three times):

| 4B q4 streams | first build: aggregate vs alone | with matmul_q4v_range |
|---|---|---|
| 2 | 1.64x, 1.49x | 1.93x, 1.79x, 1.89x |
| 3 | 2.21x, 1.95x | 2.27x |
| 4 | 2.32x, 2.55x | 2.69x |

Ids and logits were the same in every run of both builds; `/spec 9` and `/spec 3` (3 verify rows: the new kernel)
gave 9/9 and 3/3 bit-identical logits and the same KV + DeltaNet hash, and the boot prefix prefilled with block 32
and with block 1 gave the same state hash (0x626718a163622aef) as the first build.

## Consolidation on the background stream

Two boots of the 2B q8 (6G guest, 8 vCPU), each on a fresh copy of the beta data disk (A on a build of this branch
before matmul_q4v_range and `/consolidate &`, B on the final build; the 2B q8 has no Q4_0 weights), the same inputs:
`!set confidence off`, `!set consolidate 0`, "my name is Anna and my favourite number is 4711" (the model calls
remember, two rounds, ctx 1769), then

- A: `/consolidate` (the M4.5 path, foreground, on top of the current point), then "write a short story about a cat
  called Miro in about 150 words";
- B: `/consolidate &` (the background stream: back to the prompt at once), then the same story prompt right away, so
  the consolidation decodes next to the story.

| | A: `/consolidate` | B: `/consolidate &` |
|---|---|---|
| consolidation output | 189 tok, ids 0x9642c87370e31e24 | 189 tok, ids 0x9642c87370e31e24 |
| where it ran | 5925 ms in the foreground, point restored in 580 us | 15488 ms wall, 103 of its 312 rows (123 instruction + 189 output) next to the story's decode passes, 209 alone in idle slices |
| story answer | 104 tok, ids 0xc1423893a260125e, 35.27 tok/s | 104 tok, ids 0xc1423893a260125e, 31.71 tok/s |
| snapshot pool after | | 84/1024 MiB (the slot's chunks given back) |

The consolidation output and the story are the same token for token on both paths. In B the story decoded 10%
slower while the consolidation got 103 tokens done in its passes; in A the user waited 5.9 s for the consolidation
before the story. The 2B wrote a run_code tool call instead of EPISODE / FACT lines for this session on both paths
(so no episode was stored); that is the model's output for this input, the same with and without this change.

An earlier boot (B0) with `!set consolidate 2` and the story typed after 4 s idle did not catch the consolidation in
flight: the idle queue embedded the new memory texts first, the consolidation started after the story (ctx 1904, two
turns), ran alone (162 rows, 1721 ms) and gave 39 tokens; the story ids were again 0xc1423893a260125e.

## Gates

Tool-call eval (`tools/tooleval.py`, M45 protocol as in docs/T55_RESULT.md: `SLOT=200`, constrain on, confidence
on, temperature 0, a fresh copy of the beta data disk, 120 s idle before the first prompt, 4B q4, 8G guest, 8 vCPU,
the build of the speed tables),
next to a run of the same protocol on the same night with the main kernel (another task's run, kernel sources at the
same commit as this branch's base, the stock 4B q4 export):

| | main | this build |
|---|---|---|
| spike: correct / first call right / args exact | 24/30, 22, 9 | **24/30, 22, 9** |
| in-OS (32): correct / first call right | 26, 24 | **26, 24** |
| of them the 8 info prompts: correct | 5/8 | **5/8** |
| calls (spike + in-OS) | 42 + 36 | 42 + 36 |
| malformed / schema errors | 0 / 0 | 0 / 0 |
| grammar lines not valid | 1 (an unfinished 460-token call) | 1 (the same) |
| turns at the round limit (spike + in-OS) | 1 + 1 | 1 + 1 |
| decode tok/s of the call rounds, spike / in-OS (separate boots, shared box) | 35.2 / 36.73 | 38.3 / 38.99 |

The eval runs with consolidation off (`!set consolidate 0`), so no background row rides along: it checks that the
changed decode path (logits rows, pass_bg returning false, the batch kernels) leaves every call as it was.

Boot self tests: the 2B golden self-test (`SELFTEST PASS`, prompt and generated ids match) and 11 boot self-tests
(engine ABI, Q4_0 kernel on the 4B, parser, grammar, index, wasm, proc, ...) pass on both ISOs of the speed tables. The agentd
self-test (record codec, fact validation, grounding, calendar) prints FAIL on both, and it prints FAIL the same way
in the same night's boot of the main kernel (the tooleval run above): it is not caused by this branch and is left
for a separate task.

Speculative path (M4.8b): `/spec 9` gives 9/9 bit-identical logits and the same KV + DeltaNet hash on the 4B q4
(0x4e72ab3d0f2f910c) and the 2B q8 (0x8d3d0d08b4190b2e); `/spec 3` (a 3-row verify, through matmul_q4v_range on
the 4B) 3/3. The agent turns of the consolidation boots (spec on) gave the same ids on every boot, with and without
the background row in their passes (table above).

rztest (`!run rztest check A "B C"` on the beta data disk): `rztest: PASS 20/20` (2B q8, the build of the speed tables).

Hibernation (2B q8, a build of this branch before matmul_q4v_range and `/consolidate &`, which do not touch the 2B's
paths or the image): "my name is Anna and I live in Turin", `/hibernate` (ctx 1762, state hash 0x746e975649c02b20),
`!poweroff`; the next boot on the same disk: `hib: resumed #1 on main (ctx 1762), 2 points in 145 ms, state hash
0x746e975649c02b20 (same as saved)`. The image format and the key are unchanged; the engine source hash in the key
changed with this branch, so an image written by an older build is not resumed (fresh conversation, as after
M4.8b).

## Limits

- One background job (consolidation). Embedding stays on the embedder's context, not a stream; the extra slots
  are not selectable from the shell.
- An extra slot holds 4096 positions: a longer conversation is consolidated on the old path.
- The background row rides only on decode passes; the prefill of a user turn, tool results and compaction's
  re-prefill do not carry it.
- While a slot is open its chunks count as used in the snapshot pool, so a save near a full pool thins or drops
  old points a little earlier; with fewer than 256 + slot free chunks no slot opens and consolidation uses the
  old path.
- The engine source hash changed (prefill.asm): the boot snapshot cache misses once on the first boot of this build
  and a hibernation image from an older build is not resumed. The main context's KV geometry and the image format
  did not change.
