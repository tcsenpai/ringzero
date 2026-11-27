# M4.8b: speculative decoding by prompt lookup

Decode reads every weight matrix once per token (about 2.4 GB on the 4B q4, about 48 ms per token). When the
model copies text that is already in the context (a file it just read, a tool result, code it is editing),
the next tokens can be guessed from the context and checked in one batched pass, which reads the weights once
for all of them. This milestone does that, with outputs identical token for token to one token per pass.

## Method

Draft (kernel/src/llm.rs `lookup`): the context is the token array of the live positions (REC, the same array
the snapshots store) plus the emitted tokens that are not fed yet. Over the part after the boot prefix, find
the earlier place that agrees longest with the end of the context (MATCH_MIN = 4 to 32 tokens, the latest one on
a tie); the tokens that followed it are the draft, at most `match length - 1` and at most the current draft
length (8 by default, `!set spec N`). A short match drafts little. The boot prefix (system prompt and tool
JSON) is not searched: its example call (`<function=example_function_name>`) made most drafts inside tool calls
wrong.

Verify (engine/prefill.asm): `transformer_prefill(toks, n, pos0, 2)` is the batched prefill with logits for
every row from `pf_lg0` on: the final RMSNorm per row, then the LM head for all rows in one
`tp_matmul_q8_0_batch` into 16 rows of logits (15.2 MiB from the engine arena). The pass is the pending
tokens plus the draft. Each row is then picked exactly as the per-token loop picks: the row is copied to
BUF_LOGITS, `pick_next` (argmax at temperature 0, top-p with the same RNG at temperature > 0), and inside a tool
call `gram::fix` (the M4.7a grammar), then the stop checks of the old loop (EOS, Ctrl-C, `</tool_call>`, max
tokens, full context) and the streaming. The draft is accepted while the pick equals it; the first pick that
differs is the next token, as it would have been without speculation.

Why the output is the same: every row's logits are bit-identical to a `transformer_forward` at that position
(the batched matmuls repeat the per-token kernel's integer and float operations, docs/BATCHED_PREFILL_RESULT.md,
and the LM head now goes through the same batch kernel), the picks run in the same order with the same
grammar state, and the sampler is called once per emitted token, so the RNG stream is consumed the same way.
This holds for greedy and for sampling, and with the grammar on. `/spec [N]` checks the first part directly:
N positions one `transformer_forward` each against one verify pass of the same tokens.

Rollback: on a partial accept the KV rows of the rejected tokens stay in the cache and are overwritten later
(attention reads positions 0..pos only). The DeltaNet recurrent state and conv history (24 layers, 50.3 MiB
on the 4B) cannot be cut back, so it is copied to a second buffer before every verify pass and copied back on a
partial accept; the accepted tokens are then fed again at the front of the next pass (they are known, so they
cost one more row of a batched pass, not a separate prefill). A full accept needs no copy back.

Gate (no separate acceptance-rate counter): the draft length grows by 2 after a full accept and shrinks by 1
otherwise; after a pass that accepted nothing, no drafting for 8, 16, ... 64 tokens; no draft without a match
of at least 4 tokens.

Toggle and stats: `!set spec on|off|N` (default on, N = longest draft, 1..15). The `[decode]` line of every
round ends with `spec A/D accepted (P%) in V verify, R rollback, verify T ms, ids H` (`spec off, ids H` when off);
H is the FNV-1a of the emitted token ids, the equality check below. `!set seed N` sets the sampler RNG (for
equality runs at temperature > 0).

## Verifier cost and exactness (`/spec N`, 4B q4, 8 vCPU guest)

`/spec N` prefills 184-198 tokens of the boot prefix, then runs N tokens one `transformer_forward` each (logits
hashed per position, state hash after), copies the DeltaNet state back, and runs the same N tokens as one verify
pass. Two runs per N, final build:

| N (rows) | N x transformer_forward | one verify pass | ratio | logits bit-identical | KV + DeltaNet state hash |
|---|---|---|---|---|---|
| 2 | 104 / 94 ms | 68 / 64 ms | 1.5x | 2/2, 2/2 | same |
| 3 | 143 / 140 ms | 69 / 74 ms | 2.0x | 3/3, 3/3 | same |
| 5 | 226 / 243 ms | 87 / 87 ms | 2.7x | 5/5, 5/5 | same |
| 9 | 424 / 446 ms | 179 / 137 ms | 2.8x | 9/9, 9/9 | same |
| 16 | 752 / 785 ms | 268 / 300 ms | 2.7x | 16/16, 16/16 | same |

So one decode step costs about 47 ms and a verify pass about 60 ms plus 10-15 ms per extra row: a fully accepted
8-token draft gives 9 tokens for about 150 ms. A pass that accepts nothing costs one pass instead of one step
(about 15-30 ms more for the 2-4 token drafts of short matches), which is what the gate limits.

Rollback cost (`/spec`, same build, 3 runs): copying the 51456 KiB DeltaNet state (24 layers, recurrent state +
conv history) takes 1.7-2.4 ms to save and 1.6-1.9 ms to restore (the first save after boot took 22 ms, once).
That is 1-3% of a 145 ms verify pass: the save runs before every pass, the restore only on a partial accept. The
accepted tokens of a partial accept are fed again as the first rows of the next pass; in the quoting answers
above that happened 1-13 times per answer. The alternative, one state copy per row inside the pass, would cost
about 1.7 ms per row on every pass and was not built.

## Equality (spec off vs on, token ids)

Every prompt runs on a fresh conversation (`/clear`), constrain on, memory slot and consolidation off, then the
emitted ids of every round (FNV-1a in the `spec` part of the `[decode]` line) are compared between `!set spec off`
and `!set spec on`. Prompt sets (12 prompts): quoting (copy /notes/casa.txt, report /notes/misto.txt verbatim,
read /demo/stats.py and write it with a variable renamed to another file, repeat a pasted paragraph), chat (4
open questions in Italian and English), tool calls (cpuinfo, run_code, write_file, ls). Rounds are the answer
plus every tool-call round of a prompt.

| Run (final build unless noted) | temperature | spec-on runs | rounds compared | mismatches |
|---|---|---|---|---|
| se6: all sets, off/on interleaved per prompt, 3 times | 0 (greedy) | 36 | 66 | 0 |
| se9: all sets | 0.7, `!set seed 4242` per prompt | 12 | 22 | 0 |
| se2 (build before the boot-prefix exclusion) | 0 | 12 | 22 | 0 |
| se4 (build before the boot-prefix exclusion) | 0.7 | 12 | 22 | 0 |
| se5, se7 (gate variants), chat + tool | 0 | 32 + 24 | 56 + 42 | 0 |

In every run the spec-off repetitions of a prompt also gave one id sequence each (the runs are deterministic).

The first run (se1) had one differing round: the rewrite prompt made the model overwrite /demo/stats.py in the
spec-off run, so the spec-on run read a different file (its tool result prefilled 252 tokens instead of 250).
The prompt now writes to /notes/stats2.py; with the same input there was no difference in any run. With
temperature 0.7 the answers differ from the greedy ones and are the same between off and on: the sampler is
called once per emitted token in both paths, so the RNG stream is consumed the same way.

## Speed (4B q4, decode tok/s per prompt, off and on interleaved)

Fedora box (Ryzen AI 9 HX 370, KVM, 8 vCPU guest, 8 GiB), shared with an idle second 4B guest. Each prompt ran
off, on, off, on, off, on in a row (se6), mean tok/s over the 3 runs per mode; ids identical in all runs.

Answer round only (the quoting round, after the tool call that read the file):

| Prompt | tokens | off tok/s | on tok/s | speedup | accepted / drafted | verify passes | rollbacks |
|---|---|---|---|---|---|---|---|
| copy /notes/casa.txt (1520 B) | 380 | 20.84 | 53.20 | 2.55x | 326/327 (99%) | 42 | 1 |
| report /notes/misto.txt verbatim (3099 B) | 512 (max) | 20.64 | 51.02 | 2.47x | 451/462 (98%) | 59 | 1 |
| rewrite /demo/stats.py with rows -> people (write_file call) | 276 | 20.93 | 34.98 | 1.67x | 197/236 (83%) | 43 | 13 |
| repeat a pasted 54-token paragraph | 54 | 20.31 | 39.32 | 1.94x | 44/47 (94%) | 7 | 0 |

Whole prompt (every round, including the tool-call rounds), same runs:

| Set | Prompt | tokens | off tok/s | on tok/s | ratio | accepted / drafted (on, one run) |
|---|---|---|---|---|---|---|
| quote | copia il contenuto di /notes/casa.txt | 408 | 20.81 | 48.07 | 2.31x | 327/329 in 43 |
| quote | leggi /notes/misto.txt e riportane il testo esatto | 540 | 20.67 | 47.49 | 2.30x | 452/464 in 60 |
| quote | leggi /demo/stats.py, rinomina rows in people, scrivi /notes/stats2.py | 334 | 20.86 | 30.95 | 1.48x | 201/249 in 47 |
| quote | Repeat this text exactly: ... | 54 | 20.70 | 40.06 | 1.94x | 44/47 in 7 |
| chat | che cos'e un kernel? spiegalo in 5 frasi | 94 | 20.30 | 20.15 | 0.99x | 0/0 |
| chat | Explain how a CPU cache works in 3 concise sentences. | 79 | 20.16 | 19.97 | 0.99x | 0/0 |
| chat | scrivi una breve poesia sul mare | 189 | 20.93 | 20.88 | 1.00x | 6/10 in 5 |
| chat | what are the pros and cons of microkernels? | 219 | 20.77 | 19.98 | 0.96x | 5/24 in 11 |
| tool | quanti core ha la CPU? | 21 | 18.75 | 18.70 | 1.00x | 0/0 |
| tool | quanto fa 2**100? usa python | 116 | 20.88 | 26.26 | 1.26x | 51/64 in 14 |
| tool | scrivi in /notes/spec.txt la frase ... | 57 | 19.50 | 20.45 | 1.05x | 8/15 in 6 |
| tool | elenca i file in /notes | 92 | 19.71 | 19.63 | 1.00x | 7/21 in 10 |

Noise: prompts that drafted nothing do exactly the same work in both modes and still came out between 0.93x and
1.07x across the runs (se5, se6, se7: "Explain how a CPU cache works" 0.93x, 0.99x, 0.98x; "che cos'e un kernel"
0.99x, 1.07x), so chat and tool rounds are within the run-to-run noise. The temperature 0.7 run (se9, one run
per mode) gave quote 2.07-2.18x (1.54x for the rewrite), chat 0.98-1.02x, tools 0.96-1.29x.

Tuning (same prompts, interleaved): the first gate searched the whole context including the boot prefix and drafted
from 2-token matches; tool-call rounds dropped to 0.93x ("quanti core": 4/14 accepted, the drafts came from the
template's `<function=example_function_name>`). Searching only the conversation and drafting `match - 1` tokens
from matches of 3 or more fixed it (above). A stricter gate (matches of 4, back-off 8..64 tokens) gave chat
0.97-1.07x and fewer tool gains (run_code 1.19x instead of 1.26x), so it was not kept.

## Tool-call eval

`tools/tooleval.py` with the M45_RESULT protocol (fresh copy of the test disk without `/sys`, files indexed
before the first prompt, constrain on, temperature 0, `SLOT=200`), 4B q4, spec on (default):

| | before (T35 build, docs/M45_RESULT.md) | this build, spec on |
|---|---|---|
| spike: correct / first call right / args exact | 22/30, 22, 9 | **22/30**, 22, 9 |
| in-OS: correct / first call right | 23/24, 20 | **23/24**, 20 |
| malformed / schema errors / grammar lines not valid | 0 / 0 / 0 | 0 / 0 / 0 |
| prompts with the same calls and arguments as before | | 30/30 spike, 24/24 in-OS |
| decode tok/s of the rounds that made a call (spike / in-OS) | 19.79 / 19.51 | 25.75 / 27.40 |

The calls are the same as on the build before this milestone, prompt by prompt (calls and parsed arguments
compared). The two speed columns come from different days on the shared box, so read the speed row as a rough
indication only; the interleaved runs above are the speed measurement.

## Limits

- The speedup needs text to copy from the conversation. Free writing stays at the per-token speed: at most a few
  2-3 token drafts per answer, most rejected. Text from the boot prefix (system prompt, tool JSON) is not used as a
  source.
- The gate is a heuristic tuned on these 12 prompts (draft `match - 1` up to 8, +2 / -1, back-off 2..32 tokens
  after a pass that accepted nothing), not an acceptance-rate model; longer drafts did not pay (16 rows cost about 6
  decode steps, see the table above).
- Every verify pass copies 50.3 MiB of DeltaNet state first, accepted or not; per-position states inside the pass
  would avoid the copy back and the re-feed but cost one copy per row.
- Only agent turns (`generate`: answers, tool calls, compaction summaries) use it; `/bench`, `/gen`, `/fill` and the
  self-test keep the per-token loop, so their numbers are unchanged.
- The speculative counters are per round and printed only; there is no cumulative counter.
- Memory: 15.2 MiB of logits rows and 50.3 MiB for the DeltaNet copy (4B) come from the engine arena at model
  init; the snapshot pool on the 8G guest is still 1024 MiB.
- The engine source hash changed (prefill.asm), so the boot snapshot cache misses once on the first boot of this
  build and a hibernation image from an older build is not resumed (fresh conversation, user files and memory
  are kept).
