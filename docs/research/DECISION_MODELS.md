# Decision models: a third engine next to the agent and the embedder? (T49)

Date: 2026-11-27. Question from the user: small, fast "decision models" that return typed decisions (a choice, a
yes/no, a score, each with a probability), one of them called something like "laya", others built on Qwen3 0.6B
or 2B. Could one run as a third engine next to the 4B agent and the embedder, and where would it help?

Labels: **[V]** verified in this run from the source given (Hugging Face model API, `README.md`, `config.json`,
safetensors headers, code in the model repository, or our repo); **[S]** self-reported by the authors (their card
says it; the number itself is not checked); **[M]** measured in this run (section 5); **[E]** my estimate;
**[U]** unverified. Benchmarks from different cards use different sets and are not comparable with each other.

## 1. Summary

- "laya" exists: `convaiinnovations/laya` (Apache-2.0, first published 2026-09-18) [V]. It is **not** a Qwen
  model: the published checkpoints are BERT-style encoders (ModernBERT-large, 421M; mmBERT-base, 322M) with a
  decision head [V card]. Our engine has no encoder ops, so laya itself would need new kernels.
- The same idea on Qwen3-0.6B exists several times (September to November 2026). The useful pattern for us
  (Decision-2.0-Kai-0.6B, TinyJev-0.6B): a stock Qwen3-0.6B backbone, the same layer shapes as our embedder, plus
  a head of 0.5 to 1 M parameters that reads the final hidden state at a few token positions [V headers, code].
  Engine cost: the exporter (as for the embedder) and a small head in Rust; no new asm kernels [E].
- Measured offline on our 240 confeval tasks (first tool, or no tool), zero-shot, our own tool descriptions as
  the options [M]: the stock **Qwen3-Reranker-0.6B** (one yes/no per tool) picks the right first tool on 85.4%
  (held-out 89.2%), Decision-2.0-Kai-0.6B on 82.9% (83.3%), our embedder's similarity on 60.4% (61.7%). The 4B
  with the grammar is at about 87% on the same tasks (T28: 193 calls, 16 with the wrong tool, 1 wrong argument,
  14 missing calls). Both decision models give a usable probability (AUROC 0.79 and 0.81 against their own
  errors). The reranker is the same family, size and tokenizer as our embedder, Apache-2.0, and needs only two
  logits: it is the best candidate, not laya.
- Cost: asking over all 19 tools is 810 (Kai) to 2,080 (reranker) prompt tokens, an estimated 1 to 3 s in the
  kernel with batched prefill and 5 to 12 s without it [E], next to a 4B call round of 1 to 3 s. A question over
  2 or 3 candidates is about 200 to 300 tokens, 0.3 to 0.4 s batched [E].
- Verdict: no third engine now. A decision model does not beat the 4B's first call (it also does not fill the
  arguments), and in front of every turn it would cost about as much time as the call round it checks. The one
  place with a plausible gain is a second opinion on the calls whose T28 margin is below 0.80 (13 to 17% of
  calls): between the 4B's two outcomes and "none", to cut y/N questions on right calls and catch some wrong
  ones. That gain is not measured yet: go for one paired experiment on the host (section 6), no-go for kernel
  work until it shows one.

## 2. What exists (survey)

All rows: Hugging Face model API and card fetched in this run [V] unless marked. "Head" = what turns the
backbone into a decision.

| Model (repo) | Base, size | License | First published | Decision and output | Head | Evidence (self-reported unless [M]) | Latency (authors' hardware) |
|---|---|---|---|---|---|---|---|
| convaiinnovations/laya | ModernBERT-large encoder, 421M | Apache-2.0 | 2026-09-18 | choice, noul (yes/no), score; probabilities, abstain, conformal set, auto/review route | 2 transformer layers + option-marker scorer, one forward pass | typed-decisions 0.362 zero-shot (the fine-tuned checkpoint 0.766); ships over-confident, ECE 0.466 before refitting temperatures, 0.081 after [S]; "near chance on typed-decisions zero-shot" in its own limits [S] | 39.5 ms per question on a T4 GPU, 193-464 ms on CPU [S] |
| convaiinnovations/laya-multilingual | mmBERT-base encoder, 322M | Apache-2.0 | 2026-09-19 | same; card lists Italian among its languages | same | MASSIVE intent, 13 non-English languages 0.451; XNLI, 14 languages 0.731 [S] | 32.8 ms on a T4 [S] |
| vllm-sr/Decision-2.0-Kai-0.6B | Qwen3-0.6B-Base, 0.60B, full fine-tune (backbone stored as `Qwen3Model`) | Apache-2.0 | 2026-09-28 | choice (2..255 options), noul, score; a probability per option | bilinear + MLP scorer over the hidden state at the last token of each option and of the prompt, 1.05M fp32 params [V header, code] | JevArena 48.6, top of 5 models of its size [S]; **[M] section 5** | 4.9 ms median on a GPU [S]; 1.48 s median on our Mac CPU, PyTorch [M] |
| AnkitAI/TinyJev-0.6B | Qwen3-0.6B-Base, LoRA r16 merged | MIT | 2026-09-22 | choice, noul, score, calibrated | pointer head (q, k linear 1024->256), 0.52M params [V header] | OD-500 88.0%; transfer-v4 test 0.663, ECE 0.082 [S] | 85 ms a question on an M1 (MLX) [S] |
| Berk/assay-0.6b | Qwen3-0.6B-Base + LoRA | Apache-2.0 | 2026-09-22 | bool, choice, score; abstains with conformal thresholds | reads next-token logits over option labels; a 1025-param evidence head [V header] | unseen tasks 0.704 accuracy, ECE 0.037 after temperature [S] | (RTX 5090 table) [S] |
| LibertAIDAI/deem-0.6-v1 | Qwen3-0.6B-Base | Apache-2.0 | 2026-09-27 | choice, score, yes/no with abstention | letter logits [S] | standard-suite macro 0.7533 [S] | 65 ms short decisions on a desktop CPU, int8 [S] |
| LibertAIDAI/deem-0.8-v1, labqoat/qev-0.8b, chaoliangUNSW/Jev-Style-0.8B-Decision-v3 | Qwen3.5-0.8B (24 layers, Gated DeltaNet 16 K / 16 V heads of 128, full attention every 4th layer, vocab 248320) [V config] | Apache-2.0 | 2026-09-24 to 2026-10-20 | choice, noul, score | own scorers [U] | qev: 77.1% on "Benchmark Heaven" [S]; Jev-Style v3: MASSIVE 37 held-out languages 65.5% [S] | qev 3.5 ms on an L40S, ~17 ms llama.cpp on an M4 Pro [S] |
| chaoliangUNSW/Jev-Style-2B-Decision-v4 | Qwen3.5-2B (GGUF card) | Apache-2.0 | 2026-10-27 | same | bundled scorer | trained partly on soft labels distilled from TypeSafe Jev 1.14 [S]; 84.4% on its 1,000-question check [S] | (GPU) |
| Qwen/Qwen3-Reranker-0.6B | Qwen3-0.6B-Base, causal LM, vocab 151669 (the embedder's) [V config] | Apache-2.0 | 2025-05-29 | relevance: P("yes") vs P("no") at the last token for (instruction, query, document) [V card code] | none (2 logits of the tied LM head) | retrieval reranking [S]; **[M] section 5** | one forward per (query, document) pair; 3.05 s median for our 19 pairs on the Mac CPU [M] |
| Qwen/Qwen3Guard-Gen-0.6B | Qwen3-0.6B, causal LM | Apache-2.0 | 2025-09-23 | safety: generates "Safety: Safe/Unsafe/Controversial" + categories (+ "Refusal") [V card] | generated text, parsed | 119 languages, state of the art on safety sets [S] | decode of a few tokens |
| Qwen/Qwen3Guard-Stream-0.6B | Qwen3-0.6B + per-token classification heads | Apache-2.0 | 2025-09-23 | risk level and category per streamed token [V card] | `Qwen3ForGuardModel` heads | [S] | per token, alongside generation |
| katanemo/Arch-Router-1.5B | Qwen2.5-1.5B-Instruct | other (katanemo) | 2025-05-30 | route name as generated JSON `{"route": ...}` over user-defined routes [V card] | generated text | [S] | decode |
| TypeSafe Jev | closed, hosted API | commercial | (docs 2026) | "System One" model: state + typed questions (Choice, Score, Noul, Box), probabilities, no text [V docs.typesafe.ai] | unknown | laya's card quotes third-party p50 236-276 ms [S] | hosted only: not usable in ring 0 |

Notes:

- The name. The user's "laya" is laya by Convai Innovations; its card links a preprint, "From Laya to Vega: Open
  Non-Autoregressive Decision Models with Calibrated Probabilities" (Preprints 2026110781,
  doi 10.20944/preprints202611.0781.v1). The DOI resolves to preprints.org, which refused the fetch: the paper's
  content is [U]. The 0.5 release adds a "vega" engine that puts a small decision engine on top of a frozen LLM
  "reader" from 0.8B to 27B [S]; that variant was not inspected.
- All of this family follows the interface TypeSafe published for Jev (state + typed questions, `choice`,
  `noul`, `score`, one forward pass, probabilities). Several cards say so directly (TinyJev, Jev-Style, laya-serve
  is "Jev-compatible").
- Model cards in this space contradict each other on who leads; almost all numbers are on the authors' own sets
  and several were used to choose checkpoints (qev's card says so). Only section 5 is ours.
- Guard models (Qwen3Guard) answer a question ringzero does not have today: there is one local user and the
  risky actions are already gated by y/N. Routers (Arch-Router) generate JSON, which is slower than our own
  grammar-constrained 4B and adds nothing the grammar does not already give.

## 3. Where a third engine could help in ringzero

What we already have, from the repo [V]: the tool-call grammar (no malformed calls, T39 drafts), the T28 margin
(zero extra passes, AUROC 0.80 held-out against wrong calls, 0.89 against the wrong tool; a margin below 0.80
asks y/N before a writing tool; 3 of 8 held-out wrong calls had a margin of 0.94 or more and are invisible to
it), the embedder (recall, memory slot, manual lines; `RECALL_MIN` 0.55), agentd's lexical `grounded()` rule for
facts, and the `howto()` keyword rule for manual lines.

Latency scale [E, from our numbers]: the embedder (same 0.6B shapes) runs about 6 ms per token with one forward
per token (docs/EMBEDDER_RESULT.md). Batched prefill took the 2B from 32-46 to 205-226 tok/s
(docs/BATCHED_PREFILL_RESULT.md); the embedder does not use it yet. If the same 4-5x holds for the 0.6B, a
decision prompt runs at about 700-800 tok/s. A Kai-style prompt with our 19 options is about 810 tokens [M], so
about 1.0-1.6 s batched, about 5 s per token; the reranker's 19 pairs are 2,080 tokens [M], though the pairs share
their first part (system text, instruction, query), so with a snapshot after the query about 1,000 tokens remain
[E]. A question over 2-3 candidates is 200-300 tokens, 0.3-0.4 s batched. For scale: a 4B call round is 1-3 s.

| Use | What it would decide | What we have | Expected gain | Cost per use [E] | Verdict |
|---|---|---|---|---|---|
| Tool choice before the 4B generates | which tool (or none) | the 4B under the grammar, ~87% first call right | none or negative: reranker 85%, Kai 83% zero-shot [M], no arguments; the tool list is in the cached boot prefix, so pruning it saves nothing | 1-3 s every turn | no |
| Risky-call gate (second opinion) | is the 4B's pick right, between its top two outcomes and "none" | T28 margin + y/N | possible: fewer questions on right calls in the band (12 of 90 right held-out calls asked at 0.80), maybe some wrong calls caught. On the 4 confident 4B mistakes named in T28 the reranker picks the right tool with p 0.70 and 0.83 on two and is wrong on two; Kai is unsure on all four (p1 <= 0.31) [M] | 0.3-0.4 s, only on band calls (13-17% of calls) | the one candidate: measure first (section 6) |
| Needs a tool or memory at all | answer directly vs call / inject the memory slot | the 4B decides; agentd always builds the slot | small: "none" as an option is 94.2% (reranker) and 94.7% (Kai) right on this binary [M], about the 4B's level (14 missing calls); Kai's `noul` answer to the same question was unusable at 0.5 (27.9%, AUROC 0.77) [M]. Skipping the 200-token slot saves about 1 s of 4B prefill; the decision costs as much | ~1 s | no |
| Kernel event bus routing (T21, not built) | which events reach the agent | nothing yet; the roadmap asks for a deterministic perimeter | low: kernel events are structured; rules route them exactly and auditably. A model only helps for free-text payloads (log lines, file contents) | per event | not now; revisit if T21 needs free-text triage |
| Memory-write grounding | did the user really say this fact | `grounded()`: numbers must appear, half the words of 3+ letters | uncertain: catches paraphrase cases the rule misses, but the rule never invents a pass; a model adds a probability to a check that is now exact | 0.15-0.3 s per remember | not now; a candidate question for the section 6 run |
| Language detection | it vs en | the 4B answers in the user's language; the manual uses keywords | none: no failure on record | - | no |

## 4. Engine cost

### Qwen3-Reranker-0.6B: converter only, the smallest step

- Same shapes and tokenizer as the embedder (28 layers, hidden 1024, vocab 151669, tied embeddings) [V configs];
  the `--v3` export path applies as it does for the embedder [E].
- Output: P(yes) = softmax over two logits, the rows of "yes" and "no" in the tied embedding matrix times the
  final hidden state [V card code]. In the kernel: the per-token forward with `compute_logits 0`, then RMSNorm
  with `w_final` and two 1024-long dot products in Rust. Unlike the embedder, the RMS scale does not cancel here,
  so the norm must be applied in full. No asm change [E].
- Prompt: Qwen's fixed system line, `<Instruct>: ...\n<Query>: ...\n<Document>: ...`, then the assistant turn
  with an empty think block [V card]. The query comes before the document, so for several candidates the shared
  part can be prefilled once and each candidate continues from a snapshot of it (snap.rs already does this for
  the 4B) [E].
- RAM, arena, cores, boot image: as for Kai below (a second 0.6B module of about 613 MiB, the embedder's arena).

### Qwen3-0.6B decision models (Kai, TinyJev): converter only

- Backbone: `Qwen3Model`, 28 layers, hidden 1024, 16 heads / 8 KV heads of 128, FFN 3072, SiLU, tied
  embeddings, vocab 151936 [V config]. These are the shapes of Qwen3-Embedding-0.6B, which already runs as the
  second `g_ctx` (exported with `--v3`). The exporter path exists [V repo]; Kai ships its backbone in fp32
  (`backbone/model.safetensors`, 2.4 GB) [V], so we would quantize it to Q8_0 like the embedder. Accuracy after
  Q8_0 is [U].
- What the head reads: the final hidden state (after the last RMSNorm) at the last token of each
  `<option>...</option>` block and at the last prompt token (`Decision:`) [V decision_model.py `encode`,
  `forward`]. The head starts with a LayerNorm, which is invariant to the per-vector RMSNorm scale, so the
  embedder's trick (read `x * w_final` from BUF_X, skip the LM head) works here too: per-token forward, read the
  buffer after the marked tokens. No asm change for the per-token path [E]. A batched path needs the prefill
  to hand back the hidden rows of chosen positions; prefill.asm already keeps every row for the M4.8b verify
  logits, so it is a small change, but whether `transformer_prefill` runs the dense Qwen3 layout is [U].
- Head in Rust: 2 LayerNorms, 4 matrices 256x1024, one GELU over 256 values, a dot product: about 1 M
  multiply-adds per option, fp32, 4 MiB of weights [V header]. Then softmax with the shipped temperature.
- Prompt format: `Context:\n{state}\n\nTask type: {type}\nQuestion:\n{instructions}\nOptions:` + per option
  `\n<option>\n{"description":...,"key":...}\n</option>` + `\n\nSelect the single option best supported by the
  context and instructions.\nDecision:` [V code]. The state comes first, so the options cannot be cached across
  requests: every question pays for all its option tokens. A head trained with the options first would let a
  snapshot hold the option rows and cut a question to its state tokens; that needs our own training [E].
- RAM: weights about 613 MiB in Q8_0 (same size as the embedder module, 643 MB) [E]. Arena: the two 0.6B
  models never run at the same time, so the decision model can reuse the embedder's arena (113 MiB at 1024
  tokens; our prompts are about 810 tokens) [E]. The 4B's KV cache and DeltaNet state are not touched: the
  `g_ctx` swap between whole forward calls already guarantees that for the embedder [V embed.rs].
- Cores: one `g_ctx` at a time on all engine cores; a decision blocks the agent for its duration, as an embedding
  does today. It belongs between forward calls on the BSP, like `with_embedder`.
- Boot image: one more Limine module of about 613 MiB. The embedder added +3.3 to 3.7 s to the banner and left
  about 630 MiB free on the UEFI layout of a 6G guest with the 2B (docs/EMBEDDER_RESULT.md); the 4B needs an 8G
  guest (INFO.md). Whether a third 0.6B fits next to the 4B and keeps the 1 GiB snapshot pool on UEFI is [U].
- Tokenizer: Qwen3's BPE, the embedder's, already in the kernel [V repo].

### Cheaper variants that need no third model

- Head on the embedder's own backbone: train a Kai-style head on frozen Qwen3-Embedding-0.6B hidden states.
  Zero new weights (a 4 MiB head), same latency as above. laya's 0.5 "vega" engine claims a frozen reader works
  [S]. Needs training data.
- Probe on the 4B's own state: the 4B already prefills the user turn; a linear head on its last hidden state
  could predict "tool vs none" or the tool, with no extra forward pass at all, as T28 reads the logits it already
  has. Needs training data from the 4B's hidden states (host, PyTorch).

### Other bases

- Qwen3.5-0.8B decision models (deem-0.8, qev, Jev-Style 0.8B): same architecture class as our 4B (DeltaNet
  K = V = 128, 16 heads, within the engine's `Hv <= 32`) [V config], and the same tokenizer as the 4B. Converter
  only on the layer side; their scorers were not inspected [U].
- laya (ModernBERT / mmBERT): an encoder: bidirectional attention and a different tokenizer, none of which the
  engine has (docs/research/MODEL_SURVEY.md, section 1: no GELU, no sliding window; byte-level BPE only). The
  exact op list was not checked in this run [U]. Class D: new kernels. Not worth it while Qwen3 options exist.

## 5. Experiment (run in this task)

Offline on the Mac host, CPU only (PyTorch 2.15, transformers 5.24.1, 8 threads, fp32 weights), no guest. The
240 tasks of tools/confeval.py with their labels: the accepted first tools, "none" when answering without a tool
is right (`''` or `-|...`), even index tune, odd held-out. One choice question per task with 19 options: the 18
tools with the kernel's own descriptions from kernel/src/agent.rs, plus `none` ("answer directly ... or nothing
can do it"). Zero-shot: no tuning on our data, no fitted temperature. Script: `tools/dmeval.py` (`kai`,
`rerank`, `embed` write a JSON per model; `report` prints the numbers below). The reranker scores each of the 19
(request, tool) pairs and the probabilities are normalized over the tools; the embedder's softmax uses a fixed
temperature of 0.05, not fitted.

| | Kai-0.6B (decision head) | Qwen3-Reranker-0.6B (yes/no per tool) | Qwen3-Embedding-0.6B (cosine, what we have) |
|---|---|---|---|
| first tool right, all 240 | 0.829 | **0.854** | 0.604 |
| first tool right, held-out 120 | 0.833 | **0.892** | 0.617 |
| right in the top 2 (all) | 0.904 | 0.904 | 0.746 |
| tool tasks with the right tool | 162 / 199 | 171 / 199 | 119 / 199 |
| no-tool tasks answered `none` | 26 / 27 | 22 / 27 | 15 / 27 |
| AUROC, top probability vs right (all) | 0.811 | 0.786 | 0.808 |
| right when p1 >= 0.70 (share of tasks) | 0.988 (34%) | 0.953 (54%) | 0.929 (17%) |
| input tokens per task (median) | 813 | 2,078 (19 pairs) | 30 (tool vectors precomputed) |
| CPU time per task, Mac, PyTorch (median) | 1,480 ms | 3,052 ms | 164 ms |

Where the two decision models agree (194 of 240 tasks), the shared pick is right on 91.8%; when they disagree
(46), one of the two is right on 41 [M].

The 4B on the same tasks (T28, in the kernel, grammar on): 193 calls, 16 with the wrong tool and 1 more with a
wrong argument, 14 tasks without a needed call, 33 right answers without a tool: about 210 of 240 (87%) with the
right first decision. Not paired per task: the per-task 4B results of the T28 run were not kept, so the
"second opinion" question (does Kai disagree when the 4B is wrong?) can only be read on the four confident
mistakes T28 names:

| Task (T28: the 4B's confident wrong call) | Reranker's pick, p1 | Kai's pick, p1 | Embedder's pick, p1 |
|---|---|---|---|
| list /demo (4B: `run_wasm /bin/ls`, margin 0.94) | **ls 0.70** | ls 0.31 | run_wasm 0.19 |
| ricordati che lavoro come insegnante (4B: `recall`, 0.88) | forget 0.50 | **remember** 0.31 | none 0.42 |
| slow down the inference (4B: `set_setting temperature`, 1.00; right is cores or no tool) | set_setting 0.83 | none 0.28 | set_setting 0.93 |
| is the disk full? (4B: `meminfo`, 0.86; right is ls or no tool) | recall 0.55 | recall 0.19 | meminfo 0.27 |

(For "slow down the inference" the tool is right and the 4B's error was the key, which this question does not
ask.) Typical held-out errors of both: directory listings sent to `recall` ("what's in /notes?"), "use all 8
cores" to `list_cores` or `cpuinfo` instead of `set_setting`, memory questions to `none` ("come si chiama il mio
gatto?"), "copy /etc/motd ..." to `write_file`. Most are the near-ties the tool descriptions leave open, the same
ones that give the 4B low margins, so the errors may well overlap with the 4B's: the paired run is what decides.

What this does not show: the latency in the kernel (only estimated, section 3), the effect of Q8_0, results on a
tuned temperature or a fine-tuned head, and above all the paired question.

## 6. Go / no-go

- No third engine in the kernel now: no use in section 3 has a measured gain, and the only plausible one (the
  second opinion on band calls) depends on how the decision model's errors line up with the 4B's.
- Go for one prototype task, host only, no kernel change: rerun confeval on the 4B (the existing harness, which
  prints the first call and the `[confidence]` line) and keep the per-task results; on the same tasks ask the
  reranker (and Kai, for comparison) a short question over the 4B's pick, its runner-up from the margin line and
  `none`. Gate rule to test, thresholds picked on the tune split as in T28: "run without asking if margin >= 0.80,
  or if the decision model agrees with the 4B's pick at p >= X; ask if it prefers another option at p >= Y".
  Success: fewer questions on right calls at the same or better catch of wrong calls on the held-out split, with
  the short question under about 300 tokens. Add the `remember` grounding question on the in-OS fact prompts in
  the same run. Cost: one confeval run (11 minutes in the guest) plus about 10 minutes of CPU on the host.
- Only if that shows a gain: export the reranker with the embedder's path, run it per token in the embedder's
  arena, yes/no logits in Rust, and compare in the guest against a head trained on the frozen embedder (zero new
  weights) before adding a 613 MiB module.

## Sources (fetched in this run)

- https://huggingface.co/convaiinnovations/laya (card, API), https://huggingface.co/convaiinnovations/laya-multilingual
- https://doi.org/10.20944/preprints202611.0781.v1 (redirect to preprints.org, page refused: [U])
- https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B (card, `decision_config.json`, `backbone/config.json`,
  `decision_head.safetensors` header, `decision2/_vendor/dev2model/decision_model.py`, `data.py`)
- https://huggingface.co/AnkitAI/TinyJev-0.6B (card, config, `head.safetensors` header)
- https://huggingface.co/Berk/assay-0.6b (card, config, `assay_head.safetensors` header)
- https://huggingface.co/LibertAIDAI/deem-0.6-v1, https://huggingface.co/LibertAIDAI/deem-0.8-v1
- https://huggingface.co/labqoat/qev-0.8b, https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3,
  https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF, https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF
- https://huggingface.co/C-Tianyu/NanoJev
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B, https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B,
  https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B, https://huggingface.co/Qwen/Qwen3.5-0.8B (config)
- https://huggingface.co/katanemo/Arch-Router-1.5B
- https://docs.typesafe.ai/introduction (Jev)
