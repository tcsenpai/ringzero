# Model survey: small tool-calling models for the ringzero agent

Date of the survey: 2026-11-27. Scope: open-weight models up to about 4B parameters that could replace stock
Qwen3.5-2B (Q8_0) or Qwen3.5-4B (Q4_0) as the ring-0 system agent. Constraints taken from the repo: CPU only, KVM
guest, decode at or above about 19 tok/s, weights in the 2.5-3.4 GB range, typed tool calls in Qwen's XML format
under a grammar, Italian and English.

Labels used below:

- **[V]** verified in this run from the source given (model card, config.json, tokenizer.json, chat template,
  safetensors header, leaderboard CSV or local code).
- **[S]** self-reported by the model's authors (a number in their own card, run by them). Verified that the card says
  it, not that the number is right.
- **[E]** my estimate or inference, not measured.
- **[U]** unverified: could not check in this run.

Benchmark numbers from different cards are **not comparable** with each other: different BFCL versions (v3 vs v4),
thinking vs non-thinking mode, different harnesses and quantization. ringzero runs the model in non-thinking mode
(`<think>\n\n</think>` in `kernel/src/agent.rs`), while almost every number below was taken in thinking mode. The
most useful comparisons are inside one table (same authors, same protocol, several models side by side).

## 1. What the engine supports today (from the repo)

Read from `INFO.md`, `docs/ARCHITECTURE.md`, `docs/MODEL_CHOICE.md`, `engine/*.asm`, `engine/constants.inc`,
`tools/export_weights.py`, `kernel/src/llm.rs`, `kernel/src/agent.rs`. [V]

- One build (`QWEN35` defined) runs two layer types from a per-layer descriptor table (`.asmbin` v3/v4):
  - type 0, full attention: GQA, RMSNorm, QK-norm (per-head RMSNorm), RoPE with `rot_dim` (partial rotary), optional
    Qwen3.5 sigmoid output gate (`HDR_FLAG2_ATTN_OUT_GATE`), optional `1+w` norms baked by the exporter;
  - type 1, Gated DeltaNet: `Dk == Dv == 128`, conv kernel 4, `Hv = ratio * Hk` with `Hv <= 32` (M7 change for the
    4B: 16 K heads, 32 V heads).
  - MLP is SwiGLU (gate/up/down). Weights Q8_0 or Q4_0 per slot (v4), tied or untied `lm_head` (v4 `HDR_OFF_LM_HEAD`).
- **Qwen3 dense already runs on this engine**: Qwen3-Embedding-0.6B (28 full-attention layers, dim 1024) is exported
  as v3 with `--v3` and runs as a second `g_ctx` (`docs/ARCHITECTURE.md`, "Embedder"). So a Qwen3 dense causal LM is a
  configuration of existing kernels, not new kernels.
- Things the asm does **not** have: skipping QK-norm (the `HDR_FLAG_QK_NORM` bit is written by the exporter but no
  `.asm` file tests it; grep found no use), a configurable attention scale (it is `1/sqrt(head_dim)`), sliding-window
  masks, post-sublayer ("sandwich") norms, per-head scalar gates, conv kernels other than 4, SSM/Mamba scans, GELU,
  per-layer embeddings.
- Tokenizer: byte-level BPE only (GPT-2 `bytes_to_unicode`, merges hash table of 2^19 slots in the asm). No
  SentencePiece / `byte_fallback` path. The pre-tokenizer split is in Rust (`llm.rs`, "the Qwen2/3 split regex"),
  with single-digit splitting; non-ASCII bytes count as letters.
- Chat/tool format: `agent.rs` hard-codes the Qwen3.5 system block (`# Tools ... <tools> ... </tools>` and the
  `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>` instructions), ChatML
  `<|im_start|>`/`<|im_end|>`, `<tool_response>` turns, and an empty think block. `gram.rs` constrains exactly that
  XML body.
- Upstream llm.asm (github.com/alesha-pro/llm.asm, pushed 2026-11-25): the tree has only Qwen3-0.6B inference
  (`shared/transformer.asm`, `scripts/export_weights.py`), no Qwen3.5 and no other families. Our engine is already a
  superset of upstream for this purpose. [V]

Reference points measured in `docs/MODEL_CHOICE.md` [V, local]: Qwen3.5-4B Q4 module 3.36 GB, 20.9 tok/s median;
Qwen3.5-2B Q8 2.0 GB, 35.2 tok/s. That is roughly 68-70 GB/s of effective weight bandwidth, which I use below to
estimate decode speed from module size [E].

## 2. Baseline: what stock Qwen3.5 scores

| Model | BFCL-V4 | TAU2-Bench | Source |
|---|---:|---:|---|
| Qwen3.5-4B (thinking) | 50.3 | 79.9 | Qwen card [S] |
| Qwen3.5-2B (thinking) | 43.6 | 48.8 | Qwen card [S] |
| Qwen3-4B-2507 (column in the Qwen3.5-2B card) | 39.9 | 43.2 | Qwen card [S] |

Independent views of Qwen3.5-4B, each from a different vendor's table:

- NVIDIA's Nemotron-3.5-Bolt-E4B card (2026-10-05) puts Qwen3.5-4B BF16 **first** on its own BFCL-v3 (70.95) and
  Tau2 (75.06) rows, ahead of Bolt-E4B (68.51 / 55.10), Gemma4-E4B-it (66.39 / 38.50) and Nemotron-3-Nano-4B
  (61.10 / 29.20). The same table shows Qwen3.5-4B at Q4_K_M dropping to 70.56 BFCL-v3 and **61.74 Tau2**. [S, NVIDIA
  running a competitor] So 4-bit weights cost a lot on multi-turn agent tasks; this matters for our Q4_0 4B.
- LiquidAI's LFM3 card: Qwen3.5-4B BFCL V4 55.12, tau3 (4 domains) 65.98. [S]
- MiniCPM5 card: Qwen3.5-4B BFCL v4 56.8. [S]
- NeoHorse card: Qwen3.5-4B BFCL v4 61.02, tau2 84.29 (their own SGLang run, thinking on). [S]

The official BFCL leaderboard (gorilla.cs.berkeley.edu, CSV "Last Updated 2026-04-12") does **not** list any Qwen3.5,
LFM3, MiniCPM5, Granite 4.x or Nemotron 3 model, so there is no neutral BFCL number for the current small models.
Small models it does list (overall BFCL v4): Nanbeige4-3B-Thinking-2511 51.40, xLAM-2-3b-fc-r 41.22,
Qwen3-4B-Instruct-2507 (FC) 35.68, Arch-Agent-3B 35.36, Hammer2.1-3b 29.71, Qwen3-1.7B 28.41, Llama-3.2-3B 21.95. [V]

Takeaway: stock Qwen3.5-4B is already at or near the top of the 4B class in most third-party tables. A replacement
has to beat it in **our** setting (non-thinking, Q4/Q8, our 18 tools, Italian), which no card measures.

## 3. Candidates

### 3.1 Same architecture as Qwen3.5 (class A: converter only)

| Model | Base | Size | License | Released (HF createdAt) | Evidence | Tool format |
|---|---|---|---|---|---|---|
| **TokenRhythm/NeoHorse-1-4B** | Qwen3.5-4B | 4B, text-only repack | Apache-2.0 | 2026-09-05 | [S] vs Qwen3.5-4B in one table: BFCL v4 61.79 vs 61.02, tau2 88.46 vs 84.29, VitaBench 32.00 vs 21.50, PinchBench 77.33 vs 71.19, IFEval 88.35 vs 87.06; arXiv 2609.08183 | identical chat template to Qwen3.5-4B (same md5), parser `qwen3_coder` |
| **agentscope-ai/QwenPaw-Flash-4B / -2B** | Qwen3.5-4B / 2B | 4B / 2B | Apache-2.0 | 2026-03-30 | card says trained on agent trajectories for "tool invocation, command execution, memory management", "terminal operations and file system orchestration"; results are only in images, no numbers in text [U] | parser `qwen3_xml`, same arch parameters as Qwen3.5 (card) |
| InternScience/Agents-A1-4B | Qwen3.5 arch (`qwen3_5`) | 4B | Apache-2.0 | 2026-07-13 | [S] vs Qwen3.5-4B: tau2 78.2 vs 79.9 (lower), VitaBench 40.3 vs 22.0, IFEval 94.8 vs 89.8, GAIA 95.1 vs 58.3; NeoHorse's table gives it BFCL v4 46.60 | parser `qwen3_coder` |
| microsoft/FrogNano-4B-2609 | Qwen3.5-4B | 4.66B with vision | Apache-2.0 | 2026-09-17 (card: release 22-SEP-2026) | [S] SWE-bench Verified 61.5 vs 39.4 base; English only, Python repos, five-tool harness | identical chat template to Qwen3.5-4B |

Notes:

- NeoHorse's gains on BFCL are within noise (+0.77). The larger gains are on multi-turn agent sets (tau2, VitaBench,
  PinchBench), which are closer to what ringzero does. All numbers are the authors' own runs in thinking mode. [S]
- QwenPaw-Flash's stated training target (CLI commands, file I/O, memory of user preferences, multi-step tasks) is
  the closest description to ringzero's job of any card I found, and the 2B gives the fast option. No numbers I could
  read, so treat it as a candidate to test, not as evidence.
- FrogNano is the same architecture but trained for a different job (repository patches, English); not a fit, listed
  so nobody re-checks it.
- Multilingual: none of these cards measures Italian. They inherit Qwen3.5's 201-language claim; FrogNano says
  explicitly that non-English is not validated. [V for the statement, U for Italian quality]

### 3.2 Qwen3 dense (class B: existing kernels, new grammar)

| Model | Base | Size | License | Released | Evidence | Tool format |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | - | 4.0B, 36 layers, 8 KV heads | Apache-2.0 | 2025-08-05 | [S] BFCL-v3 61.9, TAU2-Retail 40.4; official BFCL v4 overall 35.68 [V] | Hermes JSON inside `<tool_call>` |
| Qwen/Qwen3-4B-Thinking-2507 | - | same | Apache-2.0 | 2025-08-05 | [S] BFCL-v3 71.2, TAU2-Retail 53.5 (thinking only) | Hermes JSON |
| katanemo/Plano-Orchestrator-4B | Qwen3-4B-Instruct-2507 | 4B | other (katanemo) | 2025-12-04 | no tool benchmark found in the card [U] | - |
| LocoreMind/LocoOperator-4B | Qwen3-4B-Instruct-2507 | 4B | MIT | 2026-02-23 | [S] codebase exploration only (Read/Grep/Glob/Bash), 65-sample internal eval | JSON |
| Team-ACE/EnvACE-Qwen3-1.7B | Qwen3-1.7B | 1.7B | Apache-2.0 | 2026-06-12 | card says it is evaluated on BFCL-v4, tau2, VitaBench; I could not parse the numbers [U] | Qwen3 native |

The Qwen3-4B base is weaker than Qwen3.5-4B on Qwen's own tables (BFCL-V4 39.9 vs 50.3, TAU2 43.2 vs 79.9) [S]. I did
not find a Qwen3 dense fine-tune with published evidence that it beats stock Qwen3.5-4B on tool use. Class B is
listed for completeness; there is no strong candidate in it.

### 3.3 Llama-family dense, close to our kernels (class C)

| Model | Size | License | Released | Evidence | Tool format | Languages |
|---|---|---|---|---|---|---|
| **openbmb/MiniCPM5-2B** | 2.52B (1.98B non-embedding), `LlamaForCausalLM`, 42 layers, 2 KV heads, head_dim 128, untied lm_head | Apache-2.0 | 2026-09-06 | [S] same table: BFCL v4 66.6 vs Qwen3.5-4B 56.8, Qwen3.5-2B 43.6; tau2 Telecom 97.1 vs 92.1; IFEval 86.7 vs 90.2 | XML but different: `<function name="X"><param name="K">V</param></function>`, CDATA for values with `<`, `&` or newlines | card lists en, zh only [V]; Italian [U] |
| ibm-granite/granite-4.2-3b | ~3.7B [E], `GraniteForCausalLM` (Llama + `attention_multiplier` 0.015625), 40 layers, head_dim 64, untied | Apache-2.0 | 2026-08-07 | [S] BFCL v4 52.41, tau3 50.99; LFM3 card: no column; MiniCPM5 table: IFEval 93.7, BFCL v4 52.2 | **same Qwen3.5 XML template** as ours, ChatML [V] | Italian in "Tested Languages" [V] |
| Nanbeige/Nanbeige4.1-3B, ToolMind-Web-3B | `LlamaForCausalLM`, 32 layers, untied | Apache-2.0 | 2026-02 / 2026-01 | not read in detail | - | en, zh |

Granite 4.2-3B is the easiest model of all to wire into the agent (same tool template), but its numbers are not
better than Qwen3.5-4B, so it is not worth the work. MiniCPM5-2B is the interesting one: plain Llama, 2B class, and
the strongest BFCL v4 claim among small dense models, but English/Chinese only on paper.

### 3.4 New architectures (class D)

| Model | Arch | Size | License | Released | Evidence | Tool format |
|---|---|---|---|---|---|---|
| **LiquidAI/LFM3-1.7B** | `Lfm3ForCausalLM`, custom code: 13 gated short-conv layers (depthwise conv, kernel 7), 6 global attention (QK-norm, per-head scalar gate `gate_proj [16, 2048]`, partial rotary 0.5), 5 sliding-window attention (window 256), sandwich norms (`post_operator_norm`, `post_ffn_norm`), tied embeddings, vocab 128k [V, config + safetensors header] | 1.73B | LFM Open License v1.0: commercial use only under USD 10M annual revenue [V] | 2026-10-23 | [S] same table: BFCL V4 **71.39** vs Qwen3.5-4B 55.12 and Qwen3.5-9B 62.83; tau3 68.01 vs 65.98; IFEval 96.66 vs 92.52; LiveClaw 45.60 vs 19.94 | **same `<tool_call><function=..><parameter=..>` body** as ours, ChatML [V] |
| Nanbeige/Nanbeige4.2-3B | custom `NanbeigeForCausalLM`: looped transformer (`num_loops` 2), "LoopSplit", mHC with depth attention, concatenated n-gram embeddings, 22 layers dim 3072 [V config, card] | 4B total, 3B non-embedding | Apache-2.0 | 2026-07-21 | [S] beats Qwen3.5-4B and 9B on its agent rows (Pinch-Bench-V2 74.7 vs 63.9, MCP-Atlas 57.8 vs 40.8); NeoHorse table: BFCL v4 67.28, tau2 85.08 | XML recommended, JSON supported; ChatML [U on exact format] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | NemotronH: mostly Mamba-2 + MLP, 4 attention layers, 42 layers | 3.97B | NVIDIA Nemotron Open Model License | 2026-03-07 | [S] BFCL v3 61.1, Tau2 Retail 34.8; below Qwen3.5-4B in NVIDIA's own Bolt table | `qwen3_coder` parser |
| google/gemma-4-E4B-it / E2B-it | Gemma 4: per-layer embeddings, 35 sliding (512) + 7 full attention, proportional RoPE, 262k vocab, SentencePiece-style BPE with `byte_fallback` [V] | 8B total / 4B effective (E4B) | Apache-2.0 | 2026-03-02 | [S] Google: Tau2 avg 42.2% (E4B), 24.5% (E2B); below Qwen3.5-4B in NVIDIA's and LiquidAI's tables | own native tokens; see `docs/research/GEMMA4_E2B_FEASIBILITY.md` |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 | Gemma 4 E4B fine-tune | E4B | NVIDIA license | 2026-10-05 | [S] its own table puts it below Qwen3.5-4B on BFCL-v3 and Tau2 | `gemma4` parser |
| XHToken/Spark-X2.5-4B | custom `spark2_5`, GELU, 27 sliding + 9 full | 4B | Apache-2.0 | 2026-08-24 | NeoHorse table: BFCL v4 63.71, tau2 77.72 [S] | en, zh |

### 3.5 Checked and dropped

- Older tool-calling fine-tunes (xLAM-2-3b-fc-r, Hammer2.1-3b, Arch-Agent-3B, ToolACE): Qwen2.5 or Llama 3.1 bases,
  2024-2025, below current models; xLAM-2 is CC-BY-NC-4.0. [V]
- Phi-4-mini-instruct (2025-02, MIT): old, no BFCL row; partial rotary 0.75 and a 200k vocab. [V]
- SmolLM3-3B (2025-07, Apache-2.0): six native languages including Italian; card BFCL 92.3 is an older BFCL subset
  and not comparable; superseded by the 2026 models. [S]
- Ministral-3-3B-Instruct-2512 (2025, Apache-2.0): Italian supported, Mistral tool format; no BFCL v4 number found.
- Qwen: no small model after Qwen3.5 (Feb 2026). Qwen3.6 (27B, 35B-A3B) and Qwen3.8 (27B and up) have no 4B
  member in the Qwen org listing as of today. Qwen-Drive-1.0-4B is a driving planner. [V]
- Jan-v3.5-4B (personality/maths), h2o-lightning-4b, Tev1-4B, Intern-Decision-4B, Mica (decision/classifier models):
  not tool agents. [V]

## 4. What it takes to run each class in our engine

Estimates are for one person who knows this codebase, including the M7-style quality gate (`tools/q4_kld.py`,
`tools/hf_agree.py`) and a `tools/tooleval.py` A/B on fedora. [E]

### Class A: same Qwen3.5 architecture (NeoHorse-1-4B, QwenPaw-Flash-4B/2B, Agents-A1-4B)

- Engine: nothing.
- Converter: `export_model_v3` already accepts `model_type` starting with `qwen3_5` (NeoHorse is `qwen3_5_text`).
  NeoHorse is a text-only repack with keys `model.layers.*`, `model.embed_tokens.weight`, `model.norm.weight`
  [V, its weight index], so the hard-coded prefix `model.language_model.` must fall back to `model.` (one line, the
  same kind of fallback already used for Qwen3-Embedding). MTP tensors are ignored. The others keep the stock
  `Qwen3_5ForConditionalGeneration` layout [V config] and should export unchanged [E].
- Template and grammar: none. NeoHorse and FrogNano ship a chat template byte-identical to Qwen3.5-4B (md5 match) [V].
  QwenPaw and Agents-A1 use the Qwen3 XML / qwen3_coder parsers [V]; template identity not checked [U].
- RAM and speed: identical to the current builds (4B Q4 3.36 GB, about 21 tok/s; 2B Q8 2.0 GB, about 35 tok/s). [E]
- Effort: **about 1 day per model**, most of it the A/B run.

### Class B: Qwen3 dense (Qwen3-4B-2507 family, EnvACE-Qwen3-1.7B)

- Engine: kernels exist (the embedder path). To check: batched prefill on an all-type-0 model, and the header chat
  ids (the exporter only fills `HDR_TOK_*` for qwen35; Qwen3 falls back to the same constants, which happen to match).
- KV cache is the real cost: Qwen3-4B has 36 full-attention layers x 8 KV heads x 128 = 147 KB per token in FP16
  [E from config], so the current 32768-token KV would need 4.8 GB. The context must drop to about 8k (1.2 GB). The
  Qwen3.5-4B only keeps KV for 8 layers.
- Decode: weights at Q4 about 2.4 GB [E], so short-context decode could be near 25 tok/s, but attention over 36
  layers grows with context and falls below the Qwen3.5 hybrid at long context [E].
- Template and grammar: Qwen3 uses `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` [V template]. `gram.rs`
  needs a second mode that constrains a JSON object (name from the tool list, then typed arguments per the schema);
  `agent.rs` needs the Qwen3 system block. About 3-5 days.
- Effort: **about 1-2 weeks**. Not recommended: no candidate here has evidence of beating Qwen3.5-4B.

### Class C: Llama-family dense (MiniCPM5-2B, Granite 4.2-3B)

- Engine (asm): honour `HDR_FLAG_QK_NORM` (skip the per-head RMSNorm when clear) in `transformer.asm` and
  `prefill.asm`; for Granite also a header field for the attention scale (0.015625 instead of 1/sqrt(64)) and a test of
  head_dim 64, which no current model uses [E]. Untied lm_head is already supported (v4).
- Converter: a Llama tensor-name map (`q_proj/k_proj/v_proj/o_proj`, `gate/up/down`, no `q_norm`), write `tids` from
  the tokenizer for non-Qwen vocabularies (MiniCPM5 eos 130073, Granite eos 100257 [V config]).
- Tokenizer: both are byte-level BPE (Ġ pieces, no byte fallback) [V], so the asm BPE works. The Rust pre-tokenizer
  must get a variant: MiniCPM5 first splits digits as `\p{N}{1,3}` then a Llama-3-style regex; Granite uses the plain
  GPT-2 ByteLevel regex [V tokenizer.json]. A wrong split gives valid but different tokens (quality loss, not a crash).
  Merge counts must fit the 2^19-slot hash table [U, not counted].
- Template and grammar: Granite 4.2 uses our exact Qwen XML system block and ChatML [V], so nothing changes.
  MiniCPM5 uses `<function name="X"><param name="K">V</param></function>` with CDATA [V]; `gram.rs` needs a new
  automaton for that shape (smaller than the JSON one) and `agent.rs` a new tools block.
- RAM and speed: MiniCPM5-2B at Q8_0 about 2.7 GB, about 25 tok/s; KV 42 layers x 2 heads x 128 = 43 KB per token,
  so 8k tokens is 350 MB, 32k is 1.4 GB [E]. Granite 4.2-3B at Q4 about 2.1 GB [E].
- Effort: **about 1-2 weeks** (asm flags 2-3 days, tokenizer split 1-2 days, grammar 2-4 days for MiniCPM5, gate and
  A/B 2 days).

### Class D: new operators (LFM3-1.7B, Nanbeige4.2-3B, Nemotron 3 Nano 4B, Gemma 4)

- **LFM3-1.7B**: new layer type "gated short conv" (in_proj to 3 x dim, depthwise causal conv of kernel 7 with a
  6-token state, multiplicative gates, out_proj); per-head scalar output gate on attention (16 values, unlike the
  Qwen3.5 per-element gate); post-sublayer norms on every layer; sliding-window mask (256) on 5 layers; an
  `embedding_norm` as final norm; a new pre-tokenizer variant (`\p{N}{1,3}`); chat ids in the header. The DeltaNet
  conv state code and the existing gate path are a starting point. The tool body is our XML format and the chat is
  ChatML [V], so the agent side is small; the card warns that turning thinking off "can be inconsistent" [S], which
  must be tested with our empty think block. RAM: Q8_0 about 1.8 GB, KV only for 6 global layers plus 5 x 256-token
  windows (small); decode about 35 tok/s or more [E]. Effort: **about 2-3 weeks**.
- **Nanbeige4.2-3B**: looped transformer (each layer run twice), mHC with depth attention, n-gram embeddings, all
  defined only in `modeling_nanbeige.py`. Two passes over 22 layers of dim 3072 means about 2x the compute and weight
  reads per token of its parameter count [E]; at Q4 that is likely under 19 tok/s [E]. Recommended max output 65k
  tokens for agent tasks (thinking). Effort: **4+ weeks**, speed risk high.
- **Nemotron 3 Nano 4B**: Mamba-2 selective scan kernels and state, a 131k vocab. Effort **4+ weeks**, and its own
  vendor shows it below Qwen3.5-4B.
- **Gemma 4 E2B/E4B**: SentencePiece-style tokenizer with byte fallback (new tokenizer path), per-layer embeddings,
  sliding windows, native tool tokens. The separate study `docs/research/GEMMA4_E2B_FEASIBILITY.md` estimates 4-6
  weeks for E2B. Tool evidence is weak (Tau2 24.5% E2B, 42.2% E4B [S]).

## 5. Ranking

1. **NeoHorse-1-4B** (class A, 1 day). Same weights shape, same template, same speed; the only Qwen3.5-4B fine-tune
   with a multi-benchmark table against its base, gains on multi-turn agent sets. Self-reported, thinking mode.
2. **QwenPaw-Flash-4B and -2B** (class A, 1 day each). Trained for exactly our kind of work (CLI, files, memory);
   the 2B would give 35 tok/s if it holds quality. No readable numbers, so it is a test candidate only.
3. **LFM3-1.7B** (class D, 2-3 weeks). Largest claimed tool-calling jump in a same-table comparison (BFCL V4 71.4 vs
   55.1 for Qwen3.5-4B), Italian listed, uses our XML tool body, smaller and faster than both current builds. Costs:
   new conv layer type and attention variants, LFM license with a USD 10M revenue cap, thinking-off behaviour unknown.
4. **MiniCPM5-2B** (class C, 1-2 weeks). Plain Llama, Apache-2.0, strong BFCL v4 claim (66.6 vs 56.8); different
   XML tool format and no Italian claim.
5. Agents-A1-4B (class A): cheap to try, but its tau2 is below the base; its gains are in search and research tasks.

Not recommended: Qwen3 dense fine-tunes (weaker base), Granite 4.2-3B (easy but not better), Nanbeige4.2-3B (speed
and architecture risk), Nemotron 3 Nano 4B and Gemma 4 E2B/E4B (weaker tool evidence, large engine work).

## 6. Recommendation

- First, run the class-A models through the existing pipeline: export NeoHorse-1-4B and QwenPaw-Flash-4B with
  `QUANT=q4`, QwenPaw-Flash-2B at Q8, pass the M7 KLD/agree gate, then `tools/tooleval.py` spike/inos/it against the
  current 4B q4 and 2B q8 (same procedure as `docs/MODEL_CHOICE.md`). This needs one converter line and no engine
  change, and it answers the only question the cards cannot: do the gains survive non-thinking mode, Q4_0 and our
  tool set. If QwenPaw-Flash-2B matches the 4B on the in-OS set, it also buys back 14 tok/s and the snapshot pool.
- Given NVIDIA's measured Tau2 drop for Qwen3.5-4B at 4-bit (75.1 to 61.7), also try the winner with Q8_0 on the
  attention and DeltaNet projections and Q4 only on the MLP, if RAM allows; this is an exporter option we already
  have (`Q4_ROLES`).
- Only if class A gives nothing: prototype LFM3-1.7B on the host first (transformers, non-thinking, our 18 tool
  schemas, Italian prompts) before writing any asm. If it beats the 4B there, it is the best long-term target
  (faster, smaller, same tool format), subject to the licence being acceptable.

## 7. Not checked in this run

- Italian quality of any candidate (no card measures it; only language lists).
- QwenPaw-Flash benchmark values (images only) and whether its template is byte-identical to Qwen3.5.
- EnvACE-Qwen3-1.7B numbers (table not parsed).
- Merge counts of non-Qwen tokenizers against the 2^19 hash table.
- Real decode speed of anything in our engine; all speeds above are scaled from module size.
- Any benchmark in non-thinking mode for the 2026 models.

## Sources fetched in this run

Local (repo): `INFO.md`, `docs/ARCHITECTURE.md`, `docs/MODEL_CHOICE.md`, `engine/constants.inc`,
`engine/transformer.asm`, `tools/export_weights.py`, `kernel/src/llm.rs`, `kernel/src/agent.rs`.

Upstream engine:
- https://api.github.com/repos/alesha-pro/llm.asm and https://api.github.com/repos/alesha-pro/llm.asm/git/trees/HEAD?recursive=1

Leaderboard and papers:
- https://gorilla.cs.berkeley.edu/leaderboard.html and https://gorilla.cs.berkeley.edu/data_overall.csv
- https://arxiv.org/abs/2605.00334 (AgentFloor, 2026-05-01, background on small models in agent pipelines)
- https://arxiv.org/abs/2608.22472 (IFFC, 2026-08-23, background)

Model cards (README.md), config.json and HF API metadata (`https://huggingface.co/api/models/<id>`) for each id below,
fetched as `https://huggingface.co/<id>/raw/main/README.md` and `.../raw/main/config.json`:
- https://huggingface.co/Qwen/Qwen3.5-4B, https://huggingface.co/Qwen/Qwen3.5-2B (also chat_template.jinja, tokenizer.json)
- https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 (also chat template, tokenizer.json), https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
- https://huggingface.co/TokenRhythm/NeoHorse-1-4B (also chat_template.jinja, model.safetensors.index.json)
- https://huggingface.co/agentscope-ai/QwenPaw-Flash-2B, https://huggingface.co/api/models/agentscope-ai/QwenPaw-Flash-4B
- https://huggingface.co/InternScience/Agents-A1-4B
- https://huggingface.co/microsoft/FrogNano-4B-2609 (also chat_template.jinja)
- https://huggingface.co/LiquidAI/LFM3-1.7B (also LICENSE, chat_template.jinja, tokenizer.json, safetensors header)
- https://huggingface.co/openbmb/MiniCPM5-2B (also chat_template.jinja, tokenizer.json)
- https://huggingface.co/ibm-granite/granite-4.2-3b (also chat_template.jinja, tokenizer.json)
- https://huggingface.co/Nanbeige/Nanbeige4.2-3B, https://huggingface.co/Nanbeige/Nanbeige4.1-3B
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16
- https://huggingface.co/google/gemma-4-E4B-it (also tokenizer.json), https://huggingface.co/google/gemma-4-E2B-it
- https://huggingface.co/XHToken/Spark-X2.5-4B
- https://huggingface.co/katanemo/Plano-Orchestrator-4B, https://huggingface.co/LocoreMind/LocoOperator-4B,
  https://huggingface.co/janhq/Jan-v3.5-4B, https://huggingface.co/Team-ACE/EnvACE-Qwen3-1.7B
- https://huggingface.co/Salesforce/xLAM-2-3b-fc-r, https://huggingface.co/MadeAgents/Hammer2.1-3b,
  https://huggingface.co/katanemo/Arch-Agent-3B, https://huggingface.co/microsoft/Phi-4-mini-instruct,
  https://huggingface.co/HuggingFaceTB/SmolLM3-3B, https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- Org listings: `https://huggingface.co/api/models?author=<org>&sort=createdAt` for Qwen, nvidia, google, ibm-granite,
  LiquidAI, HuggingFaceTB, Salesforce, MadeAgents, katanemo, janhq, microsoft, mistralai, Nanbeige, openbmb, Team-ACE;
  fine-tune listings `https://huggingface.co/api/models?filter=base_model:finetune:<id>` for Qwen3.5-4B, Qwen3.5-2B,
  Qwen3-4B-Instruct-2507.
