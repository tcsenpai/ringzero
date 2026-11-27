# Feasibility: Gemma 4 E2B on the ringzero asm engine

Date: 2026-11-27. Research only: no code, no commits. Question: can the in-kernel engine run Google's Gemma 4 E2B
instead of, or next to, Qwen3.5 2B / 4B?

Tags: **[V]** checked against a source fetched in this run (listed at the end), **[E]** my estimate, **[U]** not verified.

## 0. Verdict

**Technically a GO, as a fast experimental edition. NO-GO as a replacement of the 4B q4 working model.**

- The model exists: `google/gemma-4-E2B-it` (HF repo created 2026-03-02, last modified 2026-11-03), Apache 2.0, tech report
  arXiv 2607.02770 (v1 2026-07-02, v2 2026-07-24) [V].
- It is a plain transformer (sliding + global attention, GeGLU MLP). No DeltaNet, no MoE in E2B. Every op is an attention or
  matmul variant of what the engine already does. The genuinely new parts are Per-Layer Embeddings (PLE), KV sharing across
  layers, per-layer head and MLP sizes, GELU-tanh, logit softcapping and a SentencePiece-style tokenizer [V].
- Decode should be faster than our 2B: about 2.3B parameters are read per token (as many as the 2B), and Google ships a
  QAT checkpoint made for Q4_0. Estimate **38-45 tok/s** with Q4_0 linears, **26-29 tok/s** at Q8_0 [E].
- RAM is the catch: 2.35B of the 5.1B parameters are the PLE table, needed in memory but only 9.5 KB of it read per token.
  Text-only weights are **about 4.0 GB** (Q4_0 linears, Q8_0 embeddings and PLE) or 4.9 GB at Q8_0 [E from V counts].
- Tool calling: the published agent numbers for E2B are clearly below Qwen3.5-2B and far below Qwen3.5-4B (section 3).
  Recommended gate before any engine work: run our own spike prompt set on Gemma 4 E2B on the host (1 day).
- Effort: about **4-6 weeks** for one person to reach the same level as the Qwen editions (engine, tokenizer, agent,
  grammar, eval) [E].

## 1. Gemma 4 E2B architecture (text decoder)

Sources: HF `config.json`, `model.safetensors` header (read by HTTP range), transformers `modeling_gemma4.py` and
`configuration_gemma4.py` (main, last commit 2026-11-27), `modeling_rope_utils.py`, model card, tech report [V].

| Item | Value |
|---|---|
| Class | `Gemma4ForConditionalGeneration`, text part `gemma4_text` |
| Layers | 35. `layer_types`: 4 sliding then 1 full, repeated (full = layers 4, 9, 14, 19, 24, 29, 34; last layer always full) |
| Hidden size | 1536 |
| Attention | 8 query heads, **1 KV head (MQA)**. head_dim 256 on sliding layers, **512 on full layers** (`global_head_dim`) |
| Sliding window | 512 tokens |
| KV sharing | `num_kv_shared_layers` 20: layers 15-34 have no k/v projections and reuse the KV of the last earlier layer of the same type (layer 13 for sliding, layer 14 for full). Only 15 layers own a KV cache (12 sliding, 3 full) |
| K = V | `attention_k_eq_v` false for E2B (the model card's "unified keys and values" applies to other sizes) |
| Q/K/V norms | RMSNorm per head on q and k (with weight), **on v without weight**. Attention scale is **1.0** (no 1/sqrt(d)) |
| Layer norms | sandwich: input_layernorm, post_attention_layernorm (on the attention output, before the residual add), pre_feedforward_layernorm, post_feedforward_layernorm, plus post_per_layer_input_norm. RMSNorm is `x * rsqrt(mean(x^2)+eps) * w` (plain w, not 1+w), eps 1e-6 |
| MLP | gated, `gelu_pytorch_tanh` (GeGLU). intermediate 6144 on layers 0-14, **12288 on layers 15-34** (`use_double_wide_mlp` on the KV-shared layers) |
| RoPE | sliding: default NeoX RoPE, theta 10000, all 256 dims. Full: **proportional RoPE**, theta 1e6, `partial_rotary_factor` 0.25: frequencies `theta^(-2i/512)` for i < 64, zero for the other 192 pairs, pairs are (j, j+256) |
| Per-Layer Embeddings | table `embed_tokens_per_layer` [262144, 35 x 256] scaled by sqrt(256); plus `per_layer_model_projection` [1536 -> 8960] of the token embedding scaled by 1536^-0.5, RMSNorm per 256 chunk; the two are added and scaled by 2^-0.5. In each layer after the MLP: `h += norm(proj(gelu(gate(h)) * ple[layer]))` with gate 1536->256 and proj 256->1536 |
| Layer scalar | one learned scalar per layer multiplies the layer output (`layer_scalar`, BF16) |
| Embedding | 262144 x 1536, tied with the LM head, input scaled by sqrt(1536) (stored as a bf16 buffer: 39.25, not 39.19; transformers comment) |
| Logits | final softcapping: `30 * tanh(logits / 30)` |
| Context | 131072 (128K) |
| MatFormer | not present in the E2B config or modeling code (that was Gemma 3n). The "E" is only PLE [V by absence] |
| Drafter | Google publishes an MTP drafter, `google/gemma-4-E2B-it-assistant` (157 MB bf16, about 76M parameters, tech report table 1). It reads the main model's activations and KV; not usable by our prompt-lookup path without new work |
| Dropped towers | vision encoder about 150-168M, audio encoder about 305M (16 and 12 layers). Text-only use drops both |

Parameter counts from the safetensors header (BF16, total 5,123,178,979) [V]:

| Tensor group | Params | Read per token? |
|---|---:|---|
| PLE table `embed_tokens_per_layer` | 2348.8M | no, one 8960-value row per token |
| MLP gate/up/down (all layers) | 1557.2M | yes |
| token embedding (tied LM head) | 402.7M | yes (LM head) |
| attention q + o | 264.2M | yes |
| k + v actually used (15 layers) | 14.2M | yes (the checkpoint stores 33.0M: k/v for the 20 shared layers too, ignored at load) |
| PLE gate + proj per layer | 27.5M | yes |
| `per_layer_model_projection` | 13.8M | yes |
| norms, scalars | under 0.3M | yes |
| audio + vision | 475.7M | dropped |

Read per token: 2.28B parameters, which is the "2.3B effective" of the model card [V].

Weight sizes, text only, our formats (Q8_0 = 1.0625 B/param, Q4_0 = 0.5625) [E from V counts]:

| Plan | Read per token | File / RAM |
|---|---:|---:|
| All Q8_0 | 2.42 GB | 4.92 GB |
| Q4_0 MLP only, rest Q8_0 (our `q4`) | 1.64 GB | 4.14 GB |
| **Q4_0 all linears (QAT), embedding + PLE Q8_0** | **1.49 GB** | **3.99 GB** |
| same, PLE table Q4_0 | 1.49 GB | 2.81 GB |

Google's own QAT GGUF (`google/gemma-4-E2B-it-qat-q4_0-gguf`, `gemma-4-E2B_q4_0-it.gguf`, 3.35 GB) uses Q4_0 for every linear
including attention, Q6_K for the token embedding and the PLE table, F16 for `per_layer_model_projection` [V, GGUF header read by
range]. We have no Q6_K kernel; Q8_0 for those two tables is the closest format we run. The QAT weights before quantization are in
`google/gemma-4-E2B-it-qat-q4_0-unquantized` [V exists, contents U]. Exporting those with the llama.cpp Q4_0 rule we already use
should land on the QAT grid [U: needs a KLD check].

Tokenizer and template (`tokenizer.json`, `chat_template.jinja`, template header "Published: 2026-07-09") [V]:

- BPE with `byte_fallback`, 262144 pieces, **514,639 merges**, normalizer replaces " " with U+2581, no regex pre-split,
  256 `<0xNN>` byte pieces, decoder maps U+2581 back to space and fuses byte pieces.
- Ids: pad 0, eos 1, bos 2, `<|tool>` 46, `<tool|>` 47, `<|tool_call>` 48, `<tool_call|>` 49, `<|tool_response>` 50,
  `<tool_response|>` 51, `<|"|>` 52 (string delimiter), `<|think|>` 98, `<|channel>` 100, `<channel|>` 101, `<|turn>` 105,
  `<turn|>` 106. Generation stops on 1, 106 or 50.
- Turns: `<bos><|turn>system\n...<turn|>\n<|turn>user\n...<turn|>\n<|turn>model\n`. Thinking is on when `<|think|>` opens the
  system turn; output is `<|channel>thought\n...<channel|>answer`. With thinking off, E2B and E4B emit no channel tags (model card).
- Recommended sampling: temperature 1.0, top_p 0.95, top_k 64.

## 2. Mapping to our engine

What the engine is today [V, local tree]: `engine/transformer.asm` (vendored llm.asm, `QWEN35` build) runs layer type 0 (Qwen3
full attention: fused q|k|v, QK-norm, NeoX RoPE with partial `HDR_ROT_DIM`, GQA, optional sigmoid output gate, SwiGLU) and type 1
(Gated DeltaNet). One `head_dim`, one `n_kv_heads`, one `hidden_dim` per model in the header; per-layer descriptors hold 12 slots
(in_norm, post_norm, W_proj, W_out, W_gateup, W_down, q_norm, k_norm, DeltaNet extras). Q8_0 and Q4_0 (W4A8) matmuls, batched
prefill (`prefill.asm`, blocks of 32), FP16 KV cache, 32768 context. Tokenizer: GPT-2 byte-level BPE in asm
(`tok_encode_bpe`, byte -> base token, merge hash table, 2^19 slots), Qwen regex-like pre-split in Rust (`llm.rs chunk_end`).
Agent: Qwen XML tool format in `agent.rs` / `gram.rs`.

| Gemma 4 item | Status | Notes | Days [E] |
|---|---|---|---:|
| Plain attention + gated MLP layer | supported | the type 0 path without the DeltaNet state | 0 |
| MQA (8 q, 1 kv) | supported | GQA with ratio 8 | 0 |
| Q8_0 / Q4_0 matmuls, batched prefill | supported | | 0 |
| Per-layer head_dim (256 / 512), q_dim, kv_dim | small change, wide reach | header has one head_dim; move it into the descriptor, size buffers for the largest; touches decode, prefill, KV layout, snapshot rows | 2-3 |
| Per-layer MLP width (6144 / 12288) | small change | same as above for hidden_dim | inc. above |
| Sliding window 512 | small change | start the attention loop at pos-511 on sliding layers. Keep a full-length cache so snapshots, undo and spec rollback stay row based (ring buffer saves 0.4 GB, later) | 0.5 |
| KV sharing (layers 15-34) | small change | descriptor field "KV source layer"; skip k/v projection and KV store | 1 |
| Attention scale 1.0, v RMSNorm without weight | small change | header flag, rmsnorm with a ones vector (or a no-weight variant) | 0.5 |
| Sandwich norms (post-attn, pre/post-ffn) | small change | 3 more descriptor slots (the exporter already reserves 16) | 0.5 |
| Layer scalar, embedding scale 39.25 | small change | `scale_f32_avx2` exists | 0.25 |
| Proportional RoPE on full layers | **exporter only** | permute the rows of q_proj, k_proj and the q/k norm weights of full layers so dims {0..63, 256..319} come first; then it is our partial NeoX RoPE with rot_dim 128 and theta' = 1e6^(1/4) = 31.62 (`theta^(-2i/512) = (theta^(1/4))^(-2i/128)`). q.k dot products are unchanged by a shared permutation. Needs per-layer-type theta (two RoPE tables) | 0.5 |
| GELU-tanh gated MLP | **new kernel** (small) | `gelu_tanh_mul_avx2` next to `silu_mul_avx2`, tanh from `fast_expf_avx2`; batched variant for prefill | 1 |
| Per-Layer Embeddings | **new code path** | per token: Q8_0 row dequant of 8960 values, one 1536->8960 matmul, 35 chunk RMSNorms, add and scale; per layer: two small matmuls + GELU + mul + norm + residual. All built from existing kernels, new glue in decode and prefill | 2-3 |
| Logit softcapping | **new kernel** (small) | 262144 tanh per token (well under 1 ms). Argmax unchanged (monotone), but sampling and T28 confidence margins need it | 0.5 |
| Prefill: skip layers 15-34 for prompt tokens | optional, big win | KV-shared layers write no KV, so prompt tokens that need no logits can stop after layer 14 (transformers does the same, `_maybe_slice_kv_shared_inputs`). 0.58B params per prompt token instead of 2.28B. Speculative verify rows still run all 35 layers | 1 |
| Vocab 262144 | supported | `HDR_ARGMAX_LIMIT`, vocab trie (about 1.3 MiB) | 0 |
| Exporter (`tools/export_weights.py`) | new branch | drop towers and the 20 unused k/v pairs, permute p-RoPE, PLE tensors, layer scalars, QAT source checkpoint, new header fields | 2-3 |
| Tokenizer | **medium change** | base symbols are UTF-8 characters (piece id) with `<0xNN>` fallback instead of bytes; 514,639 merges need a 2^20 or 2^21 table (16 B entries: 16-32 MiB); U+2581 normalization; no Qwen pre-split. Chunking before each U+2581 that follows a non-U+2581 char is exact except for one piece, `>▁</`, which crosses a word boundary [V: 1 piece of 262144]; decode table at export. Test against HF on a corpus | 2-3 |
| Bring-up to an HF-matching golden test | debugging | layer taps exist (`CTX_TAP_FN`, `tools/hf_agree.py`) | 3-5 |

Engine plus tokenizer to a golden self-test: **about 15-20 days** [E].

### RAM [E]

| Part | Size |
|---|---:|
| Weights, QAT Q4_0 linears + Q8_0 embedding and PLE | 3.99 GB |
| KV cache FP16 at 32768, full-length for the 15 owner layers | 576 MiB (198 MiB with a 512-slot ring for sliding layers) |
| Qwen3-Embedding-0.6B (unchanged) | same as today |
| Snapshot points | smaller than today: no 19.3 / 50.3 MiB DeltaNet state per point, only KV rows |

So an **8G guest** like the 4B q4 (3.4 GB weights), probably 9G with the full-length KV [E]. Options to cut it: PLE table at Q4_0
(saves 1.2 GB, not QAT-trained, quality U), or keep the PLE table on the virtio disk and read one 9.5 KB row per token (saves 2.5 GB;
needs a new loader path, 2-3 days, latency per read U). Running Gemma **next to** Qwen in one boot is possible in principle (the engine
already swaps `g_ctx` for the embedder), but 4B q4 + Gemma + embedder is about 8 GB of weights: a 12G guest. A build edition
(`MODEL=gemma4-e2b`) is the sane form.

### Decode speed [E]

Decode is DRAM bound at 8 cores and follows bytes read per token (M7_Q4_RESULT.md: 2B Q8 2.0 GB at 34-36 tok/s, 2B q4all
1.33 GB at 52 tok/s, 4B q4 3.36 GB at 19-21 tok/s) [V local].

| Plan | Read per token | Pure bandwidth scaling | With 35 vs 24 layers overhead (about -10%) |
|---|---:|---:|---:|
| Gemma E2B Q8_0 | 2.42 GB | 29 | **26-29** |
| Gemma E2B Q4_0 MLP only | 1.64 GB | 42 | 36-40 |
| **Gemma E2B QAT Q4_0 all linears** | 1.49 GB | 46-47 | **38-45** |
| reference: our 2B q8 / 4B q4 | 2.0 / 3.36 GB | 34-36 / 19-21 measured | |

Long context: at 32k, 7 full-attention layers read a 32k x 512 KV (448 MiB per token at FP16), about the same as the Qwen 2B's
6 full layers (2B measured 36 -> 24 tok/s at 32k); sliding layers add almost nothing. Prefill: compute bound; with the
shared-layer skip, about 4x fewer weights per prompt token than a full pass, so faster than the 2B's 160-230 tok/s [E, U until
measured]. Speculative decoding (prompt lookup, grammar drafts) works unchanged and the rollback gets simpler (no DeltaNet state).

## 3. Tool calling

### Native format [V, chat_template.jinja and tech report table 11]

- Declarations in the system turn, one per tool:
  `<|tool>declaration:NAME{description:<|"|>...<|"|>,parameters:{properties:{KEY:{description:<|"|>...<|"|>,type:<|"|>STRING<|"|>},...},required:[<|"|>KEY<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`
  (enum as `enum:[<|"|>a<|"|>,...]`).
- Call, inside the model turn: `<|tool_call>call:NAME{key:<|"|>text<|"|>,n:5,flag:true}<tool_call|>`. Strings are wrapped in the
  `<|"|>` special token; numbers, booleans and null are bare; nested objects and lists use `{}` / `[]`. The template sorts keys.
- The model then emits `<|tool_response>` (a stop token). The host appends
  `response:NAME{value:<|"|>RESULT<|"|>}<tool_response|>` (or `key:value` pairs) and the model continues **in the same model
  turn**, no new `<|turn>model`. Several calls in a row are allowed.
- With thinking, the thought channel of a tool-call turn must be kept in the history (model card, multi-turn rules).

### What changes in ringzero [E, from agent.rs / gram.rs as they are]

- `agent.rs`: new system prompt builder from `agent::SPECS` in the declaration syntax above (replaces `TEMPLATE_TOOLS_HEAD/TAIL`),
  new turn strings (`<|turn>user`, `<|turn>model`, `<turn|>`), tool results as `response:` blocks in the model turn, stop on 50 /
  106 / 1, think toggle via `<|think|>`, new lenient parser for `call:NAME{...}` (today `parse` looks for `<function=` and
  `<parameter=`). The T54 answer round blocks token 48 (`<|tool_call>`) instead of `<tool_call>`. agentd's consolidation prompt
  (`agentd.rs:1118`) too. 2 days.
- `gram.rs`: a new automaton. The structure is simpler than Qwen's XML: trigger on token 48, then `call:`, a tool name, `{`,
  then `key:` (each key once, required ones before `}`), a value by type, `,` or `}`, then token 49. Free strings start and end
  with the `<|"|>` special token, so the value can be any text without the `</parameter>` / `<parameter=` ban of today, but the
  automaton must accept that one special token as an atom (today every special inside a call is refused except `</tool_call>`).
  Integers bare digits, enums as `<|"|>value<|"|>`. Grammar drafts (T39) carry over: `call:` + unique name rest, `{`, `:`,
  `<|"|>`, `}` and token 49 are all implied text. Confidence margins (T28) unchanged in method. 2-3 days.
- Encoder: tool results and user text must never be tokenized into `<|"|>`, `<tool_response|>` or `<turn|>` (the kernel's
  `special_at` matches specials in text today); otherwise a file can close the response block. Escape or encode such text as
  plain bytes. 0.5 day.
- Evals: port `tools/tooleval.py` / `confeval.py` expectations, rerun the spike and in-OS sets. 2 days.

Agent side: **about 6-8 days** [E].

### Quality evidence

| Benchmark | Gemma 4 E2B | Qwen3.5-2B | Qwen3.5-4B | Source |
|---|---:|---:|---:|---|
| Tau2 (avg of 3 domains) | 24.5 | 48.8 (thinking) | 79.9 (thinking) | Gemma model card; Qwen model cards [V] |
| Tau2 airline / retail / telecom | 31.0 / 34.6 / 19.7 (thinking) | - | - | Gemma tech report table 5 [V] |
| BFCL-V4 | not published | 43.6 (thinking) | 50.3 (thinking) | Qwen cards [V]; not in the Gemma card or report [V by search] |
| IFEval | 94.6 | - | - | Gemma report table 5 [V] |
| MMLU-Pro | 60.0 | 55.3 (non-thinking) | - | cards [V] |

Caveats: the Tau2 setups differ (Qwen applies the airline fixes from the Claude Opus 4.5 system card; the Gemma average does not
match the mean of its own per-domain numbers, 28.4), so this is direction, not a precise gap. The BFCL public leaderboard CSV
(last modified 2026-10-17) lists neither Gemma 4 nor Qwen3.5; it has Gemma 3 4B at 19.6% and Qwen3-4B-Instruct-2507 at 35.7%,
which is the same direction one generation earlier [V]. Gemma 4 E2B is the stronger general reasoner per byte (MMLU-Pro,
LiveCodeBench 44.0, AIME 37.5) but the weaker agent.

**Verdict on tool calling:** the native format is easy to constrain (cleaner than Qwen's XML), and our grammar removes malformed
calls on any model. What the grammar cannot fix is the choice of tool and arguments, and there all published numbers put E2B
below Qwen3.5-2B and well below the 4B q4 we use day to day. Expect Gemma E2B to be a fast chat / summarization edition, not a
better system agent [E].

## 4. Recommended next steps

1. Gate (1 day, host only): adapt `docs/spikes/toolcall/run.py` (HF transformers, already written for Qwen3.5-2B) to Gemma 4 E2B
   with the Gemma template and `processor.parse_response`, run the 30 spike prompts, compare with the Qwen tables
   (`docs/spikes/toolcall/tables.md`, MODEL_CHOICE.md: 2B 22/25 right first tool, 4B q4 22/25). Needs the 10.2 GB bf16 download or
   llama.cpp with the 3.35 GB QAT GGUF. Not run here: it is above the 6 GB download limit for this study (bf16) and the box was
   shared with QEMU 4B tests.
2. If it is within reach of the 2B on that set: exporter + engine (sections 2) behind a `GEMMA4` descriptor type, golden test
   against HF, then the agent work.
3. Measure KLD of our Q4_0 export of the QAT checkpoint against bf16 (`tools/q4_kld.py`).

## Not checked

- No weights were downloaded or run; no speed was measured. The build box was not used: every fact needed was
  readable from HF files and HTTP range reads of the safetensors and GGUF headers.
- Contents of `gemma-4-E2B-it-qat-q4_0-unquantized` and whether our Q4_0 rounding matches the QAT grid.
- Whether the BOS token is required at every prefill restart (the template always emits it once).
- Tokenizer chunking equivalence beyond the vocab scan (needs a corpus test against HF).

## Sources fetched in this run

- https://huggingface.co/google/gemma-4-E2B-it : config.json, generation_config.json, tokenizer.json, tokenizer_config.json,
  chat_template.jinja, README.md (model card), API file list, model.safetensors header (range read)
- https://huggingface.co/api/models?search=gemma-4&author=google (model list and creation dates)
- https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf : GGUF header of gemma-4-E2B_q4_0-it.gguf (first 60 MB, range read)
- https://huggingface.co/api/models/google/gemma-4-E2B-it-assistant (file list)
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py,
  configuration_gemma4.py, src/transformers/modeling_rope_utils.py (main, last gemma4 commit 2026-11-27)
- https://arxiv.org/abs/2607.02770 and https://arxiv.org/html/2607.02770 (Gemma 4 Technical Report, v2 2026-07-24)
- https://huggingface.co/Qwen/Qwen3.5-2B and https://huggingface.co/Qwen/Qwen3.5-4B model cards (README.md)
- https://gorilla.cs.berkeley.edu/data_overall.csv (BFCL leaderboard, last-modified 2026-10-17)
- Local: INFO.md, docs/ARCHITECTURE.md, docs/MODEL_CHOICE.md, docs/M7_Q4_RESULT.md, docs/FEASIBILITY-q4-qwen35-9b.md,
  engine/constants.inc, engine/transformer.asm, engine/tokenizer.asm, tools/export_weights.py, kernel/src/llm.rs, agent.rs, gram.rs,
  docs/spikes/toolcall/
