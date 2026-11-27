# T45 spike: Qwen3.5-4B's MTP head as a speculative draft

Question: is the checkpoint's own multi-token-prediction head (`mtp.*`, 1 layer) a good enough draft for greedy
speculative decoding on our DRAM-bound CPU decode, before any engine work? Answer: it is a very good draft for
tool calls (97% top-1) and a fair one for free chat (70% top-1). With the current verify cost, though, it only pays
on tool-call rounds. Details and the recommendation are at the end.

Script: `tools/spike_mtp.py`. Raw results (per prompt: generated text, step-1 accuracy, replay counters):
`docs/spikes/mtp/spike_mtp.json`.

## Checkpoint

`Qwen/Qwen3.5-4B` (HF safetensors, bf16), text config `mtp_num_hidden_layers = 1`,
`mtp_use_dedicated_embeddings = false`, tied embeddings. The 15 `mtp.*` tensors:

| tensor | shape |
|---|---|
| `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.norm.weight` | 2560 |
| `mtp.fc.weight` | 2560 x 5120 |
| `mtp.layers.0.input_layernorm.weight`, `.post_attention_layernorm.weight` | 2560 |
| `mtp.layers.0.self_attn.q_proj.weight` | 8192 x 2560 (16 heads x 256, query + output gate) |
| `mtp.layers.0.self_attn.k_proj.weight`, `.v_proj.weight` | 1024 x 2560 (4 KV heads x 256) |
| `mtp.layers.0.self_attn.o_proj.weight` | 2560 x 4096 |
| `mtp.layers.0.self_attn.q_norm.weight`, `.k_norm.weight` | 256 |
| `mtp.layers.0.mlp.gate_proj.weight`, `.up_proj.weight` | 9216 x 2560 |
| `mtp.layers.0.mlp.down_proj.weight` | 2560 x 9216 |

transformers 5.24.1 drops these on load (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so the spike
reads them with safetensors and runs its own forward.

## MTP forward (as implemented and checked)

For the token `x_{t+1}` and the main model's last hidden state `h_t` (the one that predicted `x_{t+1}`):

1. `e = RMSNorm_e(embed(x_{t+1}))`, `g = RMSNorm_h(h_t)` (Qwen3.5 zero-centred RMSNorm, `x * (1 + w)`)
2. `x = fc(cat(e, g))`. The order is embedding first: the swapped order gives 0.2% top-1, this one 70% / 98%.
3. One decoder layer, the same as a full-attention layer of the main model: gated q (`q_proj` gives query and
   gate per head, output `* sigmoid(gate)`), q/k RMSNorm per head, partial RoPE (64 of 256 dims, theta 1e7;
   text positions make the interleaved mRoPE plain RoPE), GQA 16/4, SwiGLU MLP. It has its own KV cache.
4. `out = RMSNorm(x)`, `logits = out @ embed^T` (the tied lm_head), predicting `x_{t+2}`.
5. Chained drafts (n > 1) feed `out` and the drafted token back into step 1 at the next position, with the same
   layer and KV.

Check: our layer against transformers' `Qwen3_5DecoderLayer` (full-attention type) loaded with the `mtp.layers.0`
weights, 64 rows causal, bf16: max abs diff **0.0** (reference max abs 12.1). Two choices were measured, and
neither matters for acceptance:

- main hidden after the final norm (`post`, as vLLM passes it) vs before it (`pre`): step-1 top-1 is 70.3% vs
  68.2% (chat) and 97.8% vs 96.6% (tools). The spike uses `post`.
- chained hidden `norm(out)` vs `out`: tokens per verify differ by at most 0.04 (pre_fc_norm_hidden re-normalizes
  it anyway).

## Method

- Main model: HF `AutoModelForCausalLM` (bf16, CPU, 4 torch threads), greedy `generate`, at most 128 new tokens,
  stopping at `<|im_end|>`. A forward hook on the final norm records the main hidden state of every fed position
  (prefill and each decode step, the same sequential numerics a real decode has).
- Prompt: the agent prefix exactly as `kernel/src/agent.rs` `build_prefix` builds it (tool template head, the 18
  TOOLS JSON lines, template tail, SYSTEM), then `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n
  <think>\n\n</think>\n\n` (thinking off, no memory slot, as tooleval runs with `!set slot 0`). Prompt length is
  1590-1596 tokens.
- Sets: **chat**, 10 open questions (5 Italian, 5 English, including the 4 chat prompts of M48B_RESULT.md), and
  **tools**, 10 requests from `tools/tooleval.py` INOS (remember, recall, run_code x2, ls, cat, write_file,
  set_setting, read_log, cpuinfo). Only the first round of each tool prompt is measured: the model writes the call,
  and no tool result is fed back.
- Replay: greedy verify with bit-exact batching (M4.8b) means the output is the greedy trajectory, so acceptance is
  replayed over it. The MTP KV holds one entry per accepted position, computed from the real `h_{p-1}` and `x_p`
  (the engine refills it after every verify). At each verify step from committed position j, the MTP drafts
  `d_1..d_n` (step 1 from `(h_{j-1}, x_j)`, then chained). Accepted = the leading drafts equal to the greedy
  tokens, plus the main model's own token. "Tokens per verify" = generated tokens after the first / verify steps.
- Run: Fedora box, 20 prompts, 1015 chat and 325 tool generated positions, 621 s of generation. Peak RSS 9.8 GB
  (bf16 weights + MTP), MemAvailable never below 14.6 GB.

## Acceptance (measured)

Step-1 top-1 over all generated positions (teacher forced): chat **714/1015 = 70.3%**, tools **318/325 = 97.8%**.

Greedy verify replay. "pos k" is the acceptance of draft k, given that drafts 1..k-1 were accepted.

| set | n | tokens / verify | pos 1 | pos 2 | pos 3 | verify steps with a rejected draft |
|---|---|---|---|---|---|---|
| chat | 1 | 1.703 | 423/596 = 0.710 | | | 173/596 |
| chat | 2 | 2.046 | 334/496 = 0.673 | 191/332 = 0.575 | | 303/496 |
| chat | 3 | 2.286 | 306/444 = 0.689 | 177/305 = 0.580 | 94/175 = 0.537 | 347/444 |
| tools | 1 | 1.935 | 162/168 = 0.964 | | | 6/168 |
| tools | 2 | 2.876 | 109/113 = 0.965 | 107/107 = 1.000 | | 4/113 |
| tools | 3 | 3.693 | 83/88 = 0.943 | 81/82 = 0.988 | 78/80 = 0.975 | 8/88 |

Per prompt, n = 3 gives chat 1.95-2.95 and tools 3.18-4.00 tokens per verify (the JSON has every prompt). For
reference, llama.cpp PR 22673 reports 65.9% at n = 3 on Qwen3.5-4B (GPU). Our chat set is in the same range.
Tool calls are almost fully predictable once the function name is out: tags, parameter names, `</parameter>`.

## Cost model for our engine (estimate, not measured)

Unit: one decode step, 48 ms (about 20.8 tok/s, M48B_RESULT.md, 4B q4, 8 vCPU).

- Verify pass of N rows, measured in M48B_RESULT.md (mean of two runs): N = 2 66 ms (1.375 steps), N = 3 71.5 ms
  (1.49). N = 4 is not measured: 79 ms (1.65) is interpolated between N = 3 and N = 5 (87 ms).
- MTP draft step, scaled by bytes read (decode is DRAM-bound): MTP layer about 68 MB Q4 plus the tied lm_head
  0.68 GB Q8, against about 2.4 GB for a main step. That is **d = 0.31 steps**, or **d = 0.18** with a Q4 lm_head
  (T44, about 0.36 GB). The draft's fixed costs (core sync, argmax over 248k logits) are not in d, so these
  numbers are a lower bound on the draft cost.
- The MTP KV refill for the accepted tokens rides in the first draft step as extra rows of the same weight read.
  Its compute is not counted.
- A rejected draft costs a DeltaNet restore of 1.8 ms, and today's engine feeds the accepted tokens of a partial
  accept again as rows of the next pass, at about 12.5 ms per row (M48B). Both are counted, from the replay's
  counters.

`speedup = tokens_per_verify / (verify(n+1) + n * d + p_partial * 0.0375 + refed_rows * 0.26)`

| set | n | Q8 lm_head (d = 0.31) | Q4 lm_head (d = 0.18) | free draft (d = 0, bound) |
|---|---|---|---|---|
| chat | 1 | 1.00x | 1.09x | 1.24x |
| chat | 2 | 0.93x | 1.05x | 1.37x |
| chat | 3 | 0.82x | 0.96x | 1.38x |
| tools | 1 | 1.15x | 1.25x | 1.41x |
| tools | 2 | 1.36x | 1.56x | 1.93x |
| tools | 3 | 1.43x | 1.69x | 2.24x |

At 20.8 tok/s that is, for chat, at best about 22.7 tok/s (n = 1, Q4 head). Tool-call rounds reach about 28 tok/s
(n = 2, Q8 head) to 35 tok/s (n = 3, Q4 head).

Compared with today's prompt-lookup spec (M48B_RESULT.md, measured): quoting 2.31-2.55x, tool rounds 1.00-1.26x,
chat 0.96-1.00x (it drafts almost nothing there). So MTP adds value where lookup has nothing to copy: free text
and tool-call syntax. Lookup stays far ahead on copying, where its draft is free and long.

Why chat does not pay, even at 70% acceptance: the verify pass is not free. Two rows already cost 1.375 steps,
which is the batched DeltaNet and attention compute on top of the weight read. And every draft step reads the
0.68 GB lm_head, so the free-draft bound at n = 2-3 (1.37x) is cut to about 1.0x. The T45 note's estimate of
about 1.3x on all text assumed a much cheaper verify. It is not supported for chat.

## Recommendation

**No-go for the T45 plan as written** (MTP on all text, expected about 1.3x): with the measured verify cost, free
chat comes out at 0.82-1.00x with the Q8 lm_head and at most 1.09x with a Q4 one.

**Conditional go for a narrower version:** MTP drafting only in tool-call rounds (from `<tool_call>` until
`</tool_call>`, or when lookup has no match), n = 2 with the Q8 head (estimated 1.36x) or n = 3 after T44 (1.69x).
Before building it, measure one number on the engine: the real time of an MTP draft step (MTP layer + lm_head
matvec + argmax, 8 cores). If it is above about 0.35 steps (17 ms), tool rounds drop under 1.3x and the feature is
not worth its complexity. Tool-call rounds are short (15-55 tokens here), so the wall-clock gain per agent turn
is small: about 0.5 s on a 40-token call at 1.4x (1.9 s -> 1.4 s).

Things that would change the verdict (not measured here): a cheaper verify for 2-4 rows, a draft lm_head limited
to the most frequent tokens (cuts d well below 0.18; the hit rate needs its own measurement), and acceptance
against our Q4 main model (this spike verifies against the bf16 HF model).

## Implementation outline (if the narrow version goes ahead)

1. **Exporter** (`tools/export_weights.py`): stop dropping `mtp.*`. Write fc, the layer's q/k/v/o and MLP as Q4_0
   (or Q8_0 for fc, about 14 MB), and the 5 norms as f32, in new descriptor slots after the main layers, with a
   header flag for "has MTP". Reuse the tied embedding / lm_head (no copy).
2. **Engine forward** (`engine`): `mtp_forward(rows: &[(hidden, token)], pos0) -> (hidden, logits)`. This is two
   RMSNorms plus a 5120 -> 2560 matvec, then the existing full-attention layer code (gated q, q/k norm, partial
   RoPE) and MLP code pointed at the MTP weights, then the final norm and lm_head. It needs the main model's last
   hidden state per row from decode and from the verify pass. Keep it (2560 floats per row) instead of
   discarding it after the lm_head.
3. **MTP KV**: one extra attention layer's KV (4 x 256 x 2 per position, same element type as the main KV),
   sized like the main context. It must be filled during prefill too: one batched MTP pass over the prompt after
   the main prefill. It also goes into the snapshots (boot prefix point #0, turn points), or gets rebuilt from the
   saved hidden states. Rollback is simple: it is plain KV, so truncating the position is enough, with no state
   copy.
4. **Integration with the M4.8b verify**: the drafter becomes a choice per verify. Lookup if it has a match of 3
   or more, else MTP n tokens if the round is in tool-call mode (or the gate allows it), else plain decode.
   Verify, acceptance, DeltaNet save/restore and the re-feed of a partial accept stay as they are. After the
   pass, the MTP's first draft step of the next round takes the accepted rows (their main hidden states from the
   pass) as its KV refill, in the same weight read. `!set spec` gains `mtp N` / `mtp off`.
5. **Done check** (as in the T45 note): ids identical spec off vs on (it holds by construction, as in M48B), tooleval
   unchanged, and measured tok/s on chat and tool prompts against the estimate above.
