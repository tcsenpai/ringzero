# Model eval (T59): two Qwen3.5-4B fine-tunes vs the 4B q4 working model

Question: does NeoHorse-1-4B or QwenPaw-Flash-4B, exported the same way as the working model (`MODEL=4b QUANT=q4`,
MLP Q4_0, everything else Q8_0), do better as the system agent? Candidates come from
[docs/research/MODEL_SURVEY.md](research/MODEL_SURVEY.md) (class A: same architecture, converter only).

Short answer: **neither one passes the M7 quality gate at Q4, so both are dropped.** The stock Qwen3.5-4B q4 stays
the working model. Nothing was changed in the defaults.

## Candidates

| | NeoHorse-1-4B | QwenPaw-Flash-4B | baseline |
|---|---|---|---|
| Hugging Face repo | `TokenRhythm/NeoHorse-1-4B` | `agentscope-ai/QwenPaw-Flash-4B` | `Qwen/Qwen3.5-4B` |
| License | Apache-2.0 (card metadata and LICENSE file) | Apache-2.0 (card metadata) | Apache-2.0 |
| Languages on the card | inherits Qwen3.5 | en, zh | 201 |
| Checkpoint | `Qwen3_5ForCausalLM`, text only, 2 bf16 shards, tensors under `model.` | `Qwen3_5ForConditionalGeneration` with the vision tower, 1 bf16 file (10 GB), tensors under `model.language_model.` | same layout as QwenPaw |
| `config.json` text part | identical to Qwen3.5-4B | identical except `dtype` (float32 written, weights are bf16) | |
| `chat_template.jinja` | byte-identical to Qwen3.5-4B (md5 `94f89e03...`) | byte-identical to Qwen3.5-4B (same md5) | |
| `tokenizer.json` | byte-identical | same vocab, same merges (written as pairs instead of strings), the 7 audio/tts specials moved from `tokenizer_config.json` into `tokenizer.json` (same ids) | |

Checked on the exported files: the `.asmbin` header (chat token ids, argmax limit), the tokenizer blob and the layer
table are byte-identical for all three exports; only the weights differ. So the kernel's prompt, grammar and token
ids apply unchanged.

## Converter change

`tools/export_weights.py` only needed one fallback: when a Qwen3.5 checkpoint has no
`model.language_model.embed_tokens.weight` but has `model.embed_tokens.weight`, the tensor prefix becomes `model.`
(auto-detected, no new option; the same kind of fallback the embedder already uses). QwenPaw exports unchanged
(the vision tensors are ignored). Both export to 3,361,755,072 bytes, the size of the stock 4B q4 file. Any such
file boots with `make iso MODEL=path/to/file.asmbin EDITION="..." GUEST_MEM=8G`.

## Method

All on the usual build box (8 vCPU KVM guest, 8G, BIOS), one VM at a time; other agents' guests ran before and
after each of these VMs but never during them.

1. **M7 quality gate** (docs/M7_Q4_RESULT.md): `tools/q4_kld.py --dtype bf16 --plans "q4=mlp+mlp_down"`, the
   same corpus and settings as M7 (16 x 512-token chunks of prose and code plus 2 x 4096-token chunks, 9198 scored
   positions), each model against its own HF bf16 weights. Gate: mean KLD <= 0.04 and top-1 >= 92%. Plus
   `tools/hf_agree.py` on the engine's own greedy output (golden prompt plus the five `/gen 48` prompts of M7). The
   stock 4B was run again in the same session as a control.
2. **Speed**: from the same gate boot (no disk): the boot `/bench` plus 3 x `/bench` (64 tokens, short context),
   the boot prefix prefill and `/prefill 32` (the same 1650-token prefix, batched).
3. **Tool use**: `tools/tooleval.py` with the M45 protocol (`SLOT=200`, constrain on, confidence on, temperature 0,
   `/clear` per prompt, a fresh copy of the beta data disk, 120 s idle at the prompt before the first prompt), sets
   `spike,inos,it`. Only for models that pass the gate.
4. **Italian by hand**: four open prompts in the gate boot (explain a file system in two sentences, a short email
   moving a meeting to 15:00, a one-sentence summary, three tips for sleeping better).

## Results

| | 4B q4 (baseline) | NeoHorse-1-4B q4 | QwenPaw-Flash-4B q4 |
|---|---:|---:|---:|
| M7 gate: KLD mean (all 9198 positions) | **0.0278** | 0.0444 | 0.0417 |
| M7 gate: top-1 vs own HF | **92.5%** | 91.2% | 90.3% |
| KLD p99 | 0.203 | 0.393 | 0.329 |
| KLD prose / prose-long | 0.032 / 0.032 | 0.053 / **0.091** | 0.050 / 0.053 |
| KLD code / code-long | 0.025 / 0.014 | 0.033 / 0.022 | 0.034 / 0.021 |
| PPL own HF ref -> quant | 8.193 -> 8.326 (+1.6%) | 8.550 -> 8.705 (+1.8%) | 8.988 -> 9.157 (+1.9%) |
| hf_agree, engine greedy vs own HF argmax | 200/216 = 92.6% | 206/222 = 92.8% | 202/216 = 93.5% |
| **Gate** | **pass** | **fail** | **fail** |
| Decode tok/s, median of 4 `/bench` (range) | 20.6 (19.9-21.1) | 20.3 (19.8-21.0) | 20.4 (20.4-20.8) |
| Boot prefix prefill, 1650 tokens | 22.3 s (74 tok/s) | 20.5 s (80 tok/s) | 20.6 s (80 tok/s) |
| `/prefill 32`, same 1650 tokens | 20.6 s (80 tok/s) | 20.5 s (81 tok/s) | 22.6 s (73 tok/s) |
| Prompt + tools (boot prefix) | 1650 tokens | 1650 tokens | 1650 tokens |
| spike (30): correct / first call right (of 25) / args exact (of 15) | 24, 22, 9 | not run (dropped) | not run (dropped) |
| in-OS (32): correct / first call right (of 30) | 26, 24 | not run | not run |
| of them the 8 info prompts correct | 5 | not run | not run |
| malformed / schema errors / grammar lines not valid | 0 / 0 / 1 (spike, an unfinished call) | not run | not run |
| Italian set `it` (10), first line right by hand | 8 | not run | not run |

The control run reproduces M7 (0.0275 / 92.7% then, 0.0278 / 92.5% now) and the baseline tooleval reproduces the
last T55 run (spike 24/30, 22, 9; in-OS 26/32, 24), so the harness and the corpus did not move.

Reading the gate rows: the fine-tunes lose about 1.5x more to the same RTN Q4 rule than the stock model, mostly on
prose (NeoHorse's long prose chunk: 0.091, three times the stock model). The end-to-end greedy agreement on 216-222
tokens is the same for all three (92.6-93.5%), as in M7, where q4 and q4all also tied on that check; the 9198-position
KLD is the finer measure. Speed is the same for all three within the noise of a shared box, as expected for the same
architecture and file size.

## Notable observations

- Baseline failures on the tool sets are the known ones (docs/T55_RESULT.md): spike prompts for tools the kernel
  does not have (hostname, wasm module through `run_code`, copy with `cp`), extra `ls` after a `cat` on a missing
  path, "quanto fa 2**100? usa python" going to `run_wasm`, and three system-information prompts answered without a
  call (hostname, version) plus "cosa sai di me?" answered without `recall`. One grammar line is counted not valid:
  the wasm-module spike prompt, where a 460-token `write_file` call ended unfinished.
- Italian by hand (gate boot, no disk). All three write fluent Italian. Differences:
  - Email moving the meeting to 15:00: the baseline first calls `recall` for no reason and then writes "alle 15:00
    invece che alle 14:00" (14:00 is invented); NeoHorse invents the same 14:00; QwenPaw writes it without an
    invented time.
  - One-sentence summary: baseline and QwenPaw shorten it ("a causa della pioggia e del freddo"); NeoHorse repeats
    the sentence word for word.
  - Sleep tips: NeoHorse's third tip is about not sleeping "sul divano" (the previous prompt had a cat on a sofa,
    but `/clear` had restored the boot prefix in between) and suggests "occhiali da sole per dormire"; the other two
    give plain tips. All three have one small grammar slip ("sveglia ti stesso", "svegliali").
  - File system in two sentences: all three correct.
- So QwenPaw looked as good as or slightly better than the baseline on these four prompts, NeoHorse slightly worse.
  Four prompts are not a measurement.

## Decision

| Model | Decision | Reason |
|---|---|---|
| NeoHorse-1-4B | **drop** | fails the M7 gate at Q4: KLD 0.0444 (> 0.04), top-1 91.2% (< 92%), long prose KLD 0.091 |
| QwenPaw-Flash-4B | **drop** | fails the M7 gate at Q4: KLD 0.0417 (> 0.04), top-1 90.3% (< 92%); card lists only English and Chinese |
| Qwen3.5-4B q4 | **keep** as the working model | passes the gate, reference tool scores above |

Recommendation: keep the stock 4B q4. No candidate won, so nothing to switch.

If someone wants to try these fine-tunes again, the gate failure is about how their weights take RTN Q4 on the
MLP, not about their tool skills, which were not measured here. The ways out all cost speed or work: Q8_0 everywhere
(4.5 GB, about 14 tok/s, below the 19 tok/s floor), Q4 on gate and up only (about 3.7 GB, slower than q4; M7's
stock number for that plan is KLD 0.016), or a calibrated Q4 (imatrix), which the exporter does not have. QwenPaw
also has a 2B (`QwenPaw-Flash-2B`, Q8 like the default 2B) that was not part of this run.

## Reproduce

```sh
# exporter (auto-detects the model. prefix)
uv run --no-project --with numpy python3 tools/export_weights.py --model-dir models/hf/neohorse-1-4b \
    --out models/t59-neohorse-q4_0mlp-v4.asmbin --q4 mlp,mlp_down
make iso MODEL=models/t59-neohorse-q4_0mlp-v4.asmbin EDITION="ringzero NeoHorse@4b.q4-Qwen3@0.6b (QweQwe)" GUEST_MEM=8G
# gate (HF bf16 on the host, about 10 GB RAM, no guest running)
uv run --no-project --with torch --with transformers --with numpy python3 tools/q4_kld.py \
    --model-dir models/hf/neohorse-1-4b --dtype bf16 --plans "q4=mlp+mlp_down" --work build/kld --out kld.json
MEM=8G python3 tools/boottest.py build/ringzero.iso bios build/gb 8 /bench /bench /bench "/prefill 32" \
    "/gen 8 Say hi in 5 words." "/gen 48 ..." > gb.txt   # the five /gen 48 prompts of M7
uv run --no-project --with torch --with transformers --with numpy python3 tools/hf_agree.py \
    --model-dir models/hf/neohorse-1-4b --dtype bf16 gb.txt
# tools (M45 protocol), a fresh copy of the beta data disk, 120 s idle first
SLOT=200 python3 tools/tooleval.py 127.0.0.1 PORT on eval.json spike,inos,it
```
