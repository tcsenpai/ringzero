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

# T64: four agent and tool-calling fine-tunes vs the 4B q4 working model

Question: do fine-tunes made specifically for agents or tool calling, at most 4B, do better than the stock
Qwen3.5-4B q4 as the system agent? Candidates come from the wider search in
[docs/research/AGENT_FINETUNES.md](research/AGENT_FINETUNES.md) (about 40 models listed and ranked); the top four
there that run on the engine as is were tested. All four are Qwen3.5-4B fine-tunes with the stock layer layout,
vocabulary and tool-call format; each exports to 3,361,755,072 bytes, the size of the stock 4B q4 file, with no
converter change.

Short answer: **none of them is better on our tool sets, so the stock 4B q4 stays.** AgentMercury ties the baseline,
Agents-A1 and EU-Tool pass the quality gate but score lower, SkillGym fails the gate.

## Method

Exactly the T59 pipeline above, on the same build box, one VM at a time under the shared VM lock: export with
`--q4 mlp,mlp_down`, the M7 gate (`tools/q4_kld.py` against each model's own HF bf16 weights, then `tools/hf_agree.py`
on the gate boot), decode speed from the gate boot (boot `/bench` plus 3 x `/bench`, 64 tokens) and again from the
tool session (3 x `/bench` after the prompts), and `tools/tooleval.py` with the M45 protocol (`SLOT=200`, constrain on,
confidence on, temperature 0, a fresh copy of the beta data disk, 120 s idle first), sets `spike,inos,it`. The tool
run was done for every model, also the one that failed the gate, because it is automatic.

Control: the stock 4B q4 was built from the same commit and run through the same gate boot and tool session in this
session. Its gate numbers were not run again: the weights and the exporter are the same as in T59 (0.0278 / 92.5%).
Its tool scores reproduce T59 and T55 exactly (spike 24/30, 22, 9; in-OS 26/32, 24; info 5/8).

## Results

| Model | Base | License | KLD mean | top-1 | Gate | Decode tok/s, gate boot median (range) / tool session median | spike (30): correct, first call right (25), args exact (15) | in-OS (32): correct, first call right (30); info (8) | it (10) answered without a tool | Verdict |
|---|---|---|---:|---:|---|---|---|---|---|---|
| Qwen3.5-4B q4 (control) | - | Apache-2.0 | 0.0278 (T59) | 92.5% (T59) | pass | 19.9 (18.5-21.5) / 21.5 | 24, 22, 9 | 26, 24; 5 | 10 | **keep** |
| [InternScience/Agents-A1-4B](https://huggingface.co/InternScience/Agents-A1-4B) | Qwen3.5-4B | Apache-2.0 | 0.0282 | 92.9% | pass | 19.1 (18.6-19.7) / 21.6 | 23, 20, 9 | 24, 22; 5 | 10 | drop |
| [birgermoell/Qwen3.5-4B-EU-Tool](https://huggingface.co/birgermoell/Qwen3.5-4B-EU-Tool) | Qwen3.5-4B | Apache-2.0 | 0.0282 | 92.2% | pass | 21.1 (19.4-21.4) / 21.1 | 22, 22, 9 | 25, 25; 4 | **3** (7 tool calls) | drop |
| [Minbyul/AgentMercury-Qwen3.5-4B](https://huggingface.co/Minbyul/AgentMercury-Qwen3.5-4B) | Qwen3.5-4B | Apache-2.0 | 0.0282 | 92.6% | pass | 20.3 (16.6-21.3) / 20.3 | 24, 22, 9 | **27**, 24; 5 | 10 | drop (tie) |
| [reasonwang/SkillGym-Qwen3.5-4B](https://huggingface.co/reasonwang/SkillGym-Qwen3.5-4B) | Qwen3.5-4B | Apache-2.0 | 0.0345 | **91.3%** | **fail** | 20.3 (19.3-20.5) / 19.8 | 23, 23, 9 | 24, 25; 5 | 10 | drop |

More gate detail (KLD p99; prose / prose-long KLD; PPL own HF -> quant; hf_agree):

| Model | p99 | prose / prose-long | PPL | hf_agree |
|---|---:|---|---|---|
| Agents-A1-4B | 0.262 | 0.029 / 0.032 | 9.081 -> 9.222 | 203/220 = 92.3% |
| EU-Tool | 0.218 | 0.032 / 0.042 | 8.178 -> 8.310 | 206/217 = 94.9% |
| AgentMercury | 0.206 | 0.033 / 0.034 | 8.274 -> 8.405 | 203/216 = 94.0% |
| SkillGym | 0.267 | 0.041 / 0.050 | 8.030 -> 8.198 | 206/220 = 93.6% |

No model had a malformed call, a schema error or an invalid grammar line (the control had its usual one, the
unfinished wasm-module `write_file` call). The single gate-boot reading of 16.6 tok/s (AgentMercury) and the other
low readings come from a shared box: the tool-session `/bench` of the same model is 20.2-21.4. All four stay at or
above about 19 tok/s, as expected for the same architecture and file size.

The gate failure of SkillGym is on top-1 only (KLD 0.0345 is under 0.04) and again on prose, the pattern T59 saw on
NeoHorse and QwenPaw. The three that pass are probably the ones whose weights moved least from the base [E]: AgentMercury is 200 RL
steps, and EU-Tool's own HF perplexity is the same as the stock model's (8.178 vs 8.193).

## What the tool runs show

Per prompt differences against the control are in [docs/spikes/t64/tool_diffs.txt](spikes/t64/tool_diffs.txt).

- **AgentMercury**: the same answers as the control on every prompt but one: "leggi /notes/todo.txt" reads the file
  once instead of adding an `ls`, which is +1 in-OS. One prompt is within the run-to-run noise of a 32-prompt set, so
  this is a tie, not a win.
- **Agents-A1**: answers "spegni la macchina", "riavvia", "usa 4 core per l'inferenza" and "what's 17 times 23? use
  python" in text without calling the tool, and loops `meminfo` ten times on "set the screen brightness to 80" (the
  round limit stops it). It reads files more directly (one `cat` where the control adds `ls`).
- **EU-Tool**: calls tools where none is needed: `recall` on "hi, how are you?", "what's the capital of France?",
  "Chi ha scritto la Divina Commedia?"; `run_code` on date and time arithmetic and on "traduci 'good morning' in
  italiano". So only 3 of the 10 Italian knowledge questions are answered directly. It gets more first calls right in
  the OS set (25/30, the best of all, for example `recall` on "cosa sai di me?" and `run_code` on "2**100"), but the
  over-calling outweighs it. In the gate boot it also stored the e-mail request with `remember` instead of writing
  the e-mail. Its card already warned about weak abstention (relevance 0.84).
- **SkillGym**: hits the tool round limit four times in the OS set (`ls`/`cat` loops on "elenca i file in /notes",
  `run_code` loops on the brightness prompt and on "cerca nei file qualcosa sul latte").
- Italian by hand (gate boot, four open prompts): all write fluent Italian. Agents-A1 and SkillGym repeat the
  one-sentence summary word for word; AgentMercury shortens it ("a causa della pioggia e del freddo") as the control
  does; none of the three that wrote the e-mail (EU-Tool called `remember` instead) invents an old meeting time, which the control does ("invece che alle 14:00"). Small slips:
  Agents-A1 "una maschera per il rumore", SkillGym "occhiali da sole" for sleeping, as NeoHorse in T59.

## Decision

| Model | Decision | Reason |
|---|---|---|
| Agents-A1-4B | **drop** | passes the gate; lower on both tool sets (spike 23 vs 24, in-OS 24 vs 26), skips poweroff, reboot and set_setting calls |
| Qwen3.5-4B-EU-Tool | **drop** | passes the gate; calls tools on chit-chat and knowledge questions (Italian set 3/10 without a tool), spike 22 vs 24 |
| AgentMercury-Qwen3.5-4B | **drop** (tie) | passes the gate; spike equal, in-OS 27 vs 26 (one prompt); not enough to switch, the only one worth a second look if the tool sets grow |
| SkillGym-Qwen3.5-4B | **drop** | fails the M7 gate at Q4 (top-1 91.3%, prose-long KLD 0.050); tool round limit hit 4 times |
| Qwen3.5-4B q4 | **keep** as the working model | control reproduces T55/T59 exactly |

Nothing in the defaults changed. Downloaded weights, exported files, ISOs and test disks were deleted after the runs;
raw results (gate JSON, hf_agree logs, gate-boot logs, tooleval JSON and logs, per prompt diffs) are in
[docs/spikes/t64/](spikes/t64/), one summary line per model in `summary.jsonl`.

## Reproduce

Same commands as the T59 Reproduce section, with the model directories of the four repos above
(`hf download REPO --include "*.json" --include "*.jinja" --include "*.safetensors"`), one model at a time.
