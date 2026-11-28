# Embedder fine-tune for memory recall: plan, data and gate (T74)

Date: 2026-11-28. Task T74, step (a) of the T67 decision for E2 (memory recall 95%+). Scaffold and data only:
**no training run has been started**; the user decides when. Numbers are measured unless marked **[E]**.

## Summary

- **Data generated**: 3,972 synthetic (question, positive, 3 hard negatives) records, 3,551 train and 421 dev,
  for the 11 answerable T69 classes, Italian and English, 1.5 MB in `docs/spikes/embft/` (committed).
  Pure Python, no model, deterministic (seed 7: the same bytes on every run).
- **Disjoint from the T69 eval set**: `tools/embft/check_disjoint.py` finds 0 hard leaks in 3,972 records (no
  eval question or eval entry, no shared word 6-gram with an eval question or 8-gram with an eval entry, no eval
  file path, no eval name, place or brand). The generator drops 207 records that hit one of these rules
  (137 equal to an eval question once lowercased, such as "when is my mother's birthday?", 65 sharing a 6-gram,
  5 sharing an 8-gram with an eval entry).
- **Model choice**: LoRA (r 16, alpha 32, every attention and MLP projection), merged before export, so the
  kernel's format and `.asmbin` path do not change. Reason below.
- **Export path verified**: a checkpoint written with `save_pretrained` (as `train.py` writes it) exports with
  `tools/export_weights.py --v3` to a file byte-identical to the stock export (same SHA-256), so a merged
  fine-tune drops into `make EMBED=...` with no tool change.
- **Gate verified on the base model** (base against itself, a GPU host): T69 held-out top-1 51/114 and
  no-answer 12/25, the T69 baseline exactly, so the gate reaches the real scorer; agreement 1.0, pass.
- **Expected cost of a training run [E]**: about 15 minutes and 3 to 5 GB of VRAM on a 16 GB GPU, 222 steps.

## Data

Generator: `tools/embft/gen_data.py OUT_DIR [--seed N] [--personas N]`. 120 synthetic personas (the user, 6
relatives or friends with names, cities, jobs and birthdays, a pet, files in other directories than the eval
set's), and per class one generator that writes the class's trap as the hard negatives:

| class | train | dev | IT | EN | templates | the positive | the hard negatives |
|---|---|---|---|---|---|---|---|
| para | 356 | 44 | 212 | 188 | 4 | a user fact, question paraphrased | the user's other facts |
| exact | 360 | 40 | 197 | 203 | 2 | the entry holding a host name or script name | entries with sibling tokens (nas-41 vs nas-42) |
| pointer | 358 | 42 | 202 | 198 | 1 | the file chunk | the episode that read the file, the note naming it, another file |
| sibling | 216 | 36 | 126 | 126 | 21 | a help section (manual format, generic Unix topics) | its sibling sections, one from another family |
| hub | 368 | 32 | 191 | 209 | 1 | the specific fact, for a 2-3 word question | greeting and generic episodes, /etc/motd |
| owner | 352 | 48 | 212 | 188 | 4 | the user's (or the named relative's) record | the same attribute of another person |
| raw | 102 | 18 | 60 | 60 | 20 | a `[said]` note in other words | an entry sharing the topic word |
| table | 358 | 42 | 202 | 198 | 2 | a CSV row chunk, a script chunk | other rows, the episode that ran the script |
| temporal | 354 | 46 | 203 | 197 | 1 | the newer record ("adesso", "switched to") | the older record |
| contradiction | 361 | 39 | 203 | 197 | 1 | the correcting record ("mi sbagliavo", "correzione") | the record it corrects |
| multihop | 366 | 34 | 208 | 192 | 2 | first hop: the record naming the person; second step: the record about that person | other relations' records, the first hop |
| **all** | **3,551** | **421** | 2,016 | 1,956 | 59 | | |

"Templates" counts question templates; slot values (names, cities, numbers, files) vary per persona. The dev
split holds whole personas (sibling and raw, which have no persona of their own: whole templates). The multihop
second step is written in the form `tools/memhop.py` re-embeds (question plus the name from the first hop), so
the model learns the query the kernel would send on a second hop. No-answer has no positive and is left to the
gate (abstention at the kernel floors). Files: `train.jsonl`, `dev.jsonl`, `stats.json` (counts, SHA-256 of both
files), `disjoint.txt` and `disjoint.json` (the check's report).

Weak points: the wording comes from templates (one to four per class, except sibling and raw), so a fine-tune
can learn the template instead of the class. Train and dev share templates, so the dev loss will flatter; the
T69 set is the real test. If the fine-tune overfits the wording, the next step is a paraphrase pass by a model
on a fleet host (marked as a shortcut in the generator).

## Disjointness check

`tools/embft/check_disjoint.py DATA_DIR` (exit 1 on any hard leak), run on the committed data: **0 hard leaks
of 3,972**. The eval side is everything T69 scores: 259 questions, 197 index entries (memories, manual sections,
file chunks, CSV rows), 43 file paths, 120 capitalized entity words (minus generic words such as day names and
`NAME`). Reported but not failing, the share of training questions that share a short n-gram with an eval
question:

| class | records | shares a word 3-gram | shares a 4-gram |
|---|---|---|---|
| para | 400 | 38% | 16% |
| exact | 400 | 31% | 0% |
| pointer | 400 | 12% | 0% |
| sibling | 252 | 6% | 0% |
| hub | 400 | 26% | 0% |
| owner | 400 | 100% | 53% |
| raw | 120 | 22% | 5% |
| table | 400 | 0% | 0% |
| temporal | 400 | 25% | 8% |
| contradiction | 400 | 64% | 32% |
| multihop | 400 | 55% | 26% |

The 3- and 4-gram hits are question frames ("quando e' il compleanno", "what does my brother"), not the eval
questions or their answers: owner questions are all of the form "when is RELATION's birthday?". This is the
point of the class, and the 6-gram rule keeps whole questions out. Read the owner and contradiction numbers
after a run as optimistic for those classes.

## Training script

`tools/embft/train.py --data DIR --out DIR [--dry-run] [--check-model]`, plain torch and peft:

- **Format**: the kernel's (kernel/src/embed.rs). Query `Instruct: TASK\nQuery:QUESTION`, entry = index text;
  tokens without special tokens, then EOT 151643; last token's hidden state, L2-normalized. Batches pad on the
  left; `--check-model` measured batched vs one-at-a-time vectors at min cosine 1.000000.
- **Loss**: InfoNCE, temperature 0.05, each query against its positive, its 3 hard negatives and every other
  text in the batch (32 queries, 160 texts per step).
- **LoRA, not full**: the gate wants the model to stay general, and a rank-16 update moves it less than full
  training; the optimizer state is a few MB instead of about 10 GB [E] for full AdamW on 0.6B (fp32 master
  weights and two moments), which would not leave room on a 16 GB card; `merge_and_unload` restores a plain
  Qwen3Model, so export and the kernel see no difference.
- Defaults: lr 1e-4 (10% warm-up, linear decay), 2 epochs (222 steps), bf16 autocast, gradient checkpointing,
  max 256 tokens, dev loss every 20 steps; saves the merged model in bf16 with the tokenizer and
  `train_log.json`.
- `--dry-run`: checks schema, classes, 3 negatives, no positive among the negatives, no `Instruct:` prefix in
  the data, and re-runs the leak check; prints the config and the steps; trains nothing. Run on the committed
  data: 0 errors.

## Eval hook

`memeval.py --embedder DIR run OUT.json` (and `embed`) loads a checkpoint instead of the stock embedder
(`rerankeval.EMBEDDER`). To score design A's frozen config on fine-tuned vectors:

```
memeval.py --embedder CKPT embed VEC_FT.json docs/spikes/memab/ctx.json
memeval.py ab VEC_FT.json docs/spikes/memab/config.json AB_FT.json docs/spikes/memab/ctx.json
```

The config stays the one frozen on tune in T70; the fine-tune changes the vectors only, and held-out is scored
once. The hub probes, owner tables and supersession cosines are recomputed from the new vectors by `embed`.

## Gate (the M7 idea for the embedder)

`tools/embft/gate.py --ft CKPT [--out GATE.json]`, exit 0 = pass. The base and the fine-tuned model, both in
the kernel format:

1. **General retrieval**: section heading -> section text over the repo's prose (README, INFO, docs/*.md
   without the manual and the memory research documents, 181 pairs), top-1 and MRR@10.
2. **Agreement**: cosine between the base and the fine-tuned vector of each general text.
3. **Calibration** on the T69 set with the kernel ranking: held-out answerable top-1, right no-answer
   abstentions at the kernel floors, median top-1 cosine. `RECALL_MIN` 0.55 was set on the base model's scale;
   a fine-tune that sharpens the scores moves every floor.

Pass [E thresholds, to revisit after the first run]: general top-1 drop at most 2 points and MRR drop at most
0.02, agreement mean at least 0.90, no-answer right not below the base, median top-1 cosine within 0.05.

Base numbers (measured, base against itself): general top-1 13.8%, MRR@10 0.227 (headings are short and often
generic, such as "What it does", so the task is hard; it is a regression probe, not a benchmark); T69 held-out
51/114, no-answer 12/25, median top-1 cosine 0.635; agreement 1.0. The general set is English technical prose:
add a public Italian and English retrieval set before trusting the gate on general Italian (noted in the script).

## Export into the kernel

`tools/export_weights.py --v3` needs a directory with `config.json`, one `model.safetensors` (BF16, F16 or F32;
names with or without the `model.` prefix), `tokenizer.json` and `tokenizer_config.json`. `train.py` writes
exactly that. Measured: the stock embedder loaded and saved again with `save_pretrained` exports byte-identical to
the stock export (4.4 s, 190 MB of RAM). Then:

```
HF_DIR=CKPT EXPORT_ARGS=--v3 FORCE=1 tools/fetch_model.sh models/qwen3-embed-ft-q8_0-v3.asmbin
make EMBED=models/qwen3-embed-ft-q8_0-v3.asmbin       # the manual vectors depend on $(EMBED) and are rebuilt
```

An existing memory index holds vectors of the old model: after the switch it must be rebuilt (agentd's
`rebuild`), or recall mixes two vector spaces.

## Launch (not run)

From the Mac, with a copy of the repo (no `.git`) in `rz/` and the venv made once on the GPU host
(`uv venv .venv && uv pip install --python .venv/bin/python torch transformers peft` inside the synced
directory):

```
~/.local/bin/fleet run GPU_HOST --gpu --sync rz --fetch ckpt/embft-r16/gate.json -- \
  '.venv/bin/python tools/embft/train.py --data docs/spikes/embft --out ckpt/embft-r16 &&
   .venv/bin/python tools/embft/gate.py --ft ckpt/embft-r16 --out ckpt/embft-r16/gate.json'
```

Then, as separate jobs (each under the 1 h limit): the eval hook above with `--embedder ckpt/embft-r16`, and the
export with `--fetch models/qwen3-embed-ft-q8_0-v3.asmbin`. The GPU host is the one that does not serve the
user's live speech and chat models.

## Expected cost of a run [E]

- Tokens: queries 28 characters plus the 23-token instruction, entries 64 to 70 characters (measured means), so
  about 40 tokens per text, 160 texts, about 6,400 tokens per step.
- Compute: forward 2 x 0.6 G x 6,400 = 7.7 TFLOP per step; backward through every layer plus the checkpoint
  recompute make it about 4x, 31 TFLOP per step. At 10 TFLOP/s sustained on a laptop RTX 3080 class GPU in bf16:
  about 3 s per step, 222 steps about 11 minutes, plus 12 dev passes and model load: **about 15 minutes**.
- VRAM: bf16 weights 1.2 GB, LoRA weights and AdamW state under 0.2 GB, activations with checkpointing about
  1 to 2 GB for 160 short texts, CUDA context and allocator slack about 1 GB: **3 to 5 GB of 16**.
- Host RAM: about 4.4 GB (measured for the same model load in the T73 embedding job), under the 10 GB job cap.
- Gate afterwards: 36 s, 6.4 GB of host RAM peak (measured on the base run).

## Files

- `tools/embft/gen_data.py`, `tools/embft/check_disjoint.py`, `tools/embft/train.py`, `tools/embft/gate.py`
- `tools/memeval.py --embedder`, `tools/rerankeval.py` (`EMBEDDER`)
- `docs/spikes/embft/`: `train.jsonl`, `dev.jsonl`, `stats.json`, `disjoint.txt`, `disjoint.json`, `gate_base.json`
