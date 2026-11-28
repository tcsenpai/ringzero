# Memory recall eval set and baseline (T69)

Date: 2026-11-28. Task T69, step 1 of E2 (memory recall 95%+). Input for T70, T71 and T67.

All numbers are measured on the host with `tools/memeval.py` (embedder in PyTorch fp32 on the CPU of the build
box); raw cosines in `docs/spikes/memeval/results.json`. Reproduce: `memeval.py check` (no model),
`memeval.py run OUT.json` (about 2 minutes on CPU, about 3 GB of RAM), `memeval.py report OUT.json`.

## Summary

- **Set**: 259 queries (about 110 Italian, the rest English) over a 186-entry index (138 memories, 32 manual
  sections, 16 file chunks), 12 classes, fixed tune / held-out split stratified by class (133 / 126).
- **Scorer check**: on the 93 T63 queries, the T63 corpus and T66's reading (manual queries ranked over manual
  sections, the others over the rest), the scorer gives **40/46 held-out top-1 (87.0%)**, 73/93 on all, the
  same as T66. With the kernel's BM25 gate modelled: 40/46 held-out, 75/93 all.
- **Today's kernel is lower than 87% even on the T63 set**: 34/46 (73.9%). The kernel gives a manual section to
  the model only when the message is a how-to question (`howto()`), and 15 of the 43 answerable manual queries
  are not ("quali programmi sono installati?", "how full is the context?", "/agentd log").
- **Baseline on the new set** (kernel ranking, held-out, answerable): **51/114 top-1 (44.7%)**, 76/114 top-3
  (66.7%); no-answer 8/12 held-out. With T66's split reading: 55/114 (48.2%).
- **Furthest from 95%** (held-out top-1): contradiction 0/7, raw note 1/7, table or code 1/7; then multi-hop
  2/12, wrong owner 2/10, temporal 3/12. Contradiction and temporal have top-3 at 7/7 and 9/12: both records
  are found and the older one wins, so the missing piece is order, not recall.

## Set design

Built on `tools/rerankeval.py` (corpus, file chunker, embedder) and `tools/retrieval.py` (the first 40
memories); `memeval.py` imports both and appends its entries to rerankeval's lists, so the T63 corpus is a
prefix of the new one.

**Corpus additions** (85 entries, all in agentd's index formats: `[fact] key: value`, `[said] ...`,
`[episode N] ...`, `[PATH] text` chunks):

- 21 records in 10 temporal pairs and 1 trio: an older and a newer record of the same thing (phone, car, dentist
  appointment, internet provider, head teacher, diet, running goal, rent, piano lesson, email; the Acme
  deadline as an older episode and a newer file). agentd replaces a fact of the same key, so a change arrives
  as a `[said]` note, an episode, or a fact under another key, as here.
- 16 records in 8 contradiction pairs: two records of one event that disagree (backup ran / failed, parcel day,
  train time, mother's birthday, meeting room, arrival time, library due date, dinner day).
- 14 first-hop and second-hop records for multi-hop questions (the sister's name and city, a best friend and
  where he moved, the Acme contact who is a row of `people.csv`, the new head teacher and the meeting she set).
- 8 relatives' records next to the user's own (brother's phone and job, sister's birthday and laptop, father's
  doctor, mother's city, the partner's cat, the sister's dog's food).
- 10 pointers: an episode or fact that names a file (`[episode 42] The agent read
  /home/ricetta_carbonara.txt aloud ...`).
- 12 raw `[said]` notes worded unlike the question ("non sopporto il coriandolo" for "is there a herb I can't
  stand?").
- 2 hub entries (a greeting episode, a generic question-and-answer episode).
- 2 files: `/home/stampante.txt` (printer details) and `/home/spese_novembre.csv` (expenses table).

Every new memory carries a date in `DATE` (and 3 older entries that take part in a pair). agentd stores no time
in an index text, so today's ranking cannot use the dates; they define which record is newer and are there for
designs that stamp memory lines (T70, T71).

**Queries**: the 93 T63 queries keep their gold, with one fix from T66 ("what model runs this OS?" also accepts
`/demo/data.json`), plus 166 new ones, written as a user types them. Gold is a set of keys (any chunk of a gold
file counts). A multi-hop query also lists its first hop: top-1 must be the answer, top-3 must hold both.

| Class | What it tests | n | held-out | memory | manual | files |
|---|---|---|---|---|---|---|
| para | paraphrase, no trap | 55 | 27 | 30 | 19 | 6 |
| exact | identifier or file token | 12 | 6 | 7 | 2 | 3 |
| pointer (P) | a memory naming a file vs the file | 19 | 9 | 0 | 0 | 19 |
| sibling (S) | close manual sections | 22 | 11 | 0 | 22 | 0 |
| hub (H) | short or generic query, generic entries near | 13 | 6 | 10 | 0 | 3 |
| owner (E) | the user's vs a relative's record | 20 | 10 | 20 | 0 | 0 |
| raw (R) | answer is a `[said]` sentence in other words | 14 | 7 | 14 | 0 | 0 |
| table (T) | CSV rows, code and JSON content | 15 | 7 | 0 | 0 | 15 |
| temporal | a newer record supersedes an older one; gold = newer | 24 | 12 | 22 | 0 | 2 |
| contradiction | two records of one event disagree; gold = later | 15 | 7 | 15 | 0 | 0 |
| multihop | needs two entries | 25 | 12 | 20 | 0 | 5 |
| noanswer | nothing stored answers it | 25 | 12 | 18 | 4 | 3 |
| all | | 259 | 126 | 156 | 47 | 56 |

The pool is the pool of the gold (for no-answer: where a user would expect the answer). Split: inside each
class, queries sorted by the md5 of their text, alternating tune and held-out. The T63 check keeps T63's own
even/odd split.

## Scorer

What it ports from the kernel:

- `memidx::search`: cosine order; BM25 (k1 1.5, b 0.75, the stopword list, `\w` words lowercased) over every
  entry; the gate (`identifier_like`, or the top BM25 entry has every query term and scores 1.5x the second);
  RRF with k = 60 when the gate opens; the top 24.
- `agentd::hits_split`: manual sections apart from the rest.
- `agentd::howto`, `slot`, `recall`: in **kernel** mode the list the model sees starts with the best manual
  section when the message is a how-to question and that section scores `RECALL_MIN` (0.55) or more, then the
  non-manual hits. A non-how-to question never gets a manual section, as in the kernel.
- **split** mode is T66's reading (manual queries over manual sections, the others over the rest); it is the
  upper bound of a perfect router between the manual and the rest.
- No-answer is right when none of the first 4 (`RECALL_K`) of the list reaches its floor: 0.55 for an
  embedding hit, 0.40 (`RECALL_MIN_BM25`) when BM25 joined the ranking. Facts count, although the slot pins all
  facts anyway, because `recall` returns them with their scores.

Not ported: the Q8_0 rows of the index (the host keeps fp32 vectors) and the order in which entries enter the
index (it only breaks ties).

**Check against T66** (93 T63 queries, old split):

| Reading | Corpus | Held-out top-1 | All top-1 | Held-out top-3 |
|---|---|---|---|---|
| split, no BM25 (T66) | T63, 101 entries | 40/46 (87.0%) | 73/93 (78.5%) | 45/46 |
| split, BM25 gate | T63 | 40/46 (87.0%) | 75/93 (80.6%) | 45/46 |
| kernel, BM25 gate | T63 | 34/46 (73.9%) | 65/93 (69.9%) | 37/46 |
| split, BM25 gate | new, 186 entries | 37/46 (80.4%) | 71/93 (76.3%) | 44/46 |
| kernel, BM25 gate | new | 31/46 (67.4%) | 61/93 (65.6%) | 36/46 |

The first row matches T66 exactly, so the embedder path is the same. The BM25 gate rescues 2 exact-token
queries outside the held-out half. The 6-point drop from 87.0% to 80.4% on the larger corpus comes from the new
distractors (a pointer episode beats `/home/todo.md`, an opened-file episode beats the Japan trip file).

## Baseline (kernel ranking, BM25 gate, 186 entries)

| Class | n | held-out top-1 | held-out top-3 | tune top-1 | all top-1 | all top-3 | all top-1 >= floor |
|---|---|---|---|---|---|---|---|
| para | 55 | 24/27 | 25/27 | 25/28 | 49/55 | 50/55 | 43/55 |
| exact | 12 | 4/6 | 5/6 | 3/6 | 7/12 | 10/12 | 6/12 |
| pointer | 19 | 5/9 | 7/9 | 4/10 | 9/19 | 14/19 | 9/19 |
| sibling | 22 | 6/11 | 6/11 | 4/11 | 10/22 | 10/22 | 10/22 |
| hub | 13 | 3/6 | 4/6 | 1/7 | 4/13 | 7/13 | 3/13 |
| owner | 20 | 2/10 | 6/10 | 8/10 | 10/20 | 16/20 | 9/20 |
| raw | 14 | 1/7 | 2/7 | 3/7 | 4/14 | 7/14 | 3/14 |
| table | 15 | 1/7 | 3/7 | 4/8 | 5/15 | 8/15 | 5/15 |
| temporal | 24 | 3/12 | 9/12 | 2/12 | 5/24 | 17/24 | 5/24 |
| contradiction | 15 | 0/7 | 7/7 | 1/8 | 1/15 | 14/15 | 1/15 |
| multihop | 25 | 2/12 | 2/12 | 5/13 | 7/25 | 10/25 | 6/25 |
| noanswer (right abstentions) | 25 | 8/12 | | 4/13 | 12/25 | | |
| answerable | 234 | **51/114 (44.7%)** | 76/114 | 60/120 | 111/234 | 163/234 | 100/234 |
| pool memory | 138 | 27/66 | 45/66 | 36/72 | 63/138 | 102/138 | 54/138 |
| pool manual | 43 | 14/22 | 14/22 | 10/21 | 24/43 | 24/43 | 24/43 |
| pool files | 53 | 10/26 | 17/26 | 14/27 | 24/53 | 37/53 | 22/53 |

"top-1 >= floor" is a top-1 hit that also passes the slot's floor, so it would enter the memory slot.

With T66's split reading only the manual rows change: sibling 7/11 held-out (14/22 all), para 26/27, exact 5/6,
pool manual 18/22; answerable held-out 55/114 (48.2%).

## Where it is furthest from 95%

Held-out top-1, answerable classes, from the bottom (all-queries number in brackets, the halves are small):

1. **Contradiction 0/7** (1/15). Top-3 is 7/7: both records are retrieved and the older one ranks first in 13
   of 15 (the 14th miss is a relative's record). Nothing in the index text says which is later.
2. **Raw note 1/7** (4/14). The `[said]` sentence loses to an entry that shares the topic word (`/demo/data.json`
   for "what's my shoe size?", the moka fact for the decaf note).
3. **Table or code 1/7** (5/15). The chunk embeds as "a CSV file" or "a script"; a pointer episode or the
   stats-run episode wins ("what does stats.py count by city?" goes to the episode that ran it).
4. Then multi-hop 2/12 (7/25; top-3 with both hops 2/12), wrong owner 2/10 (10/20; the sister's dog and the
   partner's cat beat the user's cat), temporal 3/12 (5/24; top-3 9/12, the same order problem as
   contradiction).

Other findings:

- **Manual routing**: in kernel mode 15 of 43 answerable manual queries never get their section because they
  are not how-to questions; the 4 manual no-answer queries are all right (nothing reaches 0.55 for a bluetooth
  mouse, email, wifi or audio).
- **No-answer**: 12/25 right in all. The misses are a generic entry above 0.55 ("how tall am I?" -> the free
  memory tool result at 0.57) and BM25 on a path ("what's in /notes/diario.txt?" -> the `/notes` listing episode
  at 0.71).
- **Floors**: 11 of the 111 answerable top-1 hits score under their floor and would not enter the slot.

## Files

- `tools/memeval.py`: the set (corpus additions, dates, 259 queries with class, pool, gold, first hop, split)
  and the scorer. `tools/rerankeval.py` gains `embedder()`, now shared by both tools.
- `docs/spikes/memeval/results.json`: the cosine of every query against every entry (259 x 186), with the
  entry keys and the instruction, so later designs can re-rank without running the embedder again.
