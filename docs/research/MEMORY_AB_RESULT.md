# Memory recall: design A (kernel rules) and design B (write-time enrichment) on the host (T70)

Date: 2026-11-28. Task T70, step 2 of E2 (memory recall 95%+). Input for T67 (the decision) and T71.

Question: on the T69 set (259 queries, docs/research/MEMORY_EVAL.md), does design A of
docs/research/MEMORY_SYSTEMS.md (ranking rules in the kernel, no model call), then A plus design B (a context line
per entry written at index time by the stock Qwen3.5-2B, BM25 and metadata views), reach 95% held-out top-1?

Numbers are measured on a second host (12 cores, 23 GB, CPU only, PyTorch fp32, transformers 5.24) unless marked
**[E]** (estimate from the engine numbers in docs/M45_RESULT.md, docs/EMBEDDER_RESULT.md and
docs/research/DECISION_MODELS.md section 3).

## Summary

- **No-go for a kernel port of A+B as the route to 95%.** Held-out answerable top-1: baseline 51/114 (44.7%),
  A 79/114 (69.3%), B 62/114 (54.4%), A+B 81/114 (71.1%). 95% Wilson intervals: A (60.3, 77.0), A+B (62.1, 78.6).
  95% is outside both intervals by about 17 points.
- **A is worth porting on its own merits**: +28 held-out queries for 1 lost, zero model calls, microseconds per
  read [E]. Three rules carry it (held-out ablation): the manual ranked apart from the how-to gate (+15),
  the hub penalty (+10) and date-aware supersession (+8). Owner penalties add 2 to 3.
- **B adds little and breaks things**: B alone +11 held-out (18 won, 7 lost); on top of A, +2. The 2B's context
  lines are often wrong or generic (section "B in detail"), and the 299 manual question rows crowd the kernel's
  24-hit window: 18 of 234 queries get an empty answer list under B.
- **Overfitting check**: configs frozen on the tune split (commit 7d1d4fa) before the single held-out run. Tune
  vs held-out: A 81/120 (67.5%) vs 79/114 (69.3%); A+B 83/120 (69.2%) vs 81/114 (71.1%). No sign that the tune
  numbers were inflated by the search.
- **Cost per write (B)**: a memory line is 130 prompt tokens after a 1,099-token shared prefix, 15 out (medians);
  7.3 s on the host, **1.0 s in ring 0 [E]**; a file chunk 197 in, 28 out, 12.7 s host, **1.8 s ring 0 [E]**;
  then about 14 more tokens to embed per entry (0.08 s [E]). **Cost per read**: A and B add no model call; A is
  a few thousand multiply-adds over the cosines the search already computes [E].

## Results

Kernel ranking (BM25 gate, how-to manual line, top 24, `hits_split`) as in T69, plus each design's rules. Cells
are top-1 (top-3); no-answer is right abstentions. Raw per-query output: docs/spikes/memab/results.json.

### Held-out (126 queries, 114 answerable)

| class | n | baseline | A | B | A+B |
|---|---|---|---|---|---|
| para | 27 | 24 (25) | 27 (27) | 21 (22) | 22 (23) |
| exact | 6 | 4 (5) | 5 (6) | 4 (5) | 5 (6) |
| pointer | 9 | 5 (7) | 6 (7) | 5 (6) | 6 (7) |
| sibling | 11 | 6 (6) | 6 (7) | 7 (8) | 9 (10) |
| hub | 6 | 3 (4) | 4 (4) | 2 (3) | 3 (5) |
| owner | 10 | 2 (6) | 4 (6) | 5 (8) | 4 (6) |
| raw | 7 | 1 (2) | 5 (6) | 5 (6) | 6 (7) |
| table | 7 | 1 (3) | 4 (7) | 1 (2) | 5 (7) |
| temporal | 12 | 3 (9) | 9 (11) | 6 (9) | 9 (11) |
| contradiction | 7 | 0 (7) | 6 (7) | 1 (7) | 5 (7) |
| multihop | 12 | 2 (2) | 3 (5) | 5 (4) | 7 (8) |
| noanswer | 12 | 8 | 12 * | 10 | 12 * |
| **answerable** | 114 | **51 (76), 44.7%** | **79 (93), 69.3%** | **62 (80), 54.4%** | **81 (97), 71.1%** |
| 95% Wilson CI, top-1 | | 35.9-53.9 | 60.3-77.0 | 45.2-63.2 | 62.1-78.6 |

### All queries (259, 234 answerable)

| class | n | baseline | A | B | A+B |
|---|---|---|---|---|---|
| para | 55 | 49 (50) | 52 (54) | 44 (46) | 43 (47) |
| exact | 12 | 7 (10) | 10 (12) | 7 (10) | 10 (12) |
| pointer | 19 | 8 (14) | 11 (15) | 9 (13) | 12 (14) |
| sibling | 22 | 10 (10) | 9 (10) | 13 (14) | 14 (15) |
| hub | 13 | 4 (7) | 6 (9) | 3 (4) | 6 (10) |
| owner | 20 | 10 (16) | 14 (16) | 13 (17) | 13 (16) |
| raw | 14 | 4 (7) | 9 (13) | 9 (11) | 13 (14) |
| table | 15 | 5 (8) | 9 (13) | 6 (7) | 11 (13) |
| temporal | 24 | 5 (17) | 20 (22) | 9 (20) | 21 (23) |
| contradiction | 15 | 1 (14) | 11 (15) | 2 (15) | 9 (15) |
| multihop | 25 | 7 (10) | 9 (15) | 9 (11) | 12 (20) |
| noanswer | 25 | 12 | 25 * | 18 | 25 * |
| **answerable** | 234 | **110 (163), 47.0%** | **160 (194), 68.4%** | **124 (168), 53.0%** | **164 (199), 70.1%** |
| 95% Wilson CI, top-1 | | 40.7-53.4 | 62.2-74.0 | 46.6-59.3 | 63.9-75.6 |

\* Not a real gain. The hub penalty lowers every score (by 0.5x or 1.0x the entry's hub value), and the floor
grid searched on tune (0.50 to 0.70) was not rescaled, so A and A+B abstain on nearly everything: of the 114
held-out answerable top-1 hits only 25 (A) and 3 (A+B) stay above the floor, against 47 for the baseline. A port
must apply the floors to the raw cosine, or retune them on the adjusted scale. No-answer is therefore unmeasured
for A and A+B.

The baseline here is the T69 scorer on cosines re-embedded in this run: 51/114 held-out as in T69, 110/234 on all
(T69: 111). The stored T69 cosines predate the wording change of a few corpus entries in commit 1b6bc52.

### Frozen configs (docs/spikes/memab/config.json)

| design | view | rules picked on tune |
|---|---|---|
| A | CSV by rows | manual apart (margin 0.06), hub 0.5, owner 0.03 / 0.06, supersession (dd >= 0.6, within 0.2), floor 0.50 |
| B | context lines | BM25 gate kept (always-on not picked), owner from the 2B line 0.06 / 0.03, floor 0.55 |
| A+B | context lines, CSV by rows | manual apart 0.06, hub 1.0, owner 0.03 / 0.1, supersession (0.6, 0.1), BM25 gate, floor 0.55 |

Rules searched but not picked on tune (left off): file-over-pointer, sibling lexical tie-break, name hop, BM25
always fused.

Held-out ablation (frozen config minus one rule, answerable top-1): A 79 -> without manual-apart 64, without hub
69, without supersession 71, without owner 77 or 76. A+B 81 -> 64, 74, 77, 81, 80.

## Which classes fail, and why

A+B held-out, all classes under 95%. The misses, by cause:

- **para 22/27 and sibling 9/11 (A alone: para 27/27)**: B's manual question rows. 32 sections got 299 extra
  vectors; for a manual question they fill most of the 24 hits, and after `hits_split` the non-manual list can
  be empty (6 queries under A+B, 18 under B). "how do I turn off the memory slot?" and "come ricomincio una nuova
  conversazione?" return nothing: the section is first but its score, lowered by the hub penalty, misses the 0.55
  how-to floor. In the kernel the fix is to keep one hit per section before the cap.
- **owner 4/10**: the owner penalty is too weak on held-out (0.03 for an entry owned by someone the query does
  not name). "come si chiama il mio gatto?" still goes to "il gatto di Giulia si chiama Mochi"; "who's my GP?"
  to the father's doctor. Tune had owner at 10/10, so the search had no reason to raise it: the tune half of this
  class is easier than the held-out half.
- **multihop 7/12**: the second hop needs a link the rules do not make: "where does my best friend live now?"
  stops at "in Giappone vado con Davide" (a Davide record, not the move to Lisbon); "what does my partner do for
  a living?" stops at the partner fact. The name-hop rule was searched and not picked.
- **temporal 9/12, contradiction 5/7**: supersession fixes most pairs (contradiction 0 -> 5, temporal 3 -> 9),
  but the entry-entry cosine of a pair (0.52 to 0.81) overlaps with unrelated records of the same genre (pointer
  episodes 0.48 to 0.70; p99 of all memory pairs 0.60). Pairs under the threshold or further than the margin
  keep the older record ("chi e' il mio preside?", "what race am I training for?"). "che macchina ho?" goes to a
  manual section through the manual-apart rule.
- **pointer 6/9, table 5/7**: most pointer misses are memories that do not name the file as a path ("Meeting
  with the client Acme every Tuesday" for the Acme call file, "[said] ho messo le idee nel file idee.md" names it
  only by basename and still wins). File-over-pointer did not gain on tune and was left off, so "[episode 44] The
  user added two items to /home/todo.md" still beats the gold file for "what's on my todo list?". Table goes from
  1 to 5 with CSV row chunks and the other rules together (not ablated apart); "what does stats.py count by city?"
  still goes to the episode that ran it.
- **hub 3/6, raw 6/7**: generic entries still win short questions ("what's my name?" -> /etc/hostname).

## B in detail: what the 2B wrote

Prompts in `memeval.py ctx`: per memory, the 61 facts and T63 memories as a shared prefix, then the record and an
instruction with one example; per file chunk, the whole file then the chunk (Contextual Retrieval); per manual
section, 6 questions. Greedy, thinking off. All lines are in docs/spikes/memab/ctx.json. Examples:

| record | 2B line |
|---|---|
| `[fact] sister_dog: il cane di mia sorella si chiama Rocco` | Sister's dog named Rocco. |
| `[said] ricordati che lavoro come insegnante di matematica al liceo` | Teacher of mathematics at the high school. |
| `User dislikes long explanations; wants the action first, details later.` | The user Ada wants to renew her TLS certificate for the home NAS before December 15. |
| `Past tool call: ls /etc showed no nginx directory, nginx is not installed.` | This record is about the user Ada's home NAS and her current system status. |
| `/home/spese_novembre.csv` row chunks | "This CSV file lists November 2026 household expenses, including a specific entry for ..." |

Good lines name the owner (the owner view from the 2B line alone gave B owner 5/10 held-out, the best of the four
designs), but some lines describe a different fact from the prefix, and file lines are dominated by the file
description, so the chunks of one file look alike. The manual questions often copy the `Asked as:` line or the
body. This matches the risk named in MEMORY_SYSTEMS.md section 6.B: a wrong line misleads every search of that
entry.

## Cost

Write path, per entry (B). Host = PyTorch fp32 on 10 threads; ring 0 = prompt / 213 tok/s (2B batched prefill) +
out / 36 tok/s (2B decode) [E], re-embedding at 6 ms per token (embedder per-token path) [E].

| entry | n | prompt tokens (median) | out tokens | host s (median) | ring 0 s [E] | re-embed [E] |
|---|---|---|---|---|---|---|
| memory | 138 | 130 after a shared 1,099 | 15 | 7.3 | 1.0 (+5.2 once for the prefix, then a snapshot) | +14 tokens, 0.08 s |
| file chunk | 27 | 197 | 28 | 12.7 | 1.8 | +14 tokens, 0.08 s |
| manual section | 32 | 187 | 67 | 25.5 | build time, 0 at run time | 299 rows baked |

For the whole corpus: 142 s (memories) + 54 s (files) of 2B time in ring 0 [E], in idle slices.

Design A, per write [E]: hub = dot products with 54 baked probe vectors (55 k multiply-adds per entry), an owner
bitmask (7 relations) and the alias table from the entry's text, the paths it names. Per read [E]: one subtract
and one mask test per entry over the cosine array `memidx::search` already fills; supersession = up to 8
entry-entry dot products on the Q8_0 rows (8 k multiply-adds); the manual decision is one comparison. Well under
1 ms; no model call.

## How agentd would implement A

All in kernel/src/agentd.rs and memidx.rs, no new model and no new file format beyond one record field:

- **Manual apart**: in `recall` and `slot`, after `hits_split`, keep the best manual section first when
  `howto(q)` (today) or when its score is at least the best non-manual score + 0.06. One comparison.
- **Hub penalty**: a table of probe questions compiled in like the manual (`PROBES`), embedded once at build
  time like the manual vectors; per index entry one f32 `hub` (mean cosine to its 10 nearest probes) stored next
  to the row when the entry is embedded; `search` ranks by `cos - lambda * hub`. The floors (`RECALL_MIN`,
  `RECALL_MIN_BM25`) must stay on the raw cosine (see the note under the tables).
- **Owner**: a relation word list (IT/EN, also read in fact keys such as `sister_dog`) and an alias table
  (name -> relation, from memories where a relation word sits within 5 words of a capitalized name), rebuilt on
  `rebuild` and updated on each new fact or note; per entry an owner bitmask; at query time the query's owner set,
  then a penalty for an entry owned by someone the query does not name, and for an entry that lacks the owner the
  query names.
- **Supersession**: records already carry a time stamp (`mstore::Rec.ts`, file mtime in the fidx table); the
  index keeps it per entry. Among the top 8 non-manual hits within 0.2 (A) of the first, those whose row has a
  cosine >= 0.6 with the first's row form a group; the newest goes first.
- **CSV rows**: in `fidx`, a `.csv` or `.tsv` file is chunked by rows with the header repeated (a chunker
  version bump, so the index is rebuilt once).
- Not worth porting from this run: file-over-pointer, sibling tie-break, name hop (never picked on tune).

## Go / no-go

- **No-go** for the E2 goal through A+B: 71.1% held-out top-1 (CI 62.1 to 78.6), against a 95% bar. Every
  answerable class is below 95%.
- **Go for a port of A alone** as a cheap step (69.3% held-out, +28 / -1 queries, no model call), with the floor
  fix above and an owner penalty measured on more queries before it is raised.
- **B not as built here**: the 2B's lines are too often wrong to be worth 1 to 2 s of idle time per entry, and
  manual question rows must be deduplicated per section before they can be tried again.
- What the misses point to (input for T71 and T67): temporal, contradiction and multi-hop need explicit
  structure (dated, superseding lines and links between entries: design C, the wiki pages), not a better score;
  owner needs the subject stored with the fact (entity-keyed facts, B.1), not inferred at read time.

## Reproduce

```
memeval.py ctx CTX.json                 # host, about 38 min of 2B time on 10 threads, 10.6 GB peak in use
memeval.py embed VEC.json CTX.json      # host, about 4 min
memeval.py tune VEC.json CFG.json CTX.json   # no model; tune split only
memeval.py ab VEC.json CFG.json OUT.json CTX.json
```

Files in docs/spikes/memab/: ctx.json (the 2B lines and their token counts and seconds), vec.json (query,
entry-entry and probe cosines of the four views; entry-entry cosines involving manual rows set to 0, unused),
config.json (frozen on tune), results.json (per query, per design: top-1, top-3, the first 3 keys; summary,
ablation, cost).
