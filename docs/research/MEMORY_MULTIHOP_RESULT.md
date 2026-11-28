# Memory recall: multi-hop candidate budget, two-step retrieval and entity-key joins on top of design A (T73)

Date: 2026-11-28. Task T73, step (b) of the T67 decision for E2 (memory recall 95%+). Input: design A's frozen
config from T70 (docs/research/MEMORY_AB_RESULT.md, `docs/spikes/memab/config.json`), 69.3% held-out top-1,
multi-hop 3/12.

Question: without any model writing, does a larger candidate budget, a second retrieval step from A's first
hit, or a join on the entities of that hit lift multi-hop, and at what read cost in ring 0?

Numbers are measured unless marked **[E]** (estimates from the engine numbers in docs/research/DECISION_MODELS.md
section 3: the embedder at about 6 ms per token on its per-token path).

## Summary

- **No-go for the kernel.** No variant changes the held-out answerable total: A 79/114 (69.3%, Wilson 60.3 to
  77.0), the best variant (entity join gated on a definition record, `join_def`) 79/114. On multi-hop it gives
  4/12 against 3/12 (Wilson 13.8 to 60.9 against 8.9 to 53.2), won by one query and paid for by a paraphrase
  query, so +1 / -1. On all 259 queries: 162/234 against 160/234.
- **The candidate budget is not the problem.** A larger window (48 or every entry, per pool) changes no top-1 on
  tune or held-out. In 7 of the 9 held-out multi-hop misses the answer is already among the first 24 cosine
  ranks (ranks 1 to 9); the failure is choosing it over the first hop, not finding it.
- **Second steps without a trigger hurt.** Every rule that may fire on any query (entity join, vector-space
  second step, re-embedded query) either never fires at its tuned setting or loses: the re-embedded query with
  the first hit's names, at its best firing setting, fires on 28 held-out queries, wins 3 and loses 12
  (temporal 9 -> 5, contradiction 6 -> 1: the second step undoes supersession).
- **Read cost [E]**: the joins are lexical, microseconds; the vector-space second step is one row of
  entry-entry products (well under 1 ms at this index size); a re-embedded second query costs a second embedder
  pass, about 0.2 to 0.33 s in ring 0 (35 to 55 tokens). None is worth it at these gains.
- **What would move multi-hop**: the link has to be stored, not guessed at read time: entity-keyed facts
  (design B.1 in MEMORY_SYSTEMS.md: the subject and the related entity as fields) or the embedder fine-tune of
  T74, whose data includes the second-step query form.

## Method

`tools/memhop.py`, on top of `memeval.rank` with A's frozen config (rows view, `docs/spikes/memab/vec.json`).
Each variant runs after A's ranking and may promote one entry to the top, keeping A's first hit second (so a
promoted second hop also satisfies the multi-hop top-3 rule, which needs both hops):

| variant | what it does | knobs searched on tune |
|---|---|---|
| budget | A with a hit window of 48 or every entry instead of 24 | window |
| join | entity-key join: names in A's first hit (or first two) not in the query, and the alias names of the relations the query names (my sister -> Chiara, from A's alias table); among the top B per pool, entries holding one of them score adjusted cosine + bonus, the best is promoted if it beats the first hit | entities (hit, alias, both), sources 1 or 2, B (8, 24, 48, all), bonus (0 to 0.3) |
| prf | two-step in vector space: adjusted cosine + w x cos(first hit row, entry row) (the stored entry-entry cosines), promoted over a threshold | B, w (0.1 to 0.5), threshold (0 to 0.3) |
| req_names | two-step retrieval: the query re-embedded with the first hit's names appended, fused as adjusted cosine + w x second cosine, promoted when it beats the first hit's fused score minus a margin | B, w (0.25 to 2), margin (0 to 0.15) |
| req_key | the same with the first hit's whole index text appended | as req_names |
| *_def | each of the above, firing only when A's first hit is a definition record (a name tied to a role: "il mio migliore amico e' Davide", "is called Giulia", "Marta e' la mia collega") and the query does not ask for the name itself; `join_def` may also join on the definition's other words (the role, when the query holds the name) | as above |

Every grid starts with the rule off, so a variant can always tie A; the objective is answerable top-1, then
top-3, on the tune split only (120 answerable queries). The frozen configs are in `docs/spikes/memhop/config.json`;
for a variant whose best point is "off", the best point with the rule on is kept as `NAME@on` and scored too.
Held-out was scored once, after tuning.

Second-step vectors: `memhop.py embed` on a GPU host (fp32, the kernel format). Its own first-step cosines
match the stored T70 ones within 0.0001, with the same top-1 entry for 258 of 259 queries (one tie), so the two
vector sets are the same space.

Disclosure: before tuning, the multi-hop queries of both splits were printed with their first-hop and answer
records (`gold@rank`, the record texts) to see the shape of the problem; the definition trigger was written from
those record formats. The trigger's knobs were picked on tune only and held-out was scored once, but the trigger
itself was not designed blind. Read the `join_def` held-out gain (+1 multi-hop) as an upper bound.

## Results

Cells are top-1 (top-3). A = T70's design A, reproduced exactly (79/114 held-out, 160/234 all). The variants
not shown (budget, join, prf, req_names, req_key and the gated prf and req forms at their tuned settings, plus
budget@on, join@on, prf@on, req_names_def@on, req_key_def@on) either never fire or fire without changing a
top-1, and equal A in every cell; the full table, 18 configs, is in `docs/spikes/memhop/ab.txt`.

### Held-out (126 queries, 114 answerable)

| class | n | A | join_def (best) | prf_def@on | req_names@on | req_key@on |
|---|---|---|---|---|---|---|
| para | 27 | 27 (27) | 26 (27) | 26 (27) | 27 (27) | 27 (27) |
| exact | 6 | 5 (6) | 5 (6) | 5 (6) | 5 (6) | 5 (6) |
| pointer | 9 | 6 (7) | 6 (7) | 6 (7) | 6 (7) | 6 (7) |
| sibling | 11 | 6 (7) | 6 (7) | 6 (7) | 7 (7) | 6 (7) |
| hub | 6 | 4 (4) | 4 (4) | 4 (4) | 4 (4) | 4 (4) |
| owner | 10 | 4 (6) | 4 (6) | 4 (6) | 4 (6) | 4 (6) |
| raw | 7 | 5 (6) | 5 (6) | 5 (6) | 3 (6) | 5 (6) |
| table | 7 | 4 (7) | 4 (7) | 4 (7) | 5 (7) | 4 (7) |
| temporal | 12 | 9 (11) | 9 (11) | 9 (11) | 5 (11) | 9 (11) |
| contradiction | 7 | 6 (7) | 6 (7) | 6 (7) | 1 (7) | 5 (7) |
| **multihop** | 12 | **3 (5)** | **4 (6)** | 4 (5) | 3 (5) | 3 (5) |
| noanswer | 12 | 12 * | 12 * | 12 * | 12 * | 12 * |
| **answerable** | 114 | **79 (93), 69.3%** | **79 (94), 69.3%** | 79 (93) | 70 (93), 61.4% | 78 (93) |
| 95% Wilson CI, answerable | | 60.3-77.0 | 60.3-77.0 | 60.3-77.0 | 52.2-69.8 | 59.4-76.2 |
| 95% Wilson CI, multihop | | 8.9-53.2 | 13.8-60.9 | 13.8-60.9 | 8.9-53.2 | 8.9-53.2 |
| fired / won / lost vs A | | | 5 / 1 / 1 | 4 / 1 / 1 | 28 / 3 / 12 | 1 / 0 / 1 |

### All queries (259, 234 answerable)

| class | n | A | join_def (best) | prf_def@on | req_names@on | req_key@on |
|---|---|---|---|---|---|---|
| para | 55 | 52 (54) | 51 (54) | 51 (54) | 53 (54) | 52 (54) |
| exact | 12 | 10 (12) | 10 (12) | 10 (12) | 10 (12) | 10 (12) |
| pointer | 19 | 11 (15) | 11 (15) | 11 (15) | 10 (15) | 11 (15) |
| sibling | 22 | 9 (10) | 9 (10) | 9 (10) | 10 (10) | 9 (10) |
| hub | 13 | 6 (9) | 6 (9) | 6 (9) | 6 (9) | 6 (9) |
| owner | 20 | 14 (16) | 14 (16) | 14 (16) | 14 (16) | 14 (16) |
| raw | 14 | 9 (13) | 9 (13) | 9 (13) | 6 (13) | 9 (13) |
| table | 15 | 9 (13) | 9 (13) | 9 (13) | 10 (13) | 9 (13) |
| temporal | 24 | 20 (22) | 20 (22) | 20 (22) | 10 (22) | 19 (22) |
| contradiction | 15 | 11 (15) | 11 (15) | 11 (15) | 2 (15) | 10 (15) |
| **multihop** | 25 | **9 (15)** | **12 (17)** | 10 (15) | 9 (15) | 9 (15) |
| noanswer | 25 | 25 * | 25 * | 25 * | 25 * | 25 * |
| **answerable** | 234 | **160 (194), 68.4%** | **162 (196), 69.2%** | 160 (194) | 140 (194), 59.8% | 158 (194) |
| 95% Wilson CI, answerable | | 62.2-74.0 | 63.0-74.8 | 62.2-74.0 | 53.4-65.9 | 61.3-73.2 |
| 95% Wilson CI, multihop | | 20.2-55.5 | 30.0-66.5 | 23.4-59.3 | 20.2-55.5 | 20.2-55.5 |

\* As in T70: A's no-answer is unmeasured, because the hub penalty lowers every score under floors that were not
rescaled (see MEMORY_AB_RESULT.md). No variant changes it.

Tune (120 answerable): A 81, `join_def` 83 (multi-hop 6 -> 8 of 13); every other variant's best point is "off".
The best point with the rule on, per ungated variant (tune): join 81 (+1 -1), prf 80 (+0 -1), req_names 70
(+2 -13), req_key 80 (+0 -1); the window never changes a top-1.

### What `join_def` did on held-out

It fired on 5 of 126 queries: "where does my best friend live now?" goes from the definition (`il mio migliore
amico e' Davide`) to "Davide si e' trasferito a Lisbona" (won); "when is our anniversary?" goes from the
partner record, which holds the anniversary, to another Giulia record (lost); "when is my girlfriend's
birthday?" moves to a wrong record (still a miss); two no-answer queries move between entries without changing
the abstention. The name-question test has a bug found after the held-out run: patterns that end in an
apostrophe ("chi e'") never match, so "chi e' Federico?" fired; it changed no score here and was left as
measured.

## Why multi-hop stays low

Held-out multi-hop misses under A, with the cosine rank of the answer: the shopping list the agent showed (8),
the best friend's city (118), his birthday (5), the girlfriend's birthday (3), the Acme contact's email (9),
the GP's prescription (6), the partner's job (28), the language Ada wants (1), the staff meeting set by the new
head teacher (5). The links are of four kinds, and only the first is a name:

1. A name in the first hop (Davide, Giulia, Elena Costa): a join can follow it, but the same join fires on
   ordinary questions whose first hit already answers (the cat's vet, the printer's toner, the build in Rust),
   and the trigger that separates the two cases is a guess about the query.
2. A role in the first hop ("medico di base", "referente di Acme"): the answer shares the role, not a name.
3. A chain of two aliases (Rocco -> my sister -> Chiara -> Parma).
4. A topic or a file (Kyoto -> the Japan trip, the shopping list -> a recipe), where entity rules have nothing
   to join.

## Read cost in ring 0 [E]

| variant | extra work per read | ring 0 [E] |
|---|---|---|
| budget | none: `memidx::search` already scores every entry; a longer sort | under 0.01 ms |
| join, join_def | a regex and a word set over 1 or 2 records, a set test over the top 8 per pool (an inverted name table for "all") | microseconds |
| prf | one row of entry-entry products on the Q8_0 rows: N x 1024 multiply-adds (0.2 M at 197 entries, 10 M at 10 k) | under 1 ms at this size, a few ms at 10 k entries |
| req_names, req_key | a second embedder pass: median 35 tokens (query plus names) or 55 (plus the record), against 34 for the query alone; then a second cosine pass | 0.21 s or 0.33 s, doubling the query's embedding time |

## Go / no-go

- **No-go** for porting any of these rules to agentd: none changes the held-out total, the only positive one
  (`join_def`) is +1 / -1 on held-out, and its trigger was written after looking at the records.
- **No-go** for a larger hit window: it changes nothing.
- Multi-hop needs the link stored at write time (entity-keyed facts, B.1) or learned by the embedder (T74). The
  T74 data includes the second-step query form (`question + name`), so a fine-tuned model can be scored with
  `memhop.py` again at no extra cost: the req variants then measure whether it learned the hop.

## Reproduce

```
memhop.py embed docs/spikes/memhop/vec2.json      # GPU host, 49 s, 4.4 GB of host RAM peak
memhop.py tune docs/spikes/memhop/vec2.json docs/spikes/memhop/config.json     # no model, tune split only
memhop.py ab docs/spikes/memhop/vec2.json docs/spikes/memhop/config.json docs/spikes/memhop/results.json
```

Files in docs/spikes/memhop/: vec2.json (second-step queries, their token counts and cosines, the agreement
check), config.json (frozen on tune), ab.txt (the full printout), results.json (per query, per variant: top-1,
top-3, whether the rule fired).
