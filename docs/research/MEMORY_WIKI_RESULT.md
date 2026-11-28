# Design C on the host: wiki pages written by the stock model (T71)

Date: 2026-11-28. Task T71, step 3 of E2 (memory recall 95%+). Input for T67. Question: can the stock Qwen3.5-2B
write and update good wiki pages (design C of MEMORY_SYSTEMS.md, section 6) from session transcripts, and does
indexing those pages fix the temporal, contradiction and multi-hop classes of the T69 set beyond kernel rules?

Labels: numbers are measured in this run unless marked **[E]** (estimate). Host: a second host of the fleet
(16 cores, 31 GB), CPU only, PyTorch 2.15 and transformers 5.24; models in fp32 compute from the bf16 weights,
8 threads (12 for the 4B), greedy decoding as agentd's consolidation. The 4B ran in bf16: in fp32 it went over
the RAM limit and the guard stopped it at 89%. Peak during the runs: 65 C (k10temp), 80% RAM. Code:
`tools/memwiki.py` (imports `memeval.py` and `rerankeval.py`); raw files in `docs/spikes/memwiki/`.

## Summary

- **The structure works**: hand-built pages (one dated line per memory, the older record of each of the 18
  temporal and contradiction pairs marked superseded) standing in for the memory entries take the T69 held-out
  answerable top-1 from **51/114 (44.7%) to 72/114 (63.2%)**: temporal 3/12 to 9/12, contradiction 0/7 to 7/7,
  multi-hop 2/12 to 6/12. Added next to the old memory entries instead, the same pages give 53/114: the older
  record still wins. The gain comes from the superseded line leaving the index, not from the page text.
- **The stock 2B did not write pages.** Three wordings of the write prompt, 4 to 8 trial sessions each, gave no usable
  page line: it copied the format line literally (`PAGE: ringzero OS`, `OLD name n`) or used the placeholder as
  the page name (`WIKI: page = Ada Moretti`), also with the answer started for it. Stopped there (3 failures).
- **Its fallback is harmful.** With agentd's own consolidation prompt (`CONS_ASK`, which the 2B does follow) and
  the kernel filing each accepted fact on the page of its key: 207 lines proposed, 166 rejected by the kernel
  rules (133 not grounded), the 41 accepted keep 31 of 136 memories; 1 of 18 pairs superseded right and 29
  supersessions wrong (`user_name` overwritten 16 times: Davide, Giulia, Chiara, Marta...). Indexed, it lowers
  held-out top-1 to 44-49/114 next to the memories and 28-36/114 instead of them.
- **The 4B follows the format but not the job.** On the same trials it wrote `WIKI: phone = iPhone 15` / `OLD:
  phone 1` and got 3 of 3 shown supersessions right. Over the 34 sessions that hold the 18 pairs, on its own
  growing pages: 78 lines proposed, 60 rejected (59 not grounded: it restates lines it is shown), 15 of 57
  memories kept, 1 of 18 pairs superseded right, 3 wrong (`user_birthday` superseded by the library due date),
  no recall gain (51/114 added, 39-40/114 replacing).
- **Idle cost** per session: 2B wiki ask about 560 tokens in (the session itself is already in the KV cache) and
  15 to 220 out; ring 0 about 5 to 6 s per session [E] (2.4 s mean for the shorter fallback ask, 7.1 s max [E];
  about 6 s mean for the 4B [E]).
- **Go / no-go: no-go for C as a model-written wiki, on the 2B and on the 4B.** Keep what made the hand-built
  pages win, as kernel rules: a date on every index entry and the older record of the same thing out of the live
  index (design A of MEMORY_SYSTEMS.md, section 6; here section 7).

## 1. Setup

**Sessions** (`memwiki.S`): the 138 memories of the T69 corpus rewritten as the 86 sessions that would have
produced them, one per date of `memeval.DATE` (undated T63 memories go in sessions before any record that
supersedes them): user turns in the user's words (Italian or English), tool calls and results in Qwen's tool
format, short agent answers. Changes over time are separate sessions on their own dates ("ho un Motorola g54,
il numero finisce con 4411" on 2026-02-10, "ho cambiato telefono, adesso ho un iPhone 15" on 2026-11-02). Every
turn names the memory it carries (or the file a tool result shows), so each written line is traced back by word
overlap (a line belongs to the memory whose turns hold the largest share of its words, at least half). Two
sessions carry nothing worth a page (a greeting, small talk). `memwiki.py check` asserts that every memory is in
exactly one session on its date.

**Write path** (design C, section 6.C): after the session, one more kernel message in `CONS_ASK`'s shape with the
page names and the numbered live lines of the 3 pages closest to the session (embedder cosine of the session's
user words and tool results against each page). The model answers one line per new fact and one line per
numbered line that is no longer true. The kernel, never the model, edits the page:

- page name through `norm_key`, line through `norm_value` (160 bytes), then `grounded()` against the session's
  user words and tool results (every number must appear, at least half the words of 3+ letters), no duplicate of
  a live line, at most 6 lines;
- a supersession needs an accepted new line on the same page in the same answer and a line that was shown;
- the kernel stamps the session date and marks `[superseded DATE]`. Pages are `wiki/<name>.md`, the
  `/var/lib/ringzero/wiki/` layout.

**Index and score**: wiki chunks per live line (`[wiki: phone] 2026-11-02 ho cambiato telefono...`) or per page
(title plus live lines, cut like files at 1000 bytes), embedded with the stock embedder (Qwen3-Embedding-0.6B,
the kernel's instruction and pooling), then T69's kernel scorer (`memeval.scored`, BM25 gate, slot order) over
the T69 index with the chunks **added**, or with the chunks **replacing** every memory entry (manual sections and
file chunks stay). A wiki chunk counts as the memories its live lines were traced to. The embedder reproduces
T69's stored cosines (checked on two entries, within 0.002).

## 2. Hand-built pages: the upper bound of the structure

`memwiki.py oracle`: one line per memory with the user's or the tool's words, the page a careful writer would
pick (52 pages, 141 lines, 123 live, at most 9 live lines on a page), the older record of each pair superseded by
the newer one's session. Kept 136/136, superseded right 18/18 by construction. Example:

```
# mom
- 2025-10-10 mia madre vive a Ravenna
- [superseded 2026-05-03] 2026-04-02 il compleanno di mia madre e' il 12 maggio
- 2026-05-01 per il compleanno di mia madre voglio regalarle un corso di ceramica
- 2026-05-03 mi sbagliavo, il compleanno di mia madre e' il 21 maggio
```

Held-out top-1 (all 259 queries in brackets), kernel ranking:

| Class | baseline | pages added, per line | pages replace memories, per line | added, per page | replace, per page |
|---|---|---|---|---|---|
| para | 24/27 (49/55) | 24/27 (49/55) | 23/27 (47/55) | 24/27 (49/55) | 23/27 (41/55) |
| exact | 4/6 (7/12) | 5/6 (7/12) | 5/6 (8/12) | 4/6 (6/12) | 4/6 (8/12) |
| pointer | 5/9 (9/19) | 4/9 (8/19) | 5/9 (10/19) | 5/9 (10/19) | 6/9 (14/19) |
| sibling | 6/11 (10/22) | 5/11 (9/22) | 6/11 (10/22) | 6/11 (10/22) | 6/11 (10/22) |
| hub | 3/6 (4/13) | 4/6 (5/13) | 3/6 (7/13) | 4/6 (6/13) | 3/6 (7/13) |
| owner | 2/10 (10/20) | 2/10 (10/20) | 5/10 (12/20) | 2/10 (10/20) | 4/10 (11/20) |
| raw | 1/7 (4/14) | 1/7 (5/14) | 3/7 (7/14) | 1/7 (4/14) | 2/7 (6/14) |
| table | 1/7 (5/15) | 1/7 (5/15) | 1/7 (4/15) | 1/7 (5/15) | 2/7 (8/15) |
| **temporal** | 3/12 (5/24) | 4/12 (6/24) | **9/12 (19/24)** | 3/12 (5/24) | **9/12 (16/24)** |
| **contradiction** | 0/7 (1/15) | 1/7 (2/15) | 5/7 (12/15) | 0/7 (1/15) | **7/7 (13/15)** |
| **multihop** | 2/12 (7/25) | 2/12 (7/25) | 2/12 (7/25) | 3/12 (8/25) | **6/12 (14/25)** |
| noanswer (right abstentions) | 8/12 (12/25) | 8/12 (12/25) | 10/12 (19/25) | 8/12 (12/25) | 10/12 (20/25) |
| **answerable** | **51/114** (111/234) | 53/114 (113/234) | 67/114 (143/234) | 53/114 (114/234) | **72/114** (148/234) |

Reading:

- **Temporal and contradiction are fixed by removal, not by pages.** Per line and per page give the same 9/12
  temporal once the superseded line is out of the live index; added next to the old entries, nothing moves (the
  older `[fact] phone: Motorola g54` keeps winning). What the page adds on top is the date on the line, which the
  4B can read in the slot.
- **Multi-hop needs the page grouping**: 6/12 per page, 2/12 per line. A page holds both hops ("il mio migliore
  amico e' Davide" and "Davide si e' trasferito a Lisbona"). This is lenient: a page chunk counts for every live
  line on it (2.4 on average), and the model then has to read the right one.
- 63.2% is still far from 95%: raw, table and owner stay low, as they do not depend on order.

## 3. The 2B writing pages: three prompt wordings, no usable line

Trial sessions 2, 14, 48, 50, 59, 77, 82, 84 (tool results; four user facts; the new head teacher; a file shown
plus a contact; the new phone with the old phone page shown; the failed backup with the backup page shown; Marta
late; a greeting), each on top of the hand-built pages of the sessions before it (`memwiki.py probe`):

| Wording | What the 2B wrote | Usable lines |
|---|---|---|
| A: `PAGE name: one short line` / `OLD name n` (first 4 sessions, from scratch) | `PAGE: ringzero OS`, `OLD name n`, then the instruction text again | 0 |
| B: `WIKI: page = fact` / `OLD: page n`, `CONS_ASK`'s wording | `WIKI: page = Ada Moretti`, `WIKI: page = phone`; on session 48 the 17 page names, one per line | 0 (4 accepted by the rules, all with page name `page`) |
| C: B with the answer started as `WIKI:` by the kernel | the same as B | 0 |

Never an `OLD` line with a number, so no supersession at all. The rules caught most of it (the page names in
session 48 are not grounded); the 4 lines they let through were `page = Marco Bianchi`, `page = phone` (the word
is inside "iPhone"), `page = backup` and `page = marta`, which shows the limit of a lexical check: right words, wrong structure.
Raw outputs: `probe_A.txt`, `probe_B.txt`, `probe_C.txt`.

## 4. The fallback the 2B can follow: facts filed by key (variant K)

`CONS_ASK` exactly as in `kernel/src/agentd.rs` (the harness check: the 2B writes its `EPISODE` line in 85 of 86
sessions, as on the kernel), plus one sentence with the keys already saved so it can reuse one. The kernel puts
each accepted `FACT` on the page named by its key and supersedes the live line of the same key (agentd replaces a
fact of the same key; here it keeps it, dated); episodes go on the page `episodes`. Full run over the 86
sessions (`k_2b/`):

| Page quality | |
|---|---|
| FACT lines proposed | 207 |
| rejected by the kernel rules | 166: 133 not grounded, 29 over the cap of 6, 2 duplicates, 2 placeholders |
| accepted | 41 lines on 11 pages, 3 not traceable to a memory |
| memories kept | **31 of 136** (37 counting the 85 episode lines) |
| memories lost | 105 |
| pairs superseded right | **1 of 18** (the email); 17 pairs never had both records written |
| supersessions wrong | **29** |

The key list made it worse, not better: from session 6 on the 2B copies the saved keys and fills them with what
it has (`user_editor = ringzero`, `user_vcpus = 1`), and same-key supersession turns that into a chain of wrong
replacements:

```
# user_name
- [superseded 2025-12-12] 2025-11-01 Davide
- [superseded 2026-01-10] 2025-12-12 Giulia
- [superseded 2026-02-01] 2026-01-10 Davide
  ... 13 more names ...
- 2026-11-26 Marta
```

One page that came out right:

```
# email
- [superseded 2026-11-12] 2024-05-01 ada.moretti@libero.it
- 2026-11-12 ada.m@proton.me
```

Indexed, K lowers recall (held-out answerable top-1): added per line 44/114, per page 49/114; replacing the
memories 28/114 and 36/114 (temporal 1-3/12, contradiction 0-1/7, multi-hop 0-2/12). Full table in
`k_2b/report.json`.

## 5. The 4B

Same harness, wording B, Qwen3.5-4B in bf16. **Trials** (`probe_B_4b.txt`): it follows the format on all 8
sessions and writes a numbered `OLD` when the old line is shown: `WIKI: phone = iPhone 15` with `OLD: phone 1`,
`WIKI: backup = bucket full` with `OLD: backup 2`, `WIKI: marta = arrives at station at 20:30` with `OLD: marta
2`, all three right. But it also restates the pages it is shown instead of the session (session 2, about the
CPU and disk, came back as six lines about Ada's email and diet; the greeting session as six machine lines), and
it translates ("Elena Costa is the new principal" for "la nuova preside ... e' Elena Costa"), which `grounded()`
rejects.

**Run** over the 34 sessions that hold a record of the 18 pairs (`ONLY=...`, `b_4b/`; 83 s per session on the
host in bf16), each on the pages the 4B itself wrote so far:

| Page quality | |
|---|---|
| lines proposed | 78; 60 rejected (59 not grounded, 1 duplicate) |
| accepted | 18 lines on 9 pages, 1 not traceable |
| memories of those sessions kept | **15 of 57** |
| OLD lines | 45 proposed, 41 rejected (mostly "no new line on that page": it supersedes lines of pages it did not write to) |
| pairs superseded right | **1 of 18** (the dentist); 1 missed; 16 pairs never had both records written |
| supersessions wrong | **3** |

Once a few pages exist, every session starts with the shown lines repeated, and the new fact lands on whatever
page is nearby:

```
# user_partner
- 2025-09-01 Giulia
- 2026-10-15 Toyota Yaris hybrid
- 2026-10-15 Panda
# user_birthday
- [superseded 2026-11-28] 2025-09-01 March 3rd
- 2026-11-28 16 dicembre
```

(the car on the partner's page; the birthday superseded by the library book's new due date). Recall, with the
pages standing in for the memories of those 34 sessions only: 40/114 per line and 39/114 per page held-out
(temporal 0/12, contradiction 0/7); added next to the memories: 51/114, no change.

## 6. Idle cost per session

Measured on the host (a CPU shared with other work): 2B with K, fp32, 8 threads: ask 164 tokens mean
(instruction and key list), 43 out mean, 220 max; 13.8 s per session of host generation, 0.2 s to pick the
pages. 4B with B, bf16, 12 threads: ask 267 tokens mean, 54 out, 83 s per session. The wiki ask (B)
with page names and shown lines is 330 to 840 tokens, 557 mean, over the trial sessions, and grows with the page
list.

Ring 0 [E], with the session already in the KV cache (the background stream shares its prefix, so only the ask
is new prefill), using DECISION_MODELS.md section 3 and MEMORY_SYSTEMS.md section 6.C: 2B batched prefill 215
tok/s, decode about 36 tok/s, embedder 6 ms per token:

| Ask | Prefill | Decode | Embed (session query + touched pages) | Per session |
|---|---|---|---|---|
| 2B, K (measured token counts) | 0.8 s | 1.2 s mean | 0.4 s | 2.4 s mean, 7.1 s max |
| 2B, B, wiki with shown lines | 2.6 s | 1.1 to 2.2 s (40 to 80 out) | 1.6 s | 5 to 6 s |
| 4B, B (measured: ask 267 tok, 54 out mean, 220 max) | about 2x the 2B rates | | 0.4 s | about 6 s mean, 15 s max |

Both run in idle time; neither touches the read path.

## 7. Go / no-go and risks

**No-go for C as a model-written wiki with the stock 2B, and with the stock 4B.** The 2B does not produce the
format at all (section 3), and the closest thing it does produce, keyed facts, has the right page in 1 of 18
changes and 29 wrong supersessions (section 4). The 4B produces the format and supersedes right when the old line
is put in front of it, but on its own pages it keeps 15 of 57 memories and gets 1 of 18 changes (section 5). A wiki the 2B maintains would be wrong more often than the memory store it is meant to
fix, and the kernel's lexical rules cannot see a wrong page or a wrong supersession: they check words, not
structure.

**Worth it on top of A + B: the rules, not the writer.** The hand-built pages say where the gain is: (1) an index
entry the kernel knows is superseded must leave the live index (temporal 3/12 to 9/12, contradiction 0/7 to 7/7
held-out), and (2) records about the same entity ranked together help multi-hop (2/12 to 6/12, lenient). Both
can be kernel rules without a model writing: a date in every index text, and "the newer record of the same
subject wins" when two entries are near-duplicates (cosine or shared key words) inside the top hits. That belongs
in T67 as part of A, measured with `memeval.py` on the same set.

Risks, from what was seen here:

- **Confident structure errors pass the rules.** `grounded()` accepted `page = phone` because "phone" is inside
  "iPhone"; a wrong page with right words is invisible to a lexical check.
- **Invented supersessions** are the worst failure: they remove a true line from the live index. Same-key
  replacement driven by model-chosen keys produced 29 in 86 sessions.
- **Copying the prompt**: the 2B repeats lists it is shown (the key list, the page names). Any ask that shows it
  existing names invites this.
- **Two copies of every fact** (facts.log and the pages) would disagree; with a 2B writer they disagree often.
- **Upper bound is lenient for multi-hop**: a page chunk counts for all its lines; the model still has to pick.

## Files

- `tools/memwiki.py`: sessions, kernel rules (`norm_key`, `norm_value`, `grounded` ported), the write path
  (`write`, `probe`), hand-built pages (`oracle`), embedding (`embed`) and the report (page quality, cost, per
  class scores). Reproduce: `memwiki.py check`; `memwiki.py oracle OUT`; `ASK=K memwiki.py write
  Qwen/Qwen3.5-2B OUT`; `memwiki.py embed OUT`; `memwiki.py report OUT`.
- `docs/spikes/memwiki/oracle_pages/`: the hand-built pages (`wiki/*.md`, `pages.json`) and `report.json`.
- `docs/spikes/memwiki/k_2b/`: the 2B run with `CONS_ASK` (every session's prompt size, output and the kernel's
  decision per line in `runs.json`), its pages and `report.json`.
- `docs/spikes/memwiki/b_4b/`: the 4B run with wording B over the 34 pair sessions.
- `docs/spikes/memwiki/probe_*.txt`: the trial outputs of the three wiki wordings and of K on the 2B, and of B
  on the 4B (`probe_B_4b.txt`).
