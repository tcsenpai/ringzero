# Memory ranking design A in the kernel (T72)

Date: 2026-11-28. Task T72 of E2 (memory recall 95%+), after the decision in T67. Design A of
docs/research/MEMORY_SYSTEMS.md, frozen on the host in T70 (docs/research/MEMORY_AB_RESULT.md,
docs/spikes/memab/config.json "A"), now runs in agentd. No model call on a read. OpenSpec change
`2026-11-28-memory-ranking-a` (archived; openspec/specs/memory, manual, shell-commands, unix-tree).

Measured on the build box (QEMU/KVM, 8 vCPU, 6 GiB guest for the 2B q8, 8 GiB for the 4B q4).

## Summary

- **In the OS, the T69 set finds the right entry first for 77 of 114 held-out answerable queries (67.5%, 95%
  Wilson CI 58.5 to 75.4)**, against 79/114 (69.3%) for the same frozen config on the host and 51/114 (44.7%) for
  the kernel ranking before. All 234 answerable: 155 (66.2%). No-answer: 8/12 held-out right, the same as before
  (the floors read the raw cosine).
- The 2-query gap (4 held-out queries lost, 2 won, list below) is mostly the embedder: the kernel embeds with the
  Q8_0 export and stores Q8_0 rows, the host used fp32 PyTorch. Kernel and host cosines differ by 0.004 on average
  (p95 0.010, max 0.018, 1248 hits compared), enough to flip a rule at its threshold (3 of the 4 lost). The fourth
  is a tie between two entries the set leaves undated: the kernel dates them when typed, the host by memory order.
- Ported: a date on every index entry (index.bin version 2, migrated from version 1), the hub penalty, the owner
  penalties, supersession, the manual section first for any query, CSV and TSV files by rows, and the floors on
  the raw cosine. Not ported: file-over-pointer (never picked on tune in T70; on the host it loses 2 to 4 held-out
  queries at every margin tried).
- Regressions: boot tests on the 2B and the 4B (BIOS, UEFI) pass, rztest 24/24, the upgrade from a 0.1.1 disk
  migrates the index in place; tooleval on the 4B q4 equals main (spike 24/30, in-OS 26/32).

## What runs where

| rule (frozen value) | code |
|---|---|
| date per entry, saved (index v2) | memidx.rs `TS`, `store`, `store_row`, `save`, `load`; agentd.rs `apply`, `said_begin`, `file_ts`, `embed_slice` |
| v1 index migration | agentd.rs `migrate` (file chunk: its file's mtime; fact: its record's time; other: the boot date) |
| hub penalty (0.5 x mean cos of the 10 nearest of 54 probes) | memrank.rs `PROBES`, `hub_of`, `penalties`; memidx.rs `set_probes`, `hubs`; probe rows baked with the manual vectors (agentd.rs `baked`, `man_vectors`) |
| owner (0.03 other owner, 0.06 missing owner) | memrank.rs `REL`, `aliases`, `owner`, `penalties` |
| ranking on cos - penalty, floors on cos | memidx.rs `search` (`Hit.cos` raw, `Hit.score` ranked) |
| supersession (top 8, within 0.2, vector cos >= 0.6) | agentd.rs `ranked`; memidx.rs `dd` |
| manual first (margin 0.06, or how-to at cos 0.55) | agentd.rs `ranked`, `recall`, `slot` |
| CSV / TSV by rows | fidx.rs `table_file`, `next_line`, `next_chunk` (chunker version 2) |
| state as a file | `/var/lib/ringzero/mem/index.tsv`: id, date, source, text head, written with every save |

Two choices the host did not have to make:

- **Every entry is dated.** On the host most T63 memories and files had no date and never joined a supersession
  group. In the kernel everything has one. In the in-OS run an entry without a date in the set is added without
  `-d`, so it gets the time it was typed, after every date of the set. `memeval.py ported` scores the host with the
  same rule: still 79/114 held-out.
- **Chunks of one file never supersede each other.** A CSV's rows share a date and look alike (vector cosine over
  0.6), so the first in-OS upgrade run dropped the right row of `/home/spese.csv` for "quanto ho speso per la cena
  fuori?" in favour of the next row. Fixed before the numbers below; on the host it changes nothing (79/114).

## In-OS against the host, per class

Held-out top-1 (top-3); no-answer = right abstentions. Host columns from `memeval.py ported`; in-OS from
`memeval.py inos` (a no-disk boot of the 2B ISO: the manual from the kernel's vectors, the other 163 entries of the
rows view typed with `/index add [-d DATE]`, then `/search` per query; per-query output in
docs/spikes/mema/inos.json, transcript docs/spikes/mema/inos-transcript.log).

| class | n | kernel before (T70 baseline) | host, frozen A | host, A as ported | **in the OS** |
|---|---|---|---|---|---|
| para | 27 | 24 (25) | 27 (27) | 27 (27) | 27 (27) |
| exact | 6 | 4 (5) | 5 (6) | 6 (6) | 5 (5) |
| pointer | 9 | 5 (7) | 6 (7) | 7 (9) | 6 (8) |
| sibling | 11 | 6 (6) | 6 (7) | 6 (7) | 6 (7) |
| hub | 6 | 3 (4) | 4 (4) | 3 (3) | 3 (3) |
| owner | 10 | 2 (6) | 4 (6) | 4 (6) | 4 (6) |
| raw | 7 | 1 (2) | 5 (6) | 5 (6) | 5 (6) |
| table | 7 | 1 (3) | 4 (7) | 4 (5) | 5 (6) |
| temporal | 12 | 3 (9) | 9 (11) | 8 (10) | 8 (10) |
| contradiction | 7 | 0 (7) | 6 (7) | 6 (7) | 5 (7) |
| multihop | 12 | 2 (2) | 3 (5) | 3 (5) | 3 (4) |
| noanswer | 12 | 8 | 8 | 8 | 8 |
| **answerable** | 114 | **51 (76), 44.7%** | **79 (93), 69.3%** | **79 (91), 69.3%** | **77 (89), 67.5%** |

"host, frozen A" is MEMORY_AB_RESULT's A column with the no-answer floor on the raw cosine (8/12; T70's 12/12 was
the floor on the lowered score, which abstains on nearly everything). "A as ported" adds the kernel's semantics:
every entry dated, superseded records out of the answer (top-3 changes, top-1 does not), the how-to floor on the
raw cosine.

The weak classes stay weak: multihop 3/12, hub 3/6, owner 4/10, sibling 6/11. They do not depend on order (T70,
T71): a second hop, the subject stored with a fact and a better embedder are the next steps (T67).

### The 2-query gap

The held-out queries where the kernel and the host (A as ported) disagree on top-1:

| query | class | host | kernel | why |
|---|---|---|---|---|
| when is my mother's birthday? | contradiction | right | wrong | the two records' vector cosine is 0.601 on the host; with the kernel's rows they do not group and the older one stays first |
| what did Acme ask for in the last call? | pointer | right | wrong | a generic newer episode ("the user asked a few questions") joins the first hit's group in the kernel and stands for it |
| HL-L2350DW | exact | right | wrong | the memory and the printer file are near-same, both undated in the set; typed after the memories, the file is newer and stands for both (the host breaks that tie by memory order) |
| when did I eat out this month? | table | right | wrong | a memory scores 0.004 over the right CSV row in the kernel |
| what does stats.py count by city? | table | wrong | right | |
| dove scrive il report lo script python? | table | wrong | right | |

Tune split, for comparison: 4 lost, 5 won.

## Disk format and migration

- index.bin version 2 = version 1 plus each entry's date (u64 Unix seconds) after the texts, in the checksum.
  Chunker version 2 (tables by rows). The baked manual blob goes to version 2 (32 sections, then the 54 probes:
  93596 bytes); `/man check`: 86/86 rows byte-identical to a fresh embedding.
- Upgrade from a 0.1.1 disk (a copy of the release `disk.img`). On the 0.1.1 2B ISO: two `/remember` facts, a
  note and a CSV file written, the index saved in idle (index.bin version 1, chunker 1, 42 entries). First boot of
  this kernel: `index: loaded 42 entries ... no re-embedding`, `index: version 1 had no dates: 8 file chunks dated
  by their file's mtime, 2 facts by their record, 32 other entries by the boot date`, `chunker 1 -> 2:
  /demo/people.csv cut again by rows` and the same for `/home/spese.csv`; 12.8 s after the kernel starts both tables are
  indexed by rows (5 and 2 entries) and the index is saved (47 entries, version 2). `/facts` and the note are as
  before; `/search quanto ho speso per la cena fuori?` gives the right row first (cos 0.717). `index.tsv` lists
  the 47 entries with their dates. Second boot: `loaded 47 entries`, no migration line; after `/remember phone
  iPhone 15` on the first boot, `/search che telefono ho?` gives `phone: iPhone 15` first.

## Regression checks

All on the final kernel, one VM at a time under the shared lock, each on its own copy of the 0.1.1 release disk.

| check | result |
|---|---|
| 2B q8, BIOS and UEFI | `SELFTEST PASS` (golden ids), every self test PASS (memrank's owner test runs in agentd's), `hub: 54 probe questions from the vectors in the kernel`, rztest PASS 24/24, `/man check` 86/86 rows byte-identical |
| 4B q4, BIOS and UEFI | every self test PASS, `SELFTEST n/a` (the golden is the 2B's, as in 0.1.1), rztest PASS 24/24, `/man check` 86/86 |
| upgrade from 0.1.1 | above: the index migrates in place, nothing re-embedded but the two tables |
| tooleval, 4B q4 (M45 protocol: 8G, `SLOT=200`, constrain on, confidence on, fresh disk, 120 s idle) | **equal to main**: spike 24/30 correct, 22 first call right, 9 args exact; in-OS 26/32 correct, 24 first call right, info 5/8; malformed 0, schema errors 0; calls 39 + 38 (0.1.1: 42 + 36), grammar lines not valid 0 (0.1.1: 1); decode of the call rounds 38.0 / 38.5 tok/s (0.1.1: 20.4 / 19.2 on a busier box). docs/spikes/mema/tooleval-4b-q4.json |
| `openspec validate --specs --strict`, `tools/surfaces_check.sh` | pass |

Cost: on 195 entries the search after the query embedding (about 200 ms, unchanged) takes 0.34 to 0.94 ms,
median 0.53 ms, over the 259 queries (penalties, owner masks and hub scores of new entries included; the
supersession and manual rules run after it, in agentd).

## Reproduce

```
memeval.py ported docs/spikes/memab/vec.json docs/spikes/memab/config.json docs/spikes/memab/ctx.json   # host, no model
MEM=6G memeval.py inos build/ringzero.iso OUTDIR      # build box, under the VM lock; about 3 min
memeval.py inos-score OUTDIR/timed.log [OUT.json]
```
