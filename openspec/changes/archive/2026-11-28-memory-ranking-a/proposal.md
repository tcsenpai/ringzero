## Why
On the T69 recall set (259 queries, docs/research/MEMORY_EVAL.md) today's kernel ranking finds the right entry first
for 44.7% of the held-out answerable queries. Design A of docs/research/MEMORY_SYSTEMS.md, ranking rules with no
model call, reached 69.3% on the host with a config frozen on the tune split (docs/research/MEMORY_AB_RESULT.md).
T67 decided to port it. T71 (docs/research/MEMORY_WIKI_RESULT.md) showed where the temporal gain comes from: a
date on every entry and the older record of the same thing out of the live answer.

## What Changes
- Every index entry carries a date: a memory record's time, a file's mtime, the time it was added. The index file
  goes to version 2 (dates after the texts). A version 1 index still loads: file chunks get their file's mtime from
  the indexed-file table, every other entry the boot date; the next idle save writes version 2.
- The search ranks by the cosine minus a penalty: half the entry's hub score (mean cosine to its 10 nearest of 54
  generic probe questions, embedded with the manual's vectors at build time) plus the owner penalties (0.03 for a
  memory of someone the query does not name, 0.06 for a memory without the owner the query names). Floors keep
  reading the raw cosine.
- Supersession: of the first 8 non-manual hits, those within 0.2 of the first whose vector has a cosine of 0.6 or
  more with the first's are one group; the newest stands for the group and the older ones leave the answer (they
  stay in the index and on disk, `/search` names them).
- The best manual section goes first for any query when its score is 0.06 over the best other hit (and still for a
  how-to message at a raw cosine of 0.55), in the recall tool and the memory slot.
- `.csv` and `.tsv` files are indexed one row per entry with the header in front (chunker version 2; a version 1
  index loads and only its tables are cut again).
- `/index add [-d YYYY-MM-DD] TEXT` (date, `\n` and `\\` escapes); `/search` prints the ranked answer with score
  and date; `/var/lib/ringzero/mem/index.tsv` lists every entry with its date, written with index.bin.

## Non-goals
- File-over-pointer (a file chunk ahead of a memory that names the file): searched on tune in T70, never picked,
  and it costs 2 to 4 held-out queries on the host; left out.
- Design B (write-time context lines by the 2B) and design C (wiki pages): no-go in T70 and T71.
- Retuning any weight or threshold on the kernel: the config is the one frozen on the host.

## Impact
- Code: kernel/src/memrank.rs (new), memidx.rs, agentd.rs, fidx.rs, shell.rs, main.rs; Makefile (manvec depends on
  memrank.rs); tools/memeval.py (`ported`, `steps`, `inos`, `inos-score`).
- Disk: index.bin version 2 (migrated in place); index.tsv new. The baked manual vectors' blob goes to version 2.
- Docs: docs/MANUAL.md, INFO.md, docs/ARCHITECTURE.md, docs/M_MEMA_RESULT.md.
