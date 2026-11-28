# Reranker over the embedder's recall (T63 spike)

Date: 2026-11-28. Question: does Qwen3-Reranker-0.6B, reranking the top 20 of Qwen3-Embedding-0.6B, put the
right memory, manual section or file chunk first more often than the embedder alone? T49
(docs/research/DECISION_MODELS.md) only tested the reranker as a tool router. Go criterion, set by the user:
+10 points top-1 on the held-out split at 0.5 s or less per query, estimated in ring 0.

Labels: numbers are measured on the host unless marked **[E]** (estimate).

## Verdict

**No-go.** On the held-out split the reranker adds 0 points of top-1 at the configuration picked on the tune
split (82.6% for both), and +2.2 points at the best configuration of all eight. The fast enough configurations
(3 candidates, about 0.4 to 0.5 s [E]) also add 0 or lose 2 points. The gain is real on top-3 (93.5% to 97.8%)
and on the tune half (+21 points), but that half has the harder queries by chance and was used to pick K, so it
does not count. No kernel work.

## Setup

- Tool: `tools/rerankeval.py` (`check` validates the data, `run OUT.json` runs the models, `report OUT.json`
  prints every number below). Raw results: `docs/spikes/rerank/results.json` (per query: the embedder's top 20
  with cosine and both reranker probabilities, token counts, host times).
- Index: 101 entries in one pool, as agentd keeps them in one memory index:
  - 55 memories: the 40 of tools/retrieval.py plus 15 new ones in agentd's formats (`[fact] k: v`,
    `[episode N] ...`, `[said] ...`), several of them near-misses of an older one (a sister's dog next to the
    user's cat, the doctor's name next to the doctor's hours, a ringzero disk check next to a `/home` one).
  - 32 manual sections, built from docs/MANUAL.md exactly as `agentd::manual()` does
    (`[manual: TITLE] body`, `Asked as:` line included).
  - 14 file chunks (one per file, all under 1000 bytes): 9 synthetic user files (`/notes`, `/home`, `/etc`) and the real demo files of
    the disk image (tools/demo as `/demo`), cut and prefixed as `fidx` does (`[PATH] text`, 1000 bytes).
- Queries: 93, Italian and English, written as a user types them: 45 for memories (the 20 of retrieval.py plus
  25 new), 29 for manual sections, 19 for files. By kind: 59 paraphrases, 14 exact tokens (`HL-L2350DW`,
  `/agentd log`, `anna.rossi@example.org`), 20 hard cases with a near-miss entry in the index, including the
  T49 error cases ("what's in /notes?", "come si chiama il mio gatto?", "come si chiama il mio medico?"). Gold
  is a set (any chunk of a gold file counts). Split: even query = tune, odd = held-out (47 / 46).
- Embedder: as kernel/src/embed.rs: plain BPE, `<|endoftext|>` appended, queries
  `Instruct: {TASK}\nQuery:{q}` with the kernel's TASK, documents bare, last-token pooling, 1024 dims, cosine.
  fp32 PyTorch, not the kernel's Q8_0, and no BM25 fusion (the kernel's RRF gate for identifier-like queries is
  not modelled).
- Reranker: the model card's prompt (system line, `<Instruct>: <Query>: <Document>:`, empty think block),
  P(yes) from the "yes" and "no" logits at the last token. Instruction: the kernel's embedder TASK, verbatim,
  not tuned. Two variants: the whole document (`full`) and the document cut to its first 96 tokens (`cap`). The
  top K of the embedder (K = 3, 5, 10, 20) are reordered by P(yes); the rest keep the cosine order.
- Host: the build box, AMD Ryzen AI 9 HX 370, CPU only, 8 threads, PyTorch 2.11, transformers 5.5.4, fp32. One
  (query, document) pair per forward, no padding.

## Numbers

Top-1, held-out (46 queries):

| pool | n | embedder top-1 | embedder top-3 | rerank cap K=5 (picked on tune) | rerank cap K=3 | rerank full K=3 | rerank full K=20 | best of the 8 (cap K=10) |
|---|---|---|---|---|---|---|---|---|
| memory | 22 | 19 | 22 | 20 | 21 | 18 | 17 | 19 |
| manual | 15 | 12 | 14 | 12 | 11 | 13 | 14 | 12 |
| files | 9 | 7 | 7 | 6 | 6 | 5 | 7 | 8 |
| **all** | 46 | **38 (82.6%)** | 43 (93.5%) | **38 (82.6%)** | 38 (82.6%) | 36 (78.3%) | 38 (82.6%) | 39 (84.8%) |

Top-3 after reranking, held-out: 43 (K=3), 44 to 45 (K=5 and more), against 43 for the embedder.

Tune split (47): embedder 28 (59.6%), rerank cap K=5 38 (80.9%). All 93: embedder 66 (71.0%) top-1 and 83
(89.2%) top-3; rerank cap K=5 76 (81.7%) and 87 (93.5%); full K=20 74 (79.6%) and 87. By kind (all 93,
embedder vs full K=20): paraphrases 43 vs 46 of 59, exact tokens 11 vs 13 of 14, hard cases 12 vs 15 of 20.
Two gold entries were outside the embedder's top 20 ("che lavoro faccio?", "ryokan Kyoto"): no reranker over
its top 20 can fix them.

How agentd uses the hits changes little: recall keeps the manual apart from the other entries
(`hits_split`). Ranked that way (memory and file queries over the non-manual entries only, manual queries over
the manual sections only), held-out top-1 goes from 28 to 28-29 of 31 and from 12 to 12-14 of 15, a post-hoc
reading on the held-out split, not a basis for go.

What the reranker gets wrong on held-out (full K=20): it moves a manual section above the fact that answers
the question ("qual e' il mio colore preferito?" to "Saved facts", "in che citta' vivo?" to "What the memory
system decided", "what CPU does this machine have?" to "CPU, cores, memory and uptime", "what's the machine's
hostname?" to the /sys and /etc section instead of `/etc/hostname`). What it fixes: "perche' mi chiede Run it
y/N?" (to "Confidence of tool calls"), "dove tiene ringzero i suoi dati?", "who lives in Milano?" (to
`/demo/people.csv`), `notes/todo.md`.

## Latency

Host (measured, median over 93 queries): an embedder query 133 ms (34 tokens). The reranker over all 20 pairs
10.7 s (`full`) and 9.0 s (`cap`), about 0.5 s per pair in fp32 PyTorch.

Ring 0 [E]: the reranker prompt has the query before the document, so the shared part (system line,
instruction, query: 77 tokens median) can be prefilled once and each candidate continues from a snapshot of it,
as snap.rs does for the 4B. Cost per query = 77 + the candidates' document and suffix tokens. Two rates from
DECISION_MODELS.md section 3: the embedder's per-token path (about 6 ms per token, docs/EMBEDDER_RESULT.md) and
batched prefill at 700 to 800 tok/s (midpoint 750), which the embedder does not use yet.

| variant | tokens per query (median) | per-token path [E] | batched [E] | batched p90 [E] |
|---|---|---|---|---|
| full K=3 | 363 | 2.18 s | 0.48 s | 0.73 s |
| full K=5 | 607 | 3.64 s | 0.81 s | 1.10 s |
| full K=20 | 2,463 | 14.8 s | 3.28 s | 3.64 s |
| cap K=3 | 285 | 1.71 s | 0.38 s | 0.52 s |
| cap K=5 | 450 | 2.70 s | 0.60 s | 0.79 s |
| cap K=20 | 1,713 | 10.3 s | 2.28 s | 2.56 s |

Only K=3 fits 0.5 s, and only with batched prefill on the 0.6B, which does not exist yet. With today's
per-token path no configuration is under 1.7 s.

## T60 (second opinion on low-margin 4B calls): skipped

T60 needs the 4B's per-task first call and T28 margin on the 240 confeval tasks, and those were not kept by the
T28 run (DECISION_MODELS.md section 5); none exist in the repo or on the build box. Getting them means building
the 4B image and running confeval in an 8 GB guest (about 11 minutes, more with boot and build), which this
host-only spike does not include, and the build box was short on RAM during this run (another job's 8 GB
guest was up and MemAvailable fell to 4 to 5 GB). T60 stays open as written, independent of this verdict.

## What would change the verdict

- A larger and balanced query set: with 46 held-out queries one query is 2.2 points, and the two halves
  differ by 23 points at the embedder alone. The all-93 reading (+10.7 points) is in the go range, but it
  includes the split used to pick K.
- An instruction that tells the reranker a stored fact beats a manual section about the same topic. That is
  tuning, to be done on a fresh tune set and checked on a fresh held-out set.
- Batched prefill for the 0.6B models: without it the cheapest useful variant costs about 1.7 s [E].

## Run notes

- The first attempt scored 20 pairs in one padded batch and asked for full-vocabulary logits at every position
  (about 10 GB); MemAvailable fell to 5.5 GB and the run was stopped by hand. The tool now asks for the last
  position only and runs one pair per forward. It frees the embedder before loading the reranker and saves
  partial results so a stopped run resumes.
- The build box is shared. The tool requires 8 GB free to start and stops before the box passes 90% (4 GB
  free) during the run. One run stopped on the 8 GB check while another job's guest was up and resumed from
  its partial file. The numbers are from complete results for all 93 queries.
