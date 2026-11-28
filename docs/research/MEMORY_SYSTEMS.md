# Agent memory systems: survey and candidate designs for agentd (T66)

Date: 2026-11-28. Task T66, input for T67 (the decision). Question: which agent-memory approaches exist, what
agentd already does, where its recall misses today, and which design could take recall top-1 from 82.6% (T63,
held-out, embedder alone) to 95% or more on a harder set.

Labels: **[V]** verified in a primary source fetched in this run (URL and date given), **[M]** measured by us
(this run or an earlier ringzero result, file named), **[E]** estimate.

Contents: 1. Summary. 2. Survey table. 3. Notes per approach. 4. What agentd already does. 5. Where recall misses
today. 6. Candidate designs. 7. Extended eval set. 8. Recommendation (input for T67, not a decision). 9. Sources.

## 1. Summary

- 23 approaches and 2 benchmarks surveyed from their primary sources (section 2). Almost every reported number
  comes from GPT-4o-mini, GPT-4.1-mini or 20B to 70B backbones; only A-Mem, SimpleMem and Memory-R1 report
  results at 1 to 3B [V]. Numbers across papers are not comparable (different judges, metrics and LLMs; the Mem0
  and Zep papers dispute each other's LoCoMo runs) [V].
- What transfers to ring 0 with a 0.6B embedder and a 2B or 4B model is the work done at **write time** with
  no model on the read path: Contextual Retrieval's context line per chunk (-35% to -49% retrieval failures
  without a reranker) [V], LongMemEval's fact-augmented keys (+9.4% recall) [V], SimpleMem's dense + BM25 +
  metadata views on the same Qwen3-Embedding-0.6B we run [V], Zep's dated facts that invalidate older ones [V].
  Designs that need the model to drive memory through many tool calls (MemGPT/Letta, MIRIX) degrade sharply on
  small models [V].

- Our misses are not paraphrase failures of the embedder in general. Re-read with the kernel's own pool split
  (manual sections ranked apart from memories and files, as `recall` and the slot already do), the T63 baseline
  is **87.0% held-out (40 of 46) and 78.5% on all 93**, not 82.6% / 71.0% [M, this run, from
  docs/spikes/rerank/results.json]. The remaining 20 misses fall into six classes (section 5); the two biggest
  are a memory that *mentions* a file beating the file itself (6) and sibling manual sections (4).
- The set has **no** temporal, contradiction, multi-hop or abstention queries, the classes where the surveyed
  systems gain most. 95% on 46 held-out queries means at most 2 misses, and one query is 2.2 points: the target
  needs a larger set first (section 7).
- Three candidate designs (section 6): **A** retrieval rules in the kernel, no model calls; **B** write-time
  enrichment (entity-keyed facts, offline manual questions, file-chunk context lines); **C** a model-maintained
  wiki of markdown pages under /var/lib/ringzero/wiki with dated, superseding lines and links.
- Recommendation for T67 (section 8): build the extended eval set, then measure A + the offline parts of B on
  the host before any kernel work; keep C as the structure for temporal and multi-hop memory, gated on a host
  experiment with the 2B writing pages.

## 2. Survey table

All rows [V] (primary source fetched in this run, section 9) unless a cell says unverified. "Calls" counts LLM
calls. "Ring 0 fit" is our judgement for a 0.6B embedder plus a 2B or 4B model, offline, on local files [E].

| Approach (date) | Mechanism | Write path | Read path | Reported numbers (benchmark, LLM) | Needs | Ring 0 fit [E] |
|---|---|---|---|---|---|---|
| Karpathy LLM wiki (gist, 2026-04-04) | Agent keeps interlinked markdown pages between raw sources and the user, plus index.md and log.md; periodic lint | Ingest: summary page, update 10-15 entity and concept pages, index, log | Read index.md, open pages, answer with citations; file good answers back | None published; "moderate scale (~100 sources, hundreds of pages)" without RAG | Many calls per ingest, files only | Storage yes; rewriting many pages is beyond a 2B, so the kernel must own the writes |
| Hindsight (paper 2025-12-14, release 2026-11-24) | Four networks: world facts, experience, opinions with confidence, entity observations | LLM extracts typed facts, resolves entities, builds a temporal entity graph | Semantic + BM25 + graph spreading + temporal, RRF, cross-encoder, token budget; reflect = extra LLM call | LongMemEval 91.4%, LoCoMo up to 89.61% (memory on GPT-OSS-120B, answers Gemini-3 Pro); 83.6% with GPT-OSS-20B on both sides vs 39% full context | >= 1 call per retain, 0 per recall, >= 1 per reflect; Postgres + pgvector | Read path yes (no LLM); write path untested below 20B |
| MemGPT (2023-10-12) / Letta (archived 2026, now letta-code) | OS-style virtual context: main context, recall storage, archival storage, paged by the LLM's function calls | At 70% context a "memory pressure" warning; the LLM saves; at 100% evict half and summarize | LLM calls search functions, possibly chained | DMR: GPT-4 32.1% -> 92.5%, GPT-4 Turbo 35.3% -> 93.4%, GPT-3.5 38.7% -> 66.9% | >= 1 call per turn plus paging calls; vector DB | Poor: on 1-3B models MemGPT gets 4-10 F1 on LoCoMo (A-Mem's table) |
| Letta sleep-time compute (2025-04-17) | Precompute over the context while idle | Idle-time LLM passes | Uses precomputed notes | About 5x less test-time compute at equal accuracy; up to +13% / +18% (stateful GSM, AIME) | Idle calls | Matches our idle consolidation |
| A-Mem (2025-02-17) | Zettelkasten notes: content, keywords, tags, context, embedding, links; neighbours "evolve" | Several calls per note: build, link, rewrite neighbours | Top-k embedding plus linked notes | LoCoMo F1 (first column, probably multi-hop): GPT-4o-mini 27.02 vs MemGPT 26.65; Qwen2.5-1.5B 18.23 vs 10.44; Qwen2.5-3B 12.57 vs 5.07; Llama 3.2 3B 17.44 vs 5.32 | About 1,100-1,400 tokens per operation at 1-3B; 1.1 s per op on Llama 3.2 1B (GPU); MiniLM embedder; vector store only | The best documented small-model fit; absolute scores are low |
| Zep / Graphiti (2025-01-20, release 2026-11-24) | Temporal knowledge graph: episodes, entities, fact edges, communities; bi-temporal; contradicted edges invalidated, kept as history | Calls for entity extraction, resolution, fact extraction, dedup, dates, invalidation | Cosine + BM25 + BFS, rerank; no LLM | DMR 94.8% (gpt-4-turbo), 98.2% (gpt-4o-mini); LongMemEval_s gpt-4o 60.2% -> 71.2%, gpt-4o-mini 55.4% -> 63.8%, context 115k -> 1.6k tokens | Many calls per write; Neo4j, FalkorDB or Neptune; BGE-m3 | The idea (dated, superseding facts) yes; the pipeline no (README warns small models fail its schemas) |
| Mem0 / Mem0g (2025-04-28) | Extract candidate facts; compare with 10 similar; LLM picks ADD / UPDATE / DELETE / NOOP; Mem0g adds a graph | 1 extraction + 1 update call per fact | Vector search (graph search in Mem0g) | LoCoMo J: Mem0 66.88, Mem0g 68.44, full context 72.90, Zep 65.99 (disputed, see Zep row of section 3), LangMem 58.10, A-Mem 48.38; p95 1.44 s | Vector DB; Neo4j for Mem0g; GPT-4o-mini | Base yes; a 2B choosing DELETE wrongly corrupts memory silently |
| Generative Agents (2023-04-07) | Memory stream; score = recency + importance + relevance (each min-max normalized, weights 1); reflection | 1 call per observation for importance 1-10; reflection when recent importance sums past 150 | Score all, take the top | Human-rated believability only | 1 call per write; flat store | Scoring yes (arithmetic); reflection quality is the weak spot |
| LangMem (repo date unverified) | Semantic (collection or profile), episodic, procedural memory; hot-path tools and a background manager | Extraction and consolidation calls | Vector search over a store; importance and recency suggested | None in the repo; about 8% under Mem0 in Mem0's paper; 50.8 in LightMem's Qwen3-30B table | >= 1 call per write; Postgres or in-memory | Feasible; JSON tool calls fragile on a 2B |
| Cognee (paper 2025-05) | Text to entities, relations, chunks, summaries in a graph plus vectors; remember / recall / improve | LLM extract per chunk, or a local GLiNER extractor without LLM | Graph + vector routing | BEAM 0.79 at 100K (20 questions), 0.67 at 10M (exploratory) | Embedded files (SQLite, LanceDB) single process | Closest framework to local files; graph quality at 2B unproven |
| MemoryOS (2025-05-30) | STM pages, MTM topic segments, long-term persona; heat score promotes and evicts | Summarize, segment, extract persona on overflow | Top segments, then top pages, plus persona | LoCoMo +49.11% F1 over baselines (GPT-4o-mini); LongMemEval 49.6 (Qwen3-30B, LightMem's table) | Many calls; ChromaDB optional; Qwen3 embeddings supported | Heat score yes; persona extraction heavy for a 2B |
| MemOS (2025-05-28, 2025-07-04) | MemCube: plaintext, KV-cache and parametric memory under one lifecycle with provenance and versions | Extraction into a tree or graph | Hybrid retrieval, optional KV injection | README: LoCoMo 88.83, LongMemEval 89.20 (backbone and judge unverified) | Neo4j + Qdrant; a local plugin with SQLite FTS5 | KV-level memory maps to our document states; numbers unverified |
| HippoRAG (2024-05-23) / HippoRAG 2 (2025-02-20) | LLM OpenIE triples to a phrase graph; Personalized PageRank from query entities; v2 adds passage nodes and an LLM triple filter | 1 call per passage (NER + OpenIE) | Embed, (v2) 1 filter call, PPR, read | Up to +20% multi-hop; v2 +7% on associative tasks over NV-Embed-v2 (Llama-3.3-70B everywhere) | In-process graph, files | PPR cheap; triples from a 2B are the weak link |
| RAPTOR (2024-01-31) | Cluster chunks (UMAP + GMM), summarize clusters, recurse into a tree; search all layers at once | 1 call per cluster per level | One embedding search, no LLM | QuALITY 82.6% with GPT-4 (prior 62.3); QASPER F1 36.6 with UnifiedQA-3B; about 4% of summaries had small errors | Vector index only | Yes for files and episodes; rebuild on change |
| Contextual Retrieval (Anthropic, 2024-09-19) | An LLM writes a 50-100 token context line per chunk from the whole document, prefixed before embedding and BM25 | 1 call per chunk | Hybrid search, optional rerank | Top-20 retrieval failure 5.7% -> 3.7% (-35%) embeddings, -> 2.9% (-49%) + BM25, -> 1.9% (-67%) + rerank (Claude 3 Haiku); generic summaries gave very little | Write-time calls only | Yes, best return for us: idle-time 2B call, nothing on the read path |
| HyDE (2022-12-20) | LLM writes a hypothetical answer; its embedding is the query | None | 1 call per query | Beats Contriever, near fine-tuned retrievers (InstructGPT) | 1 call per read | Yes, at about 1.2 s per query [E]; only when the margin is low |
| doc2query (2019-04-17) | Predict questions a document answers, append them to it before indexing | 1 generation per document | Unchanged | Not re-read in this run | Write-time only | Yes; for the manual it runs offline on the build host |
| Late chunking (2024-09-07) | Embed the whole text at token level, pool per chunk | 0 calls | Unchanged | Abstract only; numbers unverified | Long-context embedder | Unclear with our last-token pooled 0.6B |
| SimpleMem (2026-01-05) | Compact memory units; three views: dense Qwen3-Embedding-0.6B, BM25, SQL metadata; intent-planned retrieval | Calls per write; LanceDB | Planned multi-view search | LoCoMo F1 43.24 vs Mem0 34.20 (GPT-4.1-mini); LongMemEval 76.87%; Qwen2.5-1.5B 25.23 vs 23.77; Qwen2.5-3B 17.98 vs 13.03; about 600 tokens per query | LanceDB | Yes: same embedder as ours, small-model results reported |
| LightMem (2025-10-21) | Sensory filter, topic-grouped short-term memory, offline "sleep" consolidation | Few online calls, batched offline | Retrieval of consolidated memory | LongMemEval up to +7.7%, LoCoMo up to +29.3%; up to 38x fewer tokens, 30x fewer calls (GPT-4o-mini, Qwen3-30B) | Batch offline calls | Design yes (idle batching); untested under 30B |
| Memory-R1 (2025-08-27) | RL-trained memory manager (ADD / UPDATE / DELETE / NOOP) and answer agent | Trained manager call | Filtered retrieval | Llama-3.1-8B +28% F1 on LoCoMo, trained on 152 QA pairs; Qwen-2.5 3B-14B | Fine-tuning | Possible for the 4B with our own training; nothing off the shelf |
| MIRIX (2025-07-10) | Six memory types, one agent each | Many agent calls | Many agent calls | LoCoMo 85.4% (gpt-4.1-mini, picked for function calling) | Strong function calling | No |
| CSLS hub correction (2017-10-11) | Subtract each item's mean similarity to its neighbours from the cosine, against hubness | One mean per entry | Adjusted cosine | Paper is on word translation; not a memory benchmark | Arithmetic | Yes; our class H (section 5) is hubness |

Benchmarks [V]:

| Benchmark (date) | What it tests | Key finding for us |
|---|---|---|
| LoCoMo (2024-02-27) | About 300-turn, 9K-token dialogues up to 35 sessions; QA single-hop, multi-hop, temporal, open-domain, adversarial (no answer); token F1, later an LLM judge | Temporal questions are the widest gap (73% under humans); no knowledge-update questions, which Zep's critique notes |
| LongMemEval (2024-10-14) | 500 questions: extraction, multi-session, temporal, knowledge update, abstention; S (115k tokens) and M (1.5M) | Index by round, not session; fact-augmented keys +9.4% recall@k, +5.4% accuracy; time-aware query expansion +11.3% recall but an 8B model "hallucinates or misses temporal cues"; summaries in place of values hurt; an 8B reader degrades past about 3k retrieved tokens |

## 3. Notes per approach

Short notes on what matters for agentd; numbers are in the table. All [V] unless marked.

- **Karpathy's wiki** is the closest match to the Unix rule: memory as plain markdown pages, an index page and an
  append-only log, with the model as the editor and a lint pass for contradictions and stale claims. It has no
  benchmark. Its ingest step (one source touching 10 to 15 pages) assumes a frontier coding agent; for a 2B the
  kernel has to keep the edits small and checkable (design C, section 6).
- **Hindsight** splits memory by kind (world facts, experiences, opinions, entity observations) and does all
  recall without an LLM: four retrievers fused by RRF, then a cross-encoder. Its best numbers use a 120B model
  to build memory. The part we can take is the read path shape (several cheap views fused), which agentd half
  has (cosine plus a gated BM25).
- **MemGPT/Letta** puts the model in charge of paging. With a 2B that is the wrong way round: A-Mem measured
  MemGPT at 5 to 10 F1 on 1 to 3B models. agentd already decided the kernel owns memory (INFO.md, M4.5); the
  survey backs that. Letta's sleep-time compute is our idle consolidation under another name.
- **A-Mem** is the one paper that ran a linked-note memory on 1 to 3B models with a small embedder, and it beat
  MemGPT there by 2 to 3x; absolute F1 stays under 20. Its "memory evolution" rewrites older notes, which is
  the step most likely to corrupt memory with a small model.
- **Zep/Graphiti** contributes the temporal model: every fact edge has a valid time and an ingest time, and a
  contradicting fact invalidates the old edge without deleting it. That maps to an append-only log with a
  `superseded` mark, which mstore already is in spirit (`S` and `X` records).
- **Mem0** is close to agentd's facts: extract, compare with similar memories, ADD / UPDATE / DELETE / NOOP. The
  Mem0 paper and a Zep blog post (2025-05-06) dispute each other's LoCoMo runs (Zep says it was misconfigured and
  scores 75.14 J when run right); LoCoMo also has no knowledge-update questions. Read cross-paper LoCoMo numbers
  as rough.
- **Generative Agents** gives a retrieval score that is pure arithmetic (recency, importance, relevance) and a
  reflection step triggered by accumulated importance. The importance score is one small model call per write.
- **LangMem, Cognee, MemoryOS, MemOS, MIRIX** are frameworks. Their useful ideas: memory types kept apart
  (LangMem, MIRIX), a heat score for promotion and eviction (MemoryOS), provenance and versions per memory unit
  and KV-cache memory (MemOS; our document states are KV memory already), embedded file stores (Cognee). None
  reports small-model numbers we could check.
- **HippoRAG** and **RAPTOR** are retrieval structures over a corpus: a phrase graph walked by Personalized
  PageRank, and a tree of cluster summaries. Both shine on multi-hop questions over many documents, which our
  memory is not yet. RAPTOR's read path has no LLM and a 3B reader still gained, so a per-folder summary
  entry for user files is a cheap later option.
- **Contextual Retrieval** is the most transferable single result: a short context line per chunk, written once,
  cut top-20 retrieval failures by a third with embeddings alone and by half with BM25 added. Generic summaries
  did not help; the line must say what the chunk is about within its document.
- **LongMemEval** findings are directly usable: index by round, expand keys with extracted user facts, add time
  awareness, keep retrieved context small for small readers. Its time-aware query expansion needed a model
  stronger than 8B, so dates should be parsed by rules, not by the 2B.
- **SimpleMem** runs on our exact embedder (Qwen3-Embedding-0.6B) with dense, BM25 and metadata views and
  reports gains over Mem0 at 1.5B and 3B, so the multi-view idea survives at our scale.
- **CSLS** (from word translation) is the standard fix for hubness, a known property of high-dimensional nearest
  neighbour search where a few points are close to everything. The formula was not re-read in this run [E for
  the exact form we would use].

## 4. What agentd already does, mapped to the survey

Source: openspec/specs/memory/spec.md, INFO.md (agentd, manual), kernel/src/agentd.rs, kernel/src/fidx.rs [V,
read in this run].

| agentd piece | What it is | Closest surveyed idea | What it lacks against them |
|---|---|---|---|
| Journal (`journal.log`: `U`, `T`, `A`, `C`, `B`, `P` records) | Append-only raw log of every turn and tool result, with time | Generative Agents' memory stream; Letta's recall storage; LongMemEval's raw session store | Of the journal only `[said]` statements are indexed (with facts and episodes, the `KIND` list in agentd.rs); no retrieval over tool results or answers |
| `[said]` notes | A user statement with a first-person word, no question mark, no tool run, cut to 600 bytes, indexed verbatim | LongMemEval "round" granularity values | Stored as raw text, not normalized into a fact; ownership ("my" vs "my sister's") is not marked |
| Facts (`facts.log`, 64 max, `key = value`, `grounded()`) | Validated key/value, dedup, last write wins, `X` record to forget | Mem0's ADD/UPDATE/DELETE/NOOP; Letta core memory block | Flat keys, no entity or subject; the old value is replaced, not kept with a validity time; no link to the episode or file it came from |
| Episodes (`episodes.log`, from consolidation) | One or two sentences per session by the 2B, indexed as `[episode N]` | Generative Agents reflection (weakly); MemoryOS mid-term pages | No importance score, no entity links, no higher-level reflection over several episodes; an episode that names a path competes with the file it names |
| Consolidation (idle, 60 s, 220 tokens, background stream) | 2B writes `EPISODE:` and up to 8 `FACT:` lines; the kernel validates | Mem0 extraction; LangMem background manager; Letta sleep-time agent | One pass, no update of earlier memories, no conflict check beyond same-key replace |
| Memory index (`index.bin`, Q8_0, 4096 entries, 1 MiB text) | One embedding per entry, cosine | Every vector-RAG baseline in the survey | Single vector per entry, no doc expansion, no hub correction |
| Hybrid BM25 gate (RRF k = 60, only for identifier-like or strong keyword queries) | Lexical rescue for exact tokens | Zep/Graphiti hybrid search, HippoRAG 2 dense + sparse | Fine as is; T63 did not model it, so exact-token misses there are pessimistic |
| Memory slot (200 tokens: facts within 120, then up to 3 notes at cos >= 0.55) | What the model sees at the start of a conversation | Letta core memory in context; Mem0 retrieved memories in the prompt | Facts are all injected (64 max) rather than retrieved; notes have no time stamps, so the model cannot tell old from new |
| `recall` tool (top 3 non-manual + best manual for how-to) | Model-driven retrieval | MemGPT function-call paging | One query, no rewrite, no second hop |
| Manual (29-32 `[manual: TITLE]` sections, `Asked as:` lines, ranked apart) | Static docs with retrieval wordings | doc2query / HyDE on the document side | One vector per section; siblings with close topics collide |
| File chunks (`fidx`, 1000 bytes, `[PATH] text`) | User files indexed in idle time | RAPTOR leaves; Contextual Retrieval without the context line | No per-file or per-chunk context line; tables and code get no description |
| Document states (`/read`, cached KV per file) | A file's engine state loaded instead of prefilled | Letta's archival-to-context paging, at the KV level | Not used by memory today; a natural carrier for memory pages read into context |
| Unix layout (`/var/lib/ringzero/mem/*.log`, agent cannot write there) | Kernel owns memory, model proposes | Karpathy's wiki is plain files; Hindsight and Letta are databases | The stores are binary records, not readable with `cat`; only `/facts` and `/agentd` show them |

## 5. Where recall misses today

Data: the T63 per-query results (docs/spikes/rerank/results.json, 93 queries, embedder top 20 with cosine) [M].
Two readings, both from the stored cosines, no new model run:

| Reading | Held-out top-1 | Held-out top-3 | All 93 top-1 | All 93 top-3 |
|---|---|---|---|---|
| One pooled ranking (T63 as reported) | 38/46 (82.6%) | 43/46 (93.5%) | 66/93 (71.0%) | 83/93 (89.2%) |
| Kernel pool split: manual queries over manual sections, other queries over memories and files (what `recall` and the slot do) | **40/46 (87.0%)** | 45/46 (97.8%) | **73/93 (78.5%)** | 89/93 (95.7%) |

The split is the honest baseline for agentd, since manual sections never take a slot note or one of recall's top
3 (INFO.md, T22). The split removes 7 misses net (27 pooled, 20 split), almost all of them a manual section
winning a personal question. The tune half stays harder than the held-out half (70.2% against 87.0%), so the halves are not
balanced and both should be read.

### Miss classes (kernel pool split, 20 misses; held-out count in brackets)

| Class | Count | What happens | Example (query -> top-1, gold) |
|---|---|---|---|
| P. Pointer beats content | 6 (2) | An episode or fact that names a file or meeting outranks the file that holds the answer | "what's on my todo list?" -> `[episode 3] ... listed todo.txt, spesa.txt and idee.md`, gold `/notes/todo.txt`; "what did Acme ask for in the last call?" -> the weekly-meeting fact, gold `/home/meeting_acme.md` |
| S. Sibling manual sections | 4 (3) | Two manual sections share words; the wrong one wins | "come cambio la temperatura?" -> Settings, gold Temperature and thinking; "dove tiene ringzero i suoi dati?" -> What ringzero is |
| H. Hub entry | 4 (0) | One short generic entry is close to many unrelated queries | `[episode 3]` (/notes listing) is top-1 for "che tipo di risposte preferisco?" and "anna.rossi@example.org"; it is top-1 for 6 of the 20 misses in all, across classes P and H |
| E. Wrong owner (entity confusion) | 3 (0) | Same relation, other person: the sister's dog beats the user's cat and the user's partner | "come si chiama il mio gatto?" -> `[fact] sister_dog: il cane di mia sorella si chiama Rocco`, gold "Il gatto dell'utente si chiama Pixel" |
| R. Raw note, paraphrase gap | 2 (0) | The answer is a verbatim `[said]` sentence or a prose memory whose words differ; once it is not in the top 20 at all | "che lavoro faccio?" -> gold `[said] ricordati che lavoro come insegnante di matematica al liceo` not in the top 20 |
| T. Table or code content | 1 (1) | The answer is a CSV row; the chunk embeds as "a CSV file", not as its rows | "who lives in Milano?" -> `[fact] home_city: Bologna`, gold `/demo/people.csv` at rank 7 |

Notes on the classes [M unless marked]:

- Exact tokens: 3 of the misses (`notes/todo.md`, `anna.rossi@example.org`, `ryokan Kyoto`) are queries the
  kernel's BM25 gate would probably rescue (identifier-like, or every term in one entry only); T63 did not model
  the gate [E].
- Margin: the gap between the top two cosines is under 0.05 for 11 of the 20 misses and for 14 of the 73 hits
  (25 of 93 queries in all). A read-time step that runs only below that margin touches about a quarter of the
  queries and half of the misses.
- Labels: two pooled misses were arguable (the top-1 also answers: "what model runs this OS?" -> /demo/data.json,
  which lists the models; "quanto e' grande il disco?" -> the disk-check episode). The split removes the second;
  the first stays a miss in class S.
- Missing classes: no query in the set tests a stale or contradicting fact, a two-hop question, a time question
  ("what did I do last week?") or a question with no stored answer. These are LongMemEval's knowledge-update,
  multi-session, temporal and abstention categories [V, LongMemEval], and they are where the surveyed systems
  differ most. Section 7 adds them.

What 95% needs: on today's held-out half, at most 2 of 46 misses, so 4 of the 6 must go (the 3 S, 2 P, 1 T). On
all 93, at most 4 misses, so 16 of 20 must go. Both are small numbers: one query moves held-out by 2.2 points.

## 6. Candidate designs

Engine numbers used for costs (all [M] from ringzero results unless marked): embedder per-token path about 6 ms
per token, 0.61 s per 100 tokens, a query of about 34 tokens about 0.2 s (INFO.md, docs/EMBEDDER_RESULT.md,
RERANK_SPIKE.md); 2B batched prefill 200 to 226 tok/s, 2B decode about 36 tok/s (docs/M45_RESULT.md), 4B q4
decode about 20 to 21 tok/s; a consolidation today is 27 to 45 tokens in 3.3 to 4.5 s (docs/M45_RESULT.md);
the embedder has no batched prefill yet (DECISION_MODELS.md section 3; 700 to 800 tok/s if it gets the 2B's
speedup [E]).

All three keep the Unix rule: memory is files under /var/lib/ringzero, the kernel writes them, the model only
proposes, and `cat` shows them.

### A. Retrieval rules in the kernel (no model calls)

What changes:

1. **One-hop link from pointer to content.** When a slot or recall hit names a path that the file index holds
   (`/notes`, `todo.txt`, `/home/meeting_acme.md`), add that file's best chunk for the same query as the next
   line, or rank it right after the pointer. The link is a plain path match in the hit text. (HippoRAG and
   A-Mem link notes; here the link is a path, which needs no model.)
2. **Hub correction.** Store per entry its mean cosine to a fixed probe set (for example the manual's `Asked as:`
   lines and 50 generic questions, embedded once), and rank by `cos - lambda * hub` (CSLS-style; lambda tuned on
   the tune split). Costs one extra float per entry, computed when the entry is embedded.
3. **Table rows as chunks.** For `.csv` and `.tsv`, chunk by rows with the header line repeated in each chunk
   (`[/demo/people.csv] name,city,age,email | Marco Bianchi,Milano,41,...`).
4. **Model the BM25 gate in the eval** (no kernel change) so exact-token queries are scored as the kernel does.

Fixes: P (6), H (4), T (1), exact tokens. Expected gain [E]: held-out 40 -> 42 or 43 of 46 (91 to 93%), all 93
78.5% -> about 85 to 88%; S (sibling manual sections) and E (wrong owner) are not touched.
Cost: zero LLM tokens. Per write: one probe pass per new entry, about 50 dot products of 1024 (microseconds). Per
read: a path match over the top 4 hits and one more index search for the linked file (one embedding already
done, a second search under 1 ms) [E].
Risks: the link can pull a stale or large file into a 200-token slot; hub correction can hurt a true answer
that happens to be generic. Both are measurable on the eval.
Smallest experiment: add the three rules to tools/rerankeval.py's `report` over the stored cosines plus a
re-embedding of the CSV rows (host, minutes), on the T63 set and on the extended set of section 7.

### B. Write-time enrichment (doc expansion, entity keys)

What changes:

1. **Entity-keyed facts.** Fact keys get a subject: `user.cat_name = Pixel`, `sister.dog_name = Rocco`,
   `user.job = insegnante di matematica al liceo`. The kernel indexes a fact as a sentence with its owner
   ("[fact] the user's cat is called Pixel"), not as `sister_dog: ...`. Consolidation's `FACT:` line keeps its
   format with a dotted key; `norm_key` accepts one dot. A `[said]` statement that consolidation turns into a fact
   is indexed through the fact.
2. **Manual questions, offline.** For each manual section, a large model on the build host writes 8 to 12
   questions in Italian and English, including the ones that separate it from its siblings ("how do I change the
   temperature" belongs to Temperature, not Settings). Each question is embedded as its own index entry that
   points to the section (several vectors, one section). The manual is compiled in and its vectors are already
   baked, so this costs nothing at run time beyond about 300 to 400 more entries of the 4096.
3. **Context line per file.** In idle time the 2B writes one line per file ("table of 6 people: name, city, age,
   email"; "Lua demo that counts the words of a sentence"), kept in `/var/lib/ringzero/mem/files.ctx` as
   `PATH<TAB>line`, and prefixed to each chunk before embedding (Contextual Retrieval).

Fixes: E (3), S (4), R (2), T (1, with the context line), part of P. Expected gain [E]: on its own held-out 40 ->
43 of 46 (S is 3 of the 6 held-out misses), all 93 to about 85%; with A, held-out 44 to 45 of 46 (96 to 98%) and
all 93 about 90 to 93%. These are estimates from the classes, not measurements.
Cost: facts: no extra call (the consolidation prompt changes, same 220-token cap). Manual: zero at run time;
about 320 entries x 20 tokens x 6 ms = about 38 s of embedding once at first boot, or baked like the manual
vectors today. File context line: one 2B call per file, about 250 tokens of prefill (1.1 to 1.3 s) and 20 tokens
of decode (0.6 s), so about 2 s per file in idle time, then the file's chunks are re-embedded as today.
Risks: the 2B may write wrong subjects in dotted keys (`grounded()` checks values, not owners); a file context
line written by the 2B can be wrong and then misleads every chunk of that file; more manual entries raise the
chance a manual question beats a memory if the pool split is ever removed.
Smallest experiment: host only. Rewrite the T63 facts with owner sentences, generate the manual questions with a
host model, write the file context lines with the 2B in PyTorch (or by hand for a first upper bound), re-embed,
re-score.

### C. Model-maintained wiki (pages per entity and topic)

What changes: memory gains a page layer in `/var/lib/ringzero/wiki/`, one markdown file per entity or topic
(`user.md`, `pixel.md`, `acme.md`, `machine.md`, `notes.md`), readable with `cat`:

```
# acme
- 2026-11-24 weekly call: Acme wants the export in CSV, not JSON; deadline December 10 (see /home/meeting_acme.md)
- 2026-11-10 meeting every Tuesday 10:00 CET
- [superseded 2026-11-24] deadline November 30
links: [[user]] /home/meeting_acme.md
```

1. **Write path (consolidation).** After the episode and facts, the 2B gets the list of page names and the
   session, and answers with lines `PAGE name: line` (at most 4). The kernel validates each line like a fact
   (grounded in the user's words or in a tool result of the session), stamps the date, appends it to the page,
   and marks an older line of the same page `[superseded DATE]` when the 2B names it (`SUPERSEDES n`). The model
   never rewrites a page; the kernel appends and marks. Pages are capped (for example 40 lines, the oldest
   superseded lines dropped first).
2. **Index.** Each page is embedded as its title plus its live lines (chunks of 1000 bytes like files), and its
   links are read by the kernel.
3. **Read path.** Slot and recall rank pages with the other entries; a page hit brings its newest live lines
   (dated, so the model can tell old from new) and, through links, one linked file chunk (rule A.1).

Fixes: P (the page holds both the pointer and the content line), E (one page per owner), R (the `[said]` becomes
a page line), and the classes the T63 set does not test: stale or contradicting facts (superseded lines),
temporal questions (dated lines), multi-hop (links). Expected gain [E]: on the T63 set about as A + B, since
those classes are absent there; the difference shows only on the extended set, where Zep's and Mem0's reported
gains are largest on temporal and multi-hop questions [V, see section 3].
Cost: per consolidation one more 2B call, prompt about 300 to 600 tokens (session plus page names; with the
background stream the session prefix is shared) and 40 to 80 tokens out: about 1.5 to 3 s of prefill and 1.1 to
2.2 s of decode, so 3 to 5 s per session in idle time, plus re-embedding each touched page (about 200 tokens,
1.2 s per page). Per read: nothing beyond today's search. Disk: a few KB per page.
Risks: the 2B picks the wrong page or invents a supersession (the papers that report gains use GPT-4o-mini or
larger for these steps [V]); pages drift into long lists the embedder averages into mush (mitigated by
chunking by line groups and the cap); a second copy of every fact (facts.log and the page) can disagree, so one
must be the source (pages, with `/facts` as a view) or facts must be written into pages by the kernel only.
Smallest experiment: host only, on the extended set. Build the pages for the T63 memories by hand once (upper
bound of the structure), then by the 2B in PyTorch from the same sessions (what the kernel would get), embed
both, score. If the hand-built pages do not reach 95% on the extended set, the 2B-built ones will not.

### Read-time options considered and kept as add-ons

- **Query rewrite or HyDE by the 2B** (write a hypothetical memory line, embed it, search with it, fuse with the
  original): about 60 tokens of prompt (0.3 s) and 25 tokens of decode (0.7 s) plus one more embedding (0.2 s),
  about 1.2 s per query [E]. Run only when the top-2 margin is under 0.05 (a quarter of the queries, half the
  misses [M]). Helps R and E; adds latency before the first token. Measure after A and B.
- **Reranker** (T63): no-go, measured; not revisited.
- **Importance scoring and reflection** (Generative Agents): consolidation could add `IMPORTANCE: 1-10` per
  episode and, every N episodes, a reflection page line. Useful to cap the stores and order the slot, not for
  top-1 on the current set (no miss is caused by an unimportant entry winning). Fold into C if C is chosen.

## 7. Extended eval set

Goal: a set where 95% means something and where the missing classes are present. Built with the corpus and
scorer of tools/rerankeval.py (memories, manual, files in agentd's formats) and the memory list of
tools/retrieval.py.

- **Size**: at least 250 queries, split by a fixed hash of the query text (not even/odd), held-out at least 125:
  one query is 0.8 points, and at 95% the 95% confidence interval is about +/-3.8 points [E, normal
  approximation]. Italian and English in about equal parts, as today.
- **Keep** the 93 T63 queries, fix the arguable label ("what model runs this OS?" accepts /demo/data.json).
- **Add**, with new memories written in agentd's formats and with time stamps:
  - Knowledge update, 30: two records of the same key or topic at different times ("[fact] home_city: Bologna"
    in March, "[said] mi sono trasferito a Torino" in November); gold is the newer one, the older counts as a
    miss.
  - Contradiction, 15: two episodes that disagree (the backup ran / the backup failed); gold is the later one,
    or both when the question asks for history.
  - Multi-hop, 30: the answer needs two entries ("dove porto il gatto dal veterinario?" needs the cat's name,
    then the vet fact for Pixel; "what's in the file the agent showed me about Acme?" needs the episode, then the
    file). Score as top-3 containing both, and top-1 containing the second hop.
  - Temporal, 20: "what did I do on Tuesday?", "when did I last run stats.py?"; records carry dates; gold is the
    record of that day.
  - Wrong owner, 20: more relatives and their pets, devices, cities next to the user's own.
  - Abstention, 20: questions with no stored answer; scored on whether the slot stays empty (no note at cos >=
    0.55), which today's thresholds decide.
  - Tables and code, 15: rows of CSV files and functions of the demo scripts.
- **Score** with the kernel's pool split and the BM25 gate modelled (port `bm25`, `identifier_like` and the RRF
  rule from kernel/src/memidx.rs to the tool), so the numbers match what agentd would rank.
- **Report** top-1 and top-3 per class and per split; a design is accepted only if held-out top-1 is >= 95%
  and no class falls by more than one query against the baseline.

## 8. Recommendation (input for T67, not a decision)

1. **First, the eval set of section 7** (host, no models beyond the embedder). Without it 95% is a statement
   about 2 queries.
2. **Then A and the offline half of B** on the host: pool split and BM25 gate in the scorer, path links, hub
   correction, CSV rows, owner sentences for facts, generated manual questions. No model runs in ring 0 for any of
   them, and none adds latency to a turn. Our estimate is 96 to 98% held-out on the T63 set [E]; the extended set
   will say how far it falls on the new classes.
3. **C (the wiki) is the candidate for the classes the current set does not have**: temporal, knowledge
   update, multi-hop. It is the one design that turns memory into files a user can read and correct, which fits
   the Unix rule best, and its cost sits in idle time. Its risk is the 2B as a writer. Gate it on the hand-built
   versus 2B-built page experiment of section 6.C.
4. **Query rewrite only as a margin-gated add-on**, after 2 and 3, if a class is still below target.

T67 should decide: the eval set and its acceptance rule; whether facts become entity-keyed (a format change of
facts.log that needs an OpenSpec change); and whether pages in /var/lib/ringzero/wiki become the source of truth
for facts or stay a second layer.

## 9. Sources

Fetched in this run (curl), date is the paper's first version or the repo's latest release:

- Karpathy, "LLM Wiki" gist, 2026-04-04: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Hindsight, arXiv 2512.12818, 2025-12-14: https://arxiv.org/abs/2512.12818 ; repo https://github.com/vectorize-io/hindsight (v1.3.0, 2026-11-24)
- MemGPT, arXiv 2310.08560, 2023-10-12: https://arxiv.org/abs/2310.08560 ; Letta repo https://github.com/letta-ai/letta (0.16.8, 2026-05-14, archived)
- Sleep-time compute, arXiv 2504.13171, 2025-04-17: https://arxiv.org/abs/2504.13171
- A-Mem, arXiv 2502.12110, 2025-02-17: https://arxiv.org/abs/2502.12110
- Zep, arXiv 2501.13956, 2025-01-20: https://arxiv.org/abs/2501.13956 ; Graphiti https://github.com/getzep/graphiti (mcp-v1.1.5, 2026-11-24)
- Mem0, arXiv 2504.19413, 2025-04-28: https://arxiv.org/abs/2504.19413 (repo release not fetched)
- Zep's reply on the Mem0 LoCoMo run, blog post, 2025-05-06 (blog.getzep.com)
- Generative Agents, arXiv 2304.03442, 2023-04-07: https://arxiv.org/abs/2304.03442
- LangMem README and conceptual guide: https://github.com/langchain-ai/langmem (date not given)
- Cognee README and docs: https://github.com/topoteretes/cognee ; paper arXiv 2505.24478, 2025-05
- MemoryOS, arXiv 2506.06326, 2025-05-30: https://arxiv.org/abs/2506.06326
- MemOS, arXiv 2505.22101 (2025-05-28) and 2507.03724 (2025-07-04): https://arxiv.org/abs/2507.03724 ; README https://github.com/MemTensor/MemOS
- HippoRAG, arXiv 2405.14831, 2024-05-23; HippoRAG 2, arXiv 2502.14802, 2025-02-20
- RAPTOR, arXiv 2401.18059, 2024-01-31
- Contextual Retrieval, Anthropic, 2024-09-19: https://www.anthropic.com/news/contextual-retrieval
- HyDE, arXiv 2212.10496, 2022-12-20
- doc2query ("Document Expansion by Query Prediction"), arXiv 1904.08375, 2019-04-17 (title and date only)
- Late chunking, arXiv 2409.04701, 2024-09-07 (abstract only)
- SimpleMem, arXiv 2601.02553, 2026-01-05
- LightMem, arXiv 2510.18866, 2025-10-21
- Memory-R1, arXiv 2508.19828, 2025-08-27
- MIRIX, arXiv 2507.07957, 2025-07-10
- CSLS ("Word Translation Without Parallel Data"), arXiv 1710.04087, 2017-10-11 (title and date only)
- LoCoMo, arXiv 2402.17753, 2024-02-27
- LongMemEval, arXiv 2410.10813, 2024-10-14

ringzero sources: openspec/specs/memory/spec.md, openspec/specs/unix-tree/spec.md,
openspec/specs/document-states/spec.md, openspec/specs/manual/spec.md, INFO.md, docs/MANUAL.md,
kernel/src/agentd.rs, kernel/src/fidx.rs, docs/EMBEDDER_RESULT.md, docs/M45_RESULT.md,
docs/research/RERANK_SPIKE.md, docs/research/DECISION_MODELS.md, docs/spikes/rerank/results.json,
tools/rerankeval.py, tools/retrieval.py.
