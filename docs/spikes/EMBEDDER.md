# Spike: Qwen3-Embedding-0.6B next to Qwen3.5-2B (2026-11-26)

Verdict: GO. No new asm needed; about 3 exporter lines and 100-150 lines of Rust; fits in the 4 GiB guest.

## Model facts (HF files)

- Same dimensions as Qwen3-0.6B (1024 hidden, 28 layers, 16Q/8KV, head 128), tied embeddings, no lm_head, weights without the `model.` prefix. Apache-2.0, 32k context.
- Tokenizer identical to Qwen3, but every input gets `<|endoftext|>` (151643) appended: the kernel must append it by hand.
- Embedding = hidden state of that last token after the final norm, L2-normalized. Queries: `Instruct: {task}\nQuery:{q}`; documents: no prefix. Matryoshka: first d of 1024 dims, re-normalized.
- Qwen3-Reranker-0.6B shares the architecture (yes/no logits), possible later.

## Engine

- All engine state is one global context `g_ctx` (640 B) loaded with `lea rbx,[rel g_ctx]`; tokenizer pointers live inside it. Two models = Rust saves/restores the 640 B between whole forward calls on the BSP, plus `matmul_set_fp16`. No asm change.
- Qwen3 layers (type 0 without gate) still work in the QWEN35 build (v3 0.6B regression 9/9).
- After the last token, the final-norm vector is already in `BUF_X`; lm_head is wasted work (optional 2-line asm: compute_logits==2 = norm only).
- Exporter: `export_model_v3` handles Qwen3; only blocked by `assert is35`; set `pre=""` for unprefixed weights.
- Vector search: store the index as Q8_0 rows and use `tp_matmul_q8_0` (dims multiple of 128: 256 and 1024 OK).

## Costs

- Module ~610 MiB; embedder arena ~115 MiB at 1024 tokens (56 at 512). With the 2B at 32k (~430 MiB) and snapshots, ~900 MiB stays free in 4G. Watch where Limine places the second module (it can split the largest free region); go 6G if it does.
- Index for 10k chunks: 1024 dims 40/20/10.4 MiB (f32/fp16/Q8_0), 256 dims 10/5/2.6 MiB.
- Speed from the engine's Linux build, 8 threads: 89-148 tok/s prefill -> ~0.7 s per 100-token chunk, query 0.25-0.3 s (~0.1 s with the instruction prefix cached), search over 10k vectors < 5 ms. Indexing must be incremental / idle-time.

## Retrieval test (40 synthetic memories, IT/EN; 15 paraphrased + 5 exact-token queries)

| Method | semantic top1 | exact top1 |
|---|---|---|
| BM25 | 5/15 | 5/5 |
| Embedder 1024 + instruction | 15/15 | 5/5 |
| Embedder 1024, no instruction | 13/15 | 4/5 |
| Embedder 256 | 14/15 | 5/5 |
| Fusion with plain BM25 (RRF) | 8/15 | 5/5 |
| Fusion with stopword BM25 | 13/15 | 5/5 |
| Embedder, simulated Q8 W8A8 | 15/15 | 5/5 |
| Qwen3.5-2B as embedder (best pooling) | 9/15 | 4/5 |

Q8 vs fp32 cosine >= 0.998, 20/20 same top1. Fusion lesson: let BM25 in only when its top score is strong or the query looks like an identifier/path. Small synthetic set: a direction, not a benchmark.

## Plan

1. Exporter v3 for Qwen3 checkpoints; check engine vectors vs HF (cosine >= 0.998 on 10 strings).
2. Context swap in Rust; golden self-test of the 2B still passes after loading the embedder.
3. Second Limine module + mapping; log the real memory map, decide 4G vs 6G.
4. `embed()`.
5. Q8_0 index with tp_matmul_q8_0; stopword BM25; gated fusion.
6. Incremental and idle indexing.
Later: compute_logits==2, instruction cache, reranker.

Risks: second module splitting the engine region; +29-50 s UEFI boot; forgetting the end token silently degrades quality; never swap contexts mid-generation.
