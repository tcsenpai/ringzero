# Embedder in kernel result (2026-11-26)

Qwen/Qwen3-Embedding-0.6B (Apache-2.0) runs in ring 0 next to Qwen3.5-2B on the same NASM engine, per the spike
[spikes/EMBEDDER.md](spikes/EMBEDDER.md). Edition **ringzero Qwen35@2b-Qwen3@0.6b** (tag QweQwe), in the boot
banner. No new asm: the second model is a second saved engine context, swapped in Rust.

All numbers are real runs on the Fedora box (Ryzen AI 9 HX 370, QEMU 10.2.2/KVM, `-cpu host -M q35 -m 6G
-smp 8`, BIOS unless noted), on a test guest (`make run-remote REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test`)
scripted with `tools/accept.py` on the box itself. The user's 4G instance was stopped for the tests (one guest
at a time, host RAM 46-58% used during them). Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

## How to connect (instance left running on fedora, this build, 6 GiB guest)

```sh
nc LAN-ADDRESS0 4555                 # serial shell: press Enter to get "ringzero[main]> "
nc HOST-ADDRESS 4555                 # same over Tailscale (the LAN address did not answer from the Mac today)
stty -echo -icanon; nc LAN-ADDRESS0 4555; stty sane    # without the double local echo (Tab works too)
open vnc://LAN-ADDRESS0:5901         # framebuffer + PS/2 keyboard, password: aios (or vnc://HOST-ADDRESS:5901)
ssh fedoralan 'tail -f ~/coding/ringzero/build/remote-serial.log'                  # everything on serial
ssh fedoralan 'cd ~/coding/ringzero && python3 tools/screenshot.py build/remote-monitor.sock build/screen.png' && scp fedoralan:coding/ringzero/build/screen.png .
```

Restart on Fedora (stop the old guest first; `make run-remote` now needs 8 GiB available):

```sh
cd ~/coding/ringzero && echo quit | socat - UNIX-CONNECT:build/remote-monitor.sock
setsid nohup make run-remote > build/remote.log 2>&1 < /dev/null &
```

The memory index starts empty at every boot. Try: `/index add The home wifi password is stored in Bitwarden
under the entry casa-wifi.`, then `/search wifi password` or just ask "Where is my wifi password stored?".

## Steps

| Step | Status |
|---|---|
| 1. `make model` fetches and exports the embedder | DONE. `tools/fetch_model.sh` is generic (HF_REPO, HF_FILES, EXPORT_ARGS) and skips a model whose .asmbin exists. Exporter: `--v3` lets a Qwen3 checkpoint through the v3 writer, unprefixed weights (`embed_tokens.weight`, no `model.`) are detected. `models/qwen3-embed-0.6b-q8_0-v3.asmbin`: 643 181 888 B (613 MiB), flags 0x7 (tied, QK-norm, fp16 scales), sha256 `d2c4ff94ed36d30da4751f5e89b648a79a1829f64f0780af6f1eb28285e7545e`, export 4.9 s. |
| 2. Second Limine module, memory map, guest RAM | DONE. `embed.asmbin` with `module_string: embed`, modules found by path, both identity-mapped; the full map is logged at boot (`mmap:` lines). Decision: **6 GiB** (below). |
| 3. Context swap | DONE (embed.rs). `g_ctx` (640 B) + `matmul_set_fp16` saved/restored around one whole `embed()` call, on the BSP. Outside it the 2B context is always loaded, so every snapshot call sees the 2B; the embedder's forwards never touch the 2B arena, `llm::REC` or the DeltaNet state. |
| 4. `embed(text, is_query)` | DONE. Plain BPE with the embedder's tokenizer (`llm::encode_plain`), `<|endoftext|>` 151643 appended, queries `Instruct: {task}\nQuery:{q}`, one forward per token with compute_logits 0, pooled `x * w_final` from BUF_X (the RMSNorm scale cancels in the L2 normalization, so no lm_head and no asm change), Matryoshka cut + L2. Arena 113 MiB at 1024 tokens. |
| 5. Vector index + BM25 + gated fusion | DONE (memidx.rs). 4096 entries of Q8_0 rows (int8 + fp16 scales, dim 1024, `/index dim N` multiples of 128), search with `tp_matmul_q8_0`, stopword BM25, RRF only behind the gate (identifier/path-like query, or the top BM25 hit has every query term and >= 1.5x the second). |
| 6. Shell commands + agent tool | DONE. `/embed [-q] [-v] TEXT`, `/index add TEXT`, `/index stats`, `/index clear`, `/index dim N`, `/search QUERY`; agent tool `search_memory(query)` (top 3). |

## Verification

| Test | Target | Result |
|---|---|---|
| Engine vs HF fp32 (`tools/embed_ref.py`, 10 strings: 6 docs, 4 queries, IT/EN, 2-90 tokens) | cosine >= 0.998 | **PASS 10/10**, min 0.99860 (the query `notes/todo.md`), mean 0.99927; token counts equal to HF on all 10 |
| Retrieval set, semantic (15), embedding only | 15/15 | **15/15** top1 |
| Retrieval set, semantic, fused (what `/search` ranks first) | | **15/15** (gate opened on 2: "what CPU..." as identifier-like, "che stampante ho in ufficio?" as strong keyword match; both still right) |
| Retrieval set, exact-token (5) | | embedding **5/5**, fused **5/5** (gate open on all 5) |
| Embed time | | **610 ms per 100 tokens** (index average over the 40 adds, 18 tok each, 115 ms per entry); 90-token doc 553 ms; ~6 ms per token |
| Query time | | 175-258 ms, mean 217 ms (29-37 tokens, ~27 of them the instruction prefix) + search 43-71 us over 40 rows |
| Golden self-test of the 2B after the embedder loaded and the context swapped | PASS | **PASS** on every boot (the boot check embeds one string through the swap before the golden test) |
| Decode tok/s unchanged | 34.4-36 (Snapshots v1: 35.5-35.9) | `/bench` **36.37, 36.01** tok/s after embeddings in the same session; boot bench 34.0-36.9 |
| Snapshots after embedding calls | /undo /redo /save /load work | **PASS**: `/search` and `/embed -q` between turns, then `/undo` 0.58 ms, `/redo` 0.60 ms, `/clear` (forgets), `/load x` (remembers green); `/retry` of a turn with a mid-turn `search_memory` call at temp 0 gave the identical tool call and answer |
| Agent uses the index | | asked "Where is my wifi password stored?" without naming the tool: `search_memory(query=wifi password storage)` -> top hit 0.78 -> correct answer |
| Boot time to the shell | | BIOS 28.8 s (Snapshots v1 build, 4G: 23.5 s), UEFI 38.6 s (34.0 s); banner +3.7 s BIOS / +3.3 s UEFI (Limine loading 613 MiB more), the rest is the 89 extra prefix tokens of the new tool (+1.7 s) and the embedder init (0.2-0.3 s + 24-38 ms boot check). The spike feared +29-50 s on UEFI: not seen. |

### Embedding transcripts (excerpts)

```
cos 0.99958  tok hf 2 kernel 2  doc    hello
cos 0.99931  tok hf 21 kernel 21  doc    Il gatto dell'utente si chiama Pixel ed e' allergico al poll
cos 0.99951  tok hf 23 kernel 23  doc    Tool result: df -h showed /home at 91% usage, 18 GB free.
cos 0.99927  tok hf 23 kernel 23  doc    The backup job runs nightly at 02:00 with restic to a Backbl
cos 0.99924  tok hf 23 kernel 23  doc    La password del Wi-Fi di casa e' salvata in Bitwarden sotto
cos 0.99913  tok hf 37 kernel 37  query  how much disk space is left on my home partition?
cos 0.99918  tok hf 37 kernel 37  query  come si chiama il mio animale domestico?
cos 0.99860  tok hf 29 kernel 29  query  notes/todo.md
cos 0.99944  tok hf 33 kernel 33  query  What CPU does this machine have?
cos 0.99945  tok hf 90 kernel 90  doc    ringzero is an experimental AI OS: a small Rust no_std x86-6
min cos 0.99860, mean 0.99927, 10/10 >= 0.998
```

```
ringzero[main]> /search che stampante ho in ufficio?
1. #18   cos 0.710  bm25  6.15  emb+bm25 La stampante dell'ufficio e' una Brother HL-L2350DW collegata via Wi-Fi.
2. #26   cos 0.371  bm25  0.00  emb      Preferred editor is Helix with the catppuccin theme.
...
[search] 40 entries, dim 1024, query 35 tok embedded in 228.874 ms, search 48 us, BM25 strong keyword match
ringzero[main]> /search how are my files backed up?
1. #22   cos 0.536  bm25  0.00  emb      The backup job runs nightly at 02:00 with restic to a Backblaze bucket.
2. #9    cos 0.506  bm25  0.00  emb      L'utente usa Syncthing per sincronizzare il vault Obsidian tra i computer.
...
[search] 40 entries, dim 1024, query 33 tok embedded in 205.902 ms, search 45 us, BM25 not used (embedding order)
ringzero[main]> /search notes/todo.md
1. #34   cos 0.666  bm25  9.22  emb+bm25 Tool result: the file notes/todo.md contains 14 open items.
...
[search] 40 entries, dim 1024, query 29 tok embedded in 177.759 ms, search 49 us, BM25 identifier-like query
ringzero[main]> /index stats
[index] memory index: 40 of 4096 entries, dim 1024 (Q8_0 rows of 1088 B, 42 KiB), text 2 of 1024 KiB, embed avg 115 ms/entry (18 tok/entry, 610 ms per 100 tok); RAM only, lost at reboot
```

Agent tool (final instance, before it was restarted clean):

```
ringzero[main]> Where is my wifi password stored?
[tool] search_memory(query=wifi password storage)
[tool] -> 1. (score 0.78) The home wifi password is stored in Bitwarden under the entry casa-wifi.
2. (score 0.23) The user's cat is called Pixel.
3. (score 0.22) The backup job runs nightly at 02:00 with restic to a Backblaze bucket.
The home wifi password is stored in Bitwarden under the entry "casa-wifi".
[prefill] 95 tok in 1872.84 ms (50.72 tok/s)  [decode] 18 tok in 548.85 ms (30.97 tok/s, output 2 ms)  cores 8  ctx 996/32768 (3%)
```

Snapshots after embeddings (temp 0):

```
ringzero[main]> My favourite colour is green, remember it.
ringzero[main]> /search what is my favourite colour?          (embedder swap between turns)
ringzero[main]> What is my favourite colour?
Your favourite colour is green.
ringzero[main]> /save x
ringzero[main]> /embed -q where do I keep my passwords?
[embed] query dim 1024, norm 1.000000 (pooled |x*w| 969.390), 33 tok in 193.148 ms, first: -0.05723 -0.02743 ...
ringzero[main]> /undo
[undo] back to #1 "My favourite colour is green, re" (ctx 865/32768), 0.58 ms. /redo brings #2 back.
ringzero[main~1]> /redo
[redo] forward to #2 "What is my favourite colour?" (ctx 891/32768), 0.60 ms
ringzero[main]> /clear
ringzero[main]> What is my favourite colour?
I don't know your favourite colour as I don't have access to personal information about you.
ringzero[main]> /load x
[load] main at #2 "What is my favourite colour?" (ctx 891/32768), 0.66 ms
ringzero[main]> What is my favourite colour?
Your favourite colour is green.
ringzero[main]> Use the search_memory tool: where is my wifi password stored?
[tool] search_memory(query=wifi password location)
La password del Wi-Fi è salvata in Bitwarden sotto 'casa-wifi'.
ringzero[main]> /retry                                        (same tool call, same answer, same token counts)
ringzero[main]> /bench
[prefill] 25 tok in 500.07 ms (49.99 tok/s)  [decode] 64 tok in 1732.00 ms (36.37 tok/s, output 7 ms)  cores 8  ctx 88/32768 (0%)
[snap] conversation restored (ctx 1124) in 4222 us
```

## Memory map and the RAM decision

With `-m 4G` (BIOS) Limine put the embedder at the top of the high region, so the arena run shrank to 1434 MiB:
2B 406 + embedder 113 MiB, and the snapshot pool fell back from 1024 to **512 MiB** (`!mem`: free 503 MiB).

```
mmap: 0x000100000-0x0064b1000    99 MiB usable
mmap: 0x0065b8000-0x07e816000  1922 MiB kernel+modules        <- Qwen3.5-2B
mmap: 0x100000000-0x159a9d000  1434 MiB usable                <- arena
mmap: 0x159a9d000-0x180000000   613 MiB kernel+modules        <- embedder
```

Decision: **6 GiB guest** (`GUEST_MEM ?= 6G` in the Makefile for every run target, `MEM=6G` default in
`tools/boottest.py`; `make run-remote` wants 8 GiB available). The pool keeps its 1 GiB and about 2 GiB stay free
for the next milestones. Fedora cost: the 6G guest plus the host's usual load was 57% of 31 GiB with one guest.

BIOS, 6 GiB:

```
mmap: 0x000100000-0x0064b1000    99 MiB usable
mmap: 0x0065b8000-0x07e816000  1922 MiB kernel+modules        <- Qwen3.5-2B (phys 0x65b8000)
mmap: 0x07e821000-0x07f3af000    11 MiB kernel+modules        <- kernel
mmap: 0x07f3af000-0x07ff53000    11 MiB bootloader-reclaimable
mmap: 0x0b0000000-0x0c0000000   256 MiB reserved
mmap: 0x0fd000000-0x0fd3e8000     3 MiB framebuffer
mmap: 0x100000000-0x1d9a9d000  3482 MiB usable                <- arena: 2B 406 + embedder 113 + pool 1024 MiB
mmap: 0x1d9a9d000-0x200000000   613 MiB kernel+modules        <- embedder (phys 0x1d9a9d000)
!mem: RAM total 6143 MiB, free 2039 MiB, used 4104 MiB (model weights 2535 MiB, LLM engine arena 520 MiB,
      state snapshot pool 1024 MiB, kernel and firmware 25 MiB)
```

UEFI, 6 GiB, places them the other way round (embedder low at 0x5259e000, the 2B at the top, 0x187da2000), with
a 2173 MiB arena run: the pool still gets 1 GiB, about 630 MiB stay free. The kernel picks the largest usable
run either way, so neither layout needs code.

## Implementation notes

- `kernel/src/embed.rs` (~150 lines): `init` runs `model_init` on the embedder module after the 2B's and keeps
  both 640-byte contexts; `with_embedder` is the only place the embedder's context is loaded. `matmul_set_fp16`
  is set again after each swap (it is a global of matmul.asm, outside `g_ctx`). The tokenizer pointers live in
  `g_ctx`, so the embedder's own BPE is used automatically; `llm::encode_plain` is the pre-split + BPE without
  the 2B's special-token table (whose ids are outside the embedder's vocabulary).
- The pooled vector skips the final RMSNorm's scalar and the lm_head: `normalize(x * w * r) = normalize(x * w)`.
  So `transformer_forward(.., compute_logits 0)` for every token, no 151669 x 1024 matmul per embedding.
- `kernel/src/memidx.rs` (~410 lines): rows, texts and BM25 scratch are static (`.bss`, about 6 MiB). Q8_0
  quantization, fp16 conversion, `ln` and `sqrt` are written for the soft-float target (no libm, no F16C in Rust);
  a boot self-test checks them. `tp_matmul_q8_0` is called with `matmul_set_fp16(1)` and the loaded model's mode
  put back after. The scores are W8A8 cosines (query quantized by the engine, rows Q8_0).
- `search_memory` runs the embedder between two 2B forward calls of the same turn (tool round). The 2B state is
  untouched, and `/retry` reproduced such a turn exactly at temp 0.
- `tools/retrieval.py` vendors the spike's data generator with two names replaced (person, client); the gold
  ids and the queries are unchanged.

## Not done or not tested

- Incremental and idle-time indexing (spike plan step 6): `/index add` embeds synchronously (~0.1-0.6 s); not
  interruptible by Ctrl-C (an add of a 1023-token text takes about 6 s).
- Persistence of the index (RAM only, lost at reboot) and dedup of identical texts.
- `compute_logits == 2` (norm only) and the instruction-prefix KV cache (about 27 of a query's ~34 tokens are the
  prefix; caching it would cut a query to ~50 ms). The reranker.
- Search time at 4096 entries (measured only at 40: 43-71 us); the spike's estimate is < 5 ms for 10k rows.
- Texts with non-Latin scripts (the pre-tokenizer treats every non-ASCII byte as a letter; HF NFC normalization
  is not applied).
- Tab completion of `/index` subcommand arguments beyond the subcommand name.

## Risks for the next milestones (M4 virtio-blk + filesystem tools, M4.5 agentd persistent memory)

- **Index persistence format**: the rows are only valid for one embedder export and one dim. A saved index needs
  a header with the embedder file hash (sha256 above), the dim and the Q8_0 layout, and must be rebuilt (re-embed
  from the stored texts) when any of them changes.
- **Indexing cost**: 0.61 s per 100 tokens on the BSP + 7 workers, and it blocks the shell. Files from M4 (a
  1 MiB text is ~250k tokens, about 25 min) need chunking and idle-time indexing with Ctrl-C/pre-emption between
  forward calls, never inside one; and the swap must stay between whole forward calls when an idle indexer
  interleaves with agent turns.
- **Capacity**: 4096 entries, 1 MiB of text, 2 KiB per text, all static. agentd memory and file chunks will need
  the arena (or a real allocator) and eviction; at 10k rows of 1024 dims the rows are 10.4 MiB, the BM25 scan is
  linear in the text bytes (fine at 1 MiB, not at 100 MiB: an inverted index would be needed).
- **Gate tuning**: all-caps words (CPU) count as identifiers; with more memories, a strong-keyword gate can push
  a lexically similar but wrong note up. The rule is tuned on 40 synthetic notes only.
- **RAM**: 6 GiB guests from now on. A virtio-blk disk cache, file buffers and a larger index come out of the
  ~2 GiB left (BIOS) or ~630 MiB (UEFI layout). The UEFI layout is the tighter one.
- **Tool prompt growth**: every tool adds to the boot prefix (743 -> 832 tokens, +1.7 s of boot prefill). The M4
  filesystem tools will add more; keep descriptions short or cache the prefix on disk once M4 exists.
- **Determinism**: replay of thinned points depends on the 2B only; the embedder never writes the 2B state, but an
  idle indexer that changes the core count between a thin and its replay would still break the R hash check.
