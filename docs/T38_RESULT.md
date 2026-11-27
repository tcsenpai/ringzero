# T38 result: manual vectors baked into the kernel (2026-11-27)

T22 indexed the built-in manual in idle time at the first boot (26 sections, about 20 s of embedding). T38 links
the sections' vectors into the kernel, so a fresh disk has the manual in its memory index at boot with no
embedder work. Build flag `MANUAL_VECTORS=1` (default); `MANUAL_VECTORS=0` builds the T22 behaviour. Runs: Fedora
box, QEMU/KVM, 4B q4 build, 8G guests, next to the user's running instance.

## How

| Part | Where | What |
|---|---|---|
| Dump | agentd.rs `man_vectors`, `/man dump` | Embeds every section with `embed::embed` (begin + resume, the idle indexer's path) and quantizes it with `memidx::row_of` (the same `quantize` that `memidx::store` uses). Prints a 28-byte header + 26 rows of 1088 bytes as `MV <hex>` lines, then `MV end <fnv> <bytes>`. |
| Pass 1 | Makefile `$(MANVEC)`, tools/manvec.py | `make iso MANUAL_VECTORS=0 ISO=build/manvec-pass1.iso`, then `tools/manvec.py` boots it with `tools/boottest.py` (headless, no disk, `MEM=$(GUEST_MEM)`), types `/man dump`, checks size and FNV-1a, writes `build/manvec.bin` (28316 bytes). Rebuilt only when docs/MANUAL.md, the embedder file, the engine sources, embed.rs, memidx.rs or llm.rs (tokenizer pre-split) are newer. |
| Pass 2 | build.rs, agentd.rs `BAKED` | `RINGZERO_MANVEC=build/manvec.bin cargo build`: build.rs copies the file to `OUT_DIR/manvec.bin` (an empty file when the variable is empty), agentd.rs links it with `include_bytes!`. |
| Key | agentd.rs `man_key`, `mv_head` | FNV-1a over every section text (with its Asked-as line), the embedder sample hash, the engine build hash (asm sources + KV flag), the fidx chunker version and the index dim; header = `RZMV`, version, key, n, dim, row bytes. |
| Boot | agentd.rs `queue_manual`, memidx.rs `store_row` | A missing section is stored with its linked row when the header matches; otherwise it is queued as in T22. The decision line says which, and why the vectors were not used. |
| Check | `/man check` | Embeds the 26 sections again and compares each row with the linked one. |

## Measurements

| Check | Result |
|---|---|
| fresh disk, `MANUAL_VECTORS=1` | `manual: 26 sections, 0 already in the index, 26 from the vectors in the kernel (no embedder work), 0 queued`; `/index stats`: 26 entries, embed 0 ms/entry, 0 tokens; index saved at the first idle slice (26 entries in 10 ms); `/man come faccio a ibernare?` -> Hibernation (0.67) right after the boot |
| byte-identical to boot-time vectors | `/man check` on that kernel: **26/26 rows byte-identical** to a fresh embedding (20.0 s for the 26). Independent check: the 26 manual rows that the T22 kernel embedded in idle time on the user's disk (`/sys/mem/index.bin`, read on the host) are byte-identical to `build/manvec.bin`, 26 of 26 |
| fresh disk, `MANUAL_VECTORS=0` | `0 from the vectors in the kernel ..., 26 queued (vectors in the kernel not used: not built in (MANUAL_VECTORS=0))`; `/man check`: "not built in" |
| key mismatch (one key byte flipped in a copy of the blob, `MANVEC=build/manvec-bad.bin`) | `26 queued (vectors in the kernel not used: key differs (manual, embedder, engine, chunker or dim changed))`: the T22 path |
| pass 1 reproducible | `build/manvec.bin` deleted and rebuilt: identical bytes (`cmp`), fnv 10d2a72c3419d4f1 |
| build cost | `make iso MODEL=4b QUANT=q4` with pass 1: 88 s (pass-1 ISO, a 4B boot with a prefill, 26 embeddings, pass-2 kernel and ISO); the next `make iso` with the file in place: 3 s |
| kernel size | +28316 bytes of vectors (on top of the 12.3 KB manual text) |
| boot | the boot prefix and its cache key are unchanged; the 20 s of idle embedding after the first boot are gone |

## Limits

- Pass 1 needs QEMU/KVM and the guest RAM of the model being built (8G for the 4B) on the build machine; a box
  without KVM builds with `MANUAL_VECTORS=0`.
- The make dependencies list the files the vectors depend on (manual, embedder file, engine, embed.rs,
  memidx.rs, llm.rs); the key at boot covers the manual, embedder, engine, chunker and dim, not the Rust code.
  A Rust change elsewhere that altered the embedding would not be caught at boot; `/man check` shows such a
  drift. Every llm.rs change reruns pass 1.
- Changing `/index dim` changes the key: the linked rows are not used and the sections are embedded at that dim.
