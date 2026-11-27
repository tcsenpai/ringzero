# M4.8c result: precomputed document states (T27)

Goal (task text): store the prefill state of a file on the disk, like the snapshots, and load it instead of
prefilling; invalidate it on change. Done when reading a cached 4k-token file takes under 1 s, against minutes of
prefill.

Test box: the shared Linux build box (AMD, KVM), 8 vCPUs, QEMU q35, BIOS boot, virtio-blk data disk of 2 GiB
(FAT32, 4 KiB clusters, made with mkfs.vfat). 2B q8 in 6G guests, 4B q4 in 8G guests. Other agents ran VMs on the
same box between these runs, so read the times as measured on a shared machine.

## What "the state of a file" is on this model

Qwen3.5 is hybrid: 3 of every 4 layers are Gated DeltaNet (a recurrent state per head plus a short conv history,
one block R of 19.3 MiB on the 2B and 50.3 MiB on the 4B), every 4th is full attention (K/V rows per position). A
token's state depends on every token before it: the DeltaNet state directly, the attention rows through the
hidden states of the layers below. So the state of a file's tokens cannot be moved to another position or put
after another context. It is exact only at the same position after the same tokens. Then it is fully determined:
the engine is deterministic (the batched prefill, the decode and the verify pass leave bit-identical state, M4.8,
M4.8b), so the same token ids at the same positions from the same state give the same rows and the same R.

## Design

- `/read PATH` (new session command, kernel/src/docstate.rs) puts a text file of at most 64 KiB into the
  conversation: a user turn `Here is the file PATH (N bytes). Read it, my questions about it follow.` plus the
  file, encoded without special tokens (a `<|im_end|>` in the file stays text), and an assistant turn with the
  fixed text `I have read PATH.`. In a fresh conversation the 2 tokens that close the system turn come first, as
  for a question. The result is a snapshot point (`/tree` label `/read PATH`): undo, branches and hibernation work
  on it as on any turn.
- The known prefix point is the context before the file, whatever it is. Key = FNV over the boot key (model,
  boot prefix ids, engine build, KV geometry) and the token ids from the end of the boot prefix to the file's
  position, plus FNV over the file's own token ids. The first `/read` of a fresh conversation (after boot or
  `/clear`) always has the same context, so it always hits once the entry exists. A second `/read` in the same
  conversation hits only when the conversation before it repeats token for token.
- Cache file `/var/lib/ringzero/cache/docs/<16 hex>.bin`: 336-byte header (magic `RZDS`, version, boot key,
  context key, document key, base position, token count, bytes per row, planes, R bytes, word-wise FNV of the
  payload, the path), then the token ids, rows [base, base + n) of every K/V plane, and R. Only for files of 512
  tokens or more (a shorter one would be mostly R).
- Load: header, keys, geometry and size are checked before anything is read; the token ids are compared with the
  ones just encoded; rows and R are read straight into the live cache; the checksum is compared last. A refused
  entry is deleted, the current point is restored when R was already overwritten, and the file is prefilled.
- Invalidation: a changed file has other token ids, so another key; writing the new entry deletes the older entry
  of the same path, context and boot key.
- Cap 512 MiB for the directory, 64 MiB left free on the disk. Before a write, entries of another model or engine
  go first, then the least recently used (a hit rewrites the magic, which refreshes the mtime). An entry larger
  than the cap, or one that cannot fit, is not written; the read still works from the prefill.
- blk.rs read-ahead: fatfs reads a file one cluster per request (4 KiB on these volumes). A direct read that
  continues the previous one now fills a 1 MiB buffer in 64 KiB requests and the next reads copy from it; any
  write over its sectors drops it. Without it the 4B entry (185 MiB) loaded in 1.36-1.71 s.

## Measurements

Test file: a generated 14617-byte operations handbook (`/docs/handbook.txt`), 4325 tokens in the `/read` turn
(4323 when it is not the first turn of the conversation). Question after each read, temperature 0: "What is the
station code word, and who calibrates generator number 23? Answer in one sentence."

| | 2B q8 | 4B q4 |
|---|---|---|
| entry size | 69 MiB | 185 MiB |
| uncached: prefill of the 4325 tokens | 22.9-28.4 s (152-188 tok/s), 3 runs | 71.7 s (60 tok/s) |
| writing the entry after the prefill | 2.8-5.4 s | 11.8 s |
| cached, before the read-ahead: load | 0.28-0.50 s, 5 runs | 1.36-1.71 s, 2 runs |
| **cached, with the read-ahead: load** | **0.121-0.182 s**, 2 runs | **0.564-0.761 s**, 2 runs |
| cached, with the read-ahead: whole `/read` (load, commit, output) | 0.143-0.204 s | 0.621-0.821 s |
| snapshot point of the read (commit) | 6-33 ms | 10-37 ms |

## Bit-exact evidence

`/snap` prints FNV over the token ids, the KV rows and R of the live state; the stats line after an answer prints
FNV of the generated ids.

| run | read | state hash | answer ids |
|---|---|---|---|
| 2B, 4325-token file at ctx 1650, prefill | miss | 0x4fc24397a59a2629 | 0x96a1daa4f884c2a3 |
| 2B, same boot after `/clear` | hit | 0x4fc24397a59a2629 | 0x96a1daa4f884c2a3 |
| 2B, next boot, entry with a flipped byte | refused, prefill | 0x4fc24397a59a2629 | same answer text |
| 2B, after a hibernation resume and `/clear` | hit | 0x4fc24397a59a2629 | same answer text |
| 2B, next boot, entry cut by 4096 bytes | refused, prefill | 0x4fc24397a59a2629 | |
| 2B, second read of the file at ctx 5975, prefill | miss | 0x4d4a56d19d7235bc | |
| 2B, the same two reads after `/clear` | hit, hit | 0x4d4a56d19d7235bc | |
| 4B, 4325-token file at ctx 1650, prefill | miss | 0x887be5b595e069e7 | 0x83302b4dd552e92d |
| 4B, same boot after `/clear` | hit | 0x887be5b595e069e7 | 0x83302b4dd552e92d |
| 4B, next boot | hit | 0x887be5b595e069e7 | 0x83302b4dd552e92d |
| 4B, read-ahead build, after `/clear` | hit | 0x887be5b595e069e7 | 0x83302b4dd552e92d |
| 2B, read-ahead build, after `/clear` | hit | 0x4fc24397a59a2629 | 0x96a1daa4f884c2a3 |

Answers: 2B "The station code word is KESTREL-41, and Jonas Reed calibrates generator number 23." in every run;
4B "The station code word is KESTREL-41, and no one calibrates generator number 23 according to the handbook." in
every run. The same answer before and after the cache is what the test checks, not whether it is right.

An earlier 2B run with a 17060-byte version of the file (5035 tokens): prefill 31.1 s, hit 0.341 s (78 MiB), the
same state hash 0x8914dfacdc330e8d and answer ids 0x96a1daa4f884c2a3 after the miss and the hit.

## Refusals and invalidation

| case | what happened |
|---|---|
| file changed (the 17060-byte version replaced by the 14617-byte one) | `doc: /docs/handbook.txt changed, old cache entry .../25248b8937bf486b.bin removed`, prefill, new entry |
| one byte flipped 1000 bytes before the end (inside R) | `refused (checksum mismatch), deleted; prefilling`, the current point restored, same state hash as a prefill |
| entry cut by 4096 bytes | `refused (size does not match the header), deleted; prefilling`, then state hash 0x4fc24397a59a2629 as a prefill |
| other context (the same file after a question) | other key, prefill, new entry; the earlier entries stay |

The read-ahead also speeds up the other big reads: the boot snapshot cache hit went from 0.220-0.291 s (2B, 4
boots) to 0.094-0.101 s (2 boots) and from 0.747-0.863 s (4B, 2 boots) to 0.390 s (1 boot); the 4B hibernation
resume of the same 3-point image from 2614 ms to 1768 ms.

## Regression gates

The final ISOs (with the read-ahead and the new manual section, manual vectors rebuilt):

| gate | result |
|---|---|
| boot self-test | `SELFTEST PASS` on every 2B boot; the 4B boots print `SELFTEST n/a` (the golden ids are the 2B's), as on main |
| rztest (`!run rztest check A "B C"`, beta data disk copy) | `rztest: PASS 20/20` |
| speculative verifier (`/spec`) | 2B: logits bit-identical 16/16, kv+deltanet hash the same; 4B: 16/16, the same |
| hibernate and resume | 2B: 4 points with two document reads, `resumed ... (same as saved)` 0xe953b69d482277d5 on 3 boots; 4B: 3 points, 0xd657d7b59cb65a9c the same; beta disk copy: the same |
| T55 layout | layout.rs not changed; the beta disk copy booted with `layout: new disk`, `!ls /var/lib/ringzero` shows mem/, layout, bootsnap.bin, hib/ |
| tooleval, M45 protocol (4B q4 8G, `SLOT=200`, constrain on, confidence on, temperature 0, fresh beta disk copy, 120 s idle) | spike 24/30 correct, 22 first call right, 9 args exact; in-OS 26/32 correct, 24 first call right, info 5/8; calls 42 + 36; malformed 0, schema errors 0, grammar lines not valid 1 (spike) + 0; round limit 1 + 1 |

Main (docs/T55_RESULT.md, last table): spike 24/30, 22, 9; in-OS 26/32, 24, info 5/8; calls 42 + 36. The same
counts. The agent's turns do not touch the new code (no prompt, tool or decoding change).


## Limits

- An entry is valid only at its own position after its own context. The agent's `cat` tool is not cached: its
  result enters mid-turn after the model's own call, so the context before it differs on almost every turn.
  `/read` is the way in for a long file.
- Writing an entry costs seconds (fatfs writes 4 KiB clusters one request each: 2.8-5.4 s for 69 MiB, 11.8 s for
  185 MiB), on top of the first prefill.
- An entry is big: about 12 KiB (2B) or 32 KiB (4B) per token plus R. With the 512 MiB cap the 4B keeps 2 entries
  of a 4k-token file; the default 512 MiB data disk may not have the room next to the boot cache and a hibernation
  image, and then the read is not cached (the log says why).
- The read turn has a fixed English reply and is not one of the exchanges a compaction keeps: after a compaction
  the file is out of the context unless the summary mentions it.
- The key trusts the engine's determinism: if a future engine change made the decode and the prefill differ, the
  engine hash in the boot key changes and every entry misses.
