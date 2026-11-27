## 1. Code
- [x] 1.1 docstate.rs: `/read PATH`, tokens (head, file without special tokens, fixed reply), snapshot point
- [x] 1.2 cache file: header (keys, geometry, checksum, path), token ids, KV rows, R; load with every check before R
- [x] 1.3 invalidation of the old entry of a changed file, size cap, LRU eviction, disk reserve
- [x] 1.4 `/read` (list) and `/read clear`, log lines, /help entry
- [x] 1.5 snap.rs state_parts and geometry, fs.rs free_bytes

## 2. Checks
- [x] 2.1 bit-exact: state hash and greedy answer after a hit equal those after the prefill (2B and 4B)
- [x] 2.2 timings of a 4k-token file, cached and uncached, 2B and 4B
- [x] 2.3 corrupt, truncated and stale entries refused; regression gates

## 3. Docs
- [x] 3.1 README.md, INFO.md, docs/MANUAL.md, docs/ARCHITECTURE.md, site, docs/M48C_RESULT.md
