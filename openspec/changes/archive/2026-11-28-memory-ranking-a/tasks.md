## 1. Kernel
- [x] 1.1 memidx.rs: a date per entry, index version 2 with the version 1 load, hub scores from the probe rows, the search with a penalty (raw cosine kept)
- [x] 1.2 memrank.rs: probes, hub, owner rule (relation words, alias table), the frozen weights
- [x] 1.3 agentd.rs: dates through the queues, the migration, ranked answer (supersession, manual first), recall and slot on it, probes baked with the manual vectors, index.tsv
- [x] 1.4 fidx.rs: tables by rows (chunker version 2, version 1 tables cut again)
- [x] 1.5 shell.rs: `/index add [-d DATE]` with escapes, `/search` prints the ranked answer

## 2. Checks
- [x] 2.1 host: frozen A with the kernel's semantics (`memeval.py ported`)
- [x] 2.2 in the OS: the T69 set through the kernel (`memeval.py inos`), against the host's 69.3% held-out
- [x] 2.3 boot self tests on the 2B and the 4B, rztest, an upgrade from a 0.1.1 disk, tooleval on the 4B q4

## 3. Docs
- [x] 3.1 docs/MANUAL.md, INFO.md, docs/ARCHITECTURE.md, specs, docs/M_MEMA_RESULT.md, tools/surfaces_check.sh
