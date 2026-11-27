## 1. Engine
- [x] 1.1 pf_ms row table in transformer_prefill (stream fields per row, restored on return, no clear at pos 0)

## 2. Kernel
- [x] 2.1 multi.rs: slots, fork, step, the background stream, `/multi`
- [x] 2.2 llm.rs: logits rows for a verify plus the other streams; generate passes carry the background row
- [x] 2.3 agentd.rs: idle consolidation on the background stream, input does not stop it; /consolidate keeps the M4.5 path
- [x] 2.4 shell.rs: `/multi [N] [T]` and `/consolidate &` in the command table and /help

## 3. Checks
- [x] 3.1 `/multi 1..4` on the 4B q4 and the 2B q8, 8 vCPU: tok/s, ids and logits the same
- [x] 3.2 consolidation on the background stream vs /consolidate: same output ids
- [x] 3.3 gates: boot self tests, rztest, tooleval (M45 protocol), `/spec`, hibernate and resume

## 4. Docs
- [x] 4.1 INFO.md, README.md, docs/ARCHITECTURE.md (+ tools/arch_pdf.sh), docs/MANUAL.md, site, docs/M48A_RESULT.md
