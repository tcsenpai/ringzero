## ADDED Requirements

### Requirement: Multi-stream check command
`/multi [N] [T]` SHALL run N streams (all streams by default) of T greedy tokens each (64 by default, EOS ignored), first every stream alone on the agent's context and then all of them together, one row per stream per pass, and SHALL print the tok/s of both and, per stream, whether the ids and every step's logits are the same.
Source: kernel/src/shell.rs CMDS, run; kernel/src/multi.rs bench

#### Scenario: multi-stream check
- **WHEN** the user types `/multi 2`
- **THEN** the shell prints `[multi] 2 streams: the agent's context + 1 slots of 4096 positions lent by the snapshot pool (M MiB each)`, `[multi] 2 streams x 64 tokens: alone A tok/s per stream; together B tok/s aggregate (Rx), C tok/s per stream` and one `[multi] stream S: prompt P tok, ids same (fnv 0x...), logits same` line per stream
