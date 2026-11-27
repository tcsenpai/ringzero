## Why
Decode is bandwidth bound: every step reads all the weights (about 2.4 GB on the 4B q4) for one token. The batched
prefill of M4.8 and the verify pass of M4.8b already read the weights once for several rows of one conversation.
Rows of different conversations can share the same pass, so a background job of the agent (consolidation of the
session into an episode and facts) can decode next to the user's chat for little extra time, instead of only in
idle time on top of the user's snapshot point and stopping at the first keystroke.

## What Changes
- Engine: `transformer_prefill` takes an optional per-row stream table (`pf_ms`): each row runs at its own position
  on its own KV cache and DeltaNet state. Rows of one stream behave as the batched prefill does today; rows of
  different streams share every weight read. Every row's logits stay bit-identical to running that stream alone.
- Kernel: up to 3 extra stream slots (4096 positions each) next to the agent's context, open only while used, with
  memory lent by the snapshot pool (a contiguous run of free chunks, at least 256 chunks left free) and given back
  after. Nothing is set aside at boot (the 4B q4 in an 8G guest has no arena left after the pool). Not saved in
  snapshots or the hibernation image.
- agentd: an idle consolidation forks the conversation at the current point into an extra stream. It decodes
  alone in idle slices and one token per foreground decode step in the same engine pass; input at the prompt no
  longer stops it. `/consolidate` and a conversation that does not fit in the slot keep the M4.5 path (on top of
  the current point, restored after). `/consolidate &` starts the background one at once. The finish line in
  `/agentd log` names the path and the ids hash.
- New debug command `/multi [N] [T]`: N streams (default all) decoded alone and together, T tokens each (64):
  aggregate tok/s and, per stream, whether the ids and every step's logits are the same.

## Non-goals
- No scheduler for more than one background job, no user-facing multi-conversation (the extra streams are not
  selectable from the shell), no sampling on the background stream (consolidation is greedy).
- The hibernation image and the boot snapshot cache keep their format; the main context's KV geometry does not
  change.

## Impact
- Code: engine/prefill.asm (pf_ms rows), kernel/src/multi.rs (new), kernel/src/llm.rs (logits rows, generate passes
  with the background row), kernel/src/agentd.rs (consolidation on the background stream), kernel/src/shell.rs
  (`/multi`), kernel/src/snap.rs (`lend`, `take_back`), kernel/src/engine.rs, kernel/src/main.rs.
- Engine source hash changes: the boot snapshot cache misses once and an older hibernation image is not resumed.
- Docs: INFO.md, docs/ARCHITECTURE.md, docs/MANUAL.md, README.md, docs/M48A_RESULT.md, site.
