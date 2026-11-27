## Why
In the maintainer's manual test of the beta build (T54) the agent spent its 4 tool rounds on "which drivers do we
have here?" and the turn ended without an answer: the loop printed `[agent] tool round limit (4) reached` and
stopped. A question should always end with an answer, and 4 rounds is short for a model that explores the disk.

## What Changes
- Default 10 tool rounds per question (was 4), set at build time with `make iso ROUNDS=N` (RINGZERO_ROUNDS,
  1..20, clamped) and at run time with `!set rounds N` (an integer from 1 to 20, anything else refused).
- After the last tool round the shell prints `[agent] tool round limit (N) reached, answering without tools`, the
  last tool result carries the line `(No more tools in this turn: answer now with what you have; if you could not
  find out, say so.)`, and the model generates one more time with the `<tool_call>` token blocked in the sampler,
  so the turn ends with an answer.
- The rounds value is saved in the hibernation image (extra version 3; version 1 and 2 images still resume).

## Impact
- kernel/src/agent.rs ROUNDS, env_num, ask; kernel/src/llm.rs NO_TOOLS, pick_next; kernel/src/shell.rs SETTINGS,
  set_setting; kernel/src/hib.rs; Makefile; kernel/build.rs.
- Specs: agent-tools (round caps), settings (keys, rounds, hibernation), build-options (ROUNDS, CONF_ASK message,
  rebuild list).
- Docs: INFO.md, docs/MANUAL.md (Settings, new Tool round limit section), docs/ARCHITECTURE.md, docs/DEMOS.md, /help.
