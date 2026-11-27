## Why
In the maintainer's manual test of the beta build (T54) the confidence check asked y/N on 7 of 8 calls: run_code and
run_wasm had margins of 0.56 to 0.70, split between `run` and `ls` or between `_code` and `_w`. Those splits are
about the tool name's tokens, not doubt about the action, and a question on almost every program run makes the
check useless. User decision: keep the ask threshold at 0.80, but ask only before the tools that write.

## What Changes
- Below CONF_LO (default 0.80, unchanged) only write_file, mkdir, set_setting and remember ask
  `[confidence] the agent is not sure about this call. Run it? [y/N]`.
- run_code and run_wasm no longer ask: below the line they run with the notice in the `[confidence]` line, like
  the read-only tools. The notice reads `run (low-confidence notice: check the result)` for every tool that runs.
- rm, forget, poweroff and reboot keep their own y/N, as before. After a declined call every later call of a tool
  that changes the machine in the same turn still asks (F7, unchanged).
- `!set confidence on` names the tools that ask.

## Impact
- kernel/src/agent.rs ASK_LOW, conf_line, ask; kernel/src/shell.rs set_setting.
- Specs: agent-tools (low-confidence calls), decoding (action per band), settings (confidence).
- Docs: docs/T28_RESULT.md (beta behaviour), INFO.md, docs/MANUAL.md, docs/ARCHITECTURE.md.
