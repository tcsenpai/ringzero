## Why
In the maintainer's manual test of the beta build (T54) the agent answered a question about the machine with
run_code and `import os`, twice: MicroPython in ringzero has no os module, no filesystem and no network, and the
tool description did not say so.

## What Changes
- run_code's description: "Run Python code in an isolated MicroPython sandbox and return what it prints: no os
  module, no filesystem, no network, no imports from disk. For files and programs use ls, cat, write_file or
  run_wasm."
- One sentence in the system prompt: "run_code is a MicroPython sandbox without files: use ls, cat and run_wasm for
  files and programs." The boot prefix changes (a new boot snapshot cache key).

## Impact
- kernel/src/agent.rs SYSTEM, TOOLS.
- Specs: agent-tools (system prompt).
- Docs: INFO.md.
