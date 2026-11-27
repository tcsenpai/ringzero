## Why
Prompt, agent answers, tool lines, memory lines, statistics and questions all look the same on the console, so a
turn is hard to read (T46). The logo palette gives each role a color.

## What Changes
- Role colors on both consoles (ANSI SGR on serial, the framebuffer's SGR parser on VNC and the QEMU window):
  prompt bold red, user input and agent answers default, `[tool]` lines gold (yellow), `[memory]`, `[grammar]`,
  `[confidence]`, `[prefill]`/`[decode]`/`spec` statistics and `/spec`, `/prefill` bench lines gray, y/N questions
  and errors (refused settings, unknown commands, file errors) bright red.
- New setting `!set color on|off` (default on), listed in /help and Tab completion, saved in the hibernation
  image (image extra version 2; a version 1 image still resumes, with color on). Off: plain text. The escapes
  that WASI programs print themselves pass through in both cases.
- Test drivers (tools/accept.py, boottest.py, m8test.py, tooleval.py, uxtest.py) strip SGR codes before matching.

## Impact
- kernel/src/console.rs, shell.rs, agent.rs, gram.rs, llm.rs, hib.rs; tools/*.py drivers.
- Specs: console (role colors), settings (color key, hibernation), snapshots-hibernation (image contents).
