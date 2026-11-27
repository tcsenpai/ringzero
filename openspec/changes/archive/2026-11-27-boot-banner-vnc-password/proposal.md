## Why
The beta needs a boot screen that says what ringzero is (name, version, edition, models, machine) on both
consoles, in the logo palette, and the default VNC password still carries the old project name (T47).

## What Changes
- New boot banner on serial and the framebuffer: the R0 die and the name as ASCII art, then version
  `0.1.0-beta`, edition, models with their sizes and the machine (cores, RAM, firmware, framebuffer), 14 lines,
  at most 70 columns. It replaces the three `====` lines; the kernel log keeps one plain line.
- The framebuffer reads ANSI SGR (colors, bold, dim, 256 and 24-bit colors) instead of skipping every escape, so
  the banner and the escapes of WASI programs look the same as on a serial terminal.
- Kernel version comes from kernel/Cargo.toml (0.1.0-beta).
- Default `VNC_PASS` of run-remote: ringzero (was aios). The "AIOS" markers in the vendored engine sources
  become "ringzero" (comments only).

## Impact
- kernel/src/main.rs, kernel/src/console.rs, kernel/Cargo.toml, Makefile, engine/*.asm and constants.inc
  (comments only; the engine build hash changes, so the first boot of a new build prefills the system prompt
  once and an older hibernation image is not resumed).
- Specs: new capability console; build-options gains VNC_PASS.
