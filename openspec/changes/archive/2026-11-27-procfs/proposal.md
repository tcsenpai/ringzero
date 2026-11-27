## Why
ringzero presents itself as a Unix (design principle, 2026-11-27): the models already know Linux and look for
system state in files. In the maintainer's manual test the 4B asked "which drivers do we have here?" and had no file
to read. A minimal /proc gives the shell, the agent's ls and cat tools and WASI programs one place for it.

## What Changes
- A read-only virtual /proc in the kernel fs layer (kernel/src/procfs.rs, routed by fs.rs), generated at read time:
  cpuinfo (one block per core, flags from CPUID), meminfo (kB), uptime, version, devices (the active drivers) and
  bus/pci/devices (PCI bus 0). Familiar names, plain `key: value` content, not byte-exact Linux formats.
- `ls /` lists `proc/`. Writes, mkdir, rm, rename and truncate under /proc fail with `read-only file system`;
  WASI programs get EROFS. The file indexer does not walk /proc.
- WASI: opening a /proc file reads it once into the fd; read, seek and fstat work on those bytes.
- The ls and cat tool descriptions name /proc (the boot prefix changes).

## Impact
- kernel/src/procfs.rs (new), fs.rs, wasm.rs, blk.rs (location), console.rs (devices), main.rs (selftest, pub
  version consts), agent.rs (ls, cat descriptions).
- Specs: new capability proc.
- Docs: README, INFO, docs/ARCHITECTURE.md + PDF, docs/MANUAL.md (new section), site.
