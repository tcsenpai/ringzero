## Why
The design rule (user, 2026-11-27): ringzero presents itself as a Unix, because the models already know Linux.
T54 added a read-only /proc. The rest of the tree is not Unix yet: agentd's stores, the hibernation image and the
boot snapshot cache sit in `/sys` on the FAT volume (a Linux path that means something else), there is no /dev
(except `/dev/tty` for editors), no /etc and no /tmp, and five agent tools (meminfo, cpuinfo, list_cores, uptime,
read_log) exist only because the state was not in files. The 2B model reads `cat /proc/meminfo` or
`/sys/devices/system/cpu/online` without being taught, and every tool in the prompt costs boot prefix tokens.

## What Changes
- One path layer (fs.rs) routes a small mount table: `/proc` (virtual, read-only, T54), `/sys` (new, virtual
  sysfs-like, read-only), `/dev` (new, virtual devices), two virtual files inside a real `/etc`, and the FAT
  volume for everything else. The shell, the agent's ls/cat tools and every WASI program (cat, ls, head, wc,
  python3, ...) see the same tree.
- /proc gains `kmsg` (the kernel log ring) and `mounts`; `uptime` gains a readable form after the seconds;
  `cpuinfo` prints the flags once instead of once per core.
- /sys: `class/block/vda/{size,ro,device}`, `bus/pci/devices/<addr>/{vendor,device,class,driver}`,
  `devices/system/cpu/{online,possible,inference,cpuN/role}`, `power/state`.
- /dev: `null`, `zero`, `random`, `urandom`, `tty`, `console`, `vda` (raw disk, read-only).
- /etc (real FAT directory): `hostname` (writable, default `ringzero`), plus virtual read-only `os-release` and
  `passwd` (one root line).
- /tmp (real FAT directory): scratch, emptied at a fresh boot, kept when a hibernation image resumes, not indexed.
- **BREAKING (disk layout)**: ringzero's own data moves to `/var/lib/ringzero/` (`mem/*.log`, `mem/index.bin`,
  `hib/image.bin`, `bootsnap.bin`). The first boot of the new kernel migrates an old disk automatically and
  idempotently (copy, verify, commit marker, remove); an older kernel booted on a migrated disk sees an empty
  memory (documented, not prevented).
- `/var/lib/ringzero` is read-only for WASI programs (EACCES, as `/sys` is today) and for the agent's
  write_file, mkdir and rm (today the agent can overwrite `/sys/mem/facts.log`).
- Tools: meminfo, cpuinfo, list_cores and uptime leave the prompt and stay as hidden aliases (a call still works);
  read_log stays. Gated by a tooleval A/B before the apply. The system prompt and the ls/cat descriptions name
  the tree in one sentence each.

## Non-goals
- No tmpfs in RAM, no symlinks, no permissions model beyond the fixed rules above, no writable /sys or /proc.
- Settings as files (`/proc/sys/...`) and Linux-like shell output (`!ls -l` columns) are follow-ups.

## Impact
- Code: kernel/src/fs.rs (mount routing, check_rw), procfs.rs, new sysfs.rs, devfs.rs, layout.rs (migration,
  /tmp clean, /etc defaults), agentd.rs, mstore.rs, hib.rs, agent.rs (BOOT_SNAP, TOOLS, SPECS, SYSTEM, aliases,
  fs_tool guard), wasm.rs (device fds, sys_ro), fidx/agentd walk exclusions, shell.rs (help text),
  tools/wasm/rztest, tools/tooleval.py.
- Specs: proc, memory, snapshots-hibernation, programs, agent-tools (modified); unix-tree (new).
- Docs: README.md, INFO.md (layout table), docs/MANUAL.md, docs/ARCHITECTURE.md (+ tools/arch_pdf.sh),
  docs/BETA.md, site/index.html, release notes (upgrade note: hibernation state does not survive this upgrade),
  /help; a docs/T55_RESULT.md with the tooleval numbers.
