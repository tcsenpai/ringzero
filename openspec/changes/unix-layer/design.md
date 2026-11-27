## Context
T54 put a virtual read-only /proc in front of the FAT volume (procfs.rs, routed by fs.rs `is_proc`, `check_rw`,
`list`, `entry`, `read`). agentd (mstore.rs, agentd.rs), hibernation (hib.rs) and the boot snapshot cache
(agent.rs `BOOT_SNAP`) keep their files under `/sys` on the FAT volume; WASI refuses writes there (wasm.rs
`sys_ro`, EACCES) but the agent's write_file, mkdir and rm do not. `/dev/tty` exists only as a special case in
wasm.rs `path_open`. The boot prefix is 1571 tokens (system prompt + 18 tools, docs/T54_RESULT.md).

Rule applied everywhere below: familiar Linux paths and names, content in clear `key: value` or one value per
file (what sysfs does), not byte-exact Linux formats.

## Goals / Non-Goals
Goals: a Unix tree every reader sees the same way; ringzero data off Linux paths with an automatic, crash-safe
migration; fewer tools in the prompt where a file does the job. Non-goals: tmpfs, symlinks, users and modes,
writable /proc, a writable /sys beyond /sys/power/state, settings as files (`/proc/sys`, a later change), Linux-like
`!ls` output.

## Decisions

### D1. One mount table in fs.rs
`fs::check` stays the single path check. A new `fs::mount_of(rel)` returns `Proc | Sys | Dev | Etc(virtual name)
| Fat`; `read`, `entry`, `list`, `ls`, `open` and every write path branch on it, like `is_proc` today. `list("/")`
adds `proc/`, `sys/`, `dev/` (virtual) to the FAT entries; `etc/`, `tmp/`, `var/` are real FAT directories made at
boot. A FAT directory named `proc`, `sys` or `dev` is moved away by the migration (D6), so nothing real is ever
shadowed. Alternatives: a full VFS with inode traits (too much code for 4 backends); separate tools per tree
(against the rule).

### D2. The tree
Who writes: S = shell (`!write`, `!mkdir`, `!rm`), A = agent tools (write_file, mkdir, rm), W = WASI programs
(python3 and the rest), K = kernel only. "ro" = refused for everyone (shell `read-only file system`, WASI EROFS 69).

| Path | Content | Writes |
|---|---|---|
| /proc/cpuinfo | per core: processor, model name, cpu MHz, cpu cores, role; `flags` only in processor 0, the others say `flags: as processor 0` (saves ~70 tokens per core in a cat) | ro |
| /proc/meminfo | as T54 (MemTotal, MemFree, MemAvailable, ModelWeights, SnapshotPool, KernelHeap, kB) | ro |
| /proc/uptime | `3725.41 seconds (1 h 2 min 5 s)`: the first field stays a float for programs, the rest is for the model | ro |
| /proc/version, devices, bus/pci/devices | as T54 | ro |
| /proc/kmsg (new) | the kernel log ring, oldest line first (a snapshot, not Linux's draining read) | ro |
| /proc/mounts (new) | `/dev/vda / vfat rw`, `proc /proc proc ro`, `sysfs /sys sysfs ro`, `devfs /dev devfs rw` | ro |
| /sys/class/block/vda/size | 512-byte sectors (the Linux unit) | ro |
| /sys/class/block/vda/ro | `0` | ro |
| /sys/class/block/vda/device | PCI address, e.g. `0000:00:04.0` (a file, not a symlink) | ro |
| /sys/bus/pci/devices/0000:BB:DD.F/{vendor,device,class} | `0x1af4`, `0x1001`, `0x010000` | ro |
| /sys/bus/pci/devices/0000:BB:DD.F/driver | `virtio-blk`, or `none` | ro |
| /sys/devices/system/cpu/{online,possible} | `0-7` | ro |
| /sys/devices/system/cpu/inference | cores the engine uses now, e.g. `0-7` (`!set cores`) | ro |
| /sys/devices/system/cpu/cpuN/role | `BSP: shell, agent loop, engine slice` / `engine worker` / `engine worker (parked)` | ro |
| /sys/power/state | `disk` (the one state ringzero has). Writing `disk` hibernates and powers off (as Linux's suspend to disk: `/hibernate` then `!poweroff`); `mem`, `freeze` and `standby` fail with `not supported: ringzero has no suspend to RAM ...` (EINVAL for a program), anything else with `invalid argument` | S, A (asks y/N like poweroff), W (a program the agent runs asks y/N; a shell run does not) |
| /dev/null | reads end at once | accepted, discarded (S, W) |
| /dev/zero | zero bytes | accepted, discarded (S, W) |
| /dev/random, /dev/urandom | RDRAND, else the TSC-seeded xorshift of wasm.rs `rand` (not for cryptography) | accepted, discarded (S, W) |
| /dev/tty, /dev/console | WASI shell runs: raw console input as today; writes go to the console. Agent runs: ENOENT (no terminal) | console (W) |
| /dev/vda | raw disk, read-only (Q4) | ro |
| /etc/hostname | FAT file, made at boot as `ringzero` if missing; read at boot and logged as `host: NAME (/etc/hostname)` right after the disk mounts (the banner is printed before the disk is up) | S, A, W |
| /etc/os-release | virtual: `NAME=ringzero`, `ID=ringzero`, `ID_LIKE=unix`, `VERSION`, `PRETTY_NAME` (edition) | ro |
| /etc/passwd | virtual: `root:x:0:0:root:/:/bin/sh` (Q9) | ro |
| /etc/* (any other) | FAT files, the user's configuration | S, A, W |
| /tmp | FAT directory, emptied at a fresh boot, kept when a hibernation image resumes; not indexed | S, A, W |
| /var/lib/ringzero/mem/{journal,facts,episodes}.log, index.bin | agentd stores and index (was /sys/mem) | K, S (`!rm` asks) |
| /var/lib/ringzero/hib/image.bin | hibernation image (was /sys/hib) | K, S |
| /var/lib/ringzero/bootsnap.bin | boot snapshot cache (was /sys/bootsnap.bin) | K, S |
| /var/lib/ringzero/layout | `ringzero-layout 2`: the migration's commit marker | K |
| /bin, /lib, /demo, user files | unchanged | S, A, W |

Agent writes under /var/lib/ringzero fail with `permission denied: ringzero's own data (use remember and forget)`;
WASI writes there fail with EACCES (2), the rule `sys_ro` applies to /sys today. The shell keeps full access: it
is the human, and `!rm` already asks.

### D3. Readers
- Shell `!ls`, `!cat` and the agent's ls and cat go through `fs::ls` and `fs::read`. Device files list with size
  0. `cat` of a device returns `is a device: read it with a program, e.g. head -c 16 /dev/urandom` (except
  /dev/null: empty), so an endless /dev/zero never reaches a tool result.
- WASI (wasm.rs): `path_open` on /proc, /sys and the two virtual /etc files keeps the T54 path (bytes generated
  once at open, served by read, seek, fstat). A /dev path makes `Fd::Dev(kind)` with filetype character device
  (block device for vda); reads stream (zero, random) or come from the disk (vda through `blk`), and the run's
  fuel budget and Ctrl-C bound an endless read. `cat`, `ls`, `head`, `wc` (uutils) and `python3 open()` need no
  change beyond this: they already see the preopened `/`.
- Write results: EROFS for ro paths (69), EACCES for /var/lib/ringzero (2), accepted for null, zero, random,
  urandom, tty, console and every FAT path outside /var/lib/ringzero.

### D4. Who owns what in code
procfs.rs keeps /proc; new sysfs.rs and devfs.rs follow the same `entry`, `list`, `read` shape (sysfs reuses
procfs's PCI scan and `blk::location`); layout.rs holds the migration, the /tmp clean and the /etc defaults and
runs from `fs::init` right after the mount, before agentd, agent and hib touch the disk. The paths move to one
place: `layout::{MEM, HIB, BOOTSNAP}` constants used by mstore.rs, agentd.rs, hib.rs and agent.rs. The indexer
walk (`fs::walk`) skips /var/lib/ringzero, /lib and /tmp instead of /sys and /lib; virtual mounts are never walked.

### D5. Tools review
| Tool | Prompt tokens (est., JSON line) | File that replaces it | Recommendation |
|---|---|---|---|
| meminfo | ~54 | /proc/meminfo | hidden alias |
| cpuinfo | ~52 | /proc/cpuinfo | hidden alias |
| list_cores | ~51 | /sys/devices/system/cpu/*, /proc/cpuinfo role | hidden alias |
| uptime | ~43 | /proc/uptime | hidden alias |
| read_log | ~67 | /proc/kmsg | keep: `n` gives the last lines; cat returns the first 7936 bytes, the oldest |

Hidden alias: the name leaves `TOOLS` (not in the prompt) but stays in `SPECS` with a hidden flag, so the
validator accepts a call and `exec` serves it with the file's content (meminfo -> /proc/meminfo, list_cores ->
/sys/devices/system/cpu summary). Removing ~200 tokens is ~13% of the prefix: a shorter cold prefill, a smaller
boot snapshot (~12 KiB of KV per token, ~2.4 MiB) and less attention spread over tool schemas. The cost: a file
read is a two-step plan for the model (know the path, call cat) instead of one obvious tool name, and some answers
get longer (cpuinfo, kept small by D2). The hidden aliases catch the habit without paying the tokens.
Gate (tasks 1.x): tooleval A/B on the inos set plus 8 new info prompts (memory, cores, uptime, cpu model, pci, disk
size, hostname, version), counting cat on the right file as a right first tool and counting alias hits apart.
Accept if the right-first-tool rate drops by at most 1 prompt, round-limit hits do not rise and the answers to the
uptime and cores questions are right by hand; otherwise keep the four tools in the prompt with descriptions that
name their file, and update the agent-tools delta before the apply.

### D6. Migration of an existing disk
Runs once per disk, at mount, when FAT `/sys` exists or `/var/lib/ringzero/layout` is missing. fatfs `rename`
frees the old directory entry before it writes the new one (rust-fatfs `Dir::rename_internal` at the pinned rev),
so a crash in between loses the file: the migration never renames, it copies.
1. For each source with a target: `/sys/mem/{journal,facts,episodes}.log` and `index.bin` -> `/var/lib/ringzero/mem/`.
   If the target exists with the same size and FNV-1a 64 as the source, skip; else write it (`write_parts`,
   replaces a partial copy), read it back and compare size and FNV. Any mismatch or error stops the migration for
   this boot with every source intact.
2. `/sys/bootsnap.bin` is a cache: not copied. `/sys/hib/image.bin` is not copied either: its key is the boot
   key, and this change alters the boot prefix (tool list, system prompt), so no image written by an older kernel
   can resume on this one (today every prefix change already drops it). The boot key is only known after the
   model loads, later than the migration, so checking it is not worth the ordering change.
3. Unknown files left in FAT `/sys`, and any FAT `/proc` or `/dev`, are copied the same way to
   `/var/lib/ringzero/legacy/<path>`.
4. Commit: write `/var/lib/ringzero/layout` = `ringzero-layout 2`, then `fs::sync`.
5. Remove the sources (each only if its target still verifies), then the empty `/sys/mem`, `/sys/hib`, `/sys`;
   `fs::sync`. Log one `layout:` line per step and a summary (`migrated N files, M KiB, old image dropped`).
Idempotent: a crash before 4 redoes 1-3 (copies are overwrites, verified ones skipped); after 4 the next boot
resumes 5. If the migration cannot finish (disk full, I/O error), agentd runs RAM-only for this boot (the
existing no-disk mode), `/agentd` says `off (layout migration pending: REASON)`, nothing new is written to either
place and the next boot retries. Space needed: the size of /sys/mem (a few MiB), not the image or the cache.

### D7. An older kernel on a migrated disk (honest answer)
It is not prevented and not transparent. An older kernel finds no `/sys/mem`, starts empty stores there, prefills
its boot prefix and writes `/sys/bootsnap.bin`; facts and episodes look lost on it (they are intact in
/var/lib/ringzero; its indexer skips them as non-text). When the new kernel boots again, `layout` says 2 and FAT
`/sys` exists again: step 5 finds targets that do not match, so those sources are copied to
`/var/lib/ringzero/legacy/sys-<unix time>/` instead and removed, with a warning that names the folder. What the
older kernel recorded is kept but not merged (Q8). Going back on purpose: copy `/var/lib/ringzero/mem/*` to
`/sys/mem/` with mtools on the host (recipe in INFO.md) before booting the older ISO.

### D9. Merging what an older kernel wrote (Q8)
D7 keeps it apart by default. `/agentd merge-legacy [DIR]` (DIR: a folder name in /var/lib/ringzero/legacy or an
absolute path; empty = every folder with a `mem/` and no `merged` file) merges it on request:
- Compatibility first: each store present (journal.log, facts.log, episodes.log) must parse from byte 0 with this
  kernel's record codec (mstore.rs `parse`: magic, length, checksum, UTF-8); a store that does not is refused with
  `... is not in this kernel's record format (not compatible): nothing merged` and nothing of that folder changes.
  A torn tail is skipped (counted in the summary). index.bin is never imported: the merged records are queued and
  embedded again by the idle scheduler, as at a rebuild.
- Facts: the legacy store's end state (S and X records replayed), then one `remember` per key with source
  `merge-legacy` (the same key and value validation as `/remember`, the 64-fact cap); a key that already holds the
  same value, or a value written later here (record time), is kept.
- Episodes and journal: the records whose (kind, time, text) the current store does not hold are appended with
  their original kind and time, and queued for the index like the replay at boot does.
- The folder gets a `merged` file with the time and the summary; a second merge of it says `already merged`.

### D8. Model-facing text
SYSTEM gets one sentence in place of "(not Linux)": "Its file tree is Unix-like: /proc and /sys hold live system
state (cat /proc/meminfo, /proc/cpuinfo, /proc/uptime), /dev the devices, /etc the configuration, /tmp scratch
files." ls: "List a directory. /proc and /sys hold live system state." cat: "Read a text file, also under /proc,
/sys and /etc." Net prefix change is measured, expected about -170 tokens.

## Risks / Trade-offs
- Model reaches for a removed tool -> hidden alias serves it; tooleval counts how often.
- Model reads /proc/kmsg with cat and gets the oldest lines -> read_log stays and its description says "last n".
- Migration bug destroys memory -> no rename, verify before remove, commit marker, a QEMU test that kills the
  machine at each step (tasks 3.x) and an old-disk fixture.
- An older kernel after migration -> D7, documented in the release notes.
- Mount-table branches in every fs function -> one `mount_of` match per function, as `is_proc` already does.

## Migration Plan
Ship in one release with the upgrade note "the first boot moves ringzero's data to /var/lib/ringzero; a saved
hibernation is not resumed across this upgrade". Rollback: D7.

## Decisions on the open questions (user, 2026-11-27)
- Q1. Hidden aliases for meminfo, cpuinfo, list_cores and uptime, gated by the tooleval (D5); if the gate fails they
  stay in the prompt with descriptions that name their file. Result in docs/T55_RESULT.md.
- Q2. /tmp on the disk, emptied at a fresh boot, kept when a hibernation image resumes.
- Q3. /etc/hostname writable by everyone (shell, agent, programs).
- Q4. /dev/vda read-only (programs read it; cat refuses devices).
- Q5. /sys/power/state behaves like Linux: writable, `disk` hibernates and powers off; `mem`, `freeze` and `standby`
  print a clear not-supported error (no ACPI S3 or device resume in ringzero). The agent's write_file to it asks y/N
  exactly like the poweroff tool, and so does a program the agent runs; the shell and programs run from the shell
  act at once (the human typed them).
- Q6. Settings as files under /proc/sys: later, a separate change.
- Q7. The boot cache stays in /var/lib/ringzero (one protected tree).
- Q8. What an older kernel wrote after the migration is kept in legacy/ by default, and `/agentd merge-legacy [DIR]`
  merges it on request when the format is compatible (D9).
- Q9. /etc/passwd is one read-only root line; no motd.
- Q10. Agent writes under /var/lib/ringzero are refused (`permission denied`); the shell keeps access.
