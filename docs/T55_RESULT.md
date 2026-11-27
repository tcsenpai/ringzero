# T55: Unix file tree and /var/lib/ringzero

OpenSpec change `unix-layer` (archived: openspec/specs/unix-tree and the deltas to proc, memory, programs,
snapshots-hibernation, agent-tools, build-options). Every number below is from QEMU/KVM on the build box (8 vCPUs, AMD Ryzen AI 9
HX 370), 2B q8 6G guests for the functional tests, 4B q4 8G guests for the tooleval (raw JSON kept on the build
box).

## What changed

- One mount table in fs.rs (`mount_of`): /proc (procfs.rs), /sys (sysfs.rs), /dev (devfs.rs), the virtual
  /etc/os-release and /etc/passwd (layout.rs), everything else the FAT volume. `!ls /` lists `dev/ proc/ sys/` next
  to `bin/ lib/ demo/ var/ tmp/ etc/`.
- /proc gains `kmsg` and `mounts`; `uptime` reads `25.36 seconds (0 h 0 min 25 s)`; `cpuinfo` prints the flags in
  processor 0 only (`flags: as processor 0` in the others).
- /sys: class/block/vda, bus/pci/devices/ADDR, devices/system/cpu, power/state (writable: `disk` hibernates).
- /dev: null, zero, random, urandom, tty, console, vda (raw disk, read-only).
- /etc/hostname (writable, logged at boot), /tmp (emptied at a fresh boot, kept on resume).
- ringzero's data moved to /var/lib/ringzero (agentd stores and index, hibernation image, boot cache), with an
  automatic migration of a 0.1 disk and `/agentd merge-legacy` for what an older kernel writes later.
- Agent: the system prompt says the tree is Unix-like, ls and cat name /proc, /sys and /etc, and meminfo, cpuinfo,
  list_cores and uptime answer with their files. Taking those four out of the prompt (hidden aliases) failed the
  tooleval gate, so they stay in the prompt with descriptions that name the file (below).

## Boot prefix

| | main (T54) | T55 hidden aliases (14 tools) | **T55 as built (18 tools)** |
|---|---|---|---|
| system prompt + tools | 1571 tokens | 1419 tokens | **1635 tokens** |
| 2B: cold prefill, boot cache | 6.791 s, 37 MiB | 6.305 s, 35 MiB | **7.387 s, 38 MiB** |
| 4B q4: cold prefill, boot cache | 18.292 s, 99 MiB | 17.718 s, 94 MiB | **18.653 s, 101 MiB** |

The 2B main row is the 0.1.0 beta ISO (the same prompt as main).

## File tree (fresh disk, 2B)

A copy of the beta data disk (no /sys), boot log `layout: new disk, ringzero's data goes to /var/lib/ringzero` and
`host: ringzero (/etc/hostname)`, then (exact output):

| Command | Output |
|---|---|
| `!cat /proc/mounts` | `/dev/vda / vfat rw`, `proc /proc proc ro`, `sysfs /sys sysfs ro`, `devfs /dev devfs rw` |
| `!ls /sys/bus/pci/devices` | `0000:00:00.0/` `0000:00:01.0/` `0000:00:02.0/` `0000:00:03.0/` `0000:00:1f.0/` `0000:00:1f.2/` `0000:00:1f.3/` |
| `!cat /sys/class/block/vda/size`, `.../device` | `1048576`, `0000:00:03.0` |
| `!cat /sys/devices/system/cpu/online` and `!run cat ...` | `0-7` both |
| `!set cores 4`, `!cat /sys/devices/system/cpu/inference`, `.../cpu6/role` | `0-3`, `engine worker, not used for inference now (!set cores)` |
| `!cat /etc/os-release` | `NAME=ringzero`, `ID=ringzero`, `ID_LIKE=unix`, `VERSION="0.1.0 beta"`, `VERSION_ID=0.1.0`, `PRETTY_NAME="ringzero Qwen35@2b-Qwen3@0.6b (QweQwe)"` |
| `!cat /etc/passwd` | `root:x:0:0:root:/:/bin/sh` |
| `!run head -c 16 /dev/urandom` | 16 bytes, exit 0 in 38 ms |
| `!run wc -c /dev/null` | `0 /dev/null` |
| `!cat /dev/zero` | `[fs] cat: is a device: read it with a program, e.g. head -c 16 /dev/urandom` |
| python3: /sys and /proc/uptime | `0-7 True` |
| python3: `open('/dev/vda','rb').read(512)` | `b'mkfs.fat' 55aa` (the FAT boot sector) |
| python3: `os.urandom(8)`, 32 bytes of /dev/urandom, 1000 bytes to /dev/null | `8 32 1000` |
| `!write /sys/devices/system/cpu/online 3`, `/proc/uptime`, `/etc/passwd`, `!mkdir /sys/foo`, `!write /dev/vda x` | `read-only file system` each |
| `!write /dev/null hello` | `[fs] wrote 5 bytes to /dev/null` |
| `!write /sys/power/state mem` | `not supported: ringzero has no suspend to RAM (write disk to hibernate and power off)` |
| `!write /sys/power/state foo` | `invalid argument: /sys/power/state takes disk` |
| python3 write `/var/lib/ringzero/bootsnap.bin` | `PermissionError: [Errno 2] Permission denied` |
| python3 write `/sys/class/block/vda/ro`, `/etc/passwd` (append), `/dev/vda` | `OSError: [Errno 69] Read-only file system` each |
| python3 `open('/sys/power/state','w').write('mem')` | `OSError: [Errno 28] Invalid argument` |
| `!write /etc/hostname lab`, reboot | boot log `host: lab (/etc/hostname)` |
| `!write /tmp/a.txt scratch`, `/hibernate discard`, `!poweroff`, boot | `tmp: 1 entries removed from /tmp (fresh boot)` |
| `!write /tmp/b.txt keep`, `/hibernate`, `!poweroff`, boot | `hib: resumed ... (same as saved)`, `!ls /tmp` shows `b.txt  4 B` |

## Migration of a 0.1 disk (2B)

Fixture: a copy of the beta data disk booted once on the 0.1.0 beta 2B ISO (old layout): `/remember user_name Cris`,
`/remember colore_preferito verde`, `!write /notes/pet.txt Il mio cane si chiama Pixel ed e' allergico al polline.`,
one message, 150 s idle (files indexed, one consolidation: episode 1), `!poweroff` (hibernation image written). FAT
`/sys` then holds `mem/{journal,facts,episodes}.log`, `mem/index.bin`, `bootsnap.bin` and `hib/image.bin`.

First boot of the T55 ISO on a copy of it:

```
layout: 6 files in FAT /sys (the layout of ringzero 0.1): moving them to /var/lib/ringzero
layout: copied /sys/mem/journal.log -> /var/lib/ringzero/mem/journal.log (158 B, verified)
layout: copied /sys/mem/facts.log -> /var/lib/ringzero/mem/facts.log (82 B, verified)
layout: copied /sys/mem/index.bin -> /var/lib/ringzero/mem/index.bin (60209 B, verified)
layout: copied /sys/mem/episodes.log -> /var/lib/ringzero/mem/episodes.log (69 B, verified)
layout: commit marker /var/lib/ringzero/layout written
layout: migrated 4 files, 59 KiB, 2 old caches dropped (boot cache, hibernation image)
[agentd] index: loaded 39 entries from /var/lib/ringzero/mem/index.bin in 1.219 ms (6 files in the table), no re-embedding
```

`/facts` lists both facts, `/agentd` says `stores /var/lib/ringzero/mem: journal 5 records (0 KiB), 2 facts, 1
episodes`, FAT `/sys` is gone, `/search cane Pixel` ranks `[/notes/pet.txt] Il mio cane si chiama Pixel ...` first
(cos 0.464, bm25 9.68), and "come si chiama il mio cane?" gets the note in the memory slot and the answer "Il tuo
cane si chiama Pixel." The second boot logs no `layout:` line, loads the index (40 entries, no re-embedding) and
lists the same facts: the migration is idempotent.

Crash tests: test builds (`make iso LAYOUT_STOP=N`) halt after step N (a copy, the commit, a removal) and QEMU is
killed there; one more run kills the normal ISO as soon as the serial line `copied /sys/mem/facts.log` appears.
Each disk then boots the normal ISO:

| Stop | Second boot | facts.log / episodes.log MD5 |
|---|---|---|
| after 2 copies (journal, facts) | copies index.bin and episodes.log (the two verified copies skipped), commit, `migrated 2 files, 58 KiB` | same as the uninterrupted run |
| after the commit, before any removal | copies nothing, removes the sources (`migrated 0 files`) | same |
| after 2 removals | removes the other 4 sources | same |
| killed (SIGKILL) during the copies | copies index.bin and episodes.log, commit | same |
| index.bin half written (30000 of 60209 bytes) in /var/lib/ringzero/mem before the boot | `copied /sys/mem/index.bin ... (60209 B, verified)`, index loaded with no re-embedding | - |

In every case `/facts` shows the two facts, `/agentd` 5 journal records and 1 episode, the index loads its 39
entries without re-embedding and `/search cane Pixel` ranks the note first.

Disk full: the fixture with a file that leaves 16 KiB free. `layout: migration not finished (disk full): every old
file is kept, agentd runs without disk stores this boot, the next boot retries`, `/agentd`: `stores off (layout
migration pending: disk full)`, `/facts` empty, no boot cache or image written; the next boot logs the same.
After `!rm /fill.bin` (y) the next boot copies the four stores, commits and logs `migrated 4 files, 59 KiB`;
`/agentd` shows `stores /var/lib/ringzero/mem: journal 5 records (0 KiB), 2 facts, 1 episodes`.

## /sys/power/state

| Steps | Result |
|---|---|
| one message, then `!run python3 -c "open('/sys/power/state','w').write('disk')"` | `hib: disk written to /sys/power/state: hibernate and power off`, `hib: poweroff saved #1 on main (ctx 1451), 2 points ..., 21 MiB in 889 ms`, power off |
| boot | `hib: resumed #1 on main (ctx 1451), 2 points in 92 ms, state hash 0xbd081b6f8bb523aa (same as saved)`, `/tree` shows the message |
| `!write /sys/power/state disk` | the same save and power off; the next boot resumes with the same state hash |

## An older kernel after the migration, merge-legacy (2B)

The migrated disk booted on the 0.1.0 beta 2B ISO: `/facts` is empty there (`0 of 64`), `/remember pet_name
Pixel`, "il mio gatto si chiama Tom", 100 s idle (episode 1 "Il mio gatto si chiama Tom."), `!poweroff`. Then the
T55 ISO:

```
layout: 6 files in FAT /sys (...): moving them to /var/lib/ringzero
layout: copied /sys/mem/journal.log -> /var/lib/ringzero/legacy/sys-1795805006/mem/journal.log (294 B, verified)
layout: copied /sys/mem/facts.log -> /var/lib/ringzero/legacy/sys-1795805006/mem/facts.log (37 B, verified)
layout: copied /sys/mem/index.bin -> /var/lib/ringzero/legacy/sys-1795805006/mem/index.bin (61378 B, verified)
layout: copied /sys/mem/episodes.log -> /var/lib/ringzero/legacy/sys-1795805006/mem/episodes.log (45 B, verified)
layout: migrated 4 files, 60 KiB, 2 old caches dropped (boot cache, hibernation image)
layout: files an older kernel wrote are kept in /var/lib/ringzero/legacy/sys-1795805006 (not merged; /agentd merge-legacy merges them)
```

| Command | Output |
|---|---|
| `/facts` | the two facts of the new stores only |
| `/agentd merge-legacy` | `merge-legacy /var/lib/ringzero/legacy/sys-1795805006: facts 1 merged, 0 kept (same or newer here), 0 rejected; 1 episodes and 5 journal records added; queued for the index` |
| `/facts` | adds `pet_name = Pixel (merge-legacy, ...)` |
| `/agentd merge-legacy` again | `nothing to merge in /var/lib/ringzero/legacy (...)` |
| `/search gatto Tom` after 30 s idle | `[said] il mio gatto si chiama Tom` (cos 0.600) first, `[episode 2] Il mio gatto si chiama Tom.` third |
| `!write /var/lib/ringzero/legacy/bad/mem/facts.log garbage-not-a-record`, `/agentd merge-legacy bad` | `... facts.log is not in this kernel's record format (not compatible): nothing merged`; `/facts` unchanged |

## Agent writes (2B, `!set confidence off` so only the tool's own question asks)

| Request | Call and result |
|---|---|
| "scrivi la parola ciao nel file /var/lib/ringzero/mem/facts.log" | `write_file(path=/var/lib/ringzero/mem/facts.log, content=ciao)` -> `error: permission denied: ringzero's own data (use remember and forget)` |
| "cancella il file /var/lib/ringzero/layout" | `ls`, then `rm(path=/var/lib/ringzero/layout)` -> the same error, the file is intact |
| "use the write_file tool to write disk to /sys/power/state" | `[agent] wants to HIBERNATE and power off the machine (write to /sys/power/state). Proceed? [y/N] n` -> `NOT RUN ... The machine stays on.` |

In that last run the 2B wrote a JSON object instead of `disk`; the kernel asked anyway. The build now asks only for
`disk` (any other value fails in fs::write with the not-supported or invalid-argument error, without a question).

`tools/wasm/rztest` (`!run rztest check A "B C"`): `rztest: PASS 24/24`, with the new checks
`/var/lib/ringzero is read-only: Some(PermissionDenied)`, `/sys is a read-only file system` (69), `/dev/vda is
read-only` (69), `/dev/zero reads zeros`, `/dev/null takes 1 MiB`. `!run rg virtio /proc/kmsg` prints the boot log
line `disk: virtio-blk at PCI 00:03.0 (device id 1001), 1048576 sectors (512 MiB)`.

## Tooleval A/B (M45 protocol) and the alias decision

`tools/tooleval.py`, `SLOT=200`, constrain on, confidence on, temperature 0, 4B q4 8G guests, a fresh copy of the
beta data disk each, 120 s idle at the prompt before the first prompt. The in-OS set has the 24 prompts of T54
plus the 8 system information prompts of the D5 gate (memory, cores, uptime, cpu model, drivers, disk size,
hostname, version). Scoring: `cat` (or `ls`) on the right file counts as a right call, and reading the file of a
tool counts as that tool (`EQUIV`; the spike prompts name meminfo, cpuinfo and list_cores). All three runs are
scored with the same rules (`tooleval.py --rescore`).

| | main | T55 hidden aliases | **T55 as built** |
|---|---|---|---|
| spike: correct / first call right / args exact | 24/30, 22, 9 | 24/30, 22, 9 | **24/30, 22, 9** |
| in-OS (32): correct / first call right | 28, 25 | 23, 23 | **25, 23** |
| of them the 8 info prompts: correct | 5/8 | 4/8 | **4/8** |
| calls (spike + in-OS) | 51 + 36 | 41 + 43 | 47 + 38 |
| malformed / schema errors / grammar lines not valid | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| turns at the round limit (spike + in-OS) | 2 + 1 | 1 + 1 | 1 + 1 |
| hidden alias calls | - | 0 | - |
| decode tok/s of the call rounds, spike / in-OS | 39.46 / 42.48 | 34.34 / 37.19 | 39.51 / 40.48 |

With the aliases hidden the model read the files on every information prompt (spike: `cat /proc/meminfo`, `cat
/proc/cpuinfo`; in-OS: `cat /proc/uptime`, `cat /proc/cpuinfo`, `cat /proc/meminfo`) and never called a hidden
alias. The answers to the uptime and cores questions are right by hand in every run ("Il sistema è acceso da 7
minuti e 28 secondi.", "8 minuti e 41 secondi.", "La CPU ha 8 core.", "This machine has 8 CPU cores.").

Gate (design D5: at most 1 prompt worse on the first call, no more round-limit hits, uptime and cores right): the
hidden aliases lose 2 first calls against main ("quanto fa 2**100? usa python": run_wasm before run_code; "che
driver abbiamo?": `cat /proc/version` instead of /proc/devices), so the gate fails and the four tools stay in the
prompt, their descriptions naming the file (decision 1). The build with them visible loses the same 2 first calls
("2**100": `run_wasm /bin/python3`; drivers: `ls /dev`), so the drop comes from the new tree and prompt text, not
from hiding the tools; hiding them would save 216 prefix tokens (1419 instead of 1635). The correct counts also
lose prompts that read a file and then explore (`leggi /notes/todo.txt` on a disk without /notes: cat, then ls
/notes, ls /, cat) and, in the hidden-alias build, `set the screen brightness to 80` answered with run_code.

2B, by hand, the build as shipped: "quanto uptime?" calls `uptime()` and answers from `20.81 seconds (0 h 0 min 20
s)`; "che driver abbiamo?" lists /bin. With the aliases hidden the same 2B looped on `run_wasm uptime` and `run_wasm
ls` until the round limit (10 calls each).

## Follow-up: the drivers wording

Prompt text changed after the run above (the "as built" column there is the build before this change):
- system prompt: `(cat /proc/meminfo, /proc/cpuinfo, /proc/uptime; the drivers in /proc/devices), /dev the device
  files` instead of `(cat /proc/meminfo, /proc/cpuinfo, /proc/uptime), /dev the devices`;
- ls: `List a directory. /proc and /sys hold live system state, /proc/devices the drivers.`;
- system prompt: `run_code is a MicroPython sandbox without files, for calculations: ...` (`, for calculations`
  added, aimed at "2**100").

Same M45 protocol, same scoring (4B q4, 8G):

| | main | previous T55 build | **T55 now** |
|---|---|---|---|
| prompt + tools | 1571 tokens | 1635 tokens | **1652 tokens** |
| spike: correct / first call right / args exact | 24/30, 22, 9 | 24/30, 22, 9 | **24/30, 22, 9** |
| in-OS (32): correct / first call right | 28, 25 | 25, 23 | **26, 24** |
| of them the 8 info prompts: correct | 5/8 | 4/8 | **5/8** |
| calls (spike + in-OS) | 51 + 36 | 47 + 38 | 42 + 36 |
| malformed / schema errors / grammar lines not valid | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 1 |
| turns at the round limit (spike + in-OS) | 2 + 1 | 1 + 1 | 1 + 1 |
| 4B cold prefill, boot cache | 18.292 s, 99 MiB | 18.653 s, 101 MiB | 20.070 s, 101 MiB |
| 2B cold prefill, boot cache | 6.791 s, 37 MiB | 7.387 s, 38 MiB | 7.408 s, 38 MiB |

- "che driver abbiamo?": `cat /proc/devices`, answered from the block and filesystem lines (as on main).
- "quanto fa 2**100? usa python": still `run_wasm(path=/bin/python3, args=print(2**100))` first (python3 then
  fails on the argument as a file name), then the right answer. The memory slot is the same as on main, so the
  cause is the prefix text: the pick was already close on main (margin 0.61 for run_code over run_wasm) and is
  0.26 to 0.30 the other way with the T55 prompt; `, for calculations` did not flip it.
- The grammar line not valid: one `write_file` in "esegui questo modulo wasm: ..." reached 460 tokens and ended as
  `unfinished call`.
- Remaining first-call difference against main: the "2**100" prompt only. Correct-count differences: "2**100"
  (two calls) and `leggi /notes/todo.txt` (cat, then ls /notes, on a disk without /notes).

2B, by hand: "quanto uptime?" calls `uptime()` and answers from `27.07 seconds (0 h 0 min 27 s)`; "che driver
abbiamo?" now runs `ls /proc/devices` (through run_wasm) and answers "Non ho driver." (the file was listed, not
read).

## Limits

- "che driver abbiamo?" on the 2B lists /proc/devices instead of reading it and answers wrong; the 4B reads it.
- "quanto fa 2**100? usa python" picks the python3 program before run_code on the 4B (see above).
- The hostname is logged after the disk mounts, not shown in the banner (the banner is printed before the disk is
  up).
- `/agentd merge-legacy` matches records by kind, time and text: a record an older kernel wrote with the same time
  and text as one already here is taken as the same record.
- The agent's write_file to /sys/power/state asks only for `disk`; the 2B wrote JSON there in both tries, which
  fails with `invalid argument` and no question.


