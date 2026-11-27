# ringzero beta

The first public test build of ringzero: an experimental operating system with a Qwen3.5 language model and the
Qwen3-Embedding-0.6B memory model running inside its own kernel (ring 0). It is for people who want to try the
idea and tell us what breaks. It is not a daily-driver OS: it runs in QEMU/KVM, keeps your files on one FAT32
disk image, and the model can be wrong.

## Editions

| File | Edition (boot banner) | Model | Guest RAM | Decode |
|---|---|---|---:|---:|
| `ringzero-beta-4b-q4.iso` | `ringzero Qwen35@4b.q4-Qwen3@0.6b` | Qwen3.5-4B, MLP in Q4_0, the rest Q8_0 (**recommended**: best tool use) | 8G | about 20 tok/s |
| `ringzero-beta-2b-q8.iso` | `ringzero Qwen35@2b-Qwen3@0.6b` | Qwen3.5-2B Q8_0 (faster, weaker tool use) | 6G | about 34 tok/s |
| `disk.img` | - | the data disk for either ISO: 512 MiB FAT32 with python3, sqlite3, jq, lua, qjs, rg, coreutils, kibi in `/bin`, CPython's library in `/lib/python3.15`, the demo inputs in `/demo` | - | - |

Both editions include Qwen3-Embedding-0.6B for the memory index and the built-in manual with its vectors.

`make beta` builds all three into `build/beta` with a `SHA256SUMS` file. Two builds from clean (kernel, MicroPython
package, manual vectors and `build/beta` deleted before each; 90 s each with the models, the Limine and MicroPython
checkouts and the WASI programs already on the box) gave the same bytes. The sums below are from one build of the
release commit (the two-build check ran on the commit before it, which changed only the memory index):

| File | Bytes | SHA-256 |
|---|---:|---|
| `ringzero-beta-4b-q4.iso` | 4,027,955,200 | `e9ee1430abb6a68430e62548d40887dd1ef4215828c760a1ece93b3e4d396d54` |
| `ringzero-beta-2b-q8.iso` | 2,681,948,160 | `f069eb10fc6fed8164f853c9aff7140e347c1af9cb38544547d561e90f5d53a8` |
| `disk.img` | 536,870,912 | `a8ea97408859e988f5e1f8c8a678eca63beb1bb14f762fcb844762df08a10ad0` |
| `ringzero-beta-disk.img.xz` (`xz -9` of `disk.img`, for download) | 13,526,316 | `6f93d1d949ab7e15b44d6e3e8e3da387eded0ff368d024147fe0e9d9fea966f0` |

What makes it reproducible: `SOURCE_DATE_EPOCH` (default: the time of the last commit) is exported by `make beta`
and fixes the file dates xorriso writes (`--set_all_file_dates`), MicroPython's build date, the FAT volume id and
dates (`mkfs.vfat --invariant`, mtools), and the ISO's MBR disk id (Limine's `bios-install` seeds a random one from
the clock; the Makefile writes a fixed one after it). The Rust kernel, the NASM engine and the manual vectors
(`/man dump` of the pass-1 kernel) are deterministic on the same toolchain. Not covered: a different checkout path,
compiler or Rust toolchain changes the kernel bytes; the models and the WASI programs are inputs (pinned by version
and SHA-256 where downloaded, see INFO.md), not rebuilt by `make beta` when they are already there; the hashes
change with every commit (the epoch is the commit time).

## Minimum requirements

Measured on the Linux build box (Ryzen AI 9 HX 370, QEMU 10.2 with KVM, 8 vCPUs), each on a fresh copy of
`disk.img`: boot, one tool call, `!run python3 -c ...` (CPython in the WASI runtime), `!mem`.

| | 4B q4 | 2B q8 |
|---|---|---|
| guest RAM, default (`make run-*`) | 8G: 1024 MiB snapshot pool | 6G: 1024 MiB pool |
| smallest guest RAM that works (1G steps) | **7G**: boots, `cpuinfo` call, python3 ok; the snapshot pool drops to 128 MiB (about 2 undo points) | **4G**: boots, `ls` call, python3 ok; pool 256 MiB |
| one step below | 6G: `model_init failed (-4)` at boot, no prompt | 3G: `model_init failed (-4)` |
| QEMU process size on the host (RSS, after the test) | 5.9 GB at 7G, 6.8 GB at 8G | 4.0 GB at 4G, 4.8 GB at 5G |
| host RAM free to start it | guest RAM + about 1 GiB (`make run-*` asks for 10 GiB free for an 8G guest) | guest RAM + about 1 GiB (8 GiB for 6G) |
| ISO size | 4,027,955,200 bytes (3.75 GiB) | 2,681,948,160 bytes (2.50 GiB) |

- **CPU**: x86-64 with AVX2, FMA, F16C and XSAVE, passed to the guest (`-cpu host`). Without any one of them the
  kernel stops at boot with `PANIC ... engine needs AVX2+FMA+F16C+XSAVE` (tested by removing each flag with
  `-cpu host,-avx2`, `-fma`, `-f16c`). AVX-512 and AVX-VNNI are not needed: `-cpu Haswell-noTSX` (AVX2 + FMA + F16C,
  no AVX-512, no VNNI) boots and runs a tool call and python3; VNNI, when present, only speeds up the prompt prefill.
- **Disk**: `disk.img` is 512 MiB (FAT32, 64 MiB used by the programs and CPython's library, 447 MiB free). The first
  boot writes a 96 MiB prompt cache to it and every poweroff a hibernation image (91 to 197 MiB in the tests).
- **Host**: Linux with KVM. Without KVM (TCG) it is not tested and would be far too slow.

## How to run

Copy the disk first (it keeps your files, facts and the hibernation image; one disk per running guest):

```sh
cp disk.img my-disk.img
```

Serial console in the terminal (Ctrl-A X quits QEMU; inside ringzero, `!poweroff` saves the session and stops):

```sh
qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m 8G -smp 8 \
  -cdrom ringzero-beta-4b-q4.iso -boot d -no-reboot -display none -serial mon:stdio \
  -drive file=my-disk.img,if=none,id=disk0,format=raw -device virtio-blk-pci,drive=disk0
```

A window (framebuffer and PS/2 keyboard; `!kbd us` or `!kbd it` sets the layout), serial still in the terminal:

```sh
qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m 8G -smp 8 \
  -cdrom ringzero-beta-4b-q4.iso -boot d -no-reboot -vga std -display gtk -serial mon:stdio \
  -drive file=my-disk.img,if=none,id=disk0,format=raw -device virtio-blk-pci,drive=disk0
```

For the 2B edition use `ringzero-beta-2b-q8.iso` and `-m 6G`. `-smp` is the number of cores the engine uses;
`-cpu host` passes AVX2, FMA and F16C to the guest. The first boot prefills the system prompt (about 20 s on
the 4B) and caches it on the disk; later boots load it. The prompt is `ringzero[main]>`: type a sentence for
the agent, `/help` for the commands, `/man TOPIC` for the manual. Examples to paste: [DEMOS.md](DEMOS.md).

From the source tree instead (Linux with KVM; tools in [INFO.md](../INFO.md#build-and-run)):

```sh
make model MODEL=4b QUANT=q4 && make model   # both models (about 14 GB of downloads), then:
make beta                                    # build/beta: both ISOs, disk.img, SHA256SUMS
```

## Known limits

- QEMU/KVM only. Real hardware is not a beta target; only virtio-blk, the serial port, PS/2 and the framebuffer
  are driven. No network.
- The model can be wrong. Tool calls follow the tool schemas (0 malformed calls in the evals), but the model can
  pick the wrong tool or wrong values with high confidence: the `[confidence]` line catches about 5 of 8 wrong calls
  at the 0.80 line (docs/T28_RESULT.md); below it the tools that write ask y/N, program runs and reads run with a
  notice.
- Language: the system prompt asks for the language of the user's message, but after a tool result the 4B
  sometimes answers in Italian to an English request (3 of 10 in the DEMOS agent set on a fresh disk), and on a
  disk with Italian memories more often.
- After you decline a y/N, the 4B may try the same call again in the same turn; every such call asks again, and
  nothing runs without a y. The answer after a decline says it was not done, sometimes with a wrong reason.
- MicroPython (`run_code`, `!py`) has no `sqlite3`, `csv` or `statistics`: the agent sometimes tries it first and
  then switches to the WASI programs; "use sqlite3 to ..." in the request helps.
- The 2B edition calls tools less often and can answer from its own guesses ("The CPU has 4 cores" on an 8-core
  guest, without a tool call).
- Snapshot pool: every turn and every `/clear` conversation keeps a point (about 50 to 60 MiB each on the 4B). When
  the pool is full the oldest unnamed conversations are dropped; `/save NAME` keeps a point.
- kibi needs a terminal that answers the cursor-position query (a real terminal emulator over `nc` or the
  QEMU serial on stdio) and saves only into an existing directory.
- `make run-remote` listens on all interfaces with a fixed VNC password (`ringzero`) and an open serial port: use it
  on a trusted network only.
- A hibernation image resumes only on the same edition (model, prompt and engine); another ISO boots fresh.
- Files written by the agent's tools are at most 64 KiB; `cat` returns at most 8 KiB.

## Reporting issues

Open an issue on the GitHub repository with: the edition (first line of the boot banner), the host CPU
(`!cpu`), the guest RAM, what you typed, and the serial output around it (`!log 50` prints the kernel log). For a
wrong answer or a wrong tool call, the `[tool]` and `[confidence]` lines are the most useful part. Do not paste
facts you told the system that you would not publish: `/facts` lists them.

## Checklist (T34)

Run on 2026-11-27 on the Linux build box, QEMU/KVM, 8 vCPUs, one test guest at a time next to another running
guest. "Fresh disk" = a copy of the beta `disk.img`. The functional checks ran on the beta ISOs of that day; the
last kernel changes after them (the `/help` text, the snapshot pool fix) were checked again where they matter
(pool, resume, undo, tooleval, RAM).

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | ISOs and disk from the repo with `make beta`, SHA-256 sums | **pass** | three files and `SHA256SUMS` in `build/beta`, sizes and sums above |
| 2 | Bit-reproducible | **pass** | two clean builds, identical `SHA256SUMS` (what is and is not covered: above) |
| 3 | Smallest guest RAM, 4B q4 | **pass**: 7G | 7G: `cpuinfo` call and `!run python3 -c ...` -> `3.15.2 45`; 6G: `model_init failed (-4)` |
| 4 | Smallest guest RAM, 2B q8 | **pass**: 4G | 4G: `ls(path=/demo)` and python3 ok; 3G: `model_init failed (-4)` |
| 5 | CPU flags | **pass** | without AVX2, FMA or F16C: `PANIC ... engine needs AVX2+FMA+F16C+XSAVE`; `-cpu Haswell-noTSX` (no AVX-512, no VNNI): tool call and python3 ok |
| 6 | Disk image size, host RAM | **pass** | 512 MiB, 447 MiB free; QEMU RSS 5.9-6.8 GB (4B), 4.0-4.8 GB (2B) |
| 7 | Every docs/DEMOS.md block on a fresh disk, 4B q4 | **pass** after doc fixes | coreutils, ripgrep, jq, sqlite3 (one-shot and REPL), Lua (script, `-e`, REPL), python3 (`-c`, script, REPL), MicroPython lines, qjs: output as documented (`!py -f /demo/stats.py` fails with `no module named 'csv'` and `du` is unknown, both as documented). kibi: opens, edits; saving to `/notes/todo.txt` failed on a fresh disk (no `/notes`): the block now edits `/notes/home.txt`, which the setup creates (the save itself not run again). Agent block: 9 of the 10 English requests got the right tool and result; the literal translation of the sqlite request went to `run_code` and failed, so the block now uses the wording that worked ("use sqlite3 to create ...") |
| 8 | F6: language on a fresh disk | **pass** on the set, with a caveat | 5 English -> 5 English answers ("To go back one turn, type `/undo` ...", "Canberra", "This machine has 8 cores.", the /demo listing, "The sea is a vast, mysterious body of water ..."); 5 Italian -> 5 Italian ("Per ibernare il sistema, digita `/hibernate` ...", "Parigi", "La CPU ha 8 core.", "I file in /demo: ...", "Il mare e un mondo di infinite sfumature ..."). Caveat: in the agent block 3 of 10 English requests got Italian answers after a tool result (F6 note) |
| 9 | F7: declined calls | **pass** | 9 declined calls (rm twice, poweroff, reboot, forget, write_file, remember, mkdir and its retries): no answer claims success, nothing changed (`!ls`, `/facts`); a retried mkdir after a no asks again ("you declined a call in this turn ...") instead of running |
| 10 | Memory across reboot, fresh disk | **pass** | "my name is Anna and my favourite number is 4711" -> `remember` x2, `/consolidate` -> 2 facts, 1 episode; `!poweroff`; boot; `/clear`; "what is my name?" -> "Il tuo nome e Anna." (right, in Italian), "what is my favourite number?" -> 4711 |
| 11 | Hibernation resume, same state hash | **pass** | poweroff `state hash 0x6b107362c57283be`, boot `hib: resumed #30 ... (same as saved)`, `/snap` the same hash |
| 12 | Snapshot pool under many `/clear` | **pass** after a fix | before: 17 `/clear` conversations filled the 1 GiB pool and later turns, `/consolidate` and the hibernated conversation were lost; after: 22 conversations, pool at 944-959 MiB, every turn saved |
| 13 | `/undo` | **pass** | `/undo` -> `back to #1 ... 1.48 ms`, `/history` without the undone turn, `/redo` -> 1.62 ms |
| 14 | `/man` | **pass** | `/man check`: 27/27 rows byte-identical (22.6 s); `/man confidence` -> the new section (score 0.73); boot: 27 sections from the vectors in the kernel |
| 15 | `[confidence]` line | **pass** | on every call; below 0.80 a machine-changing call asks `Run it? [y/N]` |
| 16 | Speculative stats line | **pass** | `[decode] ... spec 10/11 accepted (90%) in 2 verify, 1 rollback, ... grammar 10/11` |
| 17 | tooleval, M45 protocol, SLOT=200 | **pass** | copy of a used disk without `/sys`: spike 22/30 (first call right 22, args exact 9), in-OS 23/24 (first call right 20), 0 malformed, 0 schema errors, 0 grammar lines not valid; 6 confidence questions, answered y by the harness |
| 18 | Docs complete and consistent | **pass** after fixes | INFO now lists every `/help` command (11 were missing); `/help` named the 2B on every edition (fixed); demos and examples in English (Italian only where it documents the bilingual keywords and the manual's "Asked as" lines) |
| 19 | Public text style | **pass** | no em dashes or marketing words in README.md, INFO.md, docs/BETA.md, docs/DEMOS.md; the build box address, ssh alias and machine paths were removed from INFO.md |
| 20 | Serial console typed into during boot | **pass** after a fix | input before the kernel's UART init failed its loopback self-test and left no serial console (4 of 4 boots of the 4B with the test driver); the init now retries |
| 21 | Documented QEMU command (serial) | **pass** | the "How to run" command boots the 4B ISO to the prompt with a fresh disk |
| 22 | User sign-off after their own runs | **open** | |
