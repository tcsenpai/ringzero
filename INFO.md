# ringzero: technical notes

ringzero is an experimental AI OS: a small Rust `no_std` x86-64 kernel, booted by Limine, with an LLM
(Qwen3.5-2B Q8_0) running in ring 0 as a system agent, and an embedding model (Qwen3-Embedding-0.6B Q8_0) next
to it on the same engine for a memory index. The inference engine is the NASM
assembly from [llm.asm](https://github.com/alesha-pro/llm.asm) by Alexey Fateev (alesha-pro), taken via our fork [tcsenpai/llm.asm](https://github.com/tcsenpai/llm.asm) (branch `qwen35-kernel`, which adds Qwen3.5), linked
into the kernel as objects.

Status: **M4.7 constrained tool calls + hibernation** (with M5 MicroPython and M5b WASI; edition "ringzero Qwen35@2b-Qwen3@0.6b", tag QweQwe). Boots under BIOS and UEFI, runs the APs in the engine's
worker loop, passes the Qwen3.5-2B golden self-test, then serves a shell on COM1 (and PS/2, layouts
`it`/`us`). Plain text talks to the system agent: the model calls kernel tools (memory, CPU, cores, log,
uptime, settings, poweroff/reboot with confirmation) in its native tool format, up to 10 tool rounds per question
(`!set rounds N`, 1..20, `make iso ROUNDS=N`); after the last one the model answers once more with the
`<tool_call>` token blocked, so every question ends with an answer (`[agent] tool round limit (N) reached,
answering without tools`). `!cmd` runs a machine
command (`!mem`, `!cores 4`, `!log 10`, `!kbd it`, `!set temperature 0.7`, `!poweroff`), `/cmd` a session
command (`/help`, `/clear`, `/ctx`, `/compact`, `/history`, `/bench`). Tab completes commands and their
arguments, a bare `/` or `!` lists them, up/down recall history. The context is 32768 tokens with an FP16
KV cache; past 75% fill the agent summarizes the conversation and rebuilds the context from the boot
snapshot.

Batched prefill (M4.8): prompt tokens go through the engine in blocks of 32 (`transformer_prefill`, AVX-VNNI
batch matmul), so each weight matrix is read once per block: about 200-226 tok/s instead of 32-46, bit-identical
state (`/prefill [B] [avx2|vnni]` times the boot prefix and prints the state hash; block 1 is the old path). Boot
prefix, agent turns, memory slot, tool results, compaction, consolidation and snapshot replay all use it; decode is
unchanged. Results: [docs/BATCHED_PREFILL_RESULT.md](docs/BATCHED_PREFILL_RESULT.md).

Speculative decoding (M4.8b): when the answer repeats text that is already in the conversation (a file it read,
a tool result, code it edits), the kernel drafts the next tokens by looking up the end of the context in the
earlier context and checks the draft in one batched pass with logits for every position. Each position is picked
as in the per-token loop (sampler, tool-call grammar), so the output is the same token for token, greedy or
sampled; on a partial accept the DeltaNet state is copied back (50.3 MiB on the 4B) and the KV rows are simply
overwritten later. 4B q4: quoting answers decode at 2.0-2.5x (20.8 -> 53.2 tok/s copying a 1.5 KB file), normal
chat and tool calls within the run-to-run noise. `!set spec on|off|N`, `/spec [N]` checks the verifier against
the per-token path. Results: [docs/M48B_RESULT.md](docs/M48B_RESULT.md).

Multi-stream decode (M4.8a): rows of up to 4 conversations go through one engine pass, each at its own position
on its own KV cache and DeltaNet state, so the weights are read once for all of them (`transformer_prefill` with
a per-row stream table). Stream 0 is the agent's context; up to 3 extra streams of 4096 positions are opened only
while used, with memory lent by the snapshot pool (67 MiB each on the 2B, 179 MiB on the 4B; at least 256 MiB of
the pool stay free) and given back after; they are never saved. Every stream's ids and logits are bit-identical to decoding it alone. 8 vCPU: 2 streams give 1.76-1.85x the tok/s of one
on the 4B q4 and 1.89-1.93x on the 2B q8, 4 streams 2.58-2.81x and 2.96-3.26x (on the 4B a Q4_0 kernel for 1-4
rows, `matmul_q4v_range`, keeps the weights packed). The user is agentd: an idle consolidation runs on a copy
of the conversation in an extra stream, alone in idle slices and one token next to every foreground decode step,
so typing a question does not stop it. `/multi [N] [T]` measures it. Results: [docs/M48A_RESULT.md](docs/M48A_RESULT.md).

Grammar drafts (T39): inside a tool call the speculative draft comes first from the M4.7a grammar: from its state
the kernel writes the text the call format implies up to the model's next choice (the rest of a tool or parameter
name once only one fits, `>`, `\n`, `<parameter=` while a required parameter is missing, `</parameter>`,
`</function>`, a unique enum value), tokenizes it and verifies it in the same pass; the prompt lookup runs when
that text is empty. Same picks, so the same output token for token. 4B q4: the call rounds of 12 tool prompts
decode at 35.8 tok/s instead of 20.4 (1.76x), 96% of the grammar tokens accepted. `!set gramdraft off` turns it off (lookup
drafts only). Results: [docs/T39_RESULT.md](docs/T39_RESULT.md).

Embedder: Qwen3-Embedding-0.6B runs on the same asm engine by swapping the engine context (`g_ctx`) between
whole forward calls, so the 2B's state is never touched. `/embed [-q] TEXT` prints a vector (dim 1024, about
6 ms per token, 0.61 s per 100 tokens; cosine >= 0.9986 against the HF fp32 model), `/index add TEXT` stores a
note in the RAM memory index (Q8_0 rows), `/search QUERY` returns the top 5 by embedding similarity, with BM25
joining only for identifier-like queries or a strong keyword match, and shows which signal matched. The agent
has a `recall` tool over the same index (through agentd, below). Results: [docs/EMBEDDER_RESULT.md](docs/EMBEDDER_RESULT.md).

File tree (T54 /proc, T55 the rest; the Unix principle: state as files). One mount table in fs.rs (`mount_of`)
routes every path, so `!ls` and `!cat`, the agent's `ls` and `cat` tools and WASI programs (`!run cat ...`, `!run ls
...`, python3's `open()`) see the same tree. Familiar Linux paths and names, plain content (one value per file in
/sys, `key: value` in /proc), not byte-exact Linux formats. `ls /` lists `dev/`, `proc/` and `sys/` next to the
FAT entries.

| Path | Content | Who may change it |
|---|---|---|
| `/proc/cpuinfo` | one block per core: processor, model name, cpu MHz (TSC), cpu cores, role; the CPUID flags (sse.., avx, avx2, fma, f16c, bmi2, avx512f, ...) in processor 0 only, the others say `flags: as processor 0` | nobody (read-only) |
| `/proc/meminfo` | MemTotal, MemFree, MemAvailable, ModelWeights, SnapshotPool, KernelHeap in kB | nobody |
| `/proc/uptime` | `25.36 seconds (0 h 0 min 25 s)`: the first field stays a number for programs | nobody |
| `/proc/version` | `ringzero version 0.1.0 beta (EDITION), Rust kernel, x86_64, not Linux` | nobody |
| `/proc/devices` | the active drivers: 16550 serial, PS/2 keyboard (layout), Limine framebuffer (size), virtio-blk (PCI slot, MiB), fatfs (FAT type, label, size, free), proc, sysfs, devfs | nobody |
| `/proc/bus/pci/devices` | PCI bus 0 from the config ports: `bus:dev.fn vendor:device class (code), driver` | nobody |
| `/proc/kmsg` | the kernel log ring (256 lines), oldest first, a snapshot at read time; the `read_log` tool gives the last n | nobody |
| `/proc/mounts` | `/dev/vda / vfat rw`, `proc /proc proc ro`, `sysfs /sys sysfs ro`, `devfs /dev devfs rw` | nobody |
| `/sys/class/block/vda/{size,ro,device}` | 512-byte sectors (`1048576` for 512 MiB), `0`, the PCI address | nobody |
| `/sys/bus/pci/devices/0000:00:DD.F/{vendor,device,class,driver}` | `0x1af4`, `0x1001`, `0x010000`, `virtio-blk` or `none` | nobody |
| `/sys/devices/system/cpu/{online,possible,inference}` | `0-7`; inference = the cores the engine uses now (`!set cores 4` makes it `0-3`) | nobody |
| `/sys/devices/system/cpu/cpuN/role` | `BSP: shell, agent loop, engine slice`, `engine worker`, or not used for inference now | nobody |
| `/sys/power/state` | `disk`. Writing `disk` hibernates and powers off (as Linux: the image of `/hibernate`, then `!poweroff`); `mem`, `freeze` and `standby` fail with `not supported: ringzero has no suspend to RAM` (EINVAL for programs), other values with `invalid argument` | shell and programs run from the shell at once; the agent's write_file and programs the agent runs ask y/N like the poweroff tool |
| `/dev/null`, `/dev/zero` | empty, zero bytes; writes are discarded | anyone (discarded) |
| `/dev/random`, `/dev/urandom` | RDRAND bytes (else a TSC-seeded xorshift; not for cryptography) | anyone (discarded) |
| `/dev/tty`, `/dev/console` | the console for a program run from the shell (raw input, the editor); not found for the agent's runs | writes print |
| `/dev/vda` | the raw disk, a block device for programs (`python3`: the FAT boot sector at offset 0) | nobody |
| `/etc/hostname` | a FAT file, made as `ringzero` at boot when missing, logged as `host: NAME (/etc/hostname)` | anyone |
| `/etc/os-release`, `/etc/passwd` | virtual: NAME, ID, ID_LIKE=unix, VERSION, VERSION_ID, PRETTY_NAME (the edition); `root:x:0:0:root:/:/bin/sh` | nobody |
| `/tmp` | scratch on the disk: emptied at a boot that does not resume a hibernation image, kept when one resumes; not indexed | anyone |
| `/var/lib/ringzero` | ringzero's own data: `mem/{journal,facts,episodes}.log`, `mem/index.bin`, `hib/image.bin`, `bootsnap.bin`, `cache/docs/` (document states), `layout` (the migration marker), `legacy/` | the kernel; the shell (`!rm` asks); the agent's write_file, mkdir and rm get `permission denied: ringzero's own data (use remember and forget)`, programs EACCES |

Every other change under /proc, /sys, /dev and the two virtual /etc files fails with `read-only file system`
(programs: EROFS 69). `!cat` and the agent's cat refuse a device but /dev/null (`is a device: read it with a
program, e.g. head -c 16 /dev/urandom`), so an endless /dev/zero never reaches a tool result. The file indexer
skips /var/lib/ringzero, /tmp, /lib and the virtual trees. The agent's `meminfo`, `cpuinfo`, `uptime` and
`list_cores` tools answer with /proc/meminfo, /proc/cpuinfo, /proc/uptime and /sys/devices/system/cpu, and their
descriptions name the file (taking them out of the prompt failed the T55 tooleval gate); the system prompt says the
tree is Unix-like and names /proc/devices for the drivers, the `ls` and `cat` descriptions name /proc, /sys
(`ls`: /proc/devices the drivers) and /etc, and run_code is described as the sandbox for calculations.

Disk layout migration (T55). ringzero 0.1 kept its data in FAT `/sys` (`/sys/mem`, `/sys/hib/image.bin`,
`/sys/bootsnap.bin`). The first boot of a newer kernel moves it, before agentd, the agent and hibernation touch
the disk (layout.rs): it copies `/sys/mem/{journal,facts,episodes}.log` and `index.bin` to `/var/lib/ringzero/mem/`
and any other file in FAT `/sys`, `/proc` or `/dev` to `/var/lib/ringzero/legacy/`, reads each copy back (size and
FNV-1a 64), writes `/var/lib/ringzero/layout` (`ringzero-layout 2`) and syncs, and only then removes the sources;
the old boot cache and hibernation image are dropped (the new boot prefix cannot match them). It never renames
(fatfs frees the old directory entry before it writes the new one). One `layout:` line per step and a summary
(`layout: migrated 4 files, 59 KiB, 2 old caches dropped`). A crash at any step ends, at a later boot, in the same
state; when a copy fails (disk full, I/O error) every old file stays, agentd runs without disk stores for that boot
(`/agentd`: `off (layout migration pending: REASON)`, no boot cache or image written) and the next boot retries.
An older kernel booted on a migrated disk finds an empty memory (its data is intact in /var/lib/ringzero). What it
writes to FAT `/sys` is moved by the next boot of the new kernel to `/var/lib/ringzero/legacy/sys-<time>/` (with a
warning) and not merged; `/agentd merge-legacy [DIR]` merges it on request: each store must parse with this
kernel's record codec (else `not compatible`, nothing changes), facts go through the `/remember` validation (the
newer value of a key wins), episode and journal records not held yet are appended with their own time, everything
merged is queued for the index, and the folder gets a `merged` file. Going back to an older ISO on purpose: copy
the stores to the old place on the host first, with the guest stopped:

```
mmd -i build/disk.img ::/sys ::/sys/mem
mcopy -i build/disk.img ::/var/lib/ringzero/mem/journal.log ::/var/lib/ringzero/mem/facts.log \
      ::/var/lib/ringzero/mem/episodes.log ::/var/lib/ringzero/mem/index.bin /tmp/rzmem/   # mkdir /tmp/rzmem first
mcopy -i build/disk.img /tmp/rzmem/* ::/sys/mem/
```

T54 and T55 results (the round limit, the confidence change, the tool review): [docs/T54_RESULT.md](docs/T54_RESULT.md),
[docs/T55_RESULT.md](docs/T55_RESULT.md).

Disk (M4): a virtio-blk disk (`build/disk.img`, 512 MiB FAT32, made once by `make disk`) is mounted at `/` at
boot; the banner prints its size and free space. Shell: `!ls [PATH]`, `!cat PATH`, `!write PATH TEXT` (`\n` =
new line, parents created), `!mkdir PATH`, `!rm [-r] PATH` (asks y/N), `!df`, `!sync` (poweroff and reboot sync
too). The agent has `ls`, `cat` (8 KiB tool cap), `write_file`, `mkdir` and `rm` (always asks the human y/N);
paths must be absolute, without `.` or `..`, at most 255 bytes, files at most 64 KiB. Text files written through
the shell or the agent are chunked (~1000 bytes, ~200-250 tokens) into the memory index in idle time at the
prompt, in slices that any keystroke or Ctrl-C ends within one embedder forward call; `/index stats` shows the
queue, and `/search` or `recall` return the chunks as `[/path] text`. Results:
[docs/M4_RESULT.md](docs/M4_RESULT.md).

agentd (M4.5): a kernel subsystem owns the agent's memory, because the 2B model is small; the model only reads
and writes it through typed tools (`remember`, `forget` with y/N, `recall`), and every agentd decision is logged
(`/agentd log`, kernel log level D).
- Persistent stores in `/var/lib/ringzero/mem` on the disk (T55; `/sys/mem` before, migrated at the first boot): an append-only journal of turns and tool results, facts (key ->
  value, source, time) and episodes (session summaries). Records carry a length and a checksum; at boot each file
  is scanned to its last good record and a torn tail is cut.
- The memory index is saved to `/var/lib/ringzero/mem/index.bin` with a header (embedder hash, dim, Q8_0 layout, chunker
  version, the size and mtime of each indexed file) and loaded at boot in about a millisecond; only files that
  changed on the disk (also from the host) and new records are embedded again.
- Boot snapshot cache: the prefilled system prompt + tools state (33 MiB) is saved to `/var/lib/ringzero/bootsnap.bin` and
  loaded at the next boot in about 0.2 s instead of a prefill (BIOS boot to the prompt 40.6 s cold, 11.9 s
  cached, before M4.8). Since the batched prefill (M4.8) a miss prefills the 1315-token prefix in 5.8 s instead of
  36-40 s, and a cold boot reaches the prompt in about 28 s instead of 71 s (QEMU start to prompt, shared box). The key covers the model, the prefix tokens and the engine build; any change means a prefill.
- Document states (M4.8c, `/read PATH`): a text file of up to 64 KiB goes into the conversation as a user turn
  plus a fixed reply (`I have read PATH.`), saved as a snapshot point. For 512 tokens or more the state of its
  tokens (token ids, the KV rows of their positions, the DeltaNet block R) is written to
  `/var/lib/ringzero/cache/docs/<key>.bin` (header with the keys, KV geometry, path and a checksum). The key is
  the boot key, the token ids of the conversation before the file and the file's token ids: the state of a
  token depends on everything before it, so an entry hits only at the same position after the same context (the
  first `/read` of a fresh conversation always qualifies), and the loaded state is bit-identical to the prefill.
  A 4325-token file: 2B prefill 22.9-28.4 s, hit 0.12-0.18 s (69 MiB); 4B prefill 71.7 s, hit 0.56-0.76 s
  (185 MiB) ([docs/M48C_RESULT.md](docs/M48C_RESULT.md)). blk.rs reads ahead 1 MiB on sequential reads (fatfs
  asks one 4 KiB cluster per request), which also halves the boot cache load. Cap 512 MiB with LRU eviction (entries of another model or engine
  first), 64 MiB left free on the disk; a changed file replaces its old entry; a corrupt, truncated or mismatched
  entry is refused, deleted and the file prefilled. `/read` lists the cache, `/read clear` deletes it.
- Context compiler: the first message of every conversation, and the first turn after a compaction, gets a
  memory slot of at most 200 tokens: saved facts, then notes recalled for that message (earlier statements of the
  user, episodes, file chunks). This fixes the M2.5 compaction recall failure. Recalled notes are filtered (T35):
  cosine 0.55 or more (0.40 for a keyword match), 20 characters or more (8 for a file chunk), at most 3 lines, not
  the message itself; only statements about the user (a first-person word: mi, mio, sono, my, I, ...) are kept as earlier statements,
  and not when the agent carried them out with a tool (remember / forget / recall do not count). `/agentd log` lists every
  candidate with its score and verdict.
- Idle work runs in one priority queue at the prompt (embed memory texts, embed file chunks, save the index,
  consolidate), in slices that any keystroke stops between engine steps. Consolidation (after 60 s idle, or
  `/consolidate`) has the 2B summarize the session into an episode and propose facts; the kernel validates them
  (snake_case key, short value, grounded in what the user said, dedup). The idle one runs on a copy of the
  conversation in an extra stream (M4.8a) and goes on next to the answer when the user types; `/consolidate` (and
  a conversation longer than 4096 positions) runs on top of the snapshot point, which is restored after.
Results: [docs/M45_RESULT.md](docs/M45_RESULT.md).

Built-in manual (T22): [docs/MANUAL.md](docs/MANUAL.md) is a manual written for the model (30 short sections, one
topic each, English, each with an `Asked as:` line of Italian and English wordings that is embedded but not shown).
It is compiled into the kernel (`include_str!`) and indexed one section per memory-index entry, `[manual: TITLE]
...`, in idle time at the first boot (29 entries, about 20 s of idle embedding); the entries are saved with the
index, and at every boot (and on `/index clear`) indexed sections that differ from the kernel's manual are dropped
and missing ones queued, so editing one section re-embeds only that one. Manual sections are ranked apart from the
user's memory: they never take one of the slot's recalled lines or one of `recall`'s top 3. A message that asks how
to do something (`how do I`, `how can I`, `what command`, `come faccio`, `come posso`, `come` + a first-person
verb such as `come torno`, ...) gets the best section as one slot line when it scores cos >= 0.55, within the same
200-token budget, also in the middle of a conversation (then only that line, under its own header); the shell
prints it as a `[memory] - ringzero manual, TITLE: ...` line. `recall` adds the best section after its top 3 when the
user's message (or the query) is such a how-to question and the section scores cos >= 0.55. `/man TOPIC` prints the two best sections in full, `/man` the
section titles. The system prompt and the boot prefix are unchanged. Results:
[docs/T22_MANUAL_RESULT.md](docs/T22_MANUAL_RESULT.md).

Manual vectors in the kernel (T38): with `MANUAL_VECTORS=1` (the default) the build links the 30 sections' Q8_0
rows into the kernel (`include_bytes!`, 31580 bytes), so a fresh disk has the manual in its index at boot with no
embedder work. Two passes, both in `make kernel`: pass 1 builds an ISO without vectors (`MANUAL_VECTORS=0
ISO=build/manvec-pass1.iso`), `tools/manvec.py` boots it headless once (QEMU/KVM, no disk, guest RAM
`GUEST_MEM`) and types `/man dump`: the kernel embeds every section with its own Q8 embedder (the idle indexer's
path) and prints the blob as hex lines with its FNV-1a; pass 2 is the normal kernel build with
`RINGZERO_MANVEC=build/manvec.bin`. `build/manvec.bin` is rebuilt only when docs/MANUAL.md, the embedder file, the
engine sources, embed.rs, memidx.rs or llm.rs change (about 90 s for the 4B, then the file is reused). The blob header
holds a key: FNV over every section text (Asked-as lines included), the embedder hash, the engine build hash, the
chunker version and the index dim. At boot a matching key puts the rows straight into the index (logged: `29 from
the vectors in the kernel (no embedder work)`); any mismatch, or `MANUAL_VECTORS=0`, falls back to the idle
indexing above, with the reason in the log. `/man check` embeds the sections again and compares them with the
linked rows (26/26 byte-identical). Results: [docs/T38_RESULT.md](docs/T38_RESULT.md).

Python (M5): MicroPython 1.29.0 (`ports/embed`) is compiled freestanding into the kernel and runs in ring 0
with an 8 MiB GC heap and its own 1 MiB stack from the engine arena (`RINGZERO_UPY_HEAP_MIB=16 make iso` for
another heap size). `!py CODE` runs one line (an expression's value is echoed, like the REPL), `!py -f PATH`
runs a script from the disk; each run is a fresh interpreter and ends with a status line (time, heap used).
Exceptions print a traceback (long ones cut to the first and last 5 frames) and the kernel goes on. The time
budget (10 s at the shell, 5 s for the agent) and Ctrl-C raise `KeyboardInterrupt`; a program that swallows it
is aborted 1 s later. The agent has a `run_code` tool: it gets the captured output, the traceback and the
status line (6 KiB cap). Its description and the system prompt say it is an isolated sandbox (no `os`, no
filesystem, no network, no imports from disk) and point the model at ls, cat, write_file and run_wasm for files and
programs (T54: the 4B tried `import os` for a file question). Included: builtins, f-strings, big ints, double floats, `math`, `cmath`, `sys`, `gc`,
`array`, `collections`, `struct`, `errno`; no `import` from files, no `input()`, no I/O modules yet. Results:
[docs/M5_MICROPYTHON_RESULT.md](docs/M5_MICROPYTHON_RESULT.md).

WASI (M5b): programs compiled to `wasm32-wasip1` run from the disk on [wasmi](https://github.com/wasmi-labs/wasmi)
2.0.3, an interpreter linked into the kernel (no_std), with WASI preview1 host functions mapped onto the FAT32
volume. `!wasm PATH|NAME [ARGS...] [< FILE]` (`!run` is the same command) runs one: a bare NAME is
`/bin/NAME.wasm`, else the uutils coreutils tool NAME (`!run ls -l /` runs `coreutils.wasm` with argv[0] `ls`);
`"..."` keeps spaces in an argument, `< FILE` is stdin (without it stdin is at end of file). The program sees `/`
as its preopened directory (fd 3, cwd `/`): `..` above `/` is refused, `/var/lib/ringzero` (ringzero's data) is
read-only (EACCES), the virtual /proc, /sys and /etc files EROFS, and /dev holds the devices (file tree above).
stdout and stderr go to the console; each run is a fresh instance and ends with a status line (exit code, time,
fuel, memory). The memory cap is 256 MiB per instance (`RINGZERO_WASM_MIB=512 make iso` for another), the time
budget 60 s at the shell and 10 s for the agent, and Ctrl-C stops a program (checked every 10 M fuel, about
10 ms). WASI calls not implemented (sockets, links) return ENOSYS, logged once per run. The agent has a
`run_wasm(path, args)` tool with the captured output (6 KiB cap) and the status line. `make wasm-apps` builds
or fetches the test programs (rztest, uutils coreutils, ripgrep, QuickJS-ng) and copies them to `/bin` on the
disk with mtools. Speed: about 10x slower than native on a CPU bench. Results:
[docs/M5B_WASI_RESULT.md](docs/M5B_WASI_RESULT.md).

Common software (M8): `make wasm-apps` puts the usual tools in `/bin`: uutils coreutils, ripgrep, jq (jaq),
sqlite3, Lua, CPython 3.15 (`python3`, its standard library in `/lib/python3.15`), QuickJS-ng and the kibi editor,
plus the demo inputs in `/demo`. From the shell a program has the console as its terminal: REPLs (`!run python3`,
`!run lua`, `!run sqlite3 FILE`, `!run qjs`) read typed lines (Ctrl-D ends the input, Ctrl-C stops), `isatty()` is
true, and `/dev/tty` is raw input for editors (`!run kibi FILE`, serial console only: the framebuffer skips ANSI
escapes). Waiting for input does not count against the 60 s budget. The agent sees the same programs in its
`run_wasm` tool (no terminal, empty stdin), with the output cut at about 1500 tokens and only the exit status in
the result. Open files keep their FAT handle for the run (sqlite's random I/O), the kernel heap is 192 MiB
(`RINGZERO_HEAP_MIB`) and a run that drives it low is stopped instead of the kernel. Copy-paste demos:
[docs/DEMOS.md](docs/DEMOS.md); results: [docs/M8_RESULT.md](docs/M8_RESULT.md).

Constrained tool calls (M4.7a): once the model emits `<tool_call>`, the kernel holds every next token to a
grammar built from the tool schemas, in the native format `<function=NAME><parameter=KEY>VALUE</parameter>...
</function></tool_call>`: a known tool name, its own parameter names (each once), enum values (`set_setting`
keys), integers (`read_log n`), every required parameter before `</function>`, then `</tool_call>`; free
values stay free but cannot contain `</function>` or `<parameter=`. The grammar is a byte automaton; the
token the sampler picked is checked (1-2 us) and kept when it fits, otherwise every token the automaton
rejects gets a -inf logit and the sampler picks again (one allowed token is forced). The tokenizer vocab is a
trie (ids sorted by bytes + common-prefix lengths, 1.2 MiB, built at boot in 47 ms), so a mask walks only
the prefixes the grammar allows (0.2-0.3 ms). A call whose tool name, parameter or enum/integer value had to be
chosen by the mask is not run when the tool changes the machine. Outside a tool call it costs one test per token. `!set constrain off`
keeps the grammar as a checker only (it reports where a call breaks), for A/B runs. Every call prints a
`[grammar]` line. Results: [docs/M47_RESULT.md](docs/M47_RESULT.md).

Confidence of tool calls (T28, method from ProbeLogits, arXiv:2604.11943): at every decision token of a call (tool
name, a parameter choice with an optional one among the candidates, another optional parameter or the end, an enum
value, an integer digit) the kernel takes P(best) - P(second) of the softmax over the tokens the grammar allows,
grouped by outcome (tokens leading to the same name are summed, tokens that narrow nothing left out), from the logits
the pick used (a decode step or a verify row): no extra forward pass, the sampling is unchanged. The smallest margin
of the call sets the band: below `CONF_ASK` (default 0.80) ask `[confidence] ... Run it? [y/N]` first for
set_setting, remember, write_file and mkdir (rm, forget, poweroff, reboot ask y/N anyway) and run with a notice for
every other tool (run_code, run_wasm and the read-only tools; T54: program runs asked on 7 of 8 calls in a manual
test); at or above it run. `make iso CONF_ASK=N` (percent, 0..100) moves the line; the
logged middle band runs from `CONF_ASK` to 0.80 and is empty at the default. On the 120 held-out tasks of
`tools/confeval.py`: AUROC 0.80 against wrong calls (0.89 against wrong tools); below 0.80 sit 5 of the 8 wrong
calls and 12 of the 90 right ones (the tune-split pick, `CONF_ASK=25`, catches 0 of 8). 0.74 ms per call. Every call prints a `[confidence]` line; `!set confidence on|off` (default on).
Results: [docs/T28_RESULT.md](docs/T28_RESULT.md).

Hibernation (M4.7b): `/hibernate`, and every `!poweroff` (also the agent's poweroff tool), writes the current
conversation and its whole snapshot tree to `/var/lib/ringzero/hib/image.bin` (also `echo disk > /sys/power/state`): the point and name tables, the questions
and answers, the KV rows of every point, the DeltaNet state of the current point (the others come back
thinned: visiting one replays its tokens, checked against a hash taken at save time), the raw transcript and
the agent settings (temperature, think, constrain, compaction). The key is the boot snapshot key (model,
prefix ids, engine build, KV geometry). At boot a valid image is resumed after the boot prefix: same point,
branch and tree, same state hash (token ids + KV rows + DeltaNet state), so the next answer at temperature 0
is the one it would have been; one `hib:` line says so. A mismatched or damaged image is ignored with a
notice and the boot starts fresh. `/hibernate status`; `/hibernate discard` deletes the image and skips it at
the next `!poweroff`, so the next boot starts fresh.

Engine state snapshots: every finished turn is saved as a point (`#7`) in a 1 GiB pool (KV rows of the
turn + the 19.3 MiB DeltaNet state, about 21 MiB per short turn; going back costs about 0.6 ms). The
prompt shows the branch, `ringzero[main]> `, and `ringzero[main~1]> ` after an undo. Commands: `/undo [N]`,
`/redo [N]`, `/retry`, `/tree [all]`, `/branch NAME`, `/switch NAME|#ID`, `/save NAME`, `/load NAME`,
`/drop NAME|#ID`, `/snap`. Ctrl-C, an interrupted compaction and `/bench` `/fill` `/gen` restore the
conversation instead of resetting it, and `/undo` after a compaction returns to the uncompacted context.
Only the model goes back: tool side effects stay, and `/undo` says so. Design:
[docs/design/SNAPSHOTS.md](docs/design/SNAPSHOTS.md).

Model options (M7): Qwen3.5-2B or Qwen3.5-4B, in Q8_0 or Q4_0 (W4A8, own asm kernels, bit-exact against the Q8
kernel), see [Which model for which machine](#which-model-for-which-machine) and
[docs/M7_Q4_RESULT.md](docs/M7_Q4_RESULT.md).

Architecture diagram: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Results and how to connect to a
headless instance: [docs/M47_RESULT.md](docs/M47_RESULT.md) (earlier: [M5b WASI](docs/M5B_WASI_RESULT.md), [M5 MicroPython](docs/M5_MICROPYTHON_RESULT.md), [M4.5](docs/M45_RESULT.md), [M4](docs/M4_RESULT.md), [Embedder](docs/EMBEDDER_RESULT.md), [Snapshots v1](docs/SNAPSHOTS_V1_RESULT.md), [M2.5](docs/M25_RESULT.md),
[M2+M3](docs/M2M3_RESULT.md), [M0](docs/M0_RESULT.md), [M1](docs/M1_RESULT.md)).

## Which model for which machine

Every option is built from the same tree; pick it with `MODEL` and `QUANT` on every `make` command (`make model`,
`make iso`, `make run-*`). `make model` downloads and exports only the selected model. The CPU must have
AVX2 + FMA + F16C (the engine refuses to start otherwise); AVX-VNNI, when present, speeds up the batched prompt
prefill (picked at boot, same results bit for bit). Numbers: 8 vCPU KVM guest on a Ryzen AI 9 HX 370 shared with
another guest ([docs/M7_Q4_RESULT.md](docs/M7_Q4_RESULT.md)); quality = KLD and top-1 agreement vs the HF model.

| Option | Edition | Weights | Guest RAM (default) | Host RAM free | Decode tok/s | Prefill tok/s | Quality vs HF |
|---|---|---:|---:|---:|---:|---:|---|
| `MODEL=2b` (default, `QUANT=q8`) | `Qwen35@2b-Qwen3@0.6b` | 2.0 GB | 6G | 8 GiB | 34 | 160-230 | golden self-test (= HF greedy) |
| `MODEL=4b QUANT=q4` | `Qwen35@4b.q4-Qwen3@0.6b` | 3.4 GB | 8G | 10 GiB | 19-21 | 65-90 | KLD 0.028, top-1 92.7%, PPL +1.5% |
| `MODEL=4b QUANT=q4all` | `Qwen35@4b.q4all-Qwen3@0.6b` | 2.7 GB | 8G | 10 GiB | 24-25 | 50-85 | KLD 0.061, top-1 89.8%, PPL +4.1% (smallest, lossiest) |
| `MODEL=4b QUANT=q8` | `Qwen35@4b-Qwen3@0.6b` | 4.5 GB | 8G | 11 GiB | 14 | 65-85 | KLD 0.0013, top-1 98.4% (best quality) |
| `MODEL=2b QUANT=q4` / `q4all` | `Qwen35@2b.q4...` | ~1.6 (est.) / 1.3 GB | 6G | 8 GiB | 52 (q4all) | ~200 | Q4 kernel test bench, not a product target |
| 9B | - | (~6 GB Q4) | (~10G) | - | (est. 10-17) | - | not built: [feasibility study](docs/FEASIBILITY-q4-qwen35-9b.md) |

Short version: a 16 GB machine runs any row; the 4B Q4 (`q4`) is the best quality per GB, the 4B Q8 the best
quality, the 2B the fastest. The working model (the one used day to day) is the **4B q4**: best tool selection of the
three on the tool-call sets at about 21 tok/s ([docs/MODEL_CHOICE.md](docs/MODEL_CHOICE.md)); the build default
stays the 2B. Two Qwen3.5-4B fine-tunes (NeoHorse-1-4B, QwenPaw-Flash-4B) were tested as replacements and dropped: both fail the M7 quality gate at Q4 ([docs/MODEL_EVAL.md](docs/MODEL_EVAL.md)). Four agent and tool-calling fine-tunes (Agents-A1-4B, Qwen3.5-4B-EU-Tool, AgentMercury-Qwen3.5-4B, SkillGym-Qwen3.5-4B) were tested the same way in T64: none beats the stock 4B on the tool sets, so nothing changed. Every 4B option needs an 8G guest (6G no longer boots it since M5 / M5b). `QUANT=q4` keeps the attention and DeltaNet projections in Q8_0 because quantizing them
to Q4 doubles the error (`q4all`); embeddings / lm_head are Q8_0 and norms, conv and DeltaNet gates f32 in all options.

```sh
make model MODEL=4b QUANT=q4      # Qwen/Qwen3.5-4B (~9.3 GB download) -> models/qwen35-4b-q4_0mlp-v4.asmbin (+ embedder)
make iso MODEL=4b QUANT=q4        # -> build/ringzero.iso (the banner shows the edition)
make run-remote MODEL=4b QUANT=q4 # 8G guest (GUEST_MEM overrides): the working model, headless
make model MODEL=4b QUANT=q8 && make iso MODEL=4b QUANT=q8 && make run-remote MODEL=4b QUANT=q8   # 4B Q8, 8G guest
make model MODEL=4b QUANT=q4all && make iso MODEL=4b QUANT=q4all                                   # 4B, every projection Q4
make model && make iso && make run-remote                          # the default 2B Q8
```

## Build and run

On a Linux box with `cargo` (rustup, toolchain pinned in `kernel/rust-toolchain.toml`),
`nasm`, `xorriso`, `git`, `cc` (gcc), `python3` (MicroPython's header generator), `qemu-system-x86_64` with KVM, OVMF for UEFI, and
[uv](https://docs.astral.sh/uv/) for the model export. `make wasm-apps` also needs `rustup target add wasm32-wasip1`,
`curl` and mtools (M8: also `unzip`, `tar` and `sha256sum`; it downloads wasi-sdk, Binaryen and CPython, about
330 MB, into `build/wasm/dl`).

Quickstart:

```sh
make model               # Qwen/Qwen3.5-2B (~4.5 GB) -> models/qwen35-2b-q8_0-v3.asmbin (2 GB) and
                         # Qwen/Qwen3-Embedding-0.6B (1.2 GB) -> models/qwen3-embed-0.6b-q8_0-v3.asmbin (613 MiB),
                         # each skipped when the .asmbin is there
make iso                 # kernel + Limine + both models -> build/ringzero.iso
make disk                # build/disk.img, 512 MiB FAT32 (mkfs.vfat, mtools fallback); never overwrites it
make wasm-apps           # WASI programs -> build/wasm -> /bin on the disk (guest stopped; keeps every other file)
make run-remote          # then: nc HOST 4555 (serial shell, prompt ringzero[main]>), VNC HOST:5901 (password ringzero)
make run-local           # or: the same VM in a QEMU window on the box's desktop (own disk build/disk-local.img)
```

Try at the prompt: `!write /notes/pet.txt my dog is called Pixel and he loves carrots`, `!cat /notes/pet.txt`, `!ls /`, then ask
"create a file /notes/idea.txt that says hello", or after a few seconds of idle (the file is then in the memory
index) `/clear` and "what is my dog called?". Tell it "my name is Anna and my favourite number is 4711", run `/consolidate`, then
`/facts`; `!poweroff`, boot again and ask "what is my name?": the facts and the memory index are still there, and
the second boot loads the prompt state from the disk cache, and resumes the conversation from the
hibernation image (`/snap` shows the same state hash before the poweroff and after the boot). Python: `!py 17*23`,
`!write /hello.py print("hello", 2**64)` then `!py -f /hello.py`, or ask "what is 17*23? use python". WASI (after
`make wasm-apps`): `!wasm rztest check A "B C"` (the runtime's checks), `!run ls -l /bin`, `!run wc -l < /notes/todo.txt`,
`!wasm rg -n milk /notes`, `!write /a.js console.log(6*7)` then `!wasm qjs /a.js`, or ask "use run_wasm to
count the lines of /notes/todo.txt with wc".
M8 (common software, one block per tool in [docs/DEMOS.md](docs/DEMOS.md)): `!run python3 /demo/stats.py`,
`!run sqlite3 /demo/shop.db "select 1+1;"`, `!run jq .version /demo/data.json`, `!run lua /demo/hello.lua`, the REPLs
`!run python3` / `!run lua` / `!run sqlite3 FILE` (Ctrl-D ends them), `!run kibi /notes/todo.txt` on the serial
console, or ask "which tables are in the sqlite database /demo/shop.db?" (after the sqlite3 demo created it).

A headless instance on the build box (`make run-remote`, for example with `MODEL=4b QUANT=q4` on an 8G guest), used
from another machine (HOST = the build box's address):

```sh
nc HOST 4555                         # serial shell: press Enter to get "ringzero[main]> "
stty -echo -icanon; nc HOST 4555; stty sane    # without the double local echo (Tab works too)
open vnc://HOST:5901                 # framebuffer + PS/2 keyboard, password: ringzero
tail -f build/remote-serial.log      # on the build box: everything on serial
make run-local                       # at the build box's desktop: the same VM in a QEMU window
```

agentd at the prompt:

```
/agentd                  # stores, index (saved or changed), idle queue, consolidation state, last decisions
/agentd log 40           # the last 40 decisions (slot contents, facts accepted or rejected, jobs, recovery)
/agentd merge-legacy     # merge what an older kernel wrote after the migration (legacy/sys-<time>/), once
/facts                   # saved facts: key = value (source, time)
/remember KEY VALUE      # save a fact yourself; /forget KEY deletes one (asks y/N)
/consolidate             # episode + facts from this session now (Ctrl-C stops and restores the conversation)
/consolidate &           # M4.8a: the same on the background stream: back to the prompt at once, it decodes
                         # next to the next answers (the idle one runs this way too: typing does not stop it)
!set consolidate 30      # idle seconds before an automatic consolidation (0 = only /consolidate)
!set slot 0              # turn the memory slot off (default 200 tokens)
/index clear             # throw the index away and re-embed every file and memory record in idle time
```

Tool-call grammar and hibernation:

```
!set constrain off       # A/B: the grammar only checks tool calls (default on: it constrains them)
!set confidence off      # T28: no [confidence] line and no low-confidence question (default on)
!set spec off            # one token per pass (default on: prompt-lookup drafts, verified in one pass; N = longest draft)
/spec 9                  # verifier check: 9 positions per token vs one verify pass (bit-identical logits, times)
/multi 2                 # M4.8a: 2 streams x 64 tokens alone, then together: aggregate tok/s, ids + logits same
/hibernate               # save the conversation + snapshot tree now (!poweroff does it too)
/prefill 32              # time the boot prefix prefill (block 32; /prefill 1 = per token), print the state hash
/read /docs/a.txt        # put a text file into the conversation; its state is cached and loaded next time
/read                    # the document cache (entries, MiB, cap); /read clear deletes it
/hibernate status        # is there an image; /hibernate discard deletes it and !poweroff skips it (fresh next boot)
/snap                    # point, pool, timings and the state hash of the live state
```

Machine commands and the other settings (`!help` and `/help` print the same table; every `!set` key, its values and
default are in [openspec/specs/settings/spec.md](openspec/specs/settings/spec.md)):

```
!cpu                     # CPU model, clock, features
!uptime                  # time since boot
!stats                   # threads, parking, arena, heap, context fill, KV type, keyboard (also /stats)
!park off|idle|<usec>    # worker core parking: always spin, park between requests (default), or after a spin window
!reboot                  # reboot now (!poweroff powers off and writes the hibernation image)
!set temperature 0.7     # 0..2, 0 = greedy (default); /temp 0.7 is the same
!set cores 4             # cores used for inference, 1..online (default all); !cores 4 is the same
!set keyboard us         # PS/2 layout us|it (default it); !kbd us is the same
!set think on            # model thinking on|off (default off); /think on is the same
!set compact 60          # context fill percent that triggers a compaction, 1..95 (default 75)
!set snappool 512        # MiB the snapshot pool may use, 64..pool size (default the whole pool)
!set seed 7              # sampler RNG state, positive integer (default 42; stored with every snapshot point)
!set color off           # role colors on serial and the framebuffer on|off (default on)
!set rounds 5            # tool rounds per question, 1..20 (default 10, make iso ROUNDS=N), then an answer without tools
```

On/off settings (think, constrain, confidence, gramdraft, color) take on/1/true or off/0/false; any other value is
refused with `error: <key> must be on or off` and the setting keeps its value. `!set spec` takes on, off or a
longest draft from 1 to 15 and refuses anything else. An unknown key prints the list of every key. Saved in the
hibernation image: temperature, think, constrain, compact, color, rounds (and the snapshot pool cap with the tree).

Console colors (T46, palette in [assets/branding/README.md](assets/branding/README.md)): the prompt is bold red,
`[tool]` lines gold (ANSI yellow), `[memory]`, `[grammar]`, `[confidence]` and the `[prefill]`/`[decode]` statistics
gray, y/N questions and errors bright red; user input and answers keep the default color. Serial gets ANSI SGR codes;
the framebuffer parses SGR itself (0, 1 bold glyphs, 2 dim, 22, 30-37, 39, 90-97, 38;5;N, 38;2;R;G;B; other escapes
are skipped), so WASI programs' own colors (`!run jq -C . /demo/data.json`) show on VNC too. `!set color off` gives
plain text (programs' own escapes still pass). The serial test drivers in tools/ strip SGR codes before matching.
The boot banner (R0 die, name, version 0.1.0 and the channel "beta", release name QweQwe from the edition tag, edition, models, machine) uses the same
colors.

More:

```sh
make model HF_DIR=/path/to/hf/qwen3.5-2b   # reuse a download (config.json, safetensors, tokenizer*.json)
make model EMBED_HF_DIR=/path/to/hf/qwen3-embedding-0.6b   # same for the embedder
make iso                 # kernel + Limine (fetched, pinned tag) + model -> build/ringzero.iso
RINGZERO_KV=f32 make iso     # FP32 KV cache instead of FP16 (comparison build, 768 MiB of KV)
make iso CONF_ASK=25         # T28: ask y/N below a 0.25 confidence margin instead of 0.80 (0..100)
make iso ROUNDS=6            # T54: tool rounds per question before the answer without tools (1..20, default 10)
make iso LAYOUT_STOP=5       # T55 crash test builds only: halt the migration of a 0.1 disk after step N (a copy, the commit, a removal)
make run-bios            # QEMU/KVM, q35, 6 GiB (GUEST_MEM=6G), 8 CPUs, serial on stdio (Ctrl-C to quit)
make run-uefi            # same, with OVMF
make run-uefi QEMU_DISPLAY=gtk   # to see the framebuffer in a window
make run-remote          # headless: serial on TCP 4555 (nc HOST 4555), VNC on :1 (5901, password ringzero),
                         # serial output appended to build/remote-serial.log, monitor on build/remote-monitor.sock
make run-remote REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test DISK=build/disk-test.img   # a second instance
                         # (its own disk: QEMU's image lock refuses two guests writing one image)
make run-local           # QEMU window (-display gtk, PS/2 keys, guest layout !kbd it|us), serial on the terminal
                         # (Ctrl-C goes to the guest, close the window to quit) + build/local-serial.log, monitor on
                         # build/local-monitor.sock, disk build/disk-local.img; LOCAL_DISK=build/disk.img uses the
                         # remote's disk, only while run-remote is stopped
make disk DISK=build/disk-test.img DISK_MB=256   # another data disk (whole-volume FAT32, no partition table)
mdir -i build/disk.img ::/notes          # read the disk on the host with mtools (guest stopped);
                         # fsck.vfat -a build/disk.img clears the dirty flag after a guest that was killed
tail -f build/remote-serial.log                                       # watch the remote shell
python3 tools/screenshot.py build/remote-monitor.sock build/screen.png   # framebuffer screenshot
make iso MODEL=/path/to/qwen3-0.6b-q8_0.asmbin   # any other .asmbin (EMBED= for the embedder)
DISK=build/disk.img python3 tools/boottest.py build/ringzero.iso uefi build/t 8 /bench /stats   # headless: timed serial log,
                         # types each command at the ringzero[main]> prompt, saves screen.png
```

`MODEL=2b QUANT=q8` (the default) uses `models/qwen35-2b-q8_0-v3.asmbin`, `EMBED` defaults to
`models/qwen3-embed-0.6b-q8_0-v3.asmbin` (gitignored; a symlink to an existing file works). Other models: see
[Which model for which machine](#which-model-for-which-machine).

Memory index, from the shell:

```
/index add The backup job runs nightly at 02:00 with restic to a Backblaze bucket.
/search how are my files backed up?      # 1. #0  cos 0.536  bm25  0.00  emb  The backup job ... (top 5)
/embed -q how are my files backed up?    # dim, norm, token count, ms, first values (-v: all 1024)
/index stats                             # entries, dim, bytes, embed time per 100 tokens
/index dim 256                           # Matryoshka dim (multiple of 128, empty index only); /index clear
```

## Layout

| Path | What |
|---|---|
| `kernel/src/blk.rs`, `fs.rs`, `fidx.rs` | virtio-blk on PCI, FAT32 at /, file indexer (M4) |
| `kernel/src/procfs.rs` | /proc: read-only virtual files generated at read time (cpuinfo, meminfo, uptime, version, devices, kmsg, mounts, bus/pci/devices) (T54, T55) |
| `kernel/src/sysfs.rs`, `devfs.rs` | /sys (class/block, bus/pci/devices, devices/system/cpu, power/state) and /dev (null, zero, random, urandom, tty, console, vda) (T55) |
| `kernel/src/layout.rs` | /var/lib/ringzero paths, the migration of a 0.1 disk, /etc/hostname and the virtual /etc files, /tmp (T55) |
| `kernel/src/agentd.rs`, `mstore.rs` | agentd: idle scheduler, context compiler, consolidation, fact validation, decision log, merge-legacy; record stores in /var/lib/ringzero/mem (M4.5, T55) |
| `kernel/src/multi.rs` | multi-stream decode (M4.8a): extra stream slots, fork, one pass over rows of several streams, the background stream for consolidation, `/multi` |
| `kernel/src/gram.rs` | tool-call grammar: byte automaton from the tool schemas, vocab trie, logit mask (M4.7a) |
| `kernel/src/docstate.rs` | document states: `/read`, the cache in /var/lib/ringzero/cache/docs (keys, load checks, cap, eviction) (M4.8c) |
| `kernel/src/hib.rs` | hibernation: /hibernate, poweroff hook, resume at boot (image format in snap.rs) (M4.7b) |
| `kernel/src/wasm.rs` | WASI runtime (M5b): `!wasm`/`!run`, `run_wasm`, wasmi engine per run, fuel slices (budget, Ctrl-C), WASI preview1 host functions over fs.rs; the wasm memory slot is in `mem.rs` |
| `tools/wasm_apps.sh`, `tools/wasm/rztest/` | `make wasm-apps`: builds rztest (our WASI checks, also `bench`, `loop`, `mem`, `escape`), uutils coreutils and ripgrep for wasm32-wasip1, fetches QuickJS-ng's `qjs-wasi.wasm` |
| `tools/wasm/c/`, `tools/demo/`, `tools/m8test.py` | M8: setjmp/longjmp over Binaryen's Asyncify (`setjmp.h`, `rzsjlj.c`) and libc stubs (`nosys.c`) for the C programs; the `/demo` inputs of docs/DEMOS.md; a boot test that types steps when a regex matches (REPLs, the editor) |
| `kernel/src/upy.rs`, `kernel/upy/` | MicroPython (M5): Rust side (`!py`, `run_code`, budget, stdout capture); C glue, mini libc headers, `mpconfigport.h`; `make upy` fetches MicroPython and generates the embed package into `build/upy/`, `build.rs` compiles it |
| `kernel/` | Rust kernel crate (`x86_64-unknown-none`, higher-half linker script, `build.rs` assembles the engine) |
| `engine/` | vendored NASM engine (transformer.asm has marked ringzero changes: FP16 KV, max_seq override; matmul.asm: batch kernels) + `prefill.asm` (batched prefill, M4.8; per-row stream table, M4.8a) + `cpu.asm` (`enable_avx_fpu`, float shim) + `smp.asm` (thread-pool dispatch and worker loop, adapted from the asm OS) |
| `limine.conf` | boot entry: kernel + model and embedder as Limine modules |
| `tools/manvec.py` | T38 pass 1: boots an ISO headless, `/man dump`, checks and writes `build/manvec.bin` (the manual's vectors linked into the kernel) |
| `tools/boottest.py` | headless QEMU boot test (serial with timestamps, screenshot; `"CMD ^C3"` sends Ctrl-C 3 s after CMD) |
| `tools/tooleval.py` | tool-call A/B (M4.7a): the spike prompts + in-OS prompts with `!set constrain on|off`, scored (calls, malformed, schema errors, grammar lines, decode speed); T28: answers y to a `[confidence]` question, `CONF=on|off` |
| `tools/confeval.py` | T28: 240 tool tasks with a known right first call, `run` (first call only) and `analyze` (thresholds on the tune split, AUROC and bands on the held-out split) |
| `tools/accept.py` | drives the shell over TCP (answers y/N prompts, sends Ctrl-C, waits idle, powers off), saves a transcript |
| `tools/uxtest.py` | types steps byte by byte over TCP (Tab completion, suggestions, long pasted lines), waits for a quiet prompt |
| `tools/fetch_model.sh` | `make model`: Hugging Face download (uvx hf, curl fallback) + export, skipped when the .asmbin exists |
| `tools/export_weights.py` | safetensors -> .asmbin Q8_0 exporter, vendored from llm.asm (alesha-pro, MIT; Qwen3.5 v3 export from tcsenpai/llm.asm; `--v3` for Qwen3 / the embedder). Text-only Qwen3.5 checkpoints with tensors under `model.` instead of `model.language_model.` (fine-tunes such as NeoHorse-1-4B) are detected automatically; boot the file with `make iso MODEL=path/to/file.asmbin EDITION="..." GUEST_MEM=8G` |
| `tools/embed_ref.py` | HF fp32 reference embeddings (transformers via uv) vs the kernel's `/embed -v` output, cosine per string |
| `tools/retrieval.py` | the embedder spike's retrieval set as shell steps (`/index add`, `/search`) and its scorer |
| `tools/hf_prefix.py` | renders the agent prefix with the HF chat template, compares token ids with the kernel |
| `tools/kbdtest.py`, `tools/vnckeys.py` | keyboard tests: PS/2 via QEMU `sendkey`, VNC as a keysym-only client |
| `tools/keymaps/it-altgr` | AltGr entries appended to QEMU's `it` VNC keymap (`make run-remote`) |
| `site/`, `.github/workflows/pages.yml` | the project web page (static HTML + CSS, images copied from `assets/branding/`); the workflow deploys `site/` to GitHub Pages, run by hand only (`workflow_dispatch`), not published yet |

## License

GPL-2.0-or-later (see `LICENSE`). The vendored engine is MIT upstream (`engine/LICENSE`).
Dependencies must be MIT/Apache/BSD/zlib/MPL or GPL-2.0(-or-later); no GPL-3.0-only code,
so GPL-2.0-only Linux driver code stays borrowable.

| Dependency | Version | License | Use |
|---|---|---|---|
| `limine` (crate) | 0.5.0 | MIT OR Apache-2.0 | boot protocol requests |
| `uart_16550` | 0.9.0 | MIT OR Apache-2.0 | COM1 serial |
| `raw-cpuid` | 11.6.0 | MIT | CPU feature checks |
| `noto-sans-mono-bitmap` | 0.4.0 | MIT (crate metadata) | framebuffer glyphs |
| `spin` | 0.10.1 | MIT | console lock |
| `linked_list_allocator` | 0.10.6 | Apache-2.0 OR MIT | kernel heap |
| `virtio-drivers` (rcore-os) | 0.13.0 | MIT | virtio-blk + PCI transport |
| `fatfs` (rafalh/rust-fatfs, git rev 38b3a79, 0.4.0 unreleased) | 0.4.0 | MIT | FAT32 |
| `wasmi` + `wasmi_core`, `wasmi_ir`, `wasmi_collections` | 2.0.3 | MIT OR Apache-2.0 (used under MIT) | WASI runtime: wasm interpreter (M5b) |
| transitive (wasmi): `wasmparser` 0.228, `libm` 0.2.16, `spin` 0.9.9 | | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT (`wasmparser`), MIT | wasm decoding and validation, float math |
| transitive: `bitflags` 2.13, `lock_api` 0.4, `scopeguard` 1.2, `log` 0.4, `spinning_top` 0.2, `safe-mmio` 0.3, `thiserror` 2.0, `enumn` 0.1 (+ build-time `proc-macro2`, `quote`, `syn`, `unicode-ident`) | | MIT OR Apache-2.0 (`unicode-ident`: + Unicode-3.0) | |
| transitive: `zerocopy` 0.8 | | BSD-2-Clause OR Apache-2.0 OR MIT | virtio-drivers |
| `mkfs.vfat` (dosfstools) / `mtools` (host, `make disk`, not vendored) | | GPL-3.0-or-later (host tools, not linked) | disk image |
| QEMU `it` keymap (copied at run time, not vendored) | QEMU 10 | GPL-2.0-or-later | VNC keysym map |
| Limine bootloader (binary, fetched at build) | v11.2.1 | BSD-2-Clause | BIOS + UEFI loader |
| MicroPython (fetched at build by `make upy`, pinned tag + commit check, https://github.com/micropython/micropython) | v1.29.0, commit 0fd6c57 | MIT | Python interpreter (`py/` core via `ports/embed`) |
| MicroPython `lib/libm_dbl` (musl 1.1.16 libm, same fetch) | v1.29.0 | MIT | double-precision math for MicroPython |
| llm.asm engine by alesha-pro (vendored, https://github.com/alesha-pro/llm.asm) | fork commit 1c73255 | MIT | inference |
| uutils coreutils (`make wasm-apps`: built from crates.io for wasm32-wasip1, feature `feat_wasm`; not vendored, on the disk only) | 0.13.0 | MIT | `/bin/coreutils.wasm` |
| ripgrep (same, from crates.io) | 15.2.0 | Unlicense OR MIT | `/bin/rg.wasm` |
| QuickJS-ng (`qjs-wasi.wasm` from its GitHub release, not vendored, on the disk only) | v0.18.0 | MIT | `/bin/qjs.wasm` |
| M8, all below fetched or built by `make wasm-apps` (pinned version + SHA-256 in `tools/wasm_apps.sh`), not vendored, on the disk only | | | |
| jaq (crates.io, `--no-default-features`: no mimalloc) | 3.2.0 | MIT | `/bin/jq.wasm` |
| kibi (crates.io) | 0.3.2 | MIT OR Apache-2.0 | `/bin/kibi.wasm` |
| SQLite amalgamation (sqlite.org), shell + library | 3.54.1 | Public domain | `/bin/sqlite3.wasm` |
| Lua (lua.org) | 5.4.8 | MIT | `/bin/lua.wasm` |
| CPython wasm32-wasip1 build (brettcannon/cpython-wasi-build release, python.wasm + `lib/python3.15`) | 3.15.2 (wasi-sdk 33) | PSF-2.0 (bundled expat MIT, libmpdec BSD-2-Clause, HACL* MIT) | `/bin/python3.wasm`, `/lib/python3.15` |
| wasi-sdk (clang + wasi-libc, build tool; wasi-libc is linked into sqlite3 and lua) | 34 | Apache-2.0 WITH LLVM-exception; wasi-libc MIT + Apache-2.0 (musl parts MIT) | C programs for WASI |
| Binaryen `wasm-opt` (build tool only) | version_133 | Apache-2.0 | Asyncify pass for lua's setjmp/longjmp |
| `huggingface_hub` (`hf` CLI via uvx, not vendored), `numpy` (exporter, via uv) | latest | Apache-2.0, BSD-3-Clause | `make model` |
| Qwen3.5-2B weights (Qwen, downloaded by `make model`, not vendored) | Qwen/Qwen3.5-2B | Apache-2.0 | system agent (default) |
| Qwen3.5-4B weights (Qwen, downloaded by `make model MODEL=4b`, not vendored) | Qwen/Qwen3.5-4B | Apache-2.0 | system agent (`MODEL=4b`) |
| Qwen3-Embedding-0.6B weights (Qwen, downloaded by `make model`, not vendored) | Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | embedder, memory index |
| `torch`, `transformers` (`tools/embed_ref.py`, `q4_kld.py`, `hf_agree.py`; host only, not vendored) | latest | BSD-3-Clause, Apache-2.0 | reference checks |
