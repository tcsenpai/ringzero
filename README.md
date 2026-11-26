# ringzero

ringzero is an experimental AI OS: a small Rust `no_std` x86-64 kernel, booted by Limine, with an LLM
(Qwen3.5-2B Q8_0) running in ring 0 as a system agent, and an embedding model (Qwen3-Embedding-0.6B Q8_0) next
to it on the same engine for a memory index. The inference engine is the NASM
assembly from [llm.asm](https://github.com/alesha-pro/llm.asm) by Alexey Fateev (alesha-pro), taken via our fork [tcsenpai/llm.asm](https://github.com/tcsenpai/llm.asm) (branch `qwen35-kernel`, which adds Qwen3.5), linked
into the kernel as objects.

Status: **M4.7 constrained tool calls + hibernation** (with M5 MicroPython; edition "ringzero Qwen35@2b-Qwen3@0.6b", tag QweQwe). Boots under BIOS and UEFI, runs the APs in the engine's
worker loop, passes the Qwen3.5-2B golden self-test, then serves a shell on COM1 (and PS/2, layouts
`it`/`us`). Plain text talks to the system agent: the model calls kernel tools (memory, CPU, cores, log,
uptime, settings, poweroff/reboot with confirmation) in its native tool format. `!cmd` runs a machine
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

Embedder: Qwen3-Embedding-0.6B runs on the same asm engine by swapping the engine context (`g_ctx`) between
whole forward calls, so the 2B's state is never touched. `/embed [-q] TEXT` prints a vector (dim 1024, about
6 ms per token, 0.61 s per 100 tokens; cosine >= 0.9986 against the HF fp32 model), `/index add TEXT` stores a
note in the RAM memory index (Q8_0 rows), `/search QUERY` returns the top 5 by embedding similarity, with BM25
joining only for identifier-like queries or a strong keyword match, and shows which signal matched. The agent
has a `recall` tool over the same index (through agentd, below). Results: [docs/EMBEDDER_RESULT.md](docs/EMBEDDER_RESULT.md).

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
- Persistent stores in `/sys/mem` on the disk: an append-only journal of turns and tool results, facts (key ->
  value, source, time) and episodes (session summaries). Records carry a length and a checksum; at boot each file
  is scanned to its last good record and a torn tail is cut.
- The memory index is saved to `/sys/mem/index.bin` with a header (embedder hash, dim, Q8_0 layout, chunker
  version, the size and mtime of each indexed file) and loaded at boot in about a millisecond; only files that
  changed on the disk (also from the host) and new records are embedded again.
- Boot snapshot cache: the prefilled system prompt + tools state (33 MiB) is saved to `/sys/bootsnap.bin` and
  loaded at the next boot in about 0.2 s instead of a prefill (BIOS boot to the prompt 40.6 s cold, 11.9 s
  cached, before M4.8). Since the batched prefill (M4.8) a miss prefills the 1315-token prefix in 5.8 s instead of
  36-40 s, and a cold boot reaches the prompt in about 28 s instead of 71 s (QEMU start to prompt, shared box). The key covers the model, the prefix tokens and the engine build; any change means a prefill.
- Context compiler: the first message of every conversation, and the first turn after a compaction, gets a
  memory slot of at most 200 tokens: saved facts, then notes recalled for that message (earlier statements of the
  user, episodes, file chunks). This fixes the M2.5 compaction recall failure.
- Idle work runs in one priority queue at the prompt (embed memory texts, embed file chunks, save the index,
  consolidate), in slices that any keystroke stops between engine steps. Consolidation (after 60 s idle, or
  `/consolidate`) has the 2B summarize the session into an episode and propose facts; the kernel validates them
  (snake_case key, short value, grounded in what the user said, dedup) and the user's conversation is restored
  to its snapshot point after.
Results: [docs/M45_RESULT.md](docs/M45_RESULT.md).

Python (M5): MicroPython 1.29.0 (`ports/embed`) is compiled freestanding into the kernel and runs in ring 0
with an 8 MiB GC heap and its own 1 MiB stack from the engine arena (`RINGZERO_UPY_HEAP_MIB=16 make iso` for
another heap size). `!py CODE` runs one line (an expression's value is echoed, like the REPL), `!py -f PATH`
runs a script from the disk; each run is a fresh interpreter and ends with a status line (time, heap used).
Exceptions print a traceback (long ones cut to the first and last 5 frames) and the kernel goes on. The time
budget (10 s at the shell, 5 s for the agent) and Ctrl-C raise `KeyboardInterrupt`; a program that swallows it
is aborted 1 s later. The agent has a `run_code` tool: it gets the captured output, the traceback and the
status line (6 KiB cap). Included: builtins, f-strings, big ints, double floats, `math`, `cmath`, `sys`, `gc`,
`array`, `collections`, `struct`, `errno`; no `import` from files, no `input()`, no I/O modules yet. Results:
[docs/M5_MICROPYTHON_RESULT.md](docs/M5_MICROPYTHON_RESULT.md).

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

Hibernation (M4.7b): `/hibernate`, and every `!poweroff` (also the agent's poweroff tool), writes the current
conversation and its whole snapshot tree to `/sys/hib/image.bin`: the point and name tables, the questions
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

Architecture diagram: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Results and how to connect to the
remote instance: [docs/M47_RESULT.md](docs/M47_RESULT.md) (earlier: [M5 MicroPython](docs/M5_MICROPYTHON_RESULT.md), [M4.5](docs/M45_RESULT.md), [M4](docs/M4_RESULT.md), [Embedder](docs/EMBEDDER_RESULT.md), [Snapshots v1](docs/SNAPSHOTS_V1_RESULT.md), [M2.5](docs/M25_RESULT.md),
[M2+M3](docs/M2M3_RESULT.md), [M0](docs/M0_RESULT.md), [M1](docs/M1_RESULT.md)).

## Build and run

On a Linux box with `cargo` (rustup, toolchain pinned in `kernel/rust-toolchain.toml`),
`nasm`, `xorriso`, `git`, `cc` (gcc), `python3` (MicroPython's header generator), `qemu-system-x86_64` with KVM, OVMF for UEFI, and
[uv](https://docs.astral.sh/uv/) for the model export.

Quickstart:

```sh
make model               # Qwen/Qwen3.5-2B (~4.5 GB) -> models/qwen35-2b-q8_0-v3.asmbin (2 GB) and
                         # Qwen/Qwen3-Embedding-0.6B (1.2 GB) -> models/qwen3-embed-0.6b-q8_0-v3.asmbin (613 MiB),
                         # each skipped when the .asmbin is there
make iso                 # kernel + Limine + both models -> build/ringzero.iso
make disk                # build/disk.img, 512 MiB FAT32 (mkfs.vfat, mtools fallback); never overwrites it
make run-remote          # then: nc HOST 4555 (serial shell, prompt ringzero[main]>), VNC HOST:5901 (password aios)
make run-local           # or: the same VM in a QEMU window on the box's desktop (own disk build/disk-local.img)
```

Try at the prompt: `!write /notes/todo.txt comprare il latte`, `!cat /notes/todo.txt`, `!ls /`, then ask
"crea un file /notes/idea.txt con scritto ciao" or "cosa devo comprare?" (after a few seconds of idle the file
is in the memory index). Tell it "mi chiamo Cris e il mio numero preferito e 4711", run `/consolidate`, then
`/facts`; `!poweroff`, boot again and ask "come mi chiamo?": the facts and the memory index are still there, and
the second boot loads the prompt state from the disk cache, and resumes the conversation from the
hibernation image (`/snap` shows the same state hash before the poweroff and after the boot). Python: `!py 17*23`,
`!write /hello.py print("ciao", 2**64)` then `!py -f /hello.py`, or ask "quanto fa 17*23? usa python".

Remote instance on the Fedora box (Tailscale HOST-ADDRESS):

```sh
nc HOST-ADDRESS 4555                 # serial shell: press Enter to get "ringzero[main]> "
stty -echo -icanon; nc HOST-ADDRESS 4555; stty sane    # without the double local echo (Tab works too)
open vnc://HOST-ADDRESS:5901         # framebuffer + PS/2 keyboard, password: aios
ssh fedorascale 'tail -f ~/coding/ringzero/build/remote-serial.log'                  # everything on serial
ssh fedorascale 'cd ~/coding/ringzero && make run-local'   # at the Fedora desktop: the same VM in a QEMU window
```

agentd at the prompt:

```
/agentd                  # stores, index (saved or changed), idle queue, consolidation state, last decisions
/agentd log 40           # the last 40 decisions (slot contents, facts accepted or rejected, jobs, recovery)
/facts                   # saved facts: key = value (source, time)
/remember KEY VALUE      # save a fact yourself; /forget KEY deletes one (asks y/N)
/consolidate             # episode + facts from this session now (Ctrl-C stops and restores the conversation)
!set consolidate 30      # idle seconds before an automatic consolidation (0 = only /consolidate)
!set slot 0              # turn the memory slot off (default 200 tokens)
/index clear             # throw the index away and re-embed every file and memory record in idle time
```

Tool-call grammar and hibernation:

```
!set constrain off       # A/B: the grammar only checks tool calls (default on: it constrains them)
/hibernate               # save the conversation + snapshot tree now (!poweroff does it too)
/prefill 32              # time the boot prefix prefill (block 32; /prefill 1 = per token), print the state hash
/hibernate status        # is there an image; /hibernate discard deletes it and !poweroff skips it (fresh next boot)
/snap                    # point, pool, timings and the state hash of the live state
```

More:

```sh
make model HF_DIR=/path/to/hf/qwen3.5-2b   # reuse a download (config.json, safetensors, tokenizer*.json)
make model EMBED_HF_DIR=/path/to/hf/qwen3-embedding-0.6b   # same for the embedder
make iso                 # kernel + Limine (fetched, pinned tag) + model -> build/ringzero.iso
RINGZERO_KV=f32 make iso     # FP32 KV cache instead of FP16 (comparison build, 768 MiB of KV)
make run-bios            # QEMU/KVM, q35, 6 GiB (GUEST_MEM=6G), 8 CPUs, serial on stdio (Ctrl-C to quit)
make run-uefi            # same, with OVMF
make run-uefi QEMU_DISPLAY=gtk   # to see the framebuffer in a window
make run-remote          # headless: serial on TCP 4555 (nc HOST 4555), VNC on :1 (5901, password aios),
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
make iso MODEL=/path/to/qwen3-0.6b-q8_0.asmbin   # other model file (EMBED= for the embedder)
DISK=build/disk.img python3 tools/boottest.py build/ringzero.iso uefi build/t 8 /bench /stats   # headless: timed serial log,
                         # types each command at the ringzero[main]> prompt, saves screen.png
```

`MODEL` defaults to `models/qwen35-2b-q8_0-v3.asmbin`, `EMBED` to `models/qwen3-embed-0.6b-q8_0-v3.asmbin`
(gitignored; a symlink to an existing file works).

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
| `kernel/src/agentd.rs`, `mstore.rs` | agentd: idle scheduler, context compiler, consolidation, fact validation, decision log; record stores in /sys/mem (M4.5) |
| `kernel/src/gram.rs` | tool-call grammar: byte automaton from the tool schemas, vocab trie, logit mask (M4.7a) |
| `kernel/src/hib.rs` | hibernation: /hibernate, poweroff hook, resume at boot (image format in snap.rs) (M4.7b) |
| `kernel/src/upy.rs`, `kernel/upy/` | MicroPython (M5): Rust side (`!py`, `run_code`, budget, stdout capture); C glue, mini libc headers, `mpconfigport.h`; `make upy` fetches MicroPython and generates the embed package into `build/upy/`, `build.rs` compiles it |
| `kernel/` | Rust kernel crate (`x86_64-unknown-none`, higher-half linker script, `build.rs` assembles the engine) |
| `engine/` | vendored NASM engine (transformer.asm has marked AIOS changes: FP16 KV, max_seq override; matmul.asm: batch kernels) + `prefill.asm` (batched prefill, M4.8) + `cpu.asm` (`enable_avx_fpu`, float shim) + `smp.asm` (thread-pool dispatch and worker loop, adapted from the asm OS) |
| `limine.conf` | boot entry: kernel + model and embedder as Limine modules |
| `tools/boottest.py` | headless QEMU boot test (serial with timestamps, screenshot) |
| `tools/tooleval.py` | tool-call A/B (M4.7a): the spike prompts + in-OS prompts with `!set constrain on|off`, scored (calls, malformed, schema errors, grammar lines, decode speed) |
| `tools/accept.py` | drives the shell over TCP (answers y/N prompts, sends Ctrl-C, waits idle, powers off), saves a transcript |
| `tools/uxtest.py` | types steps byte by byte over TCP (Tab completion, suggestions, long pasted lines), waits for a quiet prompt |
| `tools/fetch_model.sh` | `make model`: Hugging Face download (uvx hf, curl fallback) + export, skipped when the .asmbin exists |
| `tools/export_weights.py` | safetensors -> .asmbin Q8_0 exporter, vendored from llm.asm (alesha-pro, MIT; Qwen3.5 v3 export from tcsenpai/llm.asm; `--v3` for Qwen3 / the embedder) |
| `tools/embed_ref.py` | HF fp32 reference embeddings (transformers via uv) vs the kernel's `/embed -v` output, cosine per string |
| `tools/retrieval.py` | the embedder spike's retrieval set as shell steps (`/index add`, `/search`) and its scorer |
| `tools/hf_prefix.py` | renders the agent prefix with the HF chat template, compares token ids with the kernel |
| `tools/kbdtest.py`, `tools/vnckeys.py` | keyboard tests: PS/2 via QEMU `sendkey`, VNC as a keysym-only client |
| `tools/keymaps/it-altgr` | AltGr entries appended to QEMU's `it` VNC keymap (`make run-remote`) |

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
| transitive: `bitflags` 2.13, `lock_api` 0.4, `scopeguard` 1.2, `log` 0.4, `spinning_top` 0.2, `safe-mmio` 0.3, `thiserror` 2.0, `enumn` 0.1 (+ build-time `proc-macro2`, `quote`, `syn`, `unicode-ident`) | | MIT OR Apache-2.0 (`unicode-ident`: + Unicode-3.0) | |
| transitive: `zerocopy` 0.8 | | BSD-2-Clause OR Apache-2.0 OR MIT | virtio-drivers |
| `mkfs.vfat` (dosfstools) / `mtools` (host, `make disk`, not vendored) | | GPL-3.0-or-later (host tools, not linked) | disk image |
| QEMU `it` keymap (copied at run time, not vendored) | QEMU 10 | GPL-2.0-or-later | VNC keysym map |
| Limine bootloader (binary, fetched at build) | v11.2.1 | BSD-2-Clause | BIOS + UEFI loader |
| MicroPython (fetched at build by `make upy`, pinned tag + commit check, https://github.com/micropython/micropython) | v1.29.0, commit 0fd6c57 | MIT | Python interpreter (`py/` core via `ports/embed`) |
| MicroPython `lib/libm_dbl` (musl 1.1.16 libm, same fetch) | v1.29.0 | MIT | double-precision math for MicroPython |
| llm.asm engine by alesha-pro (vendored, https://github.com/alesha-pro/llm.asm) | fork commit 1c73255 | MIT | inference |
| `huggingface_hub` (`hf` CLI via uvx, not vendored), `numpy` (exporter, via uv) | latest | Apache-2.0, BSD-3-Clause | `make model` |
| Qwen3.5-2B weights (Qwen, downloaded by `make model`, not vendored) | Qwen/Qwen3.5-2B | Apache-2.0 | system agent |
| Qwen3-Embedding-0.6B weights (Qwen, downloaded by `make model`, not vendored) | Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | embedder, memory index |
| `torch`, `transformers` (`tools/embed_ref.py` only, via uv, not vendored) | latest | BSD-3-Clause, Apache-2.0 | embedding reference check |
