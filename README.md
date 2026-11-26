# ringzero

ringzero is an experimental AI OS: a small Rust `no_std` x86-64 kernel, booted by Limine, with an LLM
(Qwen3.5-2B Q8_0) running in ring 0 as a system agent, and an embedding model (Qwen3-Embedding-0.6B Q8_0) next
to it on the same engine for a memory index. The inference engine is the NASM
assembly from [llm.asm](https://github.com/alesha-pro/llm.asm) by Alexey Fateev (alesha-pro), taken via our fork [tcsenpai/llm.asm](https://github.com/tcsenpai/llm.asm) (branch `qwen35-kernel`, which adds Qwen3.5), linked
into the kernel as objects.

Status: **M4 disk + filesystem** (edition "ringzero Qwen35@2b-Qwen3@0.6b", tag QweQwe). Boots under BIOS and UEFI, runs the APs in the engine's
worker loop, passes the Qwen3.5-2B golden self-test, then serves a shell on COM1 (and PS/2, layouts
`it`/`us`). Plain text talks to the system agent: the model calls kernel tools (memory, CPU, cores, log,
uptime, settings, poweroff/reboot with confirmation) in its native tool format. `!cmd` runs a machine
command (`!mem`, `!cores 4`, `!log 10`, `!kbd it`, `!set temperature 0.7`, `!poweroff`), `/cmd` a session
command (`/help`, `/clear`, `/ctx`, `/compact`, `/history`, `/bench`). Tab completes commands and their
arguments, a bare `/` or `!` lists them, up/down recall history. The context is 32768 tokens with an FP16
KV cache; past 75% fill the agent summarizes the conversation and rebuilds the context from the boot
snapshot.

Embedder: Qwen3-Embedding-0.6B runs on the same asm engine by swapping the engine context (`g_ctx`) between
whole forward calls, so the 2B's state is never touched. `/embed [-q] TEXT` prints a vector (dim 1024, about
6 ms per token, 0.61 s per 100 tokens; cosine >= 0.9986 against the HF fp32 model), `/index add TEXT` stores a
note in the RAM memory index (Q8_0 rows), `/search QUERY` returns the top 5 by embedding similarity, with BM25
joining only for identifier-like queries or a strong keyword match, and shows which signal matched. The agent
has a `search_memory` tool over the same index. The index lives in RAM (lost at reboot; persistence is M4.5).
Results: [docs/EMBEDDER_RESULT.md](docs/EMBEDDER_RESULT.md).

Disk (M4): a virtio-blk disk (`build/disk.img`, 512 MiB FAT32, made once by `make disk`) is mounted at `/` at
boot; the banner prints its size and free space. Shell: `!ls [PATH]`, `!cat PATH`, `!write PATH TEXT` (`\n` =
new line, parents created), `!mkdir PATH`, `!rm [-r] PATH` (asks y/N), `!df`, `!sync` (poweroff and reboot sync
too). The agent has `ls`, `cat` (8 KiB tool cap), `write_file`, `mkdir` and `rm` (always asks the human y/N);
paths must be absolute, without `.` or `..`, at most 255 bytes, files at most 64 KiB. Text files written through
the shell or the agent are chunked (~1000 bytes, ~200-250 tokens) into the memory index in idle time at the
prompt, in slices that any keystroke or Ctrl-C ends within one embedder forward call; `/index stats` shows the
queue, and `/search` or `search_memory` return the chunks as `[/path] text`. Results:
[docs/M4_RESULT.md](docs/M4_RESULT.md).

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

Architecture diagram: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Results and how to connect to the
remote instance: [docs/M4_RESULT.md](docs/M4_RESULT.md) (earlier: [Embedder](docs/EMBEDDER_RESULT.md), [Snapshots v1](docs/SNAPSHOTS_V1_RESULT.md), [M2.5](docs/M25_RESULT.md),
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
| `MODEL=4b QUANT=q4` | `Qwen35@4b.q4-Qwen3@0.6b` | 3.4 GB | 6G (128 MiB snapshot pool; 8G for more) | 8 GiB | 19 | 65-90 | KLD 0.028, top-1 92.7%, PPL +1.5% |
| `MODEL=4b QUANT=q4all` | `Qwen35@4b.q4all-Qwen3@0.6b` | 2.7 GB | 6G | 8 GiB | 24 | 50-85 | KLD 0.061, top-1 89.8%, PPL +4.1% (smallest, lossiest) |
| `MODEL=4b QUANT=q8` | `Qwen35@4b-Qwen3@0.6b` | 4.5 GB | 8G | 11 GiB | 14 | 65-85 | KLD 0.0013, top-1 98.4% (best quality) |
| `MODEL=2b QUANT=q4` / `q4all` | `Qwen35@2b.q4...` | ~1.6 (est.) / 1.3 GB | 6G | 8 GiB | 52 (q4all) | ~200 | Q4 kernel test bench, not a product target |
| 9B | - | (~6 GB Q4) | (~10G) | - | (est. 10-17) | - | not built: [feasibility study](docs/FEASIBILITY-q4-qwen35-9b.md) |

Short version: a 16 GB machine runs any row; the 4B Q4 (`q4`) is the best quality per GB, the 4B Q8 the best
quality, the 2B the fastest. `QUANT=q4` keeps the attention and DeltaNet projections in Q8_0 because quantizing them
to Q4 doubles the error (`q4all`); embeddings / lm_head are Q8_0 and norms, conv and DeltaNet gates f32 in all options.

```sh
make model MODEL=4b QUANT=q4      # Qwen/Qwen3.5-4B (~9.3 GB download) -> models/qwen35-4b-q4_0mlp-v4.asmbin (+ embedder)
make iso MODEL=4b QUANT=q4        # -> build/ringzero.iso (the banner shows the edition)
make run-remote MODEL=4b QUANT=q4 # guest RAM from the table (GUEST_MEM=8G overrides)
make model MODEL=4b QUANT=q8 && make iso MODEL=4b QUANT=q8 && make run-remote MODEL=4b QUANT=q8   # 4B Q8, 8G guest
make model MODEL=4b QUANT=q4all && make iso MODEL=4b QUANT=q4all                                   # 4B, every projection Q4
make model && make iso && make run-remote                          # the default 2B Q8
```

## Build and run

On a Linux box with `cargo` (rustup, toolchain pinned in `kernel/rust-toolchain.toml`),
`nasm`, `xorriso`, `git`, `cc`, `qemu-system-x86_64` with KVM, OVMF for UEFI, and
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
is in the memory index). `!poweroff` and boot again: the files are still there.

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
| `kernel/` | Rust kernel crate (`x86_64-unknown-none`, higher-half linker script, `build.rs` assembles the engine) |
| `engine/` | vendored NASM engine (transformer.asm has marked AIOS changes: FP16 KV, max_seq override) + `cpu.asm` (`enable_avx_fpu`, float shim) + `smp.asm` (thread-pool dispatch and worker loop, adapted from the asm OS) |
| `limine.conf` | boot entry: kernel + model and embedder as Limine modules |
| `tools/boottest.py` | headless QEMU boot test (serial with timestamps, screenshot) |
| `tools/accept.py` | drives the shell over TCP (answers y/N prompts, sends Ctrl-C), saves a transcript |
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
| llm.asm engine by alesha-pro (vendored, https://github.com/alesha-pro/llm.asm) | fork commit 1c73255 | MIT | inference |
| `huggingface_hub` (`hf` CLI via uvx, not vendored), `numpy` (exporter, via uv) | latest | Apache-2.0, BSD-3-Clause | `make model` |
| Qwen3.5-2B weights (Qwen, downloaded by `make model`, not vendored) | Qwen/Qwen3.5-2B | Apache-2.0 | system agent (default) |
| Qwen3.5-4B weights (Qwen, downloaded by `make model MODEL=4b`, not vendored) | Qwen/Qwen3.5-4B | Apache-2.0 | system agent (`MODEL=4b`) |
| Qwen3-Embedding-0.6B weights (Qwen, downloaded by `make model`, not vendored) | Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | embedder, memory index |
| `torch`, `transformers` (`tools/embed_ref.py`, `q4_kld.py`, `hf_agree.py`; host only, not vendored) | latest | BSD-3-Clause, Apache-2.0 | reference checks |
