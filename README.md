# ringzero

ringzero is an experimental AI OS: a small Rust `no_std` x86-64 kernel, booted by Limine, with an LLM
(Qwen3.5-2B Q8_0) running in ring 0 as a system agent, and an embedding model (Qwen3-Embedding-0.6B Q8_0) next
to it on the same engine for a memory index. The inference engine is the NASM
assembly from [llm.asm](https://github.com/alesha-pro/llm.asm) by Alexey Fateev (alesha-pro), taken via our fork [tcsenpai/llm.asm](https://github.com/tcsenpai/llm.asm) (branch `qwen35-kernel`, which adds Qwen3.5), linked
into the kernel as objects.

Status: **Embedder in kernel** (edition "ringzero Qwen35@2b-Qwen3@0.6b", tag QweQwe). Boots under BIOS and UEFI, runs the APs in the engine's
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
has a `search_memory` tool over the same index. The index lives in RAM (lost at reboot) until the disk
milestone. Results: [docs/EMBEDDER_RESULT.md](docs/EMBEDDER_RESULT.md).

Engine state snapshots: every finished turn is saved as a point (`#7`) in a 1 GiB pool (KV rows of the
turn + the 19.3 MiB DeltaNet state, about 21 MiB per short turn; going back costs about 0.6 ms). The
prompt shows the branch, `ringzero[main]> `, and `ringzero[main~1]> ` after an undo. Commands: `/undo [N]`,
`/redo [N]`, `/retry`, `/tree [all]`, `/branch NAME`, `/switch NAME|#ID`, `/save NAME`, `/load NAME`,
`/drop NAME|#ID`, `/snap`. Ctrl-C, an interrupted compaction and `/bench` `/fill` `/gen` restore the
conversation instead of resetting it, and `/undo` after a compaction returns to the uncompacted context.
Only the model goes back: tool side effects stay, and `/undo` says so. Design:
[docs/design/SNAPSHOTS.md](docs/design/SNAPSHOTS.md).

Architecture diagram: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Results and how to connect to the
remote instance: [docs/EMBEDDER_RESULT.md](docs/EMBEDDER_RESULT.md) (earlier: [Snapshots v1](docs/SNAPSHOTS_V1_RESULT.md), [M2.5](docs/M25_RESULT.md),
[M2+M3](docs/M2M3_RESULT.md), [M0](docs/M0_RESULT.md), [M1](docs/M1_RESULT.md)).

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
make run-remote          # then: nc HOST 4555 (serial shell, prompt ringzero[main]>), VNC HOST:5901 (password aios)
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
make run-remote REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test   # a second instance next to it
tail -f build/remote-serial.log                                       # watch the remote shell
python3 tools/screenshot.py build/remote-monitor.sock build/screen.png   # framebuffer screenshot
make iso MODEL=/path/to/qwen3-0.6b-q8_0.asmbin   # other model file (EMBED= for the embedder)
python3 tools/boottest.py build/ringzero.iso uefi build/t 8 /bench /stats   # headless: timed serial log,
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
| transitive: `bitflags` 2.13, `lock_api` 0.4, `scopeguard` 1.2 | | MIT OR Apache-2.0 | |
| QEMU `it` keymap (copied at run time, not vendored) | QEMU 10 | GPL-2.0-or-later | VNC keysym map |
| Limine bootloader (binary, fetched at build) | v11.2.1 | BSD-2-Clause | BIOS + UEFI loader |
| llm.asm engine by alesha-pro (vendored, https://github.com/alesha-pro/llm.asm) | fork commit 1c73255 | MIT | inference |
| `huggingface_hub` (`hf` CLI via uvx, not vendored), `numpy` (exporter, via uv) | latest | Apache-2.0, BSD-3-Clause | `make model` |
| Qwen3.5-2B weights (Qwen, downloaded by `make model`, not vendored) | Qwen/Qwen3.5-2B | Apache-2.0 | system agent |
| Qwen3-Embedding-0.6B weights (Qwen, downloaded by `make model`, not vendored) | Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | embedder, memory index |
| `torch`, `transformers` (`tools/embed_ref.py` only, via uv, not vendored) | latest | BSD-3-Clause, Apache-2.0 | embedding reference check |
