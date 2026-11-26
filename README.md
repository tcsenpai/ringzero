# ringzero

ringzero is an experimental AI OS: a small Rust `no_std` x86-64 kernel, booted by Limine, with an LLM
(Qwen3.5-2B Q8_0) running in ring 0 as a system agent. The inference engine is the NASM
assembly from [llm.asm](https://github.com/alesha-pro/llm.asm) by Alexey Fateev (alesha-pro), taken via our fork [tcsenpai/llm.asm](https://github.com/tcsenpai/llm.asm) (branch `qwen35-kernel`, which adds Qwen3.5), linked
into the kernel as objects.

Status: **M2.5** (edition "ringzero Qwen35@2b"). Boots under BIOS and UEFI, runs the APs in the engine's
worker loop, passes the Qwen3.5-2B golden self-test, then serves a shell on COM1 (and PS/2, layouts
`it`/`us`). Plain text talks to the system agent: the model calls kernel tools (memory, CPU, cores, log,
uptime, settings, poweroff/reboot with confirmation) in its native tool format. `!cmd` runs a machine
command (`!mem`, `!cores 4`, `!log 10`, `!kbd it`, `!set temperature 0.7`, `!poweroff`), `/cmd` a session
command (`/help`, `/clear`, `/ctx`, `/compact`, `/history`, `/bench`). Tab completes commands and their
arguments, a bare `/` or `!` lists them, up/down recall history. The context is 32768 tokens with an FP16
KV cache; past 75% fill the agent summarizes the conversation and rebuilds the context from the boot
snapshot. Architecture diagram: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Results and how to connect
to the remote instance: [docs/M25_RESULT.md](docs/M25_RESULT.md) (earlier: [M2+M3](docs/M2M3_RESULT.md),
[M0](docs/M0_RESULT.md), [M1](docs/M1_RESULT.md)).

## Build and run

On a Linux box with `cargo` (rustup, toolchain pinned in `kernel/rust-toolchain.toml`),
`nasm`, `xorriso`, `git`, `cc`, `qemu-system-x86_64` with KVM, OVMF for UEFI, and
[uv](https://docs.astral.sh/uv/) for the model export.

Quickstart:

```sh
make model               # Qwen/Qwen3.5-2B from Hugging Face (~4.5 GB) -> models/qwen35-2b-q8_0-v3.asmbin (2 GB)
make iso                 # kernel + Limine + model -> build/ringzero.iso
make run-remote          # then: nc HOST 4555 (serial shell), VNC HOST:5901 (password aios)
```

More:

```sh
make model HF_DIR=/path/to/hf/qwen3.5-2b   # reuse a download (config.json, safetensors, tokenizer*.json)
make iso                 # kernel + Limine (fetched, pinned tag) + model -> build/ringzero.iso
RINGZERO_KV=f32 make iso     # FP32 KV cache instead of FP16 (comparison build, 768 MiB of KV)
make run-bios            # QEMU/KVM, q35, 4 GiB, 8 CPUs, serial on stdio (Ctrl-C to quit)
make run-uefi            # same, with OVMF
make run-uefi QEMU_DISPLAY=gtk   # to see the framebuffer in a window
make run-remote          # headless: serial on TCP 4555 (nc HOST 4555), VNC on :1 (5901, password aios),
                         # serial output appended to build/remote-serial.log, monitor on build/remote-monitor.sock
make run-remote REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test   # a second instance next to it
tail -f build/remote-serial.log                                       # watch the remote shell
python3 tools/screenshot.py build/remote-monitor.sock build/screen.png   # framebuffer screenshot
make iso MODEL=/path/to/qwen3-0.6b-q8_0.asmbin   # other model file
python3 tools/boottest.py build/ringzero.iso uefi build/t 8 /bench /stats   # headless: timed serial log,
                         # types each command at the ringzero> prompt, saves screen.png
```

`MODEL` defaults to `models/qwen35-2b-q8_0-v3.asmbin` (gitignored; a symlink to an existing file works).

## Layout

| Path | What |
|---|---|
| `kernel/` | Rust kernel crate (`x86_64-unknown-none`, higher-half linker script, `build.rs` assembles the engine) |
| `engine/` | vendored NASM engine (transformer.asm has marked AIOS changes: FP16 KV, max_seq override) + `cpu.asm` (`enable_avx_fpu`, float shim) + `smp.asm` (thread-pool dispatch and worker loop, adapted from the asm OS) |
| `limine.conf` | boot entry: kernel + model as a Limine module |
| `tools/boottest.py` | headless QEMU boot test (serial with timestamps, screenshot) |
| `tools/accept.py` | drives the shell over TCP (answers y/N prompts, sends Ctrl-C), saves a transcript |
| `tools/uxtest.py` | types steps byte by byte over TCP (Tab completion, suggestions, long pasted lines), waits for a quiet prompt |
| `tools/fetch_model.sh` | `make model`: Hugging Face download (uvx hf, curl fallback) + export |
| `tools/export_weights.py` | safetensors -> .asmbin Q8_0 exporter, vendored from llm.asm (alesha-pro, MIT; Qwen3.5 v3 export from tcsenpai/llm.asm) |
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
