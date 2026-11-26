# AIOS

Experimental "AI OS": a small Rust `no_std` x86-64 kernel, booted by Limine, with an LLM
(Qwen3.5-2B Q8_0) running in ring 0 as a system agent. The inference engine is the NASM
assembly from [llm.asm](https://github.com/tcsenpai/llm.asm) (branch `qwen35-kernel`), linked
into the kernel as objects.

Status: **M1**. Boots under BIOS and UEFI, runs the APs in the engine's worker loop, passes the
Qwen3.5-2B golden self-test and serves a chat REPL on COM1 (and PS/2), at the asm OS's speed.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/M0_RESULT.md](docs/M0_RESULT.md) and
[docs/M1_RESULT.md](docs/M1_RESULT.md) (how to connect to the remote instance).

## Build and run

On a Linux box with `cargo` (rustup, toolchain pinned in `kernel/rust-toolchain.toml`),
`nasm`, `xorriso`, `git`, `cc`, `qemu-system-x86_64` with KVM, and OVMF for UEFI:

```sh
make iso                 # kernel + Limine (fetched, pinned tag) + model -> build/aios.iso
make run-bios            # QEMU/KVM, q35, 4 GiB, 8 CPUs, serial on stdio (Ctrl-C to quit)
make run-uefi            # same, with OVMF
make run-uefi QEMU_DISPLAY=gtk   # to see the framebuffer in a window
make run-remote          # headless: serial on TCP 4555 (nc HOST 4555), VNC on :1 (5901, password aios)
make iso MODEL=/path/to/qwen3-0.6b-q8_0.asmbin   # other model file
python3 tools/boottest.py build/aios.iso uefi build/t 8 /bench /stats   # headless: timed serial log,
                         # types each command at the aios> prompt, saves screen.png
```

`MODEL` defaults to `~/coding/qwenbaremetal/models/qwen35-2b-q8_0-v3.asmbin`.

## Layout

| Path | What |
|---|---|
| `kernel/` | Rust kernel crate (`x86_64-unknown-none`, higher-half linker script, `build.rs` assembles the engine) |
| `engine/` | vendored NASM engine (unmodified) + `cpu.asm` (`enable_avx_fpu`, float shim) + `smp.asm` (thread-pool dispatch and worker loop, adapted from the asm OS) |
| `limine.conf` | boot entry: kernel + model as a Limine module |
| `tools/boottest.py` | headless QEMU boot test (serial with timestamps, screenshot) |

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
| Limine bootloader (binary, fetched at build) | v11.2.1 | BSD-2-Clause | BIOS + UEFI loader |
| llm.asm engine (vendored) | 1c73255 | MIT | inference |
