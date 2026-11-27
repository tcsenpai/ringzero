# ringzero 0.1.0 "QweQwe" (beta)

First public beta of ringzero, an experimental x86-64 operating system where a language model (Qwen3.5) and an
embedding model (Qwen3-Embedding-0.6B) run in ring 0 on one inference engine written in assembly.

Website: https://tcsenpai.github.io/ringzero-site/

## Downloads

| File | Where | What |
|---|---|---|
| `ringzero-beta-4b-q4.iso` | Hugging Face: https://huggingface.co/Pleistocene/ringzero | recommended: Qwen3.5-4B (Q4 MLP) + Qwen3-Embedding-0.6B |
| `ringzero-beta-2b-q8.iso` | Hugging Face | Qwen3.5-2B Q8 + Qwen3-Embedding-0.6B: smaller and faster |
| `ringzero-beta-disk.img.xz` | this release | data disk with python3, sqlite3, jq, lua, QuickJS, ripgrep, coreutils, an editor and the demo files |
| `SHA256SUMS` | both | checksums of every file |

The ISOs are larger than the 2 GiB limit of GitHub release files, so they are hosted on Hugging Face.

## Requirements

Measured on QEMU/KVM, 8 vCPUs (details in docs/BETA.md):

| | 4B q4 | 2B q8 |
|---|---|---|
| guest RAM (default) | 8 GB | 6 GB |
| smallest that works | 7 GB (fewer undo points) | 4 GB |
| host RAM free | guest RAM + about 1 GB | guest RAM + about 1 GB |

- CPU: x86-64 with AVX2, FMA, F16C and XSAVE, passed to the guest (`-cpu host`). AVX-512 and VNNI are not needed.
- Linux host with KVM. The data disk is 512 MiB.

## Run

Verify and unpack:

```sh
sha256sum -c SHA256SUMS --ignore-missing
xz -d ringzero-beta-disk.img.xz && cp ringzero-beta-disk.img my-disk.img
```

Serial console in the terminal (Ctrl-A X quits; `!poweroff` inside ringzero saves the session):

```sh
qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m 8G -smp 8 \
  -cdrom ringzero-beta-4b-q4.iso -boot d -no-reboot -display none -serial mon:stdio \
  -drive file=my-disk.img,if=none,id=disk0,format=raw -device virtio-blk-pci,drive=disk0
```

For a window replace `-display none` with `-vga std -display gtk`. For the 2B use `ringzero-beta-2b-q8.iso` and
`-m 6G`. Then type a sentence, `/help`, or `/man TOPIC`. Examples: docs/DEMOS.md.

## What works

- Talk to the system in plain language; the agent calls typed kernel tools (files, settings, memory, programs, power).
- Semantic memory: notes, files and past sessions indexed by the embedding model; facts survive a reboot.
- Snapshots of the model state with undo, redo and branches; hibernation to disk and exact resume.
- Tool calls held to the tool schemas by a grammar while they are generated; a confidence check asks before an
  uncertain call that changes the machine.
- Programs from `/bin` (WASI) and built-in MicroPython.
- Built-in manual (`/man`), speculative decoding for copied text and for tool calls.

## Known limits

- QEMU/KVM only, no network, no real-hardware support yet.
- The model can pick the wrong tool or wrong values with high confidence. Tool calls are always well formed; the
  confidence check asks before about 5 of 8 wrong calls that change the machine.
- After a tool result the 4B sometimes answers in Italian to an English request.
- The 2B edition calls tools less often and may answer from its own guesses.
- `make run-remote` exposes a serial port and VNC on all interfaces: trusted networks only.
- A hibernated session resumes only on the same edition.
- Full list: docs/BETA.md.

## Licenses

ringzero is GPL-2.0-or-later. The engine (llm.asm by alesha-pro) is MIT. The Qwen weights inside the ISOs are
Apache-2.0. Details: LICENSE.md on the Hugging Face page and INFO.md in the repository.
