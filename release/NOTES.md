# ringzero 0.1.0-beta

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

<!-- FILL from docs/BETA.md (measured): guest RAM per edition, host RAM, CPU flags, disk -->

## Run

<!-- FILL from docs/BETA.md: download, verify, unpack the disk, the QEMU command line -->

## What works

- Talk to the system in plain language; the agent calls typed kernel tools (files, settings, memory, programs, power).
- Semantic memory: notes, files and past sessions indexed by the embedding model; facts survive a reboot.
- Snapshots of the model state with undo, redo and branches; hibernation to disk and exact resume.
- Tool calls held to the tool schemas by a grammar while they are generated; a confidence check asks before an
  uncertain call that changes the machine.
- Programs from `/bin` (WASI) and built-in MicroPython.
- Built-in manual (`/man`), speculative decoding for copied text and for tool calls.

## Known limits

<!-- FILL from docs/BETA.md known limits -->

## Licenses

ringzero is GPL-2.0-or-later. The engine (llm.asm by alesha-pro) is MIT. The Qwen weights inside the ISOs are
Apache-2.0. Details: LICENSE.md on the Hugging Face page and INFO.md in the repository.
