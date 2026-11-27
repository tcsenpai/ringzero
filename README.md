# ringzero

An operating system where the AI is part of the kernel.

ringzero boots on bare x86-64 (no Linux underneath) and runs two models in ring 0, next to the scheduler
and the drivers, on one shared inference engine:

- a **language model** (Qwen3.5) that you talk to at the shell: it answers, reads and writes your files and
  runs programs;
- an **embedding model** (Qwen3-Embedding-0.6B) that gives the system a semantic memory: it indexes your
  notes, your files and your past conversations, so the OS can find things by meaning, not by file name.

Together they make an OS that understands what you ask and remembers you across reboots. It is an
experiment: the question is what an OS looks like when the models are not apps on top of it, but parts of it.

```
ringzero[main]> count the lines of /demo/stats.py with wc
[tool] run_wasm(path=wc, args=-l /demo/stats.py)
[tool] -> 24 /demo/stats.py
[exit 0]
```

## What it does

- **Talk to the machine.** Plain text goes to the system agent, which calls typed kernel tools: files,
  memory, CPU cores, logs, settings, power. Anything that deletes or powers off asks you first.
- **Semantic memory, built in.** The embedding model indexes every text file you write and every session, in
  the background while you are idle, into an index saved on the disk. "what do I need to buy?" finds the
  shopping list you wrote last week. `/search` queries it directly.
- **It remembers you.** Facts you tell it ("my name is Anna") and summaries of past sessions are kept on the
  disk and come back after a reboot. The kernel, not the model, decides what goes into the model's context,
  so the memory stays small and on topic.
- **Undo for the AI.** Every turn is a snapshot of the model's state: `/undo`, `/redo`, `/retry`, named
  branches, save and load. Going back takes under a millisecond.
- **Hibernate and resume.** Power off and the whole conversation, with its history tree, is written to disk.
  The next boot picks it up exactly where it was, checked by a state hash.
- **Tool calls that cannot break.** While the model writes a tool call, the kernel only lets through tokens
  that keep it valid against the tool's schema. In the tests: zero malformed calls.
- **Real software.** python3 (CPython 3.15), sqlite3, jq, Lua, QuickJS, ripgrep, the coreutils and a text
  editor run from `/bin` in a WASI sandbox, from the shell and for the agent. MicroPython is built in for quick
  one-liners (`!py 17*23`).
- **Fast where it matters.** The prompt is read in blocks (about 230 tokens/s on the 2B), the boot state is
  cached on disk, and the system does its housekeeping (indexing, summarizing) while you are idle.

## Try it

You need a Linux machine with KVM and about 16 GB of RAM. The full list of build tools is in
[INFO.md](INFO.md#build-and-run).

```sh
make model MODEL=4b QUANT=q4     # download and convert Qwen3.5-4B + the embedder
make iso MODEL=4b QUANT=q4       # build the bootable image
make disk                        # a 512 MiB data disk for your files and memory
make wasm-apps                   # python3, sqlite3, jq, lua, ... onto the disk
make run-local MODEL=4b QUANT=q4 # boot it in a QEMU window
```

Then, at the `ringzero[main]>` prompt:

```
my name is Anna and my favourite number is 4711
!write /notes/todo.txt buy milk
what do I need to buy?
create a table in /shop.db with sqlite, add three products with prices, and tell me the most expensive
!run python3
/undo
!poweroff
```

Boot again and ask "what is my name?". More copy-paste examples: [docs/DEMOS.md](docs/DEMOS.md).

## Which model

Pick one at build time with `MODEL` and `QUANT`. All of them run on the same engine.

| Option | Good for | Guest RAM | Speed (tokens/s) |
|---|---|---:|---:|
| `MODEL=4b QUANT=q4` | **recommended**: best tool use per GB | 8 GB | about 20 |
| `MODEL=4b QUANT=q8` | best quality | 8 GB | about 14 |
| `MODEL=2b` (default) | fastest, smaller machines | 6 GB | about 34 |

Needs an x86-64 CPU with AVX2, FMA and F16C (most desktop and laptop CPUs since 2015). Details and the other options:
[INFO.md](INFO.md#which-model-for-which-machine).

## How it is built

A small Rust `no_std` kernel booted by Limine, with the inference engine written in assembly
([llm.asm](https://github.com/alesha-pro/llm.asm) by alesha-pro, extended for Qwen3.5) linked in as objects.
The models are Qwen3.5 (chat and tools) and Qwen3-Embedding-0.6B (memory). Drivers: virtio-blk, FAT32,
serial, PS/2, framebuffer.

- [INFO.md](INFO.md): every command, the measurements, the source layout, the licenses
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the architecture diagram
- [docs/ROADMAP.md](docs/ROADMAP.md): what is done and what comes next

## Related work

- D. Son, *ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems*, 2026,
  [arXiv:2604.11943](https://arxiv.org/abs/2604.11943). Anima OS, a bare-metal x86-64 Rust kernel with LLM
  inference inside, reads the model's next-token logits as a kernel primitive for classifying agent actions.
  ringzero acts on the logits in the kernel too (the tool-call grammar mask, and the planned uncertainty signal).

## Status

Experimental. It runs in QEMU/KVM with virtio devices; real hardware is on the roadmap. Expect rough edges.

## License

GPL-2.0-or-later. The vendored engine is MIT. Models are Apache-2.0 and are downloaded, not shipped.
