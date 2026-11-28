# ringzero 0.1.1 "QweQwe" (beta)

Second beta of ringzero, an experimental x86-64 operating system where a language model (Qwen3.5) and an
embedding model (Qwen3-Embedding-0.6B) run in ring 0 on one inference engine written in assembly. Same models and
editions as 0.1.0; this release adds a Unix-like file tree, several decode streams in one engine pass and a cache
for documents read into the conversation.

Website: https://tcsenpai.github.io/ringzero-site/

## Downloads

| File | Where | What |
|---|---|---|
| `ringzero-0.1.1-4b-q4.iso` | Hugging Face: https://huggingface.co/Pleistocene/ringzero | recommended: Qwen3.5-4B (Q4 MLP) + Qwen3-Embedding-0.6B |
| `ringzero-0.1.1-2b-q8.iso` | Hugging Face | Qwen3.5-2B Q8 + Qwen3-Embedding-0.6B: smaller and faster |
| `ringzero-0.1.1-disk.img.xz` | this release | data disk with python3, sqlite3, jq, lua, QuickJS, ripgrep, coreutils, an editor and the demo files |
| `SHA256SUMS` | both | checksums of every file |

The ISOs are larger than the 2 GiB limit of GitHub release files, so they are hosted on Hugging Face. The 0.1.0
files stay there under their own names (`ringzero-beta-*`).

## What changed

- **A Unix-like file tree.** `/proc` gains `kmsg` and `mounts`; `/sys` has the disk (`class/block/vda`), the PCI
  devices, the cores and a writable `power/state` (`echo disk > /sys/power/state` hibernates and powers off);
  `/dev` has null, zero, random, urandom, tty, console and vda (the raw disk, read-only); `/etc/hostname` is
  writable and `/etc/os-release` names the edition; `/tmp` is emptied at a fresh boot and kept on resume. The
  same paths work from the shell, for the agent and for programs.
- **ringzero's data moved to `/var/lib/ringzero`** (facts, episodes, journal, memory index, hibernation image,
  boot cache, document cache). The agent's write_file, mkdir and rm are refused there; the shell's `!rm` asks.
- **`/read PATH`** puts a text file (up to 64 KiB) into the conversation and caches the engine state after it on
  the disk. Reading a 4325-token file again in a fresh conversation took 0.12-0.18 s instead of 22.9-28.4 s on the
  2B and 0.56-0.76 s instead of 71.7 s on the 4B, with exactly the same state. Sequential disk reads use a 1 MiB
  read-ahead.
- **Several streams in one engine pass.** `/multi N` decodes N prompts together, each on its own context: two
  streams give 1.76-1.93x the tokens per second of one, with the same tokens as decoding each alone.
  `/consolidate &` runs the session summary as a second stream and returns to the prompt at once; idle
  consolidation uses the same path. In the measured run the summary and the next answer were the same token for
  token as on the old path, and the answer decoded 10% slower while the summary ran next to it.
- Short notes in files (like `/notes/todo.txt` with "buy milk") are now recalled into a new conversation (file
  chunks need 8 characters instead of 20).
- The system prompt is 1652 tokens with the tools (1571 in 0.1.0): the first boot prefills it in about 20 s on the
  4B and 7.4 s on the 2B, then it is cached on the disk.

Tool use is unchanged on the 4B q4 tool-call evaluation: 24 of 30 spike prompts and 26 of 32 in-OS prompts
correct, zero malformed calls.

## Upgrading from 0.1.0

Boot the 0.1.1 ISO on your 0.1.0 disk: the first boot moves ringzero's data from FAT `/sys` to
`/var/lib/ringzero` before anything else uses the disk, reads every copy back, and only then removes the old
files. Facts, episodes, the journal and the memory index are kept; the boot cache is rebuilt (one slower first
boot) and a hibernated 0.1.0 session is not resumed. To check it worked:

- the boot log has `layout: migrated N files, ... KiB` (`!log 50` shows it again);
- `/facts` lists your facts and `/agentd` shows `stores /var/lib/ringzero/mem: ...` with your journal and episodes;
- `!ls /var/lib/ringzero/mem` lists `journal.log`, `facts.log`, `episodes.log` and `index.bin`.

If a copy fails (for example a full disk), every old file stays, the boot runs without the memory stores and says
`layout migration pending` in `/agentd`, and the next boot tries again. Going back to 0.1.0 on the same disk shows
an empty memory unless the stores are copied back first (INFO.md); what 0.1.0 writes afterwards is kept in
`/var/lib/ringzero/legacy/` and `/agentd merge-legacy` merges it.

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
xz -d ringzero-0.1.1-disk.img.xz && cp ringzero-0.1.1-disk.img my-disk.img
```

Serial console in the terminal (Ctrl-A X quits; `!poweroff` inside ringzero saves the session):

```sh
qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m 8G -smp 8 \
  -cdrom ringzero-0.1.1-4b-q4.iso -boot d -no-reboot -display none -serial mon:stdio \
  -drive file=my-disk.img,if=none,id=disk0,format=raw -device virtio-blk-pci,drive=disk0
```

For a window replace `-display none` with `-vga std -display gtk`. For the 2B use `ringzero-0.1.1-2b-q8.iso` and
`-m 6G`. Then type a sentence, `/help`, or `/man TOPIC`. Examples: docs/DEMOS.md.

## Known limits

- QEMU/KVM only, no network, no real-hardware support yet.
- The model can pick the wrong tool or wrong values with high confidence. Tool calls are always well formed; the
  confidence check sees about 5 of 8 wrong calls and asks before those that write.
- After a tool result the 4B sometimes answers in Italian to an English request.
- The 2B edition calls tools less often and may answer from its own guesses.
- An extra stream needs memory from the snapshot pool (179 MiB on the 4B plus 256 MiB kept free) and holds 4096
  positions: in a long session or with a full pool, `/consolidate &` and idle consolidation fall back to the
  foreground path.
- A `/read` cache entry is large (about 32 KiB per token on the 4B): on the default 512 MiB disk the 4B keeps about
  two entries of a 4k-token file; when there is no room the read is not cached (the log says why).
- `make run-remote` exposes a serial port and VNC on all interfaces: trusted networks only.
- A hibernated session resumes only on the same edition and build.
- Full list: docs/BETA.md.

## Licenses

ringzero is GPL-2.0-or-later. The engine (llm.asm by alesha-pro) is MIT. The Qwen weights inside the ISOs are
Apache-2.0. Details: LICENSE.md on the Hugging Face page and INFO.md in the repository.
