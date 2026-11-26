# M4 result: virtio disk, FAT32, file tools, idle file indexing (2026-11-26)

A virtio-blk disk with a FAT32 volume is mounted at `/` at boot. The shell and the agent can list, read,
write and delete files on it. Text files written through either are chunked into the memory index while the
shell is idle. Edition unchanged: **ringzero Qwen35@2b-Qwen3@0.6b** (no new model).

All numbers are real runs on the Fedora box (Ryzen AI 9 HX 370, QEMU 10.2.2/KVM, `-cpu host -M q35 -m 6G
-smp 8`, BIOS unless noted). Tests ran on a test guest (`make run-remote REMOTE_PORT=4556 REMOTE_VNC=2
REMOTE_NAME=test`), driven by `tools/accept.py` from the Mac over Tailscale (HOST-ADDRESS). The user's
instance was stopped during the tests: one guest at a time, host RAM 45-60% used. Architecture:
[ARCHITECTURE.md](ARCHITECTURE.md).

## How to connect (instance left running on fedora, this build, disk attached)

```sh
nc HOST-ADDRESS 4555                 # serial shell: press Enter to get "ringzero[main]> "
stty -echo -icanon; nc HOST-ADDRESS 4555; stty sane    # without the double local echo (Tab works too)
open vnc://HOST-ADDRESS:5901         # framebuffer + PS/2 keyboard, password: aios
ssh fedorascale 'tail -f ~/coding/ringzero/build/remote-serial.log'                  # everything on serial
```

The disk `build/disk.img` already holds the test files (`!ls /notes`). The memory index starts empty at every
boot. To fill it, rewrite a file or write a new one: `!write /notes/x.txt ...`, then wait a few seconds.

Restart on Fedora (stop the old guest first; it needs 8 GiB available):

```sh
cd ~/coding/ringzero && echo quit | socat - UNIX-CONNECT:build/remote-monitor.sock
setsid nohup make run-remote > build/remote.log 2>&1 < /dev/null &
```

At the Fedora desktop: `make run-local` opens the same VM in a QEMU window, with serial on the terminal.
It uses its own disk, `build/disk-local.img`. It can run next to run-remote if RAM allows; it refuses to start
below 8 GiB available. To use the remote's disk instead, stop run-remote first and run `make run-local
LOCAL_DISK=build/disk.img`: QEMU's image lock refuses a second guest on the same image.

## Steps

| Step | Status |
|---|---|
| 1. Kernel heap | DONE. `linked_list_allocator` 0.10.6 as `#[global_allocator]` on a 16 MiB slice of the engine arena, taken at boot before the model contexts. `/stats` shows heap use (0 KiB at rest: fatfs frees what it allocates). |
| 2. PCI + virtio-blk | DONE (blk.rs). `virtio-drivers` 0.13 `PciTransport` + `VirtIOBlk`. Config space goes through ports 0xCF8/0xCFC, bus 0 only (q35 puts the device at 00:03.0, id 1001 transitional). BARs are identity-mapped uncached (`mem::map_mmio`, 2 MiB pages, PCD/PWT). DMA queues come from the identity-mapped arena (phys == virt). Every buffer the driver shares (stack, statics, heap: not identity-mapped) goes through a 256 KiB bounce ring. Polling, one request at a time. Works under BIOS and UEFI (OVMF). |
| 3. FAT32 on the disk | DONE (fs.rs). fatfs from git (rev 38b3a79, 0.4.0): crates.io 0.3.6 needs `core_io` for no_std, which no longer builds. Whole-volume FAT32 with no partition table, LFN. A one-sector read cache, write-through, direct transfers of up to 64 KiB. `make disk` creates `build/disk.img` (512 MiB, `mkfs.vfat -F 32`, `mformat` fallback) only if it is missing; the run targets attach it as `virtio-blk-pci`. Boot log: `disk: mounted at /: Fat32 'RINGZERO', 510 MiB, used 4 KiB, free 510 MiB (cluster 4 KiB)`. |
| 4. Shell commands + agent tools | DONE. `!ls [PATH]`, `!cat PATH` (64 KiB), `!write PATH TEXT` (`\n` = new line, parents created, 64 KiB max), `!mkdir PATH` (-p), `!rm [-r] PATH` (always y/N; a non-empty directory needs -r), `!df`, `!sync`. Agent tools `ls`, `cat` (first 7936 bytes plus a truncation note, under the 8 KiB tool cap), `write_file`, `mkdir`, `rm` (files and empty directories only, always y/N by the human). Every path goes through `fs::check`: it must be absolute, with no `.`, `..` or empty components, no control characters or backslash, 255 bytes at most. Tool side effects go into the turn's effect note (`/undo` says they stay). |
| 5. Idle indexing hook | DONE (fidx.rs). `!write` and `write_file` queue text files (UTF-8, no NUL); a rewrite or `rm` first drops that file's chunks (`memidx::retain`) and its queue entry. The shell's line editor runs one slice while it waits for input at the prompt (not during y/N questions). A slice ends after 100 ms, or between two embedder forward calls (~6 ms) as soon as a byte or Ctrl-C arrives. The embedding is resumable (`embed::begin/resume`): the embedder's KV rows stay valid between slices, and the job restarts only if another embedding (`/search`, `search_memory`) ran in between. Chunks are cut at about 1000 bytes on a newline or a space, at UTF-8 boundaries, and stored as `[/path] text`. `/index stats` shows the queue. |
| 6. Persistence | DONE. A file written before `!poweroff` is there at the next boot. `!sync` and poweroff/reboot unmount and mount again, so the volume is clean for `fsck.vfat`. |

## Verification

| Test | Result |
|---|---|
| Golden self-test | **PASS** on every boot (BIOS and UEFI, with the disk attached) |
| Boot self-tests | engine ABI, memory index (+ `retain`), file indexer (chunker, path checks), tool parser: **PASS** |
| `!write /notes/todo.txt comprare il latte`, `!cat`, `!ls /` | wrote 17 bytes, `comprare il latte`, `notes/` |
| Persistence | `!write`, `!sync`, `!poweroff` (QEMU exits), boot again: `!cat /notes/todo.txt` -> `comprare il latte`, `!ls /notes` lists a file written in the previous boot. Host side: `mtype -i build/disk.img ::/notes/todo.txt` gives the same text, and `fsck.vfat -n` is clean after `!sync` + poweroff |
| Agent creates and reads a file | "crea un file /notes/idea.txt con scritto ciao" -> `write_file(path=/notes/idea.txt, content=ciao)` -> "File creato con successo."; "leggi /notes/idea.txt" -> `cat` -> `ciao` |
| rm asks y/N, 'n' keeps it | "cancella il file /notes/idea.txt" -> `rm` -> `[agent] wants to delete file /notes/idea.txt (4 B). Proceed? [y/N] n` -> kept, `!ls /notes` still lists it. Shell `!rm /notes/todo.txt` + n keeps it; `!rm /a` on a non-empty directory refuses, `!rm -r /a` asks "delete directory /a and the 3 entries inside?" |
| Path validation | `!cat notes` -> must be absolute; `!write /x/../y hi` -> '.' and '..' not allowed; `!ls /notes/todo.txt` -> not a directory; `!cat /nope` -> not found |
| Idle indexing, small file | 1520 B -> 2 chunks, 236 tok/chunk, 3.2 s of idle time (669 ms per 100 tok, the embedder's usual speed) |
| Idle indexing, larger file | 3099 B -> 5 chunks, 196 tok/chunk, 6.6 s of idle time. Commands typed while the queue was busy (`/index stats`, `!uptime`, `!cores`, Ctrl-C at the prompt) answered in 0.0-0.1 s each |
| Search finds a chunk by a paraphrase | `/search quanto ci mette il treno veloce tra Tokyo e Kyoto?` -> the travel chunk at #1 (cos 0.575; the text says "shinkansen ... circa due ore e un quarto"). `come si prepara il sugo di carne` -> the ragu recipe at #1 (0.549). `entro quando chiedo le vacanze?` -> the work chunk with "ferie estive" at #1 (0.510). `a che ora devo dare da mangiare al gatto?` -> the chunk with "Il gatto Pepe mangia ... alle 8 ... alle 7" at #1 |
| Agent `search_memory` on file chunks | "cerca nella memoria: a che ora mangia il gatto?" -> top hit `[/notes/casa.txt] ...` -> "Il gatto Pepe mangia due volte al giorno: alle 8 di mattina e alle 7 di sera." |
| Decode tok/s unchanged | boot bench 35.39 (BIOS), 35.64 (UEFI), 36.08 (final instance); `/bench` 36.91 (BIOS), 36.09 (UEFI), 35.02 after agent turns and indexing in the same session (embedder milestone: 34.0-36.9) |
| Snapshots | `/tree`, `/undo 2` 0.60 ms, `/redo 1` 0.69 ms, `/save m4test`, `/bench` then `[snap] conversation restored` in 5.7 ms: **PASS** |
| Boot prefix | 834 -> **1142 tokens** (+308 for 5 tools and the longer `search_memory` description), prefill 23.3-24.9 s (was 16.6-16.8 s) |
| Boot time to the shell (`DISK=build/disk.img tools/boottest.py`) | **BIOS 35.8 s** (was 28.8 s), **UEFI 46.8 s** (was 38.6 s). The disk adds nothing visible (probe + mount in the same 10 ms as the banner); the +7 s is the prefix prefill |
| Guest RAM (`!mem`) | total 6143 MiB, free 2022 MiB: weights 2535, engine arena 520, snapshot pool 1024, kernel heap 16, kernel + firmware 26 MiB. Bounce ring + virtqueues: < 1 MiB of arena |
| `make run-local` | Started on the Fedora Wayland desktop (`WAYLAND_DISPLAY=wayland-1`, GTK window): booted to `ringzero[main]>` with `build/disk-local.img` mounted (screendump checked), then stopped |

### Transcripts (excerpts)

```
ringzero[main]> !mkdir /a/b/c
[fs] /a/b/c created
ringzero[main]> !write /a/b/c/d.txt deep\nfile
[fs] wrote 9 bytes to /a/b/c/d.txt (queued for the memory index)
ringzero[main]> !rm -r /a
delete directory /a and the 3 entries inside? [y/N] y
[fs] deleted /a (1 index chunks dropped)

ringzero[main]> crea un file /notes/idea.txt con scritto ciao
[tool] write_file(path=/notes/idea.txt, content=ciao)
[tool] -> wrote 4 bytes to /notes/idea.txt
File creato con successo.
ringzero[main]> leggi /notes/idea.txt
[tool] cat(path=/notes/idea.txt)
[tool] -> ciao
Il contenuto è stato letto: "ciao".
ringzero[main]> cancella il file /notes/idea.txt
[tool] rm(path=/notes/idea.txt)
[agent] wants to delete file /notes/idea.txt (4 B). Proceed? [y/N] n
[tool] -> cancelled: the user answered no, /notes/idea.txt is kept
Il file non è stato cancellato perché l'utente ha risposto "no".

ringzero[main]> /index stats
[index] memory index: 5 of 4096 entries, dim 1024 (Q8_0 rows of 1088 B, 5 KiB), text 3 of 1024 KiB, embed avg
1316 ms/entry (196 tok/entry, 670 ms per 100 tok); RAM only, lost at reboot; files: 0 queued (0 bytes left),
1 indexed (5 chunks, 6586 ms of idle time)
```

## Bugs found and fixed during the milestone

- `fs::check("/")` rejected the root (an empty component); `!ls /` failed. Fixed and added to the boot self-test.
- fatfs `create_dir` creates only the last level; `!mkdir /a/b/c` and writes into new nested directories
  failed with "not found". `mkdirs` now creates each level.
- fatfs flushes the FS info sector but never clears the FAT32 dirty flag before unmount, and it cannot hand the
  storage back after `unmount()`. The disk is now a handle on a static device, so `!sync` can unmount and
  mount again. Result: `fsck.vfat` is clean after `!sync` + poweroff.

## Not done / limits

- The memory index (file chunks included) is still RAM only: after a reboot, files on disk are not
  re-indexed until they are written again (M4.5).
- File times are 1980-01-01: no RTC time provider for fatfs yet.
- A guest that is killed instead of powered off leaves the volume dirty. Data is safe (write-through), and
  `fsck.vfat -a build/disk.img` clears the flag. A volume that is dirty at mount stays dirty (fatfs keeps the flag).
- FAT names are case-insensitive, but the index keys (`[/path]`) are not: writing `/Notes/a.txt` after
  `/notes/a.txt` replaces the file but leaves the old chunks.
- Only PCI bus 0 is scanned, with polling, one request at a time, and no partition table.
- `search_memory` returns whole chunks (up to ~1 KiB each, 3 hits). A search over file chunks fed 434 tokens
  back to the model (9 s of prefill).

## Risks for M4.5 (agentd persistent memory)

- **Index persistence**: the rows can be saved next to the texts, with a header: embedder file hash (sha256 in
  EMBEDDER_RESULT), dim, Q8_0 layout, chunker version. Rebuild by re-embedding when any of them changes. At about
  0.67 s per 100 tokens, a rebuild of a few MiB of notes takes hours: it must run in idle time, and the fidx queue
  already does that. Store the file's size and mtime (or a hash) per chunk, so changes made outside the agent
  (mtools on the host) can be found at boot.
- **Write pattern**: a journal of facts and episodes appended at every turn means small writes. With
  write-through and FAT, each append rewrites the FAT sector, the directory entry and the data sector (3+
  requests). That is fine at shell speed. FAT has no journaling, though: a crash in the middle of an append can
  lose the tail or leak a cluster. Use append-only files with a length or checksum per record, and recover by
  scanning to the last good record.
- **Idle consolidation**: the 2B is far more costly than the embedder (a 300-token summary is ~10 s of decode).
  It needs the same slice discipline (stop between forward calls) and must save and restore the live snapshot
  point, because a consolidation that uses the 2B's context writes the live KV/DeltaNet state. The
  indexer only uses the embedder's context, so it never had this problem.
- **Pinned-facts slot** (fixing the compaction recall issue): facts read from the disk at boot become part of the
  system prompt. That changes the boot prefix: each pinned fact costs prefill time at every boot (now ~21 ms per
  token). Cache the prefilled boot snapshot on disk, keyed by the FNV hash of the prefix ids plus the model hash.
  This also wins back the 7 s that the M4 tools added: the root snapshot is 32 MiB, roughly 0.3 s to read over
  virtio.
- **RAM**: the heap is 16 MiB and the index arrays are static (4096 rows, 1 MiB of text). Journal buffers and a
  larger index will need the heap to grow, or index eviction. About 2 GiB of the guest are still free.
- **Concurrency of idle work**: indexer, consolidation and the y/N prompts all run on the BSP from the line
  editor. They need one queue with priorities, so a long consolidation never starves indexing, and the reverse.
