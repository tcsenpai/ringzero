# Snapshots v1 result (2026-11-26)

Engine state snapshots per [design/SNAPSHOTS.md](design/SNAPSHOTS.md), v1 scope (section 3.6): automatic
points after every turn, `/undo /redo /retry`, `/save /load /drop`, `/branch /switch /tree`, `/snap`, the
branch in the prompt, recovery instead of reset for Ctrl-C, interrupted compaction and `/bench /fill /gen`,
a point before every compaction, the pool with thinning, replay and eviction. Also in this round: the
project rename to **ringzero** (edition "ringzero Qwen35@2b" in the boot banner).

All numbers are real runs on the Fedora box (Ryzen AI 9 HX 370, QEMU 10.2.2/KVM,
`-cpu host -M q35 -m 4G -smp 8`, BIOS), on a test guest (`make run-remote REMOTE_PORT=4556 REMOTE_VNC=2
REMOTE_NAME=test`), scripted from the Mac with `tools/accept.py LAN-ADDRESS0 4556 ...`. The user's
instance was stopped during the tests (one guest only), host load 1-4. Architecture:
[ARCHITECTURE.md](ARCHITECTURE.md).

## How to connect (instance left running on the build box, this build)

```sh
nc LAN-ADDRESS0 4555                 # serial shell: press Enter to get "ringzero[main]> "
stty -echo -icanon; nc LAN-ADDRESS0 4555; stty sane    # same, without the double local echo (Tab works too)
open vnc://LAN-ADDRESS0:5901         # framebuffer + PS/2 keyboard (macOS Screen Sharing), password: aios
ssh BUILD-BOX 'tail -f ~/coding/ringzero/build/remote-serial.log'                  # everything on serial
ssh BUILD-BOX 'cd ~/coding/ringzero && python3 tools/screenshot.py build/remote-monitor.sock build/screen.png' && scp BUILD-BOX:coding/ringzero/build/screen.png .
```

Restart on Fedora (stop the old guest first):

```sh
cd ~/coding/ringzero && echo quit | socat - UNIX-CONNECT:build/remote-monitor.sock
setsid nohup make run-remote > build/remote.log 2>&1 < /dev/null &
```

The project moved: Mac `~/coding/qwenbaremetal/ringzero`, Fedora `~/coding/ringzero` (the
`models/qwen35-2b-q8_0-v3.asmbin` symlink there is absolute and still points to
`~/coding/qwenbaremetal/models/`). The ISO is now `build/ringzero.iso`, the FP32 KV build knob
`RINGZERO_KV=f32`.

## Acceptance

| Test | Result |
|---|---|
| Tell a fact, `/undo` past it: it no longer knows | PASS. "My name is Ada" (#1), `/undo 2` back to #0, "What is my name?" -> "I am the system agent for the ringzero operating system." |
| `/redo`: knows again | PASS. `/undo` then `/redo` to #1, "What is my name?" -> "Your name is Ada." Also the old line: `/switch main@2`, same question -> "Your name is Ada." |
| Design step 2 test (Ada, Bob, `/undo`) | PASS. "My name is Ada." -> "Actually, my name is Bob." -> `/undo` -> "What is my name?" -> "My name is Ada." |
| `/retry` at temp 0 gives identical output | PASS, twice: 25-token answer and 5-token answer identical text and token count (transcripts below) |
| `/branch b1`, say something, `/switch main`: main unaffected | PASS. b1: "My favourite colour is green" -> main: "I don't have personal preferences, so I don't have a favourite colour." -> b1: "Your favourite colour is green." |
| `/save x`, `/clear`, `/load x`: conversation restored | PASS. After `/clear`: "I don't know your favourite colour"; after `/load x`: "Your favourite colour is green." |
| Ctrl-C mid-answer keeps the previous turns | PASS. Story cut after 77 tokens -> "turn rolled back (ctx 908)", next question still answered "green". Ctrl-C during a 421-token prefill: rolled back to ctx 827, the next turn still knew "Ada". |
| Interrupted compaction keeps the conversation | PASS. Ctrl-C 1.5 s into `/compact` (summary): "conversation restored (ctx 960)", `/ctx` 960. |
| Point before every compaction, `/undo` after it | PASS. `/compact`: 960 -> 857 tok, point `[compact 1]`; `/undo` back to `pre-compact-1` #13, `/ctx` 960 again, "What is my favourite colour?" -> green. |
| `/bench /fill /gen` keep the conversation | PASS. After each one "[snap] conversation restored (ctx 934/960/960)" in 1.1-3.6 ms, follow-up still "green". |
| `/undo` over a tool side effect prints a note | PASS. "[undo] note: #14 ran set_setting temperature=0.3; the machine keeps it." |
| Thinning, replay of a thinned point | PASS. `!set snappool 451`: #1 and #2 thinned before the next saves; `/switch #2` replayed 62 tokens from #0 in 1.32 s, **R hash matches the stored one** (`0x52a7900202f118ad` both), "What is my name?" -> "Your name is Ada." |
| `/drop` | PASS. `/drop main` 21 MiB freed, `/drop main@5`, `/drop b1@10` 21 MiB each; `/drop #1` refused (later points). |
| Golden self-test | PASS on every boot (ids 9419,1017,0 + `<|im_end|>`). |
| Decode tok/s unchanged | PASS. `/bench` 35.53 / 35.64 / 35.82 / 35.92 tok/s (boot bench 34.65, 35.16); M2.5: 34.4-36.0. Snapshots run only between turns. |

### Times and memory per point (FP16 KV, from `!log` and `/snap`)

| Operation | Bytes | Time |
|---|---|---|
| Save a point, fresh pool chunks (short turn: 19-31 rows + R) | 19.5-20.1 MiB (21 chunks) | 4.2-8.5 ms |
| Save a point into reused chunks (freed by thinning) | 19.5-19.7 MiB | **0.99-1.03 ms** |
| Save #0, the boot prefix (741 rows + R) | 27.9 MiB (29 chunks) | 8-13 ms |
| `/undo` / `/redo` / Ctrl-C along the chain (R only) | 19.3 MiB | **0.58-0.76 ms** (first restore of #0 after boot: 4.07 ms) |
| `/switch` to another branch (R + rows above the common ancestor) | 19.6-21.0 MiB | 0.59-1.04 ms |
| Full restore after `/bench` `/fill` `/gen` (rows 0..P + R) | 28.5-30.5 MiB | 0.9-3.6 ms |
| Replay of a thinned point | 62 tokens | 1.32 s (47 tok/s) |
| Pool allocation + first touch at boot (1 GiB memset) | 1024 MiB | 2.0-5.4 s |

Memory per point: R is 19 728 KiB (20 chunks of 1 MiB) and the KV rows 12 KiB per token (all 24 planes,
85 rows per chunk), so a short turn costs 21 MiB and a 1k-token turn 32 MiB. A thinned point keeps only its
rows. `!mem` with the pool: "free 729 MiB ... LLM engine arena 407 MiB, state snapshot pool 1024 MiB".

The save time depends on the host, not on the copy: fresh chunks were zeroed at boot and not touched
since, and this host swaps to zram (10 GB in use), so the first write to those guest pages faults on the
host. Reused chunks save in about 1 ms, close to the design estimate (0.7-1.4 ms). Restores read pages
that were just written and stay at 0.6 ms. On a host without memory pressure (or real hardware) fresh
saves should take about 1 ms too. Not measured there.

## Transcripts (excerpts, `tools/accept.py`)

Undo, redo, switch to the old line:

```
ringzero[main]> My name is Ada, please remember it.
Ada, I remember you. How can I help you today?
ringzero[main]> /undo
[undo] back to #0 "boot prefix" (ctx 741/32768), 4.07 ms. /redo brings #1 back.
ringzero[main~1]> /redo
[redo] forward to #1 "My name is Ada, please remember " (ctx 779/32768), 0.68 ms
ringzero[main]> What is my name?
Your name is Ada.
ringzero[main]> /undo 2
[undo] back to #0 "boot prefix" (ctx 741/32768), 0.58 ms. /redo brings #2 back.
ringzero[main~2]> What is my name?
I am the system agent for the ringzero operating system.
ringzero[main]> /tree
#0    boot prefix                         741
|- #1    My name is Ada, please remember     779
|  #2    What is my name?                    803  main@2
`- #3    What is my name?                    774  main  <- here
pool 92/1024 MiB, 4 points (0 thinned), 2 branches, 0 saves
ringzero[main]> /undo
ringzero[main~1]> /switch main@2
[switch] main@2 at #2 "What is my name?" (ctx 803/32768), 0.64 ms
ringzero[main@2]> What is my name?
Your name is Ada.
```

Retry at temp 0:

```
ringzero[main]> Explain in one sentence what an operating system kernel is.
An operating system kernel is the core software layer that manages hardware resources and provides essential services for the rest of the operating system.
[prefill] 24 tok in 495.70 ms (48.41 tok/s)  [decode] 25 tok in 754.77 ms (31.79 tok/s, output 2 ms)  cores 8  ctx 825/32768 (2%)
ringzero[main]> /retry
[undo] back to #3 "What is my name?" (ctx 774/32768), 0.62 ms. /redo brings #5 back.
[retry] Explain in one sentence what an operating system kernel is.
An operating system kernel is the core software layer that manages hardware resources and provides essential services for the rest of the operating system.
[prefill] 24 tok in 542.62 ms (44.22 tok/s)  [decode] 25 tok in 723.97 ms (33.15 tok/s, output 1 ms)  cores 8  ctx 825/32768 (2%)
```

Branch, save, clear, load, Ctrl-C, bench:

```
ringzero[main]> /branch b1
[branch] b1 starts at #6 (from main); new turns go to b1
ringzero[b1]> My favourite colour is green, remember it.
I remember your favourite colour is green.
ringzero[b1]> /switch main
[switch] main at #6 "Explain in one sentence what an " (ctx 825/32768), 0.65 ms
ringzero[main]> What is my favourite colour?
I don't have personal preferences, so I don't have a favourite colour.
ringzero[main]> /switch b1
[switch] b1 at #7 "My favourite colour is green, re" (ctx 856/32768), 0.59 ms
ringzero[b1]> What is my favourite colour?
Your favourite colour is green.
ringzero[b1]> /save x
[save] x = #9 "What is my favourite colour?" (ctx 882/32768), kept until /drop x
ringzero[b1]> /clear
[agent] new conversation from the boot prefix (the old one stays in /tree)
ringzero[b1]> What is my favourite colour?
I don't know your favourite colour, but I can tell you that my favourite colour is blue.
ringzero[b1]> /load x
[load] b1 at #9 "What is my favourite colour?" (ctx 882/32768), 1.04 ms
ringzero[b1]> What is my favourite colour?
Your favourite colour is green.
ringzero[b1]> Write a long story about a cat who lives in a kernel.        (Ctrl-C after 2.5 s)
In the deepest layers of the ringzero kernel, where memory addresses are mapped to silicon ... His home
[prefill] 25 tok in 536.46 ms (46.60 tok/s)  [decode] 77 tok in 2261.90 ms (33.60 tok/s, output 4 ms)  cores 8  ctx 1012/32768 (3%)
^C [agent] interrupted, turn rolled back (ctx 908)
ringzero[b1]> What is my favourite colour?
Your favourite colour is green.
ringzero[b1]> /bench
A CPU cache is a small, high-speed memory chip ...
[prefill] 25 tok in 539.43 ms (46.34 tok/s)  [decode] 64 tok in 1758.37 ms (35.82 tok/s, output 5 ms)  cores 8  ctx 88/32768 (0%)
[snap] conversation restored (ctx 934) in 3591 us
ringzero[b1]> What is my favourite colour?
Your favourite colour is green.
```

Side effect note, compaction, undo after compaction:

```
ringzero[b1]> Please set the temperature setting to 0.3 with the set_setting tool.
[tool] set_setting(key=temperature, value=0.3)
[tool] -> temperature = 0.3 (top-p 0.95 sampling)
The temperature setting has been set to 0.3.
ringzero[b1]> /undo
[undo] back to #13 "What is my favourite colour?" (ctx 960/32768), 0.63 ms. /redo brings #14 back.
[undo] note: #14 ran set_setting temperature=0.3; the machine keeps it.
ringzero[b1~1]> /compact                                                  (Ctrl-C after 1.5 s)
[compact] context 960/32768 (2%, manual): summarizing...
^C [compact] interrupted, conversation restored (ctx 960)
ringzero[b1~1]> /compact
[compact] context 960/32768 (2%, manual): summarizing...
User: ringzero system agent, green favourite colour, kernel context full, about to be compacted.
Tasks: None open.
Tool results: None.
compact #1 manual: 960->857 tok (2%->2%), summary 33 tok 3168 ms, re-prefill 116 tok 2563 ms, total 5732 ms
ringzero[b1]> /tree
...
|     |  #13   What is my favourite colour?        960  pre-compact-1
|     |  #14   Please set the temperature setti   1073  b1@14
|     `- #8    What is my favourite colour?        861  main
|- #10   What is my favourite colour?        783  b1@10
`- #15   [compact 1] summary                 857  b1  <- here
ringzero[b1]> /undo
[undo] back to #13 "What is my favourite colour?" (ctx 960/32768), 0.76 ms. /redo brings #15 back.
ringzero[b1~1]> /ctx
[ctx] 960/32768 tokens (2%), boot prefix 741 tok, KV fp16, compaction at 75%, 1 compaction(s) so far, 2 exchange(s) kept for re-prefill
ringzero[b1~1]> What is my favourite colour?
Your favourite colour is green.
```

Thinning and replay (`!set snappool 451` = 84 MiB above what was in use):

```
[  243.449] A snap: thinned #1 (R freed, rows kept), 104 free chunks
[  243.450] A snap: saved #17 rows 986..1005 + R, 19956 KiB copied in 4249 us, 21 chunks
[  244.166] A snap: thinned #2 (R freed, rows kept), 103 free chunks
ringzero[b1]> /switch #2
[snap] #2 was thinned: replaying 62 tokens from #0 (about 2 s, Ctrl-C stops)
[snap] replay done, R hash matches the stored one
[switch] b1 at #2 "What is my name?" (ctx 803/32768), 1319.41 ms
[  246.563] A snap: replayed #2 62 tok, R fnv 0x52a7900202f118ad stored 0x52a7900202f118ad
ringzero[b1]> What is my name?
Your name is Ada.
```

## Implementation notes

- `kernel/src/snap.rs` (about 1000 lines with the shell commands). A point stores `parent` (storage: its rows
  continue the parent's) and `back` (history: where `/undo` goes). They differ only for a compaction
  point, which is stored under #0 (its rows restart at 741) but whose history continues from
  `pre-compact-N`, so `/undo` after a compaction returns to the uncompacted context.
- `llm::fwd` records the token id of every agent forward pass in a live array (`REC`, 128 KiB). A point
  stores the ids with its rows; a thinned point is rebuilt by feeding them again from the nearest stored
  R. The engine zeroes R at position 0, and every replay starts at 741 or later.
- The compaction re-prefill no longer keeps its own ring of the last 2 exchanges: it reads the last 2
  answered turns from the points on the current history (question and answer stored per point, capped at
  2 KiB each), so an undone turn is never re-prefilled. The auto-compaction rule (more than 2 exchanges
  since the last compaction) is counted the same way.
- Each point also stores the sampler RNG (restored with it, so a seeded run replays); `/retry` reseeds when
  temperature > 0, so a retry is a new sample.
- `/tree` keeps a linear run in one column and indents at forks (the design's example indents every
  point); `all` shows the inner points of a run.
- Typing after an undo forks: the old tip gets the kernel name `<branch>@<id>`. Switching to such a name
  makes it a real branch (no longer evictable). `/clear` starts the branch again from #0 the same way.
- `!set snappool MiB` caps the usable pool at run time (the design had a boot-time size); used to force
  thinning here.
- `!mem` and the meminfo tool report the pool apart from the engine arena.
- Ctrl-C during a prefill now prints the tokens actually fed (it printed the whole prompt before).

## Not done or not tested

- Tab completion of branch, save and `#id` arguments (design 3.2).
- The "pool full" refusal path (the turn stays unsaved, the prompt shows `*`): not reached in the tests,
  thinning always freed enough.
- Ctrl-C during the compaction re-prefill (only during the summary). Same rollback call, different place.
- Restore of a point whose replay is interrupted, the drop of kernel-named lines after 32 turns, and a
  compaction at the natural 75% crossing.
- Pool write-protection and the per-point R hash check on every switch (design 4.4, optional).

## Open risks for the next milestone (embedder via engine context swap, docs/spikes/EMBEDDER.md)

- **snap.rs reads the 2B's `g_ctx`** (K/V/DeltaNet buffers, max_seq, head dims) every time it copies. After
  the context swap, every snapshot call must run with the 2B context loaded, or it copies the embedder's
  buffers. Swap back before `commit`, `rollback`, `ensure_live` and any shell snapshot command, and never
  swap mid-turn. `kv_elt_bytes` is a global shared by both models.
- **Memory**: the pool takes 1 GiB of the 2 GiB arena region and leaves 729 MiB free. The embedder needs
  ~610 MiB for its module and ~115 MiB of arena at 1024 tokens. If Limine puts the second module inside
  the largest region, the arena shrinks; the pool then halves to 512 MiB on its own (it tries 1024, 512,
  256, ...), but it is allocated at `agent::init`, so the embedder's `model_init` must run before it or
  find no room. Decide 4G vs 6G (Fedora RAM: keep one 6G guest plus the user's under 90%).
- **Boot time**: the 1 GiB pool touch costs 2.0-5.4 s on this host, on top of the 15 s prefix prefill; the
  embedder adds its own load. Skipping the memset would move the cost to the first saves.
- **The embedder must not use `llm::fwd`** (it would overwrite the 2B's token array), and it must not run
  `transformer_forward` on the 2B context at position 0 (that zeroes the live R).
- **Determinism**: the replay hash matched at 8 threads. Idle-time indexing that changes the thread count
  between a thin and its replay would break it (record the thread count per point if needed).
- **Save times on a loaded host** (4-8 ms into cold chunks here) grow with host memory pressure; a second
  model adds more of it.
