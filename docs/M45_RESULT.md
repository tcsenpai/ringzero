# M4.5 result: agentd, persistent memory, boot snapshot cache (2026-11-26)

agentd is a kernel subsystem that owns the agent's context and memory. The 2B model is small, so the OS does the
bookkeeping deterministically; the model reads and writes memory only through typed tools (`remember`, `forget`
with y/N, `recall`), and every agentd decision is logged (`/agentd log`, kernel log level D). Edition unchanged:
**ringzero Qwen35@2b-Qwen3@0.6b** (no new model). Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

All numbers are real runs on the Fedora box (Ryzen AI 9 HX 370, QEMU 10.2.2/KVM, `-cpu host -M q35 -m 6G -smp 8`,
BIOS unless noted). Tests ran on a test guest (`make run-remote REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test
DISK=build/disk-test.img`, or a fresh `build/disk-a.img`), driven by `tools/accept.py` from the Mac over Tailscale
(HOST-ADDRESS); boot times with `tools/boottest.py`. The user's instance was stopped (`!sync` first). Another
session was running its own guests from `~/coding/ringzero-prefill` during part of the tests (host load average
5-6): runs taken under that load are marked.

## How to connect (instance left running on fedora, this build, disk attached)

```sh
nc HOST-ADDRESS 4555                 # serial shell: press Enter to get "ringzero[main]> "
stty -echo -icanon; nc HOST-ADDRESS 4555; stty sane    # without the double local echo (Tab works too)
open vnc://HOST-ADDRESS:5901         # framebuffer + PS/2 keyboard, password: aios
ssh fedorascale 'tail -f ~/coding/ringzero/build/remote-serial.log'                  # everything on serial
```

The disk `build/disk.img` holds the M4 test files plus agentd's `/sys/mem` and `/sys/bootsnap.bin`. Restart on
Fedora (power off from the shell with `!poweroff`, or stop it from the monitor; it needs 8 GiB available):

```sh
cd ~/coding/ringzero && echo quit | socat - UNIX-CONNECT:build/remote-monitor.sock
setsid nohup make run-remote > build/remote.log 2>&1 < /dev/null &
```

At the Fedora desktop: `make run-local` opens the same VM in a QEMU window, with serial on the terminal (own disk
`build/disk-local.img`; `LOCAL_DISK=build/disk.img` only while run-remote is stopped). Checked for this build:
it booted to `ringzero[main]>` in a GTK window on `wayland-1` (screendump checked), cold cache, then was stopped.

## Steps

| Step | Status |
|---|---|
| 1. agentd core | DONE (agentd.rs). One priority queue on the BSP, run from the shell's line editor while it waits for input, one slice of at most 100 ms per call: 1 embed memory texts (turns, facts, episodes), 2 embed file chunks (fidx.rs now only keeps the queue, the chunker and the table of indexed files), 3 save the index, 4 consolidate. A due consolidation gains one priority level every 50 slices it waits (aging), so a long file queue cannot starve it; memory texts and files stay ahead otherwise. Any input byte or Ctrl-C stops a slice between two engine steps (embedder forward or 2B forward); `preempt()` on the first input byte stops a consolidation in flight and restores the current snapshot point. Every decision goes to the kernel log (level D) and a 128-line ring: `/agentd` (status + last 10), `/agentd log N`. |
| 2. Persistent memory | DONE (mstore.rs, agentd.rs). `/sys/mem/journal.log` (U user turn, A answer, T tool result, C compaction summary, B boot, P poweroff), `facts.log` (S set `key\tvalue\tsource`, X forget; replayed at boot, last wins), `episodes.log` (E). Record: `'R' kind len:u32 ts:u64 sum:u32 payload`, ts from the CMOS RTC (new, sys.rs; also fatfs file times now), sum = folded FNV-1a over kind, ts and payload. Boot recovery scans each store to its last good record and truncates a torn tail. Tools: `remember(key, value)`, `forget(key)` (y/N; an unknown key gets the list of saved keys back), `recall(query)` (replaces `search_memory`: top 3 of facts, past statements, episodes and file chunks). Facts proposed by the model are validated by the kernel: snake_case key 2..32 chars, value 1..120 bytes without markup, not a placeholder, key not equal to the value, grounded (every number and at least half of the words of 3+ characters appear in what the user said since the last consolidation), dedup by key (same value: "already known"; new value: update). Shell: `/facts`, `/remember KEY VALUE`, `/forget KEY`. **Boot prefix: 1142 -> 1254 tokens** (16 tools: `search_memory` replaced by the shorter `recall`, plus `remember` and `forget`). |
| 3. Index persistence | DONE (memidx.rs save/load, agentd.rs). `/sys/mem/index.bin`: a 72-byte header (embedder sample hash, dim, row stride, layout id 0x5108 = int8 + fp16 block scales, fidx chunker version, counts, embed stats, FNV of the payload), agentd's extra lines (the store sizes the index covers; the table of indexed files with size and FAT mtime), then offsets, Q8_0 rows and texts, read straight into the static arrays. Any header mismatch or a bad checksum means a rebuild in idle time. Otherwise only records past the covered store sizes and files whose size or mtime differs (or that are new) are embedded again; files gone from the disk lose their chunks. The index is saved by the idle scheduler when it changed and nothing is queued, and at poweroff/reboot under the same condition (else the next boot re-embeds from the last saved state). Entries: `[/path] chunk`, `[fact] key: value`, `[episode N] text`, `[said] user statement` (user turns that end with `?` are not indexed: a question states no fact). `/index clear` now rebuilds everything. |
| 4. Boot snapshot cache | DONE (snap.rs `save_prefix` / `load_prefix`, agent.rs). `/sys/bootsnap.bin`: 48-byte header, token ids, rows 0..pos of the 24 K/V planes, the DeltaNet state R; 33 MiB for 1252 tokens. Key: FNV over the prefix token ids, a sample hash of the model file (size, first and last 64 KiB, 256 x 4 KiB), an engine build hash from build.rs (asm sources + KV flag), the KV element size and the context length. Mismatch, bad size or checksum: prefill and rewrite. |
| 5. Context compiler | DONE (agentd.rs `slot`). The first user message of a conversation, and the first turn after a compaction, get a slot in front of the message: `[memory from earlier conversations, kept by the kernel; use it to answer]`, every saved fact newest first within 120 tokens (all facts are pinned in this prototype), then up to 4 recalled notes for that message (cos >= 0.45, or a BM25 match when the keyword gate opens) within 200 tokens in all. Memory texts still queued are embedded first, so the last turns are searchable. The shell prints the slot as `[memory] ...` lines; the decision log has its ids, cosines, tokens and build time. `!set slot N` sets the budget, 0 turns it off. |
| 6. Idle consolidation | DONE (agentd.rs). After `!set consolidate N` idle seconds at the prompt (default 60) with new turns, or `/consolidate`: the 2B runs greedy and without thinking on top of the current snapshot point, a fixed internal instruction asks for `EPISODE:` and `FACT: key = value` lines, output goes to the capture buffer only. Then `snap::rollback` restores the current point (rows above it were scratch, R is copied back), the temperature is restored, the episode is appended and indexed, every FACT goes through the same validation as `remember`. The raw lines the model wrote are logged. Skipped (logged) when the live state is unsaved (pool full), at a fresh conversation, or without room in the context. |

## Acceptance

| Test | Result |
|---|---|
| (a) facts across a reboot, new conversation | **PASS** (final build, fresh disk). "mi chiamo Cris e il mio numero preferito e 4711", `/consolidate` (27 tok, 3.7 s): episode "Cris ha detto che il suo numero preferito è 4711.", fact `user_name = Cris`. `!poweroff`, boot (cache hit, index loaded, 3 records queued). New conversation "come mi chiamo?": slot = `user_name: Cris` + the earlier statement (47 tok, 0.44 s) -> "Mi chiamo Cris." `/clear`, "qual e il mio numero preferito?": slot = fact + statement (cos 0.66) + episode (0.58), 70 tok -> "Il mio numero preferito è 4711." (right value; the 2B says "mio" for "tuo") |
| (b) fact before /compact answered after it | **PASS**. "mia sorella si chiama Giulia e vive a Torino", 2 tool turns, `/compact`, "come si chiama mia sorella e dove vive?": with `!set slot 0` -> "Non so." although the summary held "Giulia, Torino" (the M2.5 failure, reproduced); with the slot (brother Marco, Napoli, same flow) -> "Il mio fratello si chiama Marco e vive a Napoli." (slot after compaction: 2 facts + 4 recalled, the statement first). Earlier run: "la mia auto e una Panda rossa targata AB123CD" ... `/compact` ... "che auto ho e qual e la targa?" -> "La tua auto è una Panda rossa targata AB123CD." (recalled statement cos 0.65) |
| (c) journal survives a hard kill | **PASS**. Three turns and a `!write`, then `kill -9` of QEMU. Host side: journal 55 whole records, 3598 of 3598 bytes, last record = the last answer "Ho 8 core."; `/notes/kill.txt` there; the FAT dirty flag set (fsck: boot sector differs from its backup at offset 65). Torn append simulated on the host (a record header for 100 bytes + 12 bytes of payload, 30 bytes): boot -> `recover /sys/mem/journal.log: 55 good records, torn tail of 30 bytes at offset 3598 truncated`; the pre-kill statement is found by `/search codice del garage` (cos 0.765). A crash inside a real append was not caught in the act (an append is one small write) |
| (d) index from disk, rebuild on host change | **PASS**. Boot: `index: loaded 13 entries ... in 0.750 ms`, `loaded 27 entries in 1.225 ms`, `loaded 31 entries in 19.4 ms` (under host load), all files `unchanged (kept, not re-embedded)`. Embedding the same 10 file chunks took 9.7-10.0 s of idle time (27 entries: 447 ms per entry on average, about 12 s). `/notes/todo.txt` rewritten on the host with mtools (size 17 -> 37): `scan: /notes/todo.txt changed on disk ... re-embedding it`, `5 unchanged, 1 changed`, re-embedded in 0.12 s, index saved again in 4 ms; `/search cosa devo comprare?` -> the new text at #1 (cos 0.608) |
| (e) boot time to the prompt (`DISK=<copy of disk.img> tools/boottest.py`) | **BIOS: 40.6 s cold (prefill 26.3 s + cache write 2.0 s), 11.9 s cached (load 0.195 s)**. **UEFI: 55.0 s cold (prefill 25.8 s + 2.0 s), 25.7 s cached (load 0.179 s)**. M4 was BIOS 35.8 s, UEFI 46.8 s with a 1142-token prefix. Cache loads seen: 0.159-0.238 s (0.57 s once). Same greedy answer, token for token, after a prefill and after a cache load (`diff`: identical) |
| (f) golden self-test, decode, snapshots, file tools | Golden self-test **PASS** on every boot (BIOS, UEFI, cold and cached). Boot bench 36.00 / 36.03 tok/s (BIOS cold / cached), 37.30 / 36.52 (UEFI), `/bench` 36.96 and 36.14 after boot, 35.67 in a session after agent turns and a consolidation (M4: 35.0-36.9); 30.3 and 32.5 were taken under the other session's load. `/undo 1` 0.69 ms, `/redo 1` 0.74 ms, `/save m45test`, `/consolidate`, `/bench` -> `conversation restored (ctx 1559)` in 5.5 ms, `/load m45test` 0.69 ms; agent `write_file` + `cat` of `/notes/m45.txt`, `!cat`: **PASS** |
| (g) every decision in /agentd log | **PASS**: recovery, index load or rebuild with the reason, each file scan result, boot cache hit or miss with the reason, scheduler job changes, index saves, every slot (facts, recalled ids and cosines, tokens, ms), every fact saved, updated, deduplicated or rejected with the reason, consolidation start, the raw lines the model wrote, restore time, preemption, shutdown |
| Self-tests at boot | agentd (record codec + torn tail, fact validation, grounding, calendar), memory index, file indexer, tool parser, engine ABI: **PASS** |
| Idle consolidation + preemption | `!set consolidate 8`: a turn, 8 s idle -> consolidation ran (episode + `favorite_color = verde`). Next turn, input 1.75 s into the next consolidation: `stopped by input at the prompt after 0 tok, current point restored (ctx 1583) in 648 us; retried later`; `/ctx` 1583 before and after; it ran again at the next idle period. The answer to "qual e la mia squadra del cuore?" came from the untouched conversation |
| Consolidation cost | 27-45 tokens in 3.3-4.5 s (6.4 s for 96 tokens); current point restored in 0.64-1.08 ms |
| Memory slot cost | 47-163 tokens, built in 0.19-0.44 s (query embedding + search + queued texts), plus about 21 ms per token of prefill on the first turn (47 tok/s: 1-3.5 s) |
| Tools in real turns | "la mia squadra del cuore e il Bologna" -> the model called `remember` three times: `user_name=Cris` and `favorite_color=verde` rejected (not said in this session: they came from the slot), `team=Bologna` saved. "dimentica la mia squadra del cuore" -> `forget(key=squadra_del_cuore)` -> error with the saved keys; "dimentica il fatto team" -> `forget(key=team)`, y/N, y -> forgotten |

### Transcripts (excerpts)

```
ringzero[main]> mi chiamo Cris e il mio numero preferito e 4711
Cris, il mio numero preferito è 4711.
ringzero[main]> /consolidate
[agentd] consolidating the session (Ctrl-C stops)...
[agentd] done: 1 facts (1 new or updated), 1 episodes; /facts lists them, /agentd log shows each decision
ringzero[main]> /agentd log 9
[   44.550] consolidate: start (/consolidate) on top of the current point (ctx 1298), 1 new turns, instruction 114 tok
[   48.264] consolidate #1 (/consolidate): 27 tok in 3713 ms, current point restored (ctx 1298) in 1081 us
[   48.264] consolidate #1: model wrote: EPISODE: Cris ha detto che il suo numero preferito è 4711.
[   48.264] consolidate #1: model wrote: FACT: user_name = Cris
[   48.272] episode 1 saved: Cris ha detto che il suo numero preferito è 4711.
[   48.280] fact saved user_name = Cris (source consolidation (episode 1))
ringzero[main]> !poweroff

(boot)
[agentd] index: loaded 0 entries from /sys/mem/index.bin in 0.602 ms (0 files in the table), no re-embedding
[agentd] boot: 1 facts, 1 episodes, journal 5 records (0 KiB); queued for the index: 3 memory texts, files: none
[agentd] boot cache: hit, prefix state loaded from the disk in 209 ms
ringzero[main]> come mi chiamo?
[memory] - user_name: Cris
[memory] - the user said earlier: mi chiamo Cris e il mio numero preferito e 4711
Mi chiamo Cris.
ringzero[main]> /clear
ringzero[main]> qual e il mio numero preferito?
[memory] - user_name: Cris
[memory] - the user said earlier: mi chiamo Cris e il mio numero preferito e 4711
[memory] - earlier conversation 1: Cris ha detto che il suo numero preferito è 4711.
Il mio numero preferito è 4711.

ringzero[main]> !set slot 0
ringzero[main]> mia sorella si chiama Giulia e vive a Torino
ringzero[main]> quanti core hai?   /   quanta memoria libera c'e?
ringzero[main]> /compact
User: Giulia, Torino, 8 core, 2022 MiB free RAM.
compact #1 manual: 1536->1372 tok (4%->4%), summary 31 tok 3100 ms, re-prefill 120 tok 2436 ms, total 5536 ms
ringzero[main]> come si chiama mia sorella e dove vive?
Non so.

[agentd] recover /sys/mem/journal.log: 55 good records, torn tail of 30 bytes at offset 3598 truncated
[agentd] scan: /notes/todo.txt changed on disk (size 17 -> 37, mtime 0x0 -> 0x5d7a986e): re-embedding it
[agentd] scan: 6 files on disk: 5 unchanged (kept, not re-embedded), 1 changed, 0 new, 0 removed
```

## Fact extraction: three wordings of the consolidation instruction

The 2B is weak at writing key/value facts. Measured on the same statement ("mi chiamo Cris e il mio numero
preferito e 4711"), greedy, fresh conversation:

| Attempt | Instruction | Output | Kept by the kernel |
|---|---|---|---|
| 1 (in the build) | `EPISODE:` line, then `FACT: key = value` lines (snake_case key, value in the user's words) | `FACT: user_name = Cris` (the number only in the episode), the same on two fresh disks | `user_name = Cris` |
| 2 | the same plus format examples (`FACT: user_name = Anna`, `FACT: favorite_color = blue`) | `FACT: Cris = Cris`, `FACT: 4711 = numero preferito` on a fresh disk (correct facts in a conversation that already had a memory slot) | nothing: "key only repeats the value", "key is not snake_case" |
| 3 | facts as `remember` tool calls | no EPISODE line and no call ("Cris, ho scritto una riga: ...") | nothing |

Per the 3-attempt rule the instruction stays at attempt 1. The kernel side holds: whatever the model writes,
only valid, grounded facts are stored, and statements and episodes are recalled by embedding, which is what
answered the number after the reboot. In a conversation whose slot already lists facts the 2B tends to echo the
slot as FACT lines (all deduplicated or rejected as not grounded in this session) and once skipped the EPISODE
line (logged: "no EPISODE line in the output, no episode saved").

## Bugs found and fixed during the milestone

- `norm_value` trimmed quotes before the final period, so `"Cris".` kept a quote; caught by the agentd boot
  self-test (FAIL on the first boot), fixed.
- The decision-log prefix was sliced at a fixed width and cut the first letters of each message on the console.
- A fact the model repeats with the same value was logged as "not grounded" (rejected) instead of "already known":
  dedup now runs before grounding.
- `!poweroff` could not be scripted with accept.py (it waited for a prompt that never came): `x:` steps send a line
  and read until the guest closes the connection; `w<sec>:` waits at the prompt first (idle tests).

## T35: memory slot noise fix (before the beta)

Problem (M4.7 A/B, docs/M47_RESULT.md): with the slot on, each fresh conversation of the eval got "the user said
earlier: ..." lines from the earlier eval prompts and the model followed them. Measured again on the working model
(Qwen3.5-4B q4, 8G) before the fix: in-OS exact calls 18/24 with the slot vs 22/24 without.

What the logged candidates showed (`/agentd log` now has one line per candidate: id, cosine, emb / bm25 / emb+bm25,
verdict, text): the worst noise was not low-scoring. Earlier *requests* score high against new requests of the
same kind ("crea la cartella /home/progetti" for "crea la cartella /notes/m47": 0.71; "create notes.txt with hello"
for "leggi /notes/todo.txt": 0.72), so no threshold separates them from real recall (M4.5: the statement 0.65-0.66,
an episode 0.58). The rest was short files (`ciao`, `hello`, `comprare il latte`: 0.45-0.60 against almost
anything) and BM25 hits with no meaning in common (cos 0.22-0.36).

Fix (kernel/src/agentd.rs, deterministic, every decision logged):

| Rule | Where | Why |
|---|---|---|
| a user statement becomes `[said]` only when its turn ends without a tool other than remember / forget / recall | `said_begin` / `said_tool` / `said_end` (live turns and the journal replay: a U record and its T records) | a request the agent carried out is done; logged as `said: not indexed (the turn ran a tool)` (34 of the 54 eval prompts on the 4B) |
| `[said]` also needs a first-person word (mi, mio, mia, sono, ho, ricordati, I, my, me, remember, ...) | `said()` | the 2B answered "create notes.txt with hello" without a tool, and that line came back for every /notes request |
| cosine >= 0.55 for an embedding hit (was 0.45), >= 0.40 for a BM25-gated hit (was: any) | `slot()` | short files and generic lines sit at 0.45-0.55 |
| note body >= 20 characters | `slot()` | `ciao`, `hello`, `comprare il latte` |
| not the message itself | `slot()` | the same statement typed again |
| at most 3 recalled lines (facts are pinned, not counted) | `slot()` | cap |

Not done on purpose: same-session turns are not excluded. After `/clear` the old turns are not in the context any
more, and before a consolidation `[said]` is the only carrier of a statement made before `/clear` (consolidation
covers the current line only); after `/compact` the M4.5 test (b) needs exactly those turns. The filters above
remove the noise without it.

A/B (`tools/tooleval.py`, constrain on, temperature 0, `/clear` per prompt, consolidation off; each run boots a
fresh copy of the user's disk without `/sys`, the files indexed before the first prompt; `SLOT=200` keeps the
slot's decisions per prompt; raw results on fedora in `~/coding/ringzero/build/t35/`):

| Exact calls | 4B q4 slot off | 4B slot on, old rules | **4B slot on, T35** | 2B q8 slot off | 2B slot on, T35 without the first-person rule | **2B slot on, T35** |
|---|---|---|---|---|---|---|
| spike (30) | 22 | 23 | **22** | 21 | 20 | **20** |
| in-OS (24) | 22 | 18 | **23** | 20 | 17 | **20** |
| first call right, spike (25) / in-OS (22) | 22 / 19 | 19 / 16 | **22 / 20** | 22 / 19 | 21 / 17 | **21 / 19** |
| spike exact args (15 with rules) | 9 | 8 | **9** | 9 | 8 | **8** |

Slot candidates in the 4B runs: old rules kept 155, T35 kept 10 (196 below the threshold, 25 too short, 6 BM25
below its floor). The prompts that still differ: 4B "cosa sai di me?" answered from the pinned fact
`colore_preferito: verde` instead of calling `recall` (right answer, no call), and two "usa python" prompts that
went right with the slot; 2B "how much free RAM and how many cores?" lost the `cpuinfo` call with
`/demo/data.json` (cos 0.57) in the slot.

Recall tests on the final build (fresh disk, both models):

| Test | 4B q4 (8G) | 2B q8 (6G) |
|---|---|---|
| (a) "mi chiamo Cris e il mio numero preferito e 4711", `/consolidate`, `!poweroff`, boot; "come mi chiamo?" (the conversation restored from hibernation) | `remember` x2; after the boot `recall(user_name)` -> "Ti chiami Cris." | "Mi chiamo Cris." |
| (a) `/clear`, "qual e il mio numero preferito?" (new conversation) | slot: 2 facts + statement + episode -> "Il tuo numero preferito è 4711." **PASS** | slot: fact + statement + episode -> "Il tuo numero preferito è 4711." **PASS** |
| (b) "mia sorella si chiama Giulia e vive a Torino", `/compact`, "come si chiama mia sorella e dove vive?" | `remember` x2; slot after compaction: 4 facts + the statement -> "Tua sorella si chiama Giulia e vive a Torino." **PASS** | no `remember`; slot: fact + the statement -> "La mia sorella si chiama Giulia e vive a Torino." **PASS** (the statement carries it) |

Limits: the first-person rule is a word list for Italian and English; a statement without one ("domani parto per
Roma") is no longer a `[said]` entry, only the episode and the facts of a consolidation keep it. Indexes saved
before this build keep their old `[said]` entries until `/index clear` (a rebuild replays the journal with the
new rules). The thresholds are tuned on these two eval sets and the M4.5 tests, not on a larger set.

## Not done / limits

- The memory slot recalled noise (very short files, BM25 matches on a name, earlier requests): fixed in T35
  (above).
- Every fact is pinned (all of them go into the slot within 120 tokens, newest first); there is no `/pin` and no
  eviction by relevance.
- A consolidation runs only on the current conversation's context: after `/clear` the turns of the old line are
  dropped from consolidation (they stay in the journal and as `[said]` entries).
- The journal grows without rotation; it is read whole at boot (heap).
- Index saves write the whole file (up to ~5.5 MiB for a full index) in one step of the idle scheduler.
- `/index add` notes are saved with the index but belong to no store: a rebuild drops them.
- The model hash is a sample (size + 1.1 MiB), not a full hash of the 2 GB file.
- Boot to the prompt with the cache is still 11.9 s: firmware, module load, model init, golden test and bench
  (about 9.2 s), and touching the 1 GiB snapshot pool when the root point is saved (1.4-2.7 s).

## Open risks

- **Fact extraction quality** (above): the 2B misses or garbles facts; recall of statements and episodes carries
  the rest. A bigger model, or a second pass that asks one question per candidate fact, would help.
- **Slot answers in the wrong person** ("Il mio numero preferito è 4711", "Mi chiamo Cris."): the 2B reads the
  slot as its own facts sometimes. The value is right; the wording is not.
- **Grounding is lexical**: a value said by the user in another session, or a paraphrase ("verde" vs "green"),
  is rejected; a wrong word next to a right one can pass (half of the words).
- **FAT is not journaled**: the record checksums protect the stores, but a crash can still leave the volume
  dirty or leak clusters (`fsck.vfat -a build/disk.img`).
- **Host load**: decode and prefill speed drop by 10-60% while another guest runs on the box (seen during these
  tests); boot times above were taken without it.
