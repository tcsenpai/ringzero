# M4.7 result: constrained tool calls, hibernation (2026-11-26)

Two kernel features on top of M4.5 agentd, with M5 MicroPython merged first. Edition unchanged: **ringzero
Qwen35@2b-Qwen3@0.6b** (no new model). Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

All numbers are real runs on the Fedora box (Ryzen AI 9 HX 370, QEMU 10.2.2/KVM, `-cpu host -M q35 -m 6G -smp 8`,
BIOS), on a test guest (`make run-remote REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test DISK=build/disk-test.img`,
a copy of the user's `build/disk.img`), driven from the Mac over Tailscale with `tools/accept.py` and
`tools/tooleval.py`. The user's instance was stopped (`!poweroff`) during the tests. **Another session ran its own
guest from `~/coding/ringzero-prefill` during most of the tests: host load average 10-15**, so absolute speeds are
lower and noisier than in M4.5 (boot bench 16-38 tok/s on the same build); compare numbers taken side by side.

## How to connect (instance left running on fedora, this build, the user's disk)

```sh
nc HOST-ADDRESS 4555                 # serial shell: press Enter to get "ringzero[main]> "
open vnc://HOST-ADDRESS:5901         # framebuffer + PS/2 keyboard, password: aios
ssh fedorascale 'tail -f ~/coding/ringzero/build/remote-serial.log'
```

`build/disk.img` keeps its data (not reformatted). Restart: `echo quit | socat - UNIX-CONNECT:build/remote-monitor.sock`
then `setsid nohup make run-remote > build/remote.log 2>&1 < /dev/null &`. `!poweroff` now hibernates: the next
boot resumes the conversation (`/hibernate discard` first for a fresh one).

## Step 0: M5 MicroPython merged

`worktree-agent-a0d357efb29e33717` (023108f) merged into main: conflicts in agent.rs (tool arrays: 16 M4.5 tools
+ `run_code` = 17 entries in TOOLS and SPECS), README and ARCHITECTURE (both texts kept). Boot prefix 1254 ->
1315 tokens (cache miss once, then hits). Checked: golden self-test PASS, `!py print(2+2)` -> `4` (13 ms),
"quanto fa 17*23? usa python" -> `run_code(code=17*23)` -> `391`, `/facts` (`user_name = Cris`) and the memory
slot after `/clear`, `/bench` restores the conversation.

## M4.7a constrained decoding for tool calls

| Part | Status |
|---|---|
| Grammar | DONE (gram.rs). A byte automaton built from `agent::SPECS` (the same table the validator uses): ws, `<function=`, a tool name, `>`; then `<parameter=` + an unused parameter name of that tool + `>` + a value, or `</function>` once every required parameter is given; then only the `</tool_call>` token. Values: integers (`-?digits`, read_log `n`), enums (set_setting `key`), free strings (anything but `</function>`, `<parameter=` and special tokens, until `</parameter>`). At most 4 whitespace bytes between structural parts. Active only after the model emitted `<tool_call>` |
| Masking | DONE. The sampler's pick is checked first (all its bytes through the automaton). Allowed: kept. Rejected: every token the automaton rejects gets -inf in the engine's logits buffer, then argmax or top-p picks again; with one allowed token it is forced without a pick. Greedy gives the argmax of the masked logits; top-p gives the masked distribution (reject + resample) |
| Vocab trie | DONE. 248044 text tokens sorted by their bytes + one common-prefix-length byte each (an implicit trie, the bytes stay in the model's token table): **47 ms at boot, 1211 KiB**; 0 text tokens over 128 bytes. The mask walk keeps the automaton state per depth and skips every token under a rejected prefix |
| Cost outside a call | one bool test per decoded token |
| Toggle | `!set constrain on|off` (default on). Off keeps the automaton as a passive checker; every call prints `[grammar] on|off: N tok, ... -> valid call / valid call, steered / INVALID / unfinished`, also in the kernel log |
| Self-test | `gram: tool-call grammar self-test PASS` (4 good calls, 7 bad ones: unknown tool, enum, missing value, non-integer, parameter twice, unclosed value, whitespace run) |
| Steered calls | Found in the A/B (below) and fixed: on "set the hostname to aios-dev" the model writes `key=hostname`; the mask replaced it with a valid enum key and the model then really switched the keyboard layout to `us`. Now a call in which the mask chose a tool name, parameter name, enum value or integer ("steered") is not run when the tool changes the machine (set_setting, remember, forget, write_file, mkdir, rm, poweroff, reboot, run_code): the model gets `error: not run: ... not in the schema ...`. Read-only tools still run. Checked: same prompt -> 2 steered calls not run, `!kbd` still `it` |

### A/B: spike prompts + in-OS prompts (`tools/tooleval.py`, temperature 0, `/clear` per prompt)

Spike set: the 30 prompts of `docs/spikes/toolcall/prompts.json` (written for the spike's tool list: some ask for
tools or parameters the kernel does not have, e.g. hostname, wasm). In-OS set: 24 prompts for the kernel's 17
tools (memory, Python, files, settings, log, cores, poweroff) plus 2 without a tool. y/N questions answered n.
agentd's memory slot and idle consolidation off for both runs (see the note below).

| | spike, off | spike, on | in-OS, off | in-OS, on |
|---|---|---|---|---|
| prompts (need a tool / no tool) | 30 (25 / 5) | 30 | 24 (22 / 2) | 24 |
| tool calls made | 39 | 39 | 22 | 22 |
| **calls leaving the grammar (strict format, schema)** | **5** (all `key=hostname`) | **0** | 0 | **0** |
| parser errors (`[tool] malformed call`) | 0 | 0 | 0 | 0 |
| schema errors from the validator | 4 | **0** | 0 | 0 |
| first call is the right tool (of the prompts that need one) | 23/25 | 23/25 | 19/22 | 19/22 |
| exactly the expected calls | 15/25 | 15/25 | 18/22 | 18/22 |
| no-tool prompts answered without a tool | 5/5 | 5/5 | 2/2 | 2/2 |
| tokens masked | - | 1 (1 call) | - | 0 |
| decode tok/s of the call rounds (host load 10-15) | 26.22 | 23.08 | 26.21 | 26.60 |

Target met: **0 malformed or out-of-schema calls with constrain on**. The base model already writes the format
well (no parser error in 100 calls either way; the spike's missing `</tool_call>` did not show up), so the
grammar changed exactly one call: the call path is the same token for token in every other prompt, which is
why the correct-tool rates are identical. The wrong-tool cases are model choices the grammar cannot fix
(`ls` after a failed `cat`, `run_code` loops, no call for "how many cores do I have?").

Earlier run with agentd's memory slot on (default), constrain off: correct-tool rate much lower (spike exact
14/30, in-OS 7/24): the slot of each fresh conversation carried "the user said earlier: ..." lines from the
previous eval prompts, and the 2B followed them (e.g. "ricordati che il mio colore preferito e il verde" ->
"ho letto il file /home/todo.md", no `remember` call). Grammar-breaking calls the same 5. Not a grammar issue,
but a real agentd risk (below).

### Cost per token

| Measure | Value |
|---|---|
| check of the picked token (fast path), per token inside a call | avg 1.45 us, 0.77-5.4 us over 130 calls (under load) |
| full mask, boot bench (`gram: full mask cost`), quieter host | call start 305 us (70 tokens allowed), tool name after `re` 227 us (15), after the name 215 us (70), enum value 241 us (69), **free value 3576 us (all 248044 allowed)** |
| same, under load average 11 | 1431 / 1444 / 1010 / 1435 us, free value 39463 us |
| masks in the A/B | 1 mask, 418 us, in 61 calls with constrain on |
| decode, `/bench` alternated off / on (64 tokens, no tool call) | off 35.16, 35.63, 37.13 tok/s; on 37.12, 37.05, 29.47 (load spike) tok/s; M4.5: 36-37 |

A full mask costs about 1% of a 30 ms decode step in structural states; inside a free value (rare: only when a
token closes the value and then breaks the grammar) it walks the whole trie, about 12%.

## M4.7b hibernation

| Part | Status |
|---|---|
| Image | DONE (snap.rs `hib_save` / `hib_load`, hib.rs). `/sys/hib/image.bin`: 64-byte header (magic RZHB, version, key, state hash, sizes, pool size, FNV of the rest), metadata (head, cur, tick, next id, pool cap, root length; every point but the root: id, parent, history link, rows, RNG, visit, exchange flag, compaction number, R hash, its question, answer and side-effect notes; every branch and save name), the 1 MiB pool chunks of every point's KV rows (token ids included), the DeltaNet state R of the current point only, then the agent's extra blob (temperature, think, constrain, compaction percent and count, raw transcript). The root is the boot prefix this boot loaded or prefilled |
| Key | FNV of the boot snapshot key (model sample hash, prefix ids, engine build hash, KV element size, max seq, context length) + "ringzero hibernate v1"; the pool size and the root length are checked too |
| Save | `/hibernate`; every `!poweroff` and the agent's poweroff tool (sys::poweroff, before agentd's shutdown and the disk sync). The live state is first made equal to the current point (a mid-turn poweroff keeps the last finished turn). A failed write deletes the partial file |
| Resume | at boot after the root point: header, key, sizes, metadata parsed and checked before anything changes, chunks read straight into newly allocated pool chunks and hashed, then the tables are committed and `restore(cur)` rebuilds the live KV rows and R. One line: `hib: resumed #N on BRANCH (ctx P), K points in T ms, state hash H (same as saved)` |
| Other points | come back thinned; visiting one replays its tokens from the nearest stored R and checks the R hash taken at save time |
| Fallback | any mismatch or error: `hib: image ... not resumed (why), fresh conversation; /hibernate discard deletes it`, chunks freed, the boot goes on at the root |
| Discard | `/hibernate discard` deletes the image and turns the poweroff hook off for the session (else the next `!poweroff` would write a new one); `/hibernate` turns it on again. `/hibernate status` |
| State hash | FNV over the live token ids, the rows 0..pos of all 24 K/V planes and R; `/snap` prints it |

### Acceptance

| Test | Result |
|---|---|
| (a) conversation continues after poweroff + boot, same state hash, same tree | **PASS**. "mi chiamo Cris e ho un gatto di nome Pippo", "quanta RAM libera?" (meminfo), `/branch alt` + "che processore ho?" (cpuinfo), `/switch main`, `/save keep1`, "come si chiama il mio gatto e quanti anni potrebbe avere?" -> "Il tuo gatto si chiama Pippo.", `/undo 1`; `/snap` hash **0x65e5979e00cfd53c**; `!poweroff` -> image 31 MiB in 2866 ms. Boot: `hib: resumed #3 on main (ctx 1490), 6 points in 500 ms, state hash 0x65e5979e00cfd53c (same as saved)`, prompt `ringzero[main~1]>`, `/tree all` the same 6 points, names (main@1, keep1, alt, main) and current point (4 now thinned), `/snap` the same hash |
| (b) identical next answer at temperature 0 | **PASS**. The same question after the resume -> "Il tuo gatto si chiama Pippo." (ctx 1526 both times). On the final build again: state hash 0x5834b5da4441f737 saved and resumed (220 ms); "quale e il mio gatto? e quanto RAM ho libera?" -> `recall(query=gatto)` -> "Il tuo gatto è Pippo." (ctx 1787), `/undo 1`, `!poweroff` (33 MiB, 2334 ms), boot (161 ms, same hash), same question -> the same call, the same tool result and "Il tuo gatto è Pippo." (ctx 1787) |
| (c) thinned point after a resume | **PASS**. `/switch alt` -> `#4 was thinned: replaying 119 tokens from #3`, `replay done, R hash matches the stored one` (4.8 s under load) |
| (d) restore time, disk size, save time | resume **161-500 ms** (31-33 MiB image, 6-7 points; the cached boot prefix 0.23-0.48 s comes before it); image **30-33 MiB** for 6-7 points (19.3 MiB R of the current point + 1 MiB per point + metadata); save at poweroff **2.3-2.9 s** |
| (e) mismatch falls back cleanly | **PASS**. One data byte flipped on the host (mtools, offset 200000): `not resumed (checksum mismatch), fresh conversation`. Key byte flipped (offset 9): `not resumed (key differs (model, prefix, engine or KV geometry changed))`. Both boots: `ringzero[main]>`, `/tree` = the root only, a tool turn works |
| (f) discard | **PASS**. `/hibernate discard`, `/hibernate status` -> no image, `!poweroff` -> `hib: poweroff without an image (/hibernate discard)`, next boot: no hib line, fresh |

## General checks (final build)

| Check | Result |
|---|---|
| Golden self-test | PASS on every boot (merge build, M4.7 builds, every resume) |
| Self-tests | engine ABI, memory index, file indexer, agentd, tool parser, **tool-call grammar**: PASS |
| Decode | `/bench` 35.2-37.1 tok/s alternating constrain off/on (M4.5: 36-37); boot bench 37.70 at low load, 16.10 under load 11 |
| Snapshots | `/undo 1` 0.57-1.22 ms, `/redo 1` 1.18 ms, `/branch`, `/switch`, `/save`, `/bench` restores the conversation |
| File tools | agent `write_file(/notes/m47.txt, "ciao m47")` + `cat` -> `ciao m47` |
| agentd memory | `/facts` `user_name = Cris` after reboots; "come mi chiamo?" -> "Cris" from the slot; "e il mio numero preferito?" -> `recall` -> "Il tuo numero preferito è 4711." |
| MicroPython | `!py print(2+2)` -> 4; `run_code(code=17 * 23)` -> 391 |

## Risks and limits

- A grammar makes a call valid, not right: a value outside the schema becomes a valid one the model did not
  mean. Side-effect tools do not run steered calls; read-only ones do (a steered `cat` path is impossible, paths
  are free strings, but a steered `read_log n` or tool name is possible).
- Free string values are not checked beyond the closing tags (paths, keys of `remember`): the kernel validator and
  the tools still do that.
- The full mask inside a free value walks the whole trie (3.6 ms, 39 ms under heavy load). Rare; a shortcut
  for tokens without `<` would make it cheap if it ever matters.
- The A/B shows the base model rarely needs the grammar on these sets (1 of 100 calls); the benefit is the
  guarantee, measured as 5 -> 0 broken calls.
- agentd's memory slot hurt tool selection badly in back-to-back runs (earlier eval prompts recalled as "the
  user said earlier"); worth a follow-up: questions and commands are not facts about the user.
- Hibernation writes about 20 MiB + 1 MiB per point at every `!poweroff` (2-3 s); on the 512 MiB disk a tree near
  the 256-point limit would need up to about 280 MiB. A full disk means no image (the partial file is deleted).
- Non-current points come back thinned: the first visit replays their tokens (about 45 tok/s, slower under load).
- The resumed state depends on the boot prefix state being bitwise the same as at save time. It was (cache hit),
  and the state hash check at resume would show any difference; a prefill on a different core count could
  differ.
- `!reboot` does not hibernate (only poweroff, as specified). agentd's in-memory session text (for the next
  consolidation) is not in the image; the journal on disk is.
- UEFI resume was not tested separately (the same code path after boot).
