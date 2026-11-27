# T22 result: built-in OS manual in memory (2026-11-27)

"How do I ..." questions about ringzero are answered from a manual that ships with the kernel, through the same
memory index that holds the user's files and conversations. Edition unchanged: **ringzero Qwen35@4b.q4-Qwen3@0.6b**
(no new model). All runs: Fedora box (Ryzen AI 9 HX 370, QEMU/KVM, `-cpu host -M q35 -m 8G -smp 8`, BIOS), 4B q4,
on a test guest (`make run-remote MODEL=4b QUANT=q4 REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test DISK=<copy>`)
next to the user's running instance, driven by `tools/accept.py` and `tools/tooleval.py` over the serial port.

## What was built

| Part | Where | What |
|---|---|---|
| Manual | docs/MANUAL.md | 26 sections (`## Title`), one topic each, English, 246-607 bytes each, 12.3 KB in all. Each section names its commands and answers a question alone. Its first line `Asked as: ...` lists Italian and English wordings: it is embedded with the section (cross-lingual retrieval) and not shown to the model. |
| Delivery | kernel/src/agentd.rs | `include_str!("../../docs/MANUAL.md")`: the manual is part of the kernel image, so it always matches the build. No Limine module, no file on the disk. |
| Indexing | agentd.rs `queue_manual` | At boot (after the index load and the file scan) and on a rebuild: indexed `[manual: TITLE] ...` entries whose text is not in the kernel's manual are dropped, sections not in the index are queued as memory texts (idle job 1). The entries are saved in `/sys/mem/index.bin` with the rest (existing header logic). The text itself is the version check: a changed section is re-embedded, the others are kept. |
| Ranking | agentd.rs `hits_split` | One search (top 24) split into manual sections and everything else, so the manual never takes one of the user's slot lines or one of `recall`'s top 3. |
| Gate | agentd.rs `howto` | A message asks how to do something: `how do i`, `how can i`, `how to`, `how does`, `what command`, `which command`, `quale comando`, `in che modo`, `come si fa`, `come posso`, ..., or `come` followed by a first-person verb (`come torno`, `come faccio`, `come cambio`; not `come mi chiamo`, `come si chiama`, `come stai`). |
| Slot | agentd.rs `slot`, `man_slot`; agent.rs | A how-to message gets the best manual section as one slot line when its cosine is >= 0.55 (RECALL_MIN), within the slot's 200-token budget. On the first message of a conversation it comes after the pinned facts; later in a conversation (no memory slot there) the line comes alone under `[from the ringzero manual, kept by the kernel; use it to answer]`. Printed as `[memory] - ringzero manual, TITLE: ...`; every candidate is in `/agentd log` (`slot manual #id cos ...: kept`). |
| recall | agentd.rs `recall` | Top 3 of the user's memory as before, then the best manual section when the user's message (or the query) passes the gate and the section scores >= 0.55. |
| Shell | shell.rs | `/man TOPIC`: the two best sections in full with their score; `/man`: the 26 titles. |
| Prompt | - | Unchanged: no system prompt or tool description change, so the boot prefix and the boot cache key are the same. |

## 1. Ten how-to questions (4B q4, temperature 0, `/clear` before each)

On a copy of the user's disk (1 fact, `user_name = Cris`); the manual was indexed in idle time first (26 entries).
"Section" is the slot line the kernel added (shown as a `[memory]` line in the transcript), with its cosine.

| # | Question | Section in the slot (cos) | Answer (short) | Correct |
|---|---|---|---|---|
| 1 | come torno indietro di un turno? | Undo and redo a turn (0.78) | "usa il comando `/undo` ... `/undo N`" | y |
| 2 | how do I run python? | Running Python (0.70) | `!py <codice>`, `!py -f <PATH>`, `!run python3 /<PATH>`, REPL with Ctrl-D, `run_code` (in Italian) | y |
| 3 | come faccio a ibernare? | Hibernation (0.67) | "digita `/hibernate` ... `/hibernate status` ... `/hibernate discard`" | y |
| 4 | how do I see what the memory system decided? | What the memory system decided (0.73) | `/agentd log N` and what it lists (in Italian) | y |
| 5 | come cambio layout tastiera? | Keyboard layout (0.74) | `!set keyboard it` / `us`, `!kbd` shows the current one | y |
| 6 | how do I list the programs in /bin? | Programs in /bin (0.84) | `!ls /bin` | y |
| 7 | come cancello una cartella con tutto il contenuto? | Writing and deleting files (0.59) | `!rm -r PATH` | y |
| 8 | how do I start a new conversation? | New conversation (0.78) | `/clear`, facts and index stay (in Italian) | y |
| 9 | come vedo quanto contesto ho usato? | Context size and compaction (0.67) | `/ctx`, out of 32768 tokens | y |
| 10 | how do I turn off the memory slot? | Memory slot and consolidation (0.57) | `!set slot 0` | y |

10/10 correct, each with the section visible. Also checked: the same hibernation question as the second turn of a
conversation (manual-only slot, cos 0.67, `/hibernate` answer), "what command shows the kernel log?" mid-conversation
(Kernel log, cos 0.85, `!log`, `!log 5`), "how can I change the sampling temperature?" (Settings, `!set temperature`)
and "come posso vedere lo spazio libero sul disco?" (Listing and reading files, `!df`).

One fix during the run: the first hibernation section said "`/hibernate` saves ..." and the 4B answered
`!hibernate`; the section now says "Type `/hibernate` (a session command: it starts with a slash, not with `!`)", and
the answer became `/hibernate`. The next boot logged `manual: 26 sections, 25 already in the index, 1 queued, 1 old
entries dropped`: only that section was embedded again.

Cosines (`/search`) of the best manual section for every tool-eval prompt and the recall-test questions: 0.34-0.85,
for example 0.85 "set the keyboard layout to us", 0.84 "show the last ten log lines", 0.74 "how many cores do I
have?", 0.67 "hi, how are you?", 0.62 "come mi chiamo?". The "CPU, cores, memory and uptime" section scored
0.55-0.74 against greetings and personal questions. So no threshold separates how-to questions from tool requests
or chat; the gate on the wording does, and the threshold only drops weak sections.

## 2. Tool-call eval (M45 protocol)

`tools/tooleval.py` unchanged, `SLOT=200`, constrain on, temperature 0, `/clear` per prompt, consolidation off; a
fresh copy of the user's disk without `/sys`, files and manual indexed before the first prompt.

| | before (docs/M48B_RESULT.md) | T22 build |
|---|---|---|
| spike: correct / first call right / args exact | 22/30, 22, 9 | **22/30**, 22, 9 |
| in-OS: correct / first call right | 23/24, 20 | **23/24**, 20 |
| malformed / schema errors / grammar lines not valid | 0 / 0 / 0 | 0 / 0 / 0 |
| prompts whose slot got a manual line | | 0 of 54 |
| prompts with the same calls and arguments as the first T22 run | | 54/54 |
| decode tok/s of the rounds that made a call (spike / in-OS) | 25.75 / 27.40 | 26.13 / 28.74 |

No eval prompt passes the how-to gate, so the slot of every eval prompt is the one it had before T22. A first run
on the build before the last recall change (manual line in `recall` when it scored at least as high as the first
hit) gave the same numbers: 22/30, 22, 9; 23/24, 20; 0 / 0 / 0.

## 3. Recall tests (docs/M45_RESULT.md), fresh disk

| Test | Result |
|---|---|
| (a) "mi chiamo Cris e il mio numero preferito e 4711", `/consolidate`, `!poweroff`, boot (index loaded: `manual: 26 sections, 26 already in the index, 0 queued`; conversation resumed from hibernation, same state hash); "come mi chiamo?" | `recall(user_name)` -> facts `user_name`, `user_favorite_number` and episode 1, no manual line (the message is not a how-to question) -> "Ti chiami Cris." **PASS** |
| (a) `/clear`, "qual e il mio numero preferito?" | slot: 2 facts + statement + episode -> "Il tuo numero preferito è 4711." **PASS** |
| (b) "mia sorella si chiama Giulia e vive a Torino", "quanti core hai?", "quanta memoria libera c'e?", `/compact`, "come si chiama mia sorella e dove vive?" | `remember` x2; slot after compaction: 4 facts + the statement (cos 0.72) -> "Tua sorella si chiama Giulia e vive a Torino." **PASS** |

## 4. Boot time and prefix

`DISK=<copy of the user's disk> MEM=8G tools/boottest.py ISO bios ...`, BIOS, boot cache hit, alternating the build
before T22 and this build, on the shared box (the user's 8G guest running):

| | before T22 | T22 |
|---|---|---|
| boot prefix (system prompt + 18 tools) | 1478 tokens, fnv 0xb854650e7e902767 | 1478 tokens, fnv 0xb854650e7e902767 (same) |
| boot cache | hit (key unchanged) | hit (key unchanged) |
| QEMU start to `ringzero[main]>`, run 1 / run 2 | 24.55 s / 21.35 s | 22.57 s / 19.88 s |
| manual check at boot (`manual: 26 sections, 26 already in the index`) | - | 30-40 ms after the index load |
| first boot on a disk without the manual in its index | - | 26 sections queued; embedded in about 20 s of idle time at the prompt (21.4 s -> 41.9 s in the decision log, then the index is saved) |
| index size on the user's disk | 24 entries | 50 entries (+26), index.bin 35 -> 73 KiB |

The difference between the two builds is within the run-to-run spread (the second run of each was 3 s faster than
the first). The kernel image grows by the 12.3 KB manual text.

## Limits

- The gate is a word list for Italian and English. A how-to question worded otherwise ("dimmi il comando per
  ibernare", "posso tornare indietro?") gets no manual line in the slot; the model can still call `recall`, which
  then returns the section only if the query itself passes the gate.
- One manual line per message. A question that needs two sections (for example Python and the programs in /bin)
  gets only the best one.
- The `Asked as:` lines were written with these kinds of questions in mind; the 10 questions were not tuned one by
  one, but a larger set of held-out questions was not run (4 extra questions above).
- English questions were answered in Italian 3 times out of 5 (2, 4, 8), with an English manual line in the slot.
  This was not compared with a run without the manual; it belongs to the F6 check of the beta gate (English
  requests get English replies).
- First boot (or `/index clear`): the 26 sections take about 20 s of idle embedding. A how-to question asked before
  that embeds the queued sections first (the slot and `recall` flush the memory queue; Ctrl-C stops it).
- The manual is part of the kernel: changing it means a rebuild. It does not cover INFO.md's build and host
  commands (`make ...`), only what can be done at the ringzero prompt.
