# M8 common software in /bin

Result: `make wasm-apps` now puts the software people expect on a small Unix-like system in `/bin` on the data
disk: uutils coreutils, ripgrep, jq (jaq), sqlite3, Lua 5.4, CPython 3.15 with its standard library, QuickJS-ng
and the kibi editor. All of them run from the shell with realistic commands; the REPLs of python3, lua and sqlite3
work with typed input, kibi edits and saves a file over the serial console. The agent runs them through
`run_wasm`: the 2B answered the sqlite3 request of the task (create, insert, select the most expensive product)
and used wc, seq and jq correctly in at least one run; it did not get python3, lua or rg right in any run (it
reaches for MicroPython, or writes a wrong command line). The runtime changes for this: a 192 MiB kernel heap with
a load check and a stop-the-run flag instead of an OOM panic, console stdin and `/dev/tty`, a file handle cache, a
token cap on the agent's output, a more forgiving command-line splitter. Golden self-test PASS on every boot,
decode speed unchanged within the noise (table below).

## What runs

All rows were run on the Fedora box under QEMU/KVM (q35, 6 GiB, 8 vCPUs, BIOS) from the serial console by
`tools/m8test.py`; the transcripts are `build/t1` .. `build/t13` in `~/coding/ringzero-m8`. Times are the
status line of each run (module load included), with the M7 agent's guests running on the same box.

| Program | Version | Size | Command | Result |
|---|---|---|---|---|
| uutils coreutils | 0.13.0 | 9.9 MB | `!run ls -l /bin`, `wc -l`, `sort -r`, `head -n 3`, `cut -d , -f 1,2`, `sha256sum`, `sort < FILE`, `cp`, `mv` | correct output, 58-190 ms each. `du` is not in the `feat_wasm` set |
| ripgrep | 15.2.0 | 3.3 MB | `!run rg -n latte /notes`, `rg -i -c roma /demo`, `rg -l "def \|import " /demo` | correct, 56-90 ms |
| jq (jaq) | 3.2.0 | 3.6 MB | `!run jq .version /demo/data.json`, `".models[] \| .name"`, `"[.models[].size_mb] \| add"`, `-r ".apps[]"` | correct, 77-181 ms; colors on the terminal |
| sqlite3 | 3.54.1 | 1.5 MB | `!run sqlite3 /demo/shop.db "create table ...; insert ..."`, `"select ... order by price desc;"`, `-json`, then the REPL: `select * from items where price > 2;`, `.tables`, `.quit` | correct, 110-233 ms per command; the db file lives on the FAT disk |
| Lua | 5.4.8 | 0.5 MB | `!run lua /demo/hello.lua` (gmatch, table.sort, closures, `pcall(error)`, io.open write + read), `lua -e ...`, the REPL with `pcall(error, "caught")` | correct, 23-134 ms; `fib(20)` 20 ms |
| CPython | 3.15.2 | 31 MB + 13 MB stdlib | `!run python3 -c "import sys; print(sys.version)"`, `!run python3 /demo/stats.py` (csv, re, statistics, collections, json, writes a file), `-c "import json, re; ..."`, the REPL (`import datetime, textwrap`, a list comprehension) | correct. Start-up 454 ms; stats.py 4555 ms the first time (compiles and writes `__pycache__`), 956 ms after |
| QuickJS-ng | 0.18.0 | 1.6 MB | `!run qjs /demo/hello.js`, `qjs -e "console.log(...)"`, the REPL (`1 + 2`, `.quit`) | correct; the REPL echoes each character twice (see below) |
| kibi | 0.3.2 | 0.2 MB | `!run kibi /notes/todo.txt`: the cursor-position query answered, text typed, Ctrl-S, Ctrl-Q, then `!cat` | `27B written to /notes/todo.txt`, the file has the text. Serial console only |
| MicroPython (M5, in the kernel) | 1.29.0 | | `!py print(sum(x * x for x in range(10)))` | `285`; `!py -f /demo/stats.py` stops at `import csv` (no csv, statistics): pure-Python scripts that need the stdlib go to `python3` |

Disk: `/bin` holds 53 MB (11 files), `/lib/python3.15` 13 MB in 594 files plus the `__pycache__` python3 writes;
`!df` after all runs: 104528 KiB used of 510 MiB.

## Agent

`run_wasm`'s description lists the programs and five sample command lines (`wc -l /a.txt, jq .name /a.json,
sqlite3 /a.db "select 1;", lua /a.lua, python3 /a.py`). Requests one per conversation (`/clear` in between), on a
disk without earlier memories (`/sys` removed), Qwen3.5-2B Q8, greedy. Final kernel (`build/t13`):

| Request | Calls | Result |
|---|---|---|
| quante righe ha /notes/casa.txt? | `cat(/notes/casa.txt)` | `7 righe.` (right, without run_wasm) |
| crea un db sqlite /spesa.db con una tabella prodotti (nome, prezzo) con pane 2.5, latte 1.2, caffe 4.8 e dimmi qual e il prodotto piu caro | `run_wasm(path=sqlite3 /spesa.db "CREATE TABLE ...; INSERT ...")` -> `[exit 0]`, `run_wasm(path=sqlite3 /spesa.db "SELECT * FROM prodotti ORDER BY prezzo DESC;")` -> `caffe\|4.8 ...` | `Il prodotto più caro è la caffe a 4.8.` |
| estrai il campo version da /data.json | `cat(/data.json)` | `La versione è 0.8.` (right, without run_wasm) |
| ... con jq | `run_wasm(path=/data.json, args=jq .version)` twice, `run_code(import json)` twice | failed (program and file swapped). With the earlier description (`build/t10`, `t12`): `run_wasm(path=jq -r '.version' /data.json)` -> `0.8` on the third call |
| conta le righe di /notes/casa.txt con wc | `run_wasm(path=wc -l /notes/casa.txt)` | `7 /notes/casa.txt` |
| quali tabelle ci sono nel database sqlite /demo/shop.db? | `run_wasm(path=sqlite3 /demo/shop.db "SELECT name FROM sqlite_master WHERE type='table';")` | `Ci sono 1 tabella nel database: items.` |
| usa run_wasm per eseguire seq 1 3000 | `run_wasm(path=seq, args=1 3000)` | output cut to 1500 tokens (below); then the model wandered off with `rg -0 /notes` |
| esegui lo script /demo/stats.py con python3 | `run_code(wc -l ...)`, `run_wasm(wc ...)`, `run_wasm(jq .)` | failed: the model does not connect python3 with run_wasm |
| esegui /demo/hello.lua con lua | `run_code(import os)`, `run_code(import sys)` x3 | failed: same, it reaches for MicroPython |
| usa rg per cercare la parola latte nella cartella /notes | `run_wasm(path=rg /notes/casa.txt, args=latte)` | rg ran with the file as the pattern; the model then said "latte" is not in the file (wrong) |

So the agent used sqlite3, coreutils (wc, seq) and jq through `run_wasm` with the right answer in at least one
run each; python3, lua and rg were not used correctly by the 2B in any run.

What the model gets wrong is mostly the command line: SQL without quotes and without the database file, a made-up
path (`/path/spesa.db`), `-c` code wrapped in an extra pair of quotes, `lua` with an expression instead of `-e`,
`rg -r FILE PATTERN` (`-r` is `--replace`). Three fixes in the tool made some of its habits work:

- the whole command in `path` (`path=jq -r '.version' /data.json`, `path=seq 1 3000`) is split into program and
  arguments;
- the program name repeated as the first argument (`args=sqlite3 /spesa.db ...`) is dropped;
- the splitter takes `'...'` as well as `"..."`, and `\"`, `\\` as in a shell (`\d` stays, for regexes).

agentd's memory slot sometimes puts an earlier request of the same boot into the prompt ("the user said
earlier: usa run_wasm per eseguire seq 1 3000"), and the model then answers that one. That is agentd's recall, not
the runtime; it is the main reason the agent tests ran each request after `/clear` on a fresh disk.

Token cap: `usa run_wasm per eseguire seq 1 3000` -> the output is cut at 1500 tokens of the loaded tokenizer (a
line end in the second half when there is one), with `[... 12394 more bytes of output dropped (the tool keeps about
1500 tokens): narrow the command, e.g. head, wc -l, rg -c]`. The next prefill is 1564 tokens in 10.7-19.5 s;
M5b's 6 KiB of the same output was 6218 tokens and 36 s. The model sees only `[exit N]` (or the trap / stop
reason) after the output; the full status line goes to the kernel log.

## After the merge onto main: the 4B q4 (working model)

M8 merged onto main after M7 (Qwen3.5-4B Q4_0 MLP, 8G guest, docs/MODEL_CHOICE.md). Same requests, one per
conversation, greedy, constrain on, memory slot on (default), on a copy of the user's disk with `/sys` removed
(`build/t35/m8_4b` on fedora, `tools/m8test.py`, MEM=8G). The demo blocks of docs/DEMOS.md (coreutils, rg, jq,
sqlite3, lua, python3, qjs) all gave the same output as on the 2B build (stats.py 5.1 s on its first run).

| Request | 2B q8 (above) | 4B q4 |
|---|---|---|
| quante righe ha /notes/casa.txt? | `cat` -> `7 righe.` | `run_wasm(path=wc, args=-l /notes/casa.txt)` -> `7 righe.` |
| crea un db sqlite /spesa.db ... prodotto piu caro | two `run_wasm` sqlite3 calls -> right | one `run_wasm(path=/bin/sqlite3, args=/spesa.db "CREATE ...; INSERT ...; SELECT ... ORDER BY prezzo DESC LIMIT 1;")` -> `caffe\|4.8` -> `Il prodotto più caro è il caffè a 4.8.` |
| estrai il campo version da /data.json | `cat` -> `0.8` | `run_wasm(path=jq, args='.version' /data.json)` -> `"0.8"` -> right |
| ... con jq | failed (program and file swapped) | the same jq call -> right |
| conta le righe di /notes/casa.txt con wc | `run_wasm` -> right | `run_wasm(path=wc, args=/notes/casa.txt)` -> `7 29 177` -> `7 righe` |
| quali tabelle ci sono nel database sqlite /demo/shop.db? | right | `run_wasm(path=sqlite3, args=/demo/shop.db ".tables")` -> `items` -> right |
| esegui lo script /demo/stats.py con python3 | failed (reached for wc, jq) | `run_wasm(path=python3, args=/demo/stats.py)` -> the JSON, summed up right |
| esegui /demo/hello.lua con lua | failed (MicroPython) | `run_wasm(path=/bin/lua, args=/demo/hello.lua)` -> right |
| usa rg per cercare la parola latte nella cartella /notes | failed (file as the pattern) | `run_wasm(path=rg, args=-l /notes latte)`: pattern and path swapped, exit 2; the answer was still right because the memory slot had `/notes/todo.txt: comprare il latte` |

So the 4B uses `run_wasm` correctly for coreutils, sqlite3, jq, python3 and lua (8 of 9; the 2B: 3 of 9 with
`run_wasm`, plus 2 right through `cat`), and only rg's argument order is still wrong. RAM with the 192 MiB heap:
4B q4 on 8G `free 1651 MiB` at the prompt and after the runs, 2B q8 on 6G `free 1564 MiB`; python3 peaks at 93 MiB
of heap on both.

## Runtime changes (kernel/src/wasm.rs, mem.rs, fs.rs, console.rs)

| Risk from M5b | Change | Check |
|---|---|---|
| kernel heap OOM halts the kernel; CPython does not fit | heap 64 -> 192 MiB (`RINGZERO_HEAP_MIB`), module limit 24 -> 48 MiB. A module is loaded only when 4x its size + 12 MiB is free (the load makes single big allocations); during the run the allocator sets a flag when less than 12 MiB is left and `stop()` ends the run at the next fuel slice (`stopped: the kernel heap ran low`) | a 100 MiB heap build refuses python3 cleanly, see Measurements |
| stdin is a file or EOF: no REPLs | shell runs get the console: fd 0 is a line reader (echo, backspace, Enter, Ctrl-D = end of input, Ctrl-C stops the run, escape sequences ignored), `/dev/tty` is raw (one byte per read, no echo), `fd_fdstat_get` drops the seek/tell rights on console fds so `isatty()` is true. Time spent waiting for input is added to the deadline. Agent runs keep stdin empty and no tty | python3, lua, sqlite3 and qjs REPLs; kibi |
| every read/write reopens the file by path (sqlite random I/O) | an open fd keeps its fatfs `File` (opened on first use); a write, a truncate, an unlink or a rename on the same path closes the other handles, so none keeps a stale size or cluster chain; fs.rs gets `open`, `len`, `read_at`, `write_file_at`, `set_len` (read_into and the WASI writes use them) | sqlite3 create/insert/select and the REPL on `/demo/shop.db` and `/t.db` |
| lua needs setjmp/longjmp; wasmi has no exception handling (checked: no such option in wasmi 2.0.3's Config) | `tools/wasm/c/setjmp.h` + `rzsjlj.c`: setjmp unwinds the stack to main with Binaryen's Asyncify and rewinds at once (saving the locals in jmp_buf), longjmp unwinds and rewinds from that copy (the state machine of Ruby's wasm/setjmp.c, our code). `wasm-opt --asyncify -O2` after linking; in the other order (`-O2 --asyncify`) every longjmp trapped (`unreachable`), probably because -O2 inlined rz_setjmp into its caller first (not checked further) | hello.lua (pcall of error, nil index, recursion 150 deep, coroutines) under wasmtime and ringzero |
| 6 KiB of digits is 6000 tokens for the agent | `tok_cut`: binary search for the longest prefix within 1500 tokens of `llm::encode_plain` | seq 1 3000 above |
| editors: no tty | kibi opens `/dev/tty` (raw console); it asks the terminal for the window size with `ESC [6n`, so it needs a terminal that answers: the serial console (nc, a terminal emulator). The framebuffer console now skips ANSI escape sequences instead of drawing them | kibi row above |

Other: `PYTHON_BASIC_REPL=1` in the environment (CPython's new REPL needs termios and spent 9 s importing before
it gave up), a `.wasm` without the `\0asm` header gets a one-line error instead of wasmi's dump of both magic
numbers, agentd's file indexer skips `/lib` (it had queued the 594 stdlib files for embedding). `filestat` now
gives dev 1 and an inode (FNV-1a of the lowercased path): with 0 for every file, `!run cp /demo/data.json
/data.json` said `'/demo/data.json' and '/data.json' are the same file`.

Regression checks on the final kernel (`build/t15`): `!wasm rztest check A "B C"` PASS 20/20, `rztest escape`
(every `..` refused), `rztest stdin < /notes/casa.txt` (177 bytes, 7 lines), `cp`, `mv`, python3
`shutil.copy`, and sqlite3 on a fresh db: 2000 rows inserted, `update ... where x % 3 = 0`, `sum(x)` = 2667333
(right), `pragma integrity_check` = ok.

## Measurements

Baseline is main at `35e521d` (M5b), built in its own directory; both ISOs booted alternately on the same box
without a data disk, five rounds, each boot running the boot bench and three `/bench` (64-token decode). The M7
agent's 8 GiB guest was busy on the same host the whole time (load average 12-17 on 24 threads), so single
numbers swing from 17 to 36 tok/s:

| | baseline `35e521d` | M8 |
|---|---|---|
| 64-token decode, 20 runs each: median / best | 29.1 / 35.6 tok/s | 29.1 / 36.7 tok/s |
| agent boot prefix (system prompt + 18 tools) | 1413 tokens | 1478 tokens (+65: the program list and sample command lines in run_wasm) |
| `!mem` free (6 GiB guest) | 1694 MiB | 1566 MiB (-128: kernel heap 64 -> 192 MiB) |
| kernel heap | 64 MiB | 192 MiB |
| kernel `.text` | 2 820 660 B | 2 842 036 B (+21 376) |
| golden self-test | PASS | PASS on every boot of the M8 ISOs (agent, shell, bench and 100 MiB heap boots) |

The engine is unchanged and wasm never runs during a forward pass; the decode numbers agree within the noise.

Programs (status line of the shell runs, module load included, warm disk cache):

| Run | Time | Kernel heap peak | Linear memory |
|---|---|---|---|
| `wc -l /notes/casa.txt` (coreutils, 9.9 MB) | 58-96 ms | 22 MiB | 4.4 MiB |
| `rg -n latte /notes` | 56 ms | 7 MiB | 2.1 MiB |
| `jq .version /demo/data.json` | 77 ms | 8 MiB | 1.7 MiB |
| `sqlite3 /demo/shop.db "select ..."` | 110-201 ms | 3 MiB | 0.4 MiB |
| `lua /demo/hello.lua` | 134 ms | 2 MiB | 4.1 MiB |
| `python3 -c "..."` | 454-645 ms | 92 MiB | 40 MiB (its initial size) |
| `python3 /demo/stats.py` first / later | 4555 / 956 ms | 93 MiB | 40 MiB |
| `qjs /demo/hello.js` | 158 ms | 3 MiB | 2.4 MiB |

Disk (`make wasm-apps` on an empty 512 MiB image): `/bin` 53 MB in 11 files, `/lib/python3.15` 13 MB in 594
files, `/demo` 3 KB; 104528 KiB used after all the test runs (with `__pycache__`, the test databases and notes).

Low heap: a kernel built with `RINGZERO_HEAP_MIB=100` first panicked on `!run python3` (one allocation of the load
failed with 96 MiB free). With the 4x check the same build answers `/bin/python3.wasm: loading it needs about 135
MiB of kernel heap, 99 MiB is free (100 MiB in all, RINGZERO_HEAP_MIB)` and the next `!run wc -l` runs normally.
The in-run flag (`stopped: the kernel heap ran low`) was not reached by any of these programs: their heap use
stops growing once loaded.

## What does not work, and why

- **Shell pipes**: there are none (`!run` runs one program; `< FILE` is the only redirection). A pipeline is two
  runs and a file, or one python3/lua script.
- **The agent and command lines**: see above. For sqlite3, wc, jq and seq the 2B picks the right program and
  often a wrong command line; for python3 and lua it uses MicroPython's `run_code` instead. The tool round limit
  (4) ends the retries. The 4B (M7) is the next thing to try here.
- **qjs REPL**: QuickJS-ng has its own line editor and expects a raw terminal; with our line-mode fd 0 every
  character shows twice and Ctrl-D does not end it (`.quit` and Ctrl-C do). A per-program raw stdin would fix it.
- **kibi on the framebuffer**: the QEMU window has no ANSI support (escape sequences are skipped now, not drawn),
  and nobody answers the `ESC [6n` size query there, so kibi would wait for the answer (not tried). Serial
  console only.
- **coreutils `du`, `df`, `chown`, ...**: not in uutils' `feat_wasm` set.
- **CPython**: this build has no `zlib` (so `gzip` and compressed `zipfile` fail), no `_sqlite3` (the `sqlite3`
  module; the CLI is there) and no `termios`. `subprocess`, `socket` and `threading` import, but nothing can be
  spawned, connected or started (`RuntimeError: can't start new thread`): WASI preview1 has no processes or
  threads, and our runtime no sockets. The first import of each stdlib module compiles it (stats.py: 4.6 s the
  first time, 1 s after).
- **Speed**: everything runs on wasmi's interpreter, about 10x slower than native (M5b); CPython start-up is
  0.45 s, a small script 1 s.
- **Pinned but not reproducible bit for bit**: the cargo builds (coreutils, ripgrep, jaq, kibi) use `--locked`
  and crates.io's checksums, the C builds depend on wasi-sdk 34 and Binaryen 133 (pinned by SHA-256); the output
  files are not hashed.

## Risks for the beta

- The tool output cap uses the tokenizer of the loaded model: with another model (4B) the 1500 tokens are its
  tokens, which is what we want, but the cap is not re-tuned for it.
- The stop-on-low-heap flag covers the heap growing during a run. A single allocation larger than the free heap
  still panics: that is what a 100 MiB heap did on CPython's load before the 4x check. The check (4x the file +
  12 MiB free) is measured on these programs, not proven: a module whose translated code grows far beyond 4x its
  size could still get there.
- Typed input during a run goes to the program: a REPL left open blocks the shell, as on any system; Ctrl-C ends
  it. A program that does not read stdin leaves typed bytes in the type-ahead for the shell.
- `make wasm-apps` downloads about 330 MB (wasi-sdk, Binaryen, CPython) the first time.
- CPython comes from a core developer's release builds (brettcannon/cpython-wasi-build), not from python.org.
