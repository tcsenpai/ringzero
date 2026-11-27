# ringzero demos: common software in /bin

Every program below is a `wasm32-wasip1` build that runs on ringzero's WASI runtime (wasmi in ring 0, see
[M5B_WASI_RESULT.md](M5B_WASI_RESULT.md) and [M8_RESULT.md](M8_RESULT.md)). `make wasm-apps` builds or fetches them
(pinned versions and SHA-256, see the dependency table in INFO.md) and copies them to `/bin` on the data disk,
with CPython's standard library in `/lib/python3.15` and the demo inputs below in `/demo` (`tools/demo/`).

How to type them: each block is one or more lines for the ringzero prompt (serial console: `nc HOST 4555`, or the
QEMU window). `!run NAME ARGS` runs `/bin/NAME.wasm` (a coreutils tool when there is no such file); `"..."` keeps
spaces in one argument, `< FILE` is stdin. Programs started from the shell have the console as their terminal:
REPLs read typed lines (Enter sends a line, Ctrl-D ends the input, Ctrl-C stops the program). The agent runs the
same programs with its `run_wasm` tool, without a terminal (stdin is empty) and with a 10 s budget.

Some blocks use a note file and a JSON file; create them first:

```
!write /notes/home.txt buy milk\npay the electricity bill\ncall the plumber about the bathroom\ntake the dog out\nwash the car\nfix the kitchen door\nwater the plants\n
!run cp /demo/data.json /data.json
```

## coreutils (uutils 0.13.0)

```
!run ls -l /bin
!run wc -l /notes/home.txt
!run sort -r /notes/home.txt
!run head -n 3 /demo/people.csv
!run cut -d , -f 1,2 /demo/people.csv
!run sha256sum /demo/people.csv
!run ls /lib/python3.15/json
```

There are no pipes in the shell: one program per command, `< FILE` for stdin (`!run sort < /notes/home.txt`).
`/bin/coreutils.wasm` is the uutils `feat_wasm` set: `du`, `df`, `chown` and the other tools that need missing
WASI calls are not in it (`coreutils: unknown program 'du'`).

## ripgrep (15.2.0)

```
!run rg -n milk /notes
!run rg -i -c roma /demo
!run rg -l "def |import " /demo
```

## jq (jaq 3.2.0, a jq clone)

```
!run jq .version /demo/data.json
!run jq ".models[] | .name" /demo/data.json
!run jq "[.models[].size_mb] | add" /demo/data.json
!run jq -r ".apps[]" /demo/data.json
```

`/bin/jq.wasm` is jaq: the jq language with a few differences (see jaq's README), colors on a terminal.

## sqlite3 (3.54.1)

A database file on the disk, one command per run:

```
!run sqlite3 /demo/shop.db "create table if not exists items(name text, price real); insert into items values('bread', 2.5), ('milk', 1.2), ('coffee', 4.8);"
!run sqlite3 /demo/shop.db "select name, price from items order by price desc;"
!run sqlite3 -json /demo/shop.db "select count(*) as n, round(avg(price), 2) as avg from items;"
```

Interactive (the prompt is `SQLite-3.54 shop.db->`):

```
!run sqlite3 /demo/shop.db
select * from items where price > 2;
.tables
.quit
```

## Lua (5.4.8)

```
!run lua /demo/hello.lua
!run lua -e "print(_VERSION, math.maxinteger, ('%.3f'):format(math.pi))"
```

REPL:

```
!run lua
t = {} for i = 1, 5 do t[i] = i * i end
print(table.concat(t, " "))
print(pcall(error, "caught"))
```

then Ctrl-D.

## Python (CPython 3.15.2)

```
!run python3 -c "import sys; print(sys.version)"
!run python3 /demo/stats.py
!run python3 -c "import json, re; print(json.dumps(re.findall(r'\d+', 'a1b22c333')))"
```

`stats.py` reads `/demo/people.csv` with `csv`, uses `re`, `statistics`, `collections` and `json`, and writes
`/demo/report.json`. REPL (the basic one: the new REPL needs termios):

```
!run python3
import datetime, textwrap
print(datetime.date(2026, 11, 26).strftime("%A"))
[x * x for x in range(10) if x % 2]
```

then Ctrl-D. The first run of a module compiles it and writes `__pycache__` next to it on the disk; later runs
load the `.pyc`.

MicroPython (in the kernel, M5) runs small pure-Python scripts too, with fewer modules: `!py -f /demo/stats.py` does
not run there (no `csv`, `statistics`), `!py print(sum(x * x for x in range(10)))` does.

## QuickJS-ng (0.18.0)

```
!run qjs /demo/hello.js
!run qjs -e "console.log([3, 1, 2].sort().map((x) => x * 2))"
```

`!run qjs` alone is its REPL (`.quit` or Ctrl-C leaves it). It has its own line editor, so each typed character
shows twice (ours and its echo); Ctrl-D does not end it.

## kibi (0.3.2, a small terminal editor)

```
!run kibi /notes/home.txt
```

Ctrl-S saves, Ctrl-Q quits, Ctrl-F finds. kibi saves only into a directory that exists (`/notes` exists after the
setup block above). It needs an ANSI terminal: use it from the serial console (`nc`, a
terminal emulator), not from the QEMU window (the framebuffer console does not interpret escape sequences).

## The agent

The agent picks the program with `run_wasm`. Requests checked on the 4B q4 beta (`ringzero-beta-4b-q4.iso`, a
fresh disk after the setup block above, greedy, one per conversation: `/clear` between them; 2026-11-27):

```
use sqlite3 to create /shop2.db with a table products (name, price) holding bread 2.5, milk 1.2, coffee 4.8, and tell me the most expensive product
```

```
which tables are in the sqlite database /demo/shop.db?
```

```
count the lines of /notes/home.txt with wc
```

```
use run_wasm to run seq 1 3000
```

The last one shows the output cap: about 1500 tokens (and 6 KiB) are kept, with a note on what was dropped.
`how many lines does /notes/home.txt have?` (through `wc`) and `extract the version field from /data.json` (through
`cat`) get the right answer too, and so do:

```
run the script /demo/stats.py with python3
```
```
run /demo/hello.lua with lua
```
```
extract the version field from /data.json with jq
```

`create a sqlite db /shopping.db with a table products ... and tell me the most expensive product` (without "use
sqlite3") sent the 4B to MicroPython's `run_code` first, which has no `sqlite3` module, and it did not recover
within the 4 tool rounds. After a tool result the answer is sometimes in Italian even for an English request (3 of
these 10). The 2B results, with the Italian wordings of the M8 run, are in [M8_RESULT.md](M8_RESULT.md).
