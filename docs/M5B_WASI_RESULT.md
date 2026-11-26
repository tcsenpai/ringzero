# M5b WASI runtime: wasm32-wasip1 programs in ring 0

Result: programs compiled to `wasm32-wasip1` run on ringzero from the FAT32 disk, on wasmi 2.0.3 (an interpreter
linked into the kernel) with WASI preview1 host functions. Commands are `!wasm PATH|NAME [ARGS] [< FILE]` (and
`!run`) at the shell, and the agent's `run_wasm` tool. Our Rust test app passes 20 of 20 checks. Three real
programs run: uutils coreutils (multicall), ripgrep and QuickJS-ng. A `..` escape is refused. A runaway loop
is stopped by the time budget and by Ctrl-C, and the memory cap holds at 256 MiB. The golden self-test passes
and decode speed is unchanged. The costs are 309 MiB of RAM, 1.87 MB of kernel text and 96 tokens of boot
prefix. The CPU bench runs about 10x slower than native.

## Runtime choice

wasmi 2.0.3 (MIT OR Apache-2.0, used under MIT): pure Rust, builds `no_std` + `alloc` for `x86_64-unknown-none`,
validates modules and translates functions lazily to a register bytecode. Three features make it fit a kernel
without OS threads or signals:

- **fuel + resumable calls**: `call_resumable` returns `OutOfFuel` at a slice boundary and `resume` goes on, so
  the kernel checks the budget and Ctrl-C between slices of 10 M fuel (5 to 10 ms) and simply drops the store to
  stop. No longjmp and no timer interrupt are needed.
- **StoreLimits** for the memory cap: `memory.grow` past the cap returns -1, as the program expects.
- the host-function API (`Linker::func_wrap`, `Memory::data_and_store_mut`) and `Error::i32_exit` for `proc_exit`.

Not taken: wasm3 (C, in maintenance mode, its WASI layer wants a libc/POSIX underneath), WAMR (C, needs a platform
port), and wasmtime with the Pulley interpreter (no_std is newer and much bigger; still the later option the
milestone task names). `wasmi_wasi` is std-only (wasi-common), so the WASI layer is ours (`kernel/src/wasm.rs`,
about 900 lines).

Two wasmi settings were needed on this target:

| Problem | Cause | Fix |
|---|---|---|
| `!wasm rztest bench` -> `#GP` at a `ret` in `wasmi::...::i64_mul_rri` | the default `auto-dispatch` picks wasmi's tail-call dispatch for x86-64, but on `x86_64-unknown-none` the handlers pass arguments on the stack, so LLVM emits `call` instead of a sibling call: one native frame per executed wasm instruction, until the 1 MiB stack overflows into the arena | feature `portable-dispatch` (a loop) |
| `!wasm rg ...` -> `trap: ran out of fuel while calling a resumable function: required_fuel=16779` | lazy translation charges fuel (7 + 2 per byte), and running out during a translation comes back as an error, not as a resumable `OutOfFuel` | `CustomFuelCosts` with 0 fuel for translation and validation: only executed code pays |
| coreutils: `not a valid wasm module: unexpected SIMD opcode: 0xfd` | the uutils build has simd128 code | feature `simd` (+0.6 MB of text) |

## What was built

- **`kernel/src/wasm.rs`**: `!wasm`/`!run` (`shell_cmd`), the `run_wasm` tool, the command-line splitter
  (spaces, `"..."`, `< FILE`; boot self-test), and one `run`: read the `.wasm` file (up to 24 MiB and a third of
  the free heap), a fresh `Engine` + `Module` + `Store` + `Linker`, instantiate, `_start` in fuel slices. On
  `OutOfFuel` it checks Ctrl-C (COM1 + PS/2, polled every 10 ms) and the deadline, then refills
  `max(10 M, required_fuel)` and resumes. The run is on MicroPython's 1 MiB stack (`upy::on_stack`), because the
  boot stack is 64 KiB. The status line gives the exit code or trap, time, fuel used, final memory size and the cap.
- **Program lookup**: a path is used as given. A bare NAME is `/bin/NAME`, then `/bin/NAME.wasm`, then
  `/bin/coreutils.wasm` with argv[0] = NAME. uutils is a multicall binary, so `!run ls` and the agent's
  `run_wasm(path=/bin/ls)` run coreutils' ls, like the links in a Unix /bin. argv[0] is the file stem
  (`coreutils`), env is `HOME=/ PWD=/ PATH=/bin TERM=dumb`.
- **WASI preview1** (all `wasi_snapshot_preview1`, no preview2):

| Group | Calls | Notes |
|---|---|---|
| process | `args_*`, `environ_*`, `proc_exit` (`Error::i32_exit`), `sched_yield` | |
| time | `clock_time_get`, `clock_res_get` (1 us) | realtime = RTC at the start of the run + TSC; monotonic = TSC since boot |
| random | `random_get` | RDRAND (10 tries per 8 bytes), else xorshift64* seeded from the TSC (logged at boot which one) |
| stdio | `fd_write`/`fd_read` on 0/1/2 | stdout + stderr -> console, or a 6 KiB buffer for the agent (never cut inside UTF-8, dropped bytes counted); stdin = the `<` file (read-only), else always EOF |
| files | `path_open` (CREAT, EXCL, TRUNC, DIRECTORY, APPEND; write = right FD_WRITE), `fd_read/write/pread/pwrite/seek/tell/close/renumber`, `fd_fdstat_get/set_flags`, `fd_filestat_get/set_size/set_times`, `fd_sync/datasync/advise`, `path_filestat_get/set_times` | an open file is (path, position): every call reopens it through fs.rs (`read_into`, `write_at`), so `!sync` and remounts never leave a stale handle. FAT has one mtime, so set_times is accepted and ignored |
| dirs | `fd_prestat_get`/`fd_prestat_dir_name` (fd 3 = `/`), `fd_readdir` (cookie = entry index), `path_create_directory`, `path_remove_directory`, `path_unlink_file`, `path_rename` (replaces the target, as POSIX does), `path_readlink` (EINVAL: no links on FAT) | |
| events | `poll_oneoff` | clock subscriptions sleep (with the budget and Ctrl-C checked), fd subscriptions are ready at once |
| the rest | `sock_*`, `path_symlink`, `path_link`, `fd_allocate`, `proc_raise`, ... | any other `wasi_snapshot_preview1` import gets a stub that returns ENOSYS (52) and logs `wasm: NAME is not implemented (ENOSYS)` once per run. Imports from other modules fail instantiation with wasmi's message |

- **Sandbox**: paths resolve against the directory fd. `.` and `..` are folded, and a `..` above `/` returns
  ENOTCAPABLE (76) with a log line. Absolute paths in `path_open` are refused: wasi-libc and Rust std already
  strip the `/` against the preopen. Writes, creates, unlinks, renames and mkdirs under `/sys` (agentd's stores,
  the boot snapshot, the hibernation image) return EACCES. The same `fs::check` as the shell applies (255 B, no
  control characters).
- **Memory** (`mem.rs`): the kernel heap went from 16 to 64 MiB (module bytes, wasmi's module and translated code,
  the store). The wasm slot (256 MiB, `RINGZERO_WASM_MIB` at build time) is taken from the arena at boot. The global
  allocator is now a small wrapper around `linked_list_allocator`. While wasm.rs instantiates a module, the first
  allocation of whole 64 KiB pages (the memory's initial size) takes the slot, and its reallocs stay in place.
  wasmi grows a memory by doubling a `Vec<u8>`. In the heap, a 128 -> 256 MiB step would need both buffers at
  once, and from 1088 KiB the doubling steps would stop at 136 MiB (the next step asks for 272 MiB).
- **fs.rs**: `entry` (is_dir, size, mtime, also for directories; `meta` uses it now), `list`, `touch`, `write_at`
  (gap past the end zero-filled), `rename`.
- **Hooks into existing files**: `main.rs` (`mod wasm;`, `wasm::init()` + splitter self-test after `upy::init`),
  `shell.rs` (two `CMDS` rows, one match arm), `agent.rs` (one tool JSON line, one `SPECS` row, one `exec` arm,
  `FX` += `run_wasm` so a grammar-steered call does not run; arrays 17 -> 18), `upy.rs` (`on_stack`, its capture
  uses `SBuf::push_cut` shared with wasm.rs), `log.rs` (`push_cut`), `Makefile` (`wasm-apps`),
  `tools/boottest.py` (`"CMD ^C3"` sends Ctrl-C 3 s after CMD).
- **Test programs** (`make wasm-apps`, `tools/wasm_apps.sh`): `rztest` (ours, `tools/wasm/rztest`, 131 KB),
  uutils coreutils 0.13.0 (`cargo install --target wasm32-wasip1 --features feat_wasm`, 9.9 MB), ripgrep 15.2.0
  (3.3 MB), QuickJS-ng v0.18.0 (the release's `qjs-wasi.wasm`, 1.6 MB). Release profile with strip, no debug info.
  They are copied to `/bin` with `mmd`/`mcopy -o`: nothing is formatted, and the target refuses while a guest has
  the image open.

Agent tool (one line in the prefix, 1315 -> 1411 tokens, +96):

```
{"name": "run_wasm", "description": "Run a WASI program from the disk (in /bin, e.g. coreutils) and return what it prints.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "args": {"type": "string", "description": "command-line arguments, space separated"}}, "required": ["path"]}}
```

## Verification (Fedora, QEMU/KVM q35, 6 GiB, 8 vCPUs, BIOS; the M7 agent's 6 GiB guest was running)

All steps were typed over serial by `tools/boottest.py` on a test disk (`build/disk-m5b.img`: `make wasm-apps` +
`/notes.txt`, `/hello.js`, `/src/{main.rs,readme.txt}`). The final run is `build/m5b-final-timed.log` on the box.

| Check | Result |
|---|---|
| golden self-test | `SELFTEST PASS (prompt ids match, generated ids match vs golden 9419,1017,0 + <\|im_end\|>)` on every boot of the M5b ISO |
| other boot self-tests | `wasm: command-line splitter self-test PASS`; engine M0, memory index, file indexer, agentd, tool-call parser, grammar: PASS |
| `!wasm rztest check A "B C"` | `rztest: PASS 20/20`: args, env, mkdir, write, read back, append + metadata (29 B), seek + read, relative path (cwd /), read_dir, metadata is_dir, rename, remove_file, remove_dir, missing file -> NotFound, `/sys is read-only: Some(PermissionDenied)`, sandbox, clock realtime (`1795723916 s since the epoch`), monotonic + sleep 50 ms (`50 ms`), random_get, ENOSYS. stderr line on the console. `[wasm] exit 0 in 569 ms` |
| sandbox (`rztest escape`, also inside `check`) | `open "../x"`, `"/../x"`, `"a/../../x"`, `"../../etc/passwd"`: `refused (Capabilities insufficient (os error 76))`; `write "../escape.txt": refused`; kernel log `wasm: sandbox: '../x' climbs above the preopened /, refused` |
| unimplemented call | `wasm: sock_accept is not implemented (ENOSYS)` logged once for two calls, both return 52 |
| stdin from a file | `!wasm rztest stdin < /notes.txt` -> `stdin: 47 bytes, 4 lines` |
| `!wasm rztest loop` (budget) | `[wasm] stopped: time budget exceeded in 60002 ms (budget 60000 ms)` |
| `!wasm rztest loop` + Ctrl-C after 3 s | `[wasm] interrupted (Ctrl-C) in 2981 ms`; the next command runs normally |
| memory cap | `!wasm rztest mem 250` -> `250 MiB allocated and touched`, memory 257088 KiB; `rztest mem 300` -> `mem: allocation failed after 254 MiB` (exit 3, memory 261184 KiB of the 256 MiB cap). Before the slot fix the same test stopped at 134 MiB (doubling step past the slot) |
| `!wasm rztest sleep 200` | `slept 200 ms (asked 200)` (poll_oneoff) |
| agent: `use run_wasm to run coreutils ls -l /bin and tell me which file is the biggest` | `[tool] run_wasm(path=/bin/ls, args=-l /bin)` -> the listing (multicall fallback) -> `Il file più grande è coreutils.wasm con 9936108 byte.` |
| agent: `conta le righe di /notes.txt con il programma wc di coreutils` | `run_wasm(path=/bin/wc, args=-l /notes.txt)` -> `4 /notes.txt` -> `Il file /notes.txt contiene 4 righe.` |
| agent: `usa run_wasm per eseguire seq 1 3000` | the output cut at 6 KiB + `[... 7749 more bytes of output dropped]`, then the status line |
| MicroPython after the shared changes | `!py print(2+2)` -> `4`; agent `run_code(code=17*23)` -> `391` |
| `!mem` | free 1694 MiB (baseline 2003), arena 792 MiB (536), kernel heap 64 MiB (16) |

Real programs:

| Program | Command | Result |
|---|---|---|
| uutils coreutils 0.13.0 | `!wasm coreutils ls -l /` | the root listing (`drwxrwxrwx 1 somebody somegroup 0 Nov 26 2026 bin`, ...), 126 ms with module load (9.9 MB) |
| | `!run cat /notes.txt`, `!wasm coreutils wc -l /notes.txt`, `!run coreutils sort -r /notes.txt`, `!run ls /bin`, `seq 1 3000` (agent) | correct output, 50-130 ms each |
| ripgrep 15.2.0 | `!wasm rg -n main /src` | `/src/main.rs:1:fn main() {`, 39 ms |
| | `!wasm rg -i todo /` | `/src/readme.txt:TODO: write docs` (walks the whole disk), 109 ms |
| QuickJS-ng v0.18.0 | `!wasm qjs /hello.js` (recursive fib, arrow functions, `JSON.stringify`, `Date.now`) | `hello from QuickJS on ringzero, fib(25) = 75025 in 208 ms`, `{"os":"ringzero","list":[1,4,9]}` |

What failed, and why:

- **coreutils** at first: SIMD opcodes, not accepted until wasmi's `simd` feature was on (now it runs).
- **ripgrep** at first: the lazy-translation fuel trap (fixed, see above).
- **agent + rg**: the model wrote `run_wasm(path=/bin/rg, args=-l /src -i TODO)`, so rg took `TODO` as a path
  (`IO error ... os error 44`) and the model answered that nothing was found. That is a model error, and rg's own
  error came back correctly.
- Not tried here (M8): CPython.wasm (needs its stdlib tree on the disk; its size against the 24 MiB module limit
  is not checked), sqlite3 and lua (C: they need wasi-sdk; lua's errors use setjmp/longjmp, which wasi-libc only
  offers through the exception-handling proposal, untested here), jq/jaq, editors (no tty).

Speed (`rztest bench`: sieve to 2 000 000 + 20 M rounds of an integer hash, same source natively on the host):

| | native (Fedora host) | ringzero wasm | ratio |
|---|---|---|---|
| earlier, busier host | 41-44 ms | 530-551 ms | 12.5-13x |
| later, quieter host (load average 5) | 24-31 ms | 267 ms | 9-11x |

wasmi runs about 1-1.8 G fuel per second here (the bench is 508 M fuel; a tight loop gets 108 G fuel in 60 s). Fuel
numbers are wasmi's charges, not instruction counts: QuickJS reports 51 G fuel for a 260 ms run, so something it
does is charged far above its cost. We did not look into which operation it is. It does not affect the budget,
which is measured in time.

Decode speed and boot, baseline ISO (`70ec408`, built before M5b) vs M5b, alternating boots on the same box and
disk:

| | baseline `70ec408` | M5b |
|---|---|---|
| boot bench, 64-token decode | 37.30, 34.56 tok/s | 34.84, 36.23 tok/s |
| `/bench`, 64-token decode (3 per boot, 2 boots) | 35.98 35.95 36.63 36.45 37.84 37.55 (mean 36.7) | 36.12 37.13 37.77 38.07 37.26 36.68 (mean 37.2) |
| agent boot prefix | 1315 tok, prefilled in 6.54 s | 1411 tok (+96 for `run_wasm`), 6.92 s |
| `!mem` free | 2003 MiB | 1694 MiB (-309: slot 256, heap +48, kernel image +5) |
| `!mem` LLM engine arena | 536 MiB | 792 MiB |
| kernel `.text` | 954 692 B | 2 820 660 B (+1.87 MB: wasmi with portable dispatch + simd, wasm.rs) |

Decode is within the host noise. The engine did not change, and wasm never runs during a forward pass.

## Limits and risks (for M8)

- **Ring 0, but contained by the interpreter**: wasm code cannot reach kernel memory. Every load and store is
  bounds-checked by wasmi, and the host functions check every guest pointer (EFAULT). The risk is a bug in wasmi
  or in our host functions, which run in ring 0 with the whole machine in reach. This is tighter than
  MicroPython, whose interpreter is C without bounds checks.
- **Kernel heap OOM halts the kernel**: wasmi allocates its module and translated code with infallible `alloc`.
  The module limit (a third of the free heap, at most 24 MiB) keeps the known programs well inside the 64 MiB, but
  a module whose translated code grows far beyond its size could still exhaust the heap. That panics, it does not
  trap. CPython-size modules need a bigger heap, or the module bytes moved to the arena.
- **One memory per instance in the slot**: a second memory, or a module with 0 initial pages, lives in the heap
  under the same cap but with the doubling problem (and the heap size). The slot's reallocs return the same
  pointer even when the doubling step asks for more than the slot. That is safe only because StoreLimits keeps the
  memory length within the slot, and only the linear memory is ever placed there (`arm_wasm_slot` around
  instantiate). The `shortcut:` comment in mem.rs marks this.
- **No time slicing inside host calls or translation**: a single huge `fd_read`/`fd_write` (capped at 1 MiB per
  read) or the lazy translation of a giant function runs to its end before the next budget check. Big files are
  read by reopening and seeking on every call: fine for tools, slow for a database doing many small random reads
  (sqlite).
- **stdin is a file or EOF**: no interactive stdin yet, so REPLs (qjs without a script, lua) and editors need a
  console line reader on fd 0 and some tty answers (`fd_fdstat_get` reports a character device, nothing more).
- **Output into the agent's context**: 6 KiB of dense output (3000 numbers) became 6218 tokens and took 36 s to
  prefill. M8 tools should keep their agent output short (`head`, `wc`, `rg -c`), or the cap should be in tokens.
- **No `fd_allocate`, links, sockets or threads**: rust std and wasi-libc tolerate the ENOSYS answers seen so far.
  Programs built with `wasm32-wasip1-threads`, or that need preview2 (components), do not run.
- **Speed**: about 10x native on integer code here; anything float-heavy pays the soft-float build of the kernel
  (wasmi's float ops go through compiler_builtins, not SSE).
