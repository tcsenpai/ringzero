# programs Specification

## Purpose
Baseline of how ringzero runs user programs in ring 0: WASI preview1 programs on the wasmi interpreter and MicroPython snippets, from the shell and from the agent's tools, with their budgets, limits, output caps and file access rules.

## Requirements

### Requirement: WASI shell commands
The system SHALL run a WASI program from the shell with `!wasm PATH|NAME [ARGS...] [< FILE]`, SHALL treat `!run` as the same command, and SHALL print the usage line `usage: !wasm PATH|NAME [ARGS...] [< FILE]  (NAME = /bin/NAME.wasm, else the coreutils tool NAME)` when no program is given. Output goes straight to the console and the run ends with one `[wasm] ...` status line.
Source: kernel/src/shell.rs CMDS, run; kernel/src/wasm.rs shell_cmd

#### Scenario: empty command prints usage
- **WHEN** the user types `!run` with no arguments
- **THEN** the shell prints the `usage: !wasm PATH|NAME ...` line and runs nothing

#### Scenario: status line after a run
- **WHEN** `!wasm /bin/rztest.wasm` finishes
- **THEN** the console shows a final line starting with `[wasm] ` that holds the outcome, the time in ms, the budget, the fuel in M, the memory in KiB, the cap in MiB and the kernel heap peak in MiB

### Requirement: Program name resolution
The system SHALL resolve a program word that has no `/` as `/bin/NAME`, then `/bin/NAME.wasm`, and otherwise as `/bin/coreutils.wasm` with argv[0] set to NAME (the multicall coreutils tool). A word that contains `/` is used as a path, outside /bin without a fallback. argv[0] is the base name without `.wasm`.
Source: kernel/src/wasm.rs status, program

#### Scenario: bare name found in /bin
- **WHEN** the program word is `rg` and `/bin/rg.wasm` exists
- **THEN** the system runs `/bin/rg.wasm` with argv[0] `rg`

#### Scenario: coreutils fallback
- **WHEN** the program word is `ls` and neither `/bin/ls` nor `/bin/ls.wasm` exists
- **THEN** the system runs `/bin/coreutils.wasm` with argv[0] `ls`

### Requirement: Command-line splitting and stdin redirect
The system SHALL split a command line on spaces and tabs, keep spaces inside `"..."` and `'...'`, honour `\"` and `\\` outside single quotes, and take `< FILE` or `<FILE` as stdin. It SHALL reject the line with `unterminated quote` or `'<' needs a file name`. A boot self-test of the splitter logs `wasm: command-line splitter self-test PASS` or `FAIL`.
Source: kernel/src/wasm.rs program, split, tok_cut, selftest; kernel/src/main.rs kmain

#### Scenario: redirect
- **WHEN** the command is `wc -l < /a.txt`
- **THEN** argv is `wc`, `-l` and stdin reads `/a.txt`

#### Scenario: unterminated quote
- **WHEN** the command is `a 'b`
- **THEN** the run is refused with `unterminated quote`

### Requirement: WASI time budget in fuel slices
The system SHALL give a shell WASI run 60000 ms and an agent `run_wasm` run 10000 ms, SHALL run the program in fuel slices of 10,000,000 checking the deadline between slices, and SHALL stop a run past its deadline with `stopped: time budget exceeded`.
Source: kernel/src/wasm.rs SHELL_MS, SLICE, out, stop, run, status

#### Scenario: runaway loop from the agent
- **WHEN** a `run_wasm` program loops forever
- **THEN** it is stopped after about 10000 ms and the tool result ends with `[stopped: time budget exceeded]`

#### Scenario: REPL waits for the user
- **WHEN** a shell run blocks reading a typed line for 90 s
- **THEN** the run is not stopped by the budget, because time waiting for typed input is added to the deadline (kernel/src/wasm.rs:319-327)

### Requirement: WASI interruption by Ctrl-C
The system SHALL poll for Ctrl-C at most every 10 ms during a WASI run, including inside `poll_oneoff` sleeps and console reads, and SHALL end the run with the status `interrupted (Ctrl-C)`.
Source: kernel/src/wasm.rs out, stop, tty_wait, poll_oneoff, status

#### Scenario: Ctrl-C during a long run
- **WHEN** the user presses Ctrl-C while `!run python3` is executing
- **THEN** the run stops and the status line starts with `interrupted (Ctrl-C)`

### Requirement: WASI memory cap
The system SHALL cap the linear memory of every WASI instance at `RINGZERO_WASM_MIB` MiB (build time, default 256, clamped to 16..2048), reserved from the arena at boot, and SHALL enforce it through the store limits. When the arena cannot provide the slot the WASI runtime SHALL be off and every run SHALL fail with `the WASI runtime is off (no arena memory at boot)`.
Source: kernel/src/wasm.rs init, run

#### Scenario: default cap
- **WHEN** the kernel is built without `RINGZERO_WASM_MIB`
- **THEN** the boot log says `memory cap 256 MiB per instance` and a `memory.grow` past 256 MiB fails inside the program

### Requirement: WASI module loading limits
The system SHALL refuse a module larger than 48 MiB, SHALL refuse to load when about 4 times the file size plus the kernel heap low mark is not free in the kernel heap, SHALL refuse a file that does not start with the `\0asm` magic with a message that lists the programs in /bin, and SHALL refuse a module without `_start` with `no _start: not a WASI command`.
Source: kernel/src/wasm.rs MOD_MAX, run

#### Scenario: not a wasm file
- **WHEN** the user runs `!wasm /notes.txt`
- **THEN** the error says `/notes.txt: not a wasm program (the programs are in /bin: ...)`

#### Scenario: module too large
- **WHEN** a .wasm file of 60 MiB is run
- **THEN** the run is refused with `larger than the 49152 KiB limit for a module`

### Requirement: WASI kernel heap guard
The system SHALL stop a running WASI program, instead of the kernel, when the kernel heap runs low, with the status `stopped: the kernel heap ran low (N MiB in all)`, and SHALL not start a program whose loading already left the heap almost full.
Source: kernel/src/wasm.rs Why, stop, status, run

#### Scenario: heap runs low mid-run
- **WHEN** the kernel heap low flag is raised during a run
- **THEN** the run stops at the next slice boundary with `stopped: the kernel heap ran low`

### Requirement: Supported WASI fd calls
The system SHALL implement these `wasi_snapshot_preview1` fd imports: fd_advise, fd_close, fd_datasync, fd_fdstat_get, fd_fdstat_set_flags, fd_filestat_get, fd_filestat_set_size, fd_filestat_set_times, fd_pread, fd_prestat_get, fd_prestat_dir_name, fd_pwrite, fd_read, fd_readdir, fd_renumber, fd_seek, fd_sync, fd_tell, fd_write. Setting file times is accepted and ignored.
Source: kernel/src/wasm.rs strings, link

#### Scenario: set times
- **WHEN** a program calls fd_filestat_set_times on an open file
- **THEN** the call returns 0 and the file's mtime is unchanged

### Requirement: Supported WASI path, process and other calls
The system SHALL implement args_get, args_sizes_get, environ_get, environ_sizes_get, clock_res_get, clock_time_get, path_create_directory, path_filestat_get, path_filestat_set_times, path_open, path_readlink, path_remove_directory, path_rename, path_unlink_file, poll_oneoff, proc_exit, random_get and sched_yield, and SHALL return ENOSYS (52) from every other import, logged once per run.
Source: kernel/src/wasm.rs strings, link

#### Scenario: symlinks
- **WHEN** a program calls path_readlink
- **THEN** it returns EINVAL (28): FAT has no symlinks

#### Scenario: unknown import
- **WHEN** a module imports `sock_accept`
- **THEN** the module links, the call returns 52 and the kernel log has one `sock_accept is not implemented (ENOSYS)` line

### Requirement: WASI file sandbox
The system SHALL give every WASI run one preopened directory `/` as fd 3, SHALL resolve paths relative to a directory fd, SHALL refuse absolute paths and any `..` that climbs above `/` with ENOTCAPABLE (76), and SHALL log a `sandbox: ... climbs above the preopened /, refused` warning for the latter.
Source: kernel/src/wasm.rs file, resolve, fd_prestat_get, fd_prestat_dir_name, run

#### Scenario: escape attempt
- **WHEN** a program opens `../../etc/passwd` from fd 3
- **THEN** path_open returns 76 and nothing is opened

### Requirement: Program stdin, terminal and environment
The system SHALL give a `run_wasm` run a stdin at EOF unless `< FILE` is given, and a shell run the console as a terminal: fd 0 reads typed lines, `/dev/tty` gives raw bytes, and isatty() is true. The environment SHALL be `HOME=/`, `PWD=/`, `PATH=/bin`, `TERM=dumb`, `PYTHON_BASIC_REPL=1`.
Source: kernel/src/wasm.rs drop_handles, tty_wait, fd_fdstat_get, path_open, run

#### Scenario: agent run reads stdin
- **WHEN** `run_wasm` runs `cat` with no `<`
- **THEN** the first read returns 0 bytes and the program exits

#### Scenario: editor in an agent run
- **WHEN** an agent run opens `/dev/tty`
- **THEN** path_open returns ENOENT (44)

### Requirement: Agent run_wasm output and status
The system SHALL capture a `run_wasm` call's stdout and stderr into at most 6144 bytes and keep at most 1500 tokens of it, SHALL then append `[... N more bytes of output dropped (the tool keeps about 1500 tokens): ...]`, and SHALL end the result with `[exit N]`, `[trap: MSG]`, or the stop reason without timings.
Source: kernel/src/wasm.rs TOK_CAP, tool, tok_cut

#### Scenario: long output
- **WHEN** `run_wasm` runs `seq 1 3000`
- **THEN** the result holds about 1500 tokens of numbers, the dropped-bytes marker and `[exit 0]`

#### Scenario: command in path
- **WHEN** the model calls `run_wasm` with path=`wc -l /a.txt` and no args
- **THEN** the system runs `wc` with arguments `-l /a.txt`, and a repeated program name in args is dropped (kernel/src/wasm.rs:1066-1080)

### Requirement: WASI exit and trap outcome
The system SHALL report `exit 0` when `_start` returns, `exit N` when the program calls proc_exit(N), and `trap: MSG` with wasmi's message for any other trap or host error; the interpreter state is dropped after every run and every run uses a fresh Engine and Store.
Source: kernel/src/wasm.rs Run, status, run, link

#### Scenario: proc_exit
- **WHEN** a program calls proc_exit(3)
- **THEN** the shell status line starts with `exit 3`

### Requirement: MicroPython shell command
The system SHALL run MicroPython with `!py CODE` or `!py -f PATH`, SHALL refuse a `-f` file larger than 256 KiB, not text, or not UTF-8 with `[py] PATH: ...`, SHALL print `usage: !py CODE | !py -f PATH` when no code is given, and SHALL run a single-line source in single-input mode (values echo). Every run is a fresh interpreter.
Source: kernel/src/shell.rs CMDS, run; kernel/src/upy.rs run, shell_cmd

#### Scenario: one-liner
- **WHEN** the user types `!py 1+1`
- **THEN** the console shows `2` and a `[py] ok in N ms (budget 10000 ms), GC heap ...` line

#### Scenario: big script file
- **WHEN** the user runs `!py -f /big.py` on a 300 KiB file
- **THEN** the shell prints `[py] /big.py: larger than 256 KiB` and runs nothing

### Requirement: MicroPython time budget and abort
The system SHALL give a `!py` run 10000 ms and a `run_code` run 5000 ms, SHALL raise KeyboardInterrupt when the budget passes or Ctrl-C is pressed (polled every 10 ms), and SHALL abort the run 1000 ms later if it still runs, or at once on a second Ctrl-C.
Source: kernel/src/upy.rs SHELL_MS, GRACE_MS, on_stack, upy_check, Run, status

#### Scenario: program swallows KeyboardInterrupt
- **WHEN** a `run_code` program catches KeyboardInterrupt in an endless loop
- **THEN** it is aborted about 1000 ms after the 5000 ms budget and the status says `aborted: time budget exceeded and KeyboardInterrupt was ignored`

#### Scenario: program stops on the interrupt
- **WHEN** a `!py` loop does not catch KeyboardInterrupt and runs past 10000 ms
- **THEN** the status says `stopped: time budget exceeded`

### Requirement: MicroPython memory
The system SHALL give MicroPython a fixed GC heap of `RINGZERO_UPY_HEAP_MIB` MiB (build time, default 8, clamped 1..256) and a private 1 MiB stack from the arena at boot, shared with the WASI runtime (never run at the same time), and SHALL answer `MicroPython is not available (no arena memory at boot)` when the arena could not provide them.
Source: kernel/src/upy.rs STACK, heap_mib, init, upy_on_stack, on_stack, run; kernel/build.rs upy

#### Scenario: default heap
- **WHEN** the kernel is built without `RINGZERO_UPY_HEAP_MIB`
- **THEN** the boot log says `GC heap 8 MiB + stack 1024 KiB from the arena`

### Requirement: Agent run_code output
The system SHALL capture the stdout and traceback of a `run_code` call into at most 6144 bytes, SHALL append `[... N more bytes of output dropped]` when output was cut, and SHALL end the result with `[python: STATUS]` where STATUS is the run status line. There is no token cap on `run_code` output beyond the byte cap.
Source: kernel/src/upy.rs upy_exec, OUT_CAP, shell_cmd, tool

#### Scenario: print loop
- **WHEN** `run_code` prints 10000 lines
- **THEN** the result holds the first 6144 bytes, the dropped-bytes marker and a `[python: ok in ...]` line

### Requirement: Program set on the data disk
`make wasm-apps` SHALL build or fetch (pinned by version and SHA-256) rztest, coreutils 0.13.0, rg 15.2.0, qjs v0.18.0, jq (jaq 3.2.0), kibi 0.3.2, sqlite3 3.54.1, lua 5.4.8 and python3 3.15.2, copy them into `/bin`, the stdlib into `/lib/python3.15` and tools/demo into `/demo`, and refuse while a guest has the disk open.
Source: tools/wasm_apps.sh fetch, crate, cc_init; Makefile target disk, target wasm-apps

#### Scenario: guest running
- **WHEN** `make wasm-apps` runs while QEMU has the disk open
- **THEN** it prints `refusing: a guest has ... open` and exits with an error

#### Scenario: checksum mismatch
- **WHEN** a downloaded archive does not match its pinned SHA-256
- **THEN** the file is deleted and the script stops with `checksum mismatch: URL`

### Requirement: ringzero data is read-only for programs
The system SHALL refuse with EACCES (2) every WASI write, create, truncate, mkdir, unlink, rmdir or rename whose path is under `/var/lib/ringzero` (case-insensitive), while reads there stay allowed; /proc, /sys (but /sys/power/state) and /dev/vda refuse changes with EROFS (69).
Source: kernel/src/wasm.rs data_ro, path_open, path_op, path_rename

#### Scenario: write to the boot cache
- **WHEN** a program opens `var/lib/ringzero/bootsnap.bin` for writing
- **THEN** path_open returns 2 and the file is unchanged

## Open questions
- OPEN: pure C loops inside MicroPython (a huge bigint power such as `10**10**8`, a giant str()) never reach the VM hook, so the budget and Ctrl-C may not stop them; stated only in a comment (kernel/src/upy.rs:4-7) and docs/M5_MICROPYTHON_RESULT.md, not confirmable from the code paths.
- docs/M5B_WASI_RESULT.md says rztest passes 20 of 20 checks and the CPU bench runs about 10x slower than native: measurements, not confirmable in code.
- docs/M5B_WASI_RESULT.md says one 10 M fuel slice takes 5 to 10 ms: the slice size is in code, its duration is not.
- poll_oneoff reports fd_read and fd_write subscriptions as always ready (kernel/src/wasm.rs:679), so a REPL that polls stdin does not block there; whether any shipped program relies on real readiness was not checked.
- Whether a given data disk holds all listed programs depends on `make wasm-apps` having run; the kernel only checks the file at run time.
- `run_code` and `run_wasm` are tools that change the machine (steered and declined calls), but a low-confidence call of either runs with a notice instead of a question (see the decoding spec, T54); no other confirmation applies to them.
