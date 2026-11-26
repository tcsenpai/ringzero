# M5 code execution: MicroPython freestanding in ring 0

Result: MicroPython 1.29.0 runs inside the ringzero kernel, with no OS under it. `!py CODE` and `!py -f PATH` at
the shell and the agent's `run_code` tool. Each run is a fresh interpreter. Exceptions print a traceback and the
kernel goes on. The time budget and Ctrl-C stop runaway code, including code that catches `KeyboardInterrupt`.
The golden self-test passes and decode speed is unchanged. The cost is 9 MiB of arena and 63 tokens of boot prefix.

## What was built

- **Source**: MicroPython is fetched at build time, not vendored. `make upy` clones tag `v1.29.0`, checks that
  HEAD is commit `0fd6c573ea815774668bbb16b8e197c8822368b2`, and runs `ports/embed` (`kernel/upy/micropython_embed.mk`)
  with our `kernel/upy/mpconfigport.h`. That generates the self-contained package (`py/`, `genhdr/`,
  `shared/runtime/gchelper*`) into `build/upy/micropython_embed`. `kernel/build.rs` compiles the package,
  `lib/libm_dbl` (musl libm, MIT) and our glue with `cc`, archives the result as `libupy.a` and links it into the
  kernel. License: MIT (README dependency table).
- **Freestanding flags**: `-ffreestanding -nostdinc` (gcc's own `stddef/stdint/stdarg/limits/float`, plus
  `kernel/upy/include/`: tiny `string.h math.h stdlib.h assert.h errno.h stdio.h unistd.h`), `-mcmodel=kernel`
  (the image is at `0xffffffff80000000`), `-mno-red-zone`, `-fno-pic -fno-stack-protector -fno-math-errno -O2`.
  SSE2 stays on for doubles. The kernel enables SSE/AVX at boot. Rust is soft-float, so no float ever crosses the
  FFI.
- **libc**: `memcpy memmove memset memcmp` come from Rust's `compiler_builtins`. `kernel/upy/libc.c` has
  `strlen strcmp strncmp strchr memchr fabs nan abort errno`. NLR is MicroPython's own `nlrx64.c` (asm, no
  setjmp). GC register scan is `gchelper_generic.c` (x86-64 asm).
- **Config** (`mpconfigport.h`): ROM level EXTRA (f-strings, set ops, ...), double floats, mpz big ints, `math`,
  `cmath`, detailed errors with line numbers, stack check. Off: external import, `input()`, `io`, sys stdio
  objects, threads, `.mpy` loading, finalisers, `uctypes`. Modules present: builtins, `sys` (platform
  `ringzero`), `gc`, `micropython`, `math`, `cmath`, `array`, `collections`, `struct`, `errno`.
- **Memory**: `upy::init` (main.rs, after the kernel heap, before the models and the snapshot pool) takes the GC
  heap (8 MiB, `RINGZERO_UPY_HEAP_MIB` at build time, clamped to 1..256) and a 1 MiB stack from the engine arena.
- **Glue** (`kernel/upy/glue.c`): `upy_exec` switches to the private stack (`upy_on_stack`, 10 lines of asm),
  then runs `gc_init` + `mp_init` + lexer/parse/compile/call + `mp_deinit`. A one-line source is parsed as REPL
  input (`!py 17*23` prints `391`), longer code as a file. `mp_stack_set_limit` = stack - 32 KiB, so deep
  recursion is a `RuntimeError`. Uncaught exceptions are printed by our `print_exc`: the same format as
  `mp_obj_print_exception`, but cut to the first and last 5 frames.
- **Budget and Ctrl-C**: `MICROPY_VM_HOOK_LOOP/RETURN` call `upy_poll` every 256 backward jumps or returns, and
  that calls Rust's `upy_check`. `upy_check` reads the TSC, and polls COM1 + PS/2 for Ctrl-C every 10 ms (a port
  read is a VM exit). On Ctrl-C or past the deadline it returns 1, and the glue calls `mp_sched_keyboard_interrupt`
  (a normal `KeyboardInterrupt` with traceback). If the program is still running 1 s later (it caught the
  exception), or on a second Ctrl-C, it returns 2. The glue then `__builtin_longjmp`s back to the run's frame,
  resets `nlr_top` and tears the interpreter down. The next run starts fresh. Budgets: 10 s at the shell, 5 s for
  `run_code`.
- **Output**: `mp_hal_stdout_tx_strn*` call `upy_out`. At the shell it goes to the console. For the agent it goes
  into a 6 KiB buffer: never cut inside a UTF-8 sequence, dropped bytes are counted.
- **Hooks into existing files**: `main.rs` (`mod upy;`, `upy::init()`), `shell.rs` (one `CMDS` row, one match
  arm), `agent.rs` (one tool JSON line, one `SPECS` row, one `exec` arm; arrays 14 -> 15), `build.rs` (`upy()`),
  `Makefile` (`upy` target, `kernel: upy`).

Agent tool (one line in the prefix):

```
{"name": "run_code", "description": "Run Python code (MicroPython) and return what it prints.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}
```

The tool result is the captured output, then the traceback if there is one, then
`[python: ok in 0 ms (budget 5000 ms), GC heap 2 KiB used of 8 MiB]`.

## Verification (Fedora, QEMU/KVM q35, 6 GiB, 8 vCPUs; two other agents' 6 GiB guests were running)

Transcripts: two boots of the M5 ISO (run1: Python checks, run2: traceback cut, agent, bench) and one boot of the
baseline ISO (HEAD `0d0abf6` without M5, same box, right after). All steps were typed over serial.

| Check | Result |
|---|---|
| `!py print(2+2)` | `4`, `[py] ok in 13 ms (budget 10000 ms), GC heap 0 KiB used of 8 MiB` |
| `!py 17*23` (REPL-style echo) | `391` |
| `!py -f /test.py` (from the FAT32 disk: fib, 2**100, math, f-string, sys, list comprehension) | `fib(90) = 2880067194370816120`, `2**100 = 1267650600228229401496703205376`, `sqrt(2) = 1.4142135623730951 pi = 3.141592653589793 sin(pi/6) = 0.5`, `  f-string ok [1, 2, 3] ringzero micropython`, the primes < 60; ok in 67 ms, heap 16 KiB |
| `!py -f /err.py` (ValueError 3 calls deep) | `before the error`, `Traceback (most recent call last):` with 4 `File "/err.py", line N, in ...` lines, `ValueError: invalid syntax for integer with base 10: 'three'`; `[py] error`; the shell goes on |
| `!py 1/0` | `ZeroDivisionError: divide by zero`, kernel goes on |
| `!py -f /rec.py` (unbounded recursion) | `RuntimeError: maximum recursion depth exceeded` after 2539 frames; traceback cut to 5 + `[... 2529 frames not shown ...]` + 5, 172 ms (35.6 s before the cut: the full traceback went out over serial from C, where the budget cannot reach) |
| `!py x = [0] * (10**8)` | `MemoryError: memory allocation failed, allocating 800000000 bytes` |
| `!py while True: pass` | `KeyboardInterrupt`, `[py] stopped: time budget exceeded in 10043 ms` |
| `!py while True: pass` + Ctrl-C after 2 s | `KeyboardInterrupt`, `[py] interrupted (Ctrl-C) in 2032 ms` |
| `!py -f /evil.py` (catches `KeyboardInterrupt` in a loop) | 70 lines `caught KeyboardInterrupt, going on ...`, then `[py] aborted: time budget exceeded and KeyboardInterrupt was ignored in 11002 ms`; the next `!py print(1)` prints `1` (fresh interpreter after the hard abort) |
| agent: `quanto fa 17*23? usa python` | `[tool] run_code(code=17*23)` -> `391` + status line; answer `391` (3.2 s turn) |
| agent: `usa python per calcolare la somma dei primi 100 numeri primi` | the model wrote an 18-line `is_prime`/`sum_primes` program; `run_code` -> `1060` (it summed the primes <= 100, the model's reading of the question); answer `La somma dei primi 100 numeri primi è 1060.` |
| golden self-test | `SELFTEST PASS (prompt ids match, generated ids match vs golden 9419,1017,0 + <|im_end|>)` on both M5 boots |
| other boot self-tests | engine M0, memory index, file indexer, tool-call parser: PASS |

Speed and memory, baseline (no M5) vs M5, same box, same hour:

| | baseline `0d0abf6` | M5 |
|---|---|---|
| boot bench, 64-token decode | 35.13 tok/s | 33.41 tok/s (run2), 29.51 tok/s (run1, more host load) |
| `/bench` at the prompt, 64-token decode | 36.80 tok/s | 36.30 tok/s |
| agent boot prefix | 1140 tok, prefilled in 24.08 s | 1203 tok (+63 for `run_code`), 25.88 s |
| `!mem` "LLM engine arena" | 520 MiB | 529 MiB (+8 MiB heap + 1 MiB stack) |
| `!mem` free | 2022 MiB | 2013 MiB |
| kernel `.text` | 480 895 B | 766 975 B (+286 KB MicroPython + libm + glue) |
| `!py` fresh interpreter + `print(2+2)` | | 13 ms at the shell (most of it serial output); `run_code` reports 0 ms |

Decode is within the host noise. The engine code did not change, and Python does not run during a forward pass.

## Limits and risks

- **Ring 0**: Python code runs in ring 0, on the BSP, in the kernel's address space. MicroPython has no way to
  reach raw memory with this config (`uctypes`, `machine`, native emitters and inline asm are off; no `.mpy`
  loading). But a bug in the interpreter (a C memory-safety bug triggered by crafted Python) can corrupt
  anything: model weights, page tables, the disk cache. It cannot be contained, because there is no user mode
  and no separate address space. The agent can run any code the model writes. Today that is only computation
  (no file or network modules), but every module added later widens this. Moving the interpreter to ring 3 with
  its own page tables is the real fix.
- **The budget only acts at VM hook points.** A single long C operation, such as `10**(10**7)` (the multiply is
  bounded by the 8 MiB heap, but `str()` of a huge int is quadratic) or a giant `sorted()`, does not pass through
  the hook and can hold the CPU past the budget. Ctrl-C is not seen either until it returns. A timer-interrupt
  watchdog that redirects to the abort path would close this gap.
- After a hard abort, every C function that the longjmp skipped just never returns. That is safe here because
  the interpreter state is thrown away (`gc_init`/`mp_init` on the next run) and MicroPython holds no kernel
  locks. Any future hook that takes a kernel lock from Python must not be aborted this way.
- Ctrl-C during `run_code` stops the Python code. The agent turn goes on with the `interrupted` result (a second
  Ctrl-C stops the turn).
- No `import` of files from the disk yet (`mp_lexer_new_from_file` raises `OSError`). Read-only FS access, `json`,
  `re` and `random` (extmod) are left for later.
