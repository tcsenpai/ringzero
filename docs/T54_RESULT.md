# T54: release fixes from the maintainer's manual test

The maintainer's manual test of the beta build (4B q4) showed three problems: the confidence check asked y/N on 7
of 8 calls (run_code and run_wasm margins of 0.56 to 0.70, split between `run` and `ls` or between `_code` and
`_w`), the agent spent its 4 tool rounds on "which drivers do we have here?" and never answered, and run_code tried
`import os` twice (MicroPython in ringzero has no os). T54 changes:

1. Below the 0.80 confidence line only the tools that write ask (write_file, mkdir, set_setting, remember);
   run_code and run_wasm run with the notice, like the read-only tools (docs/T28_RESULT.md, beta section).
2. 10 tool rounds per question (was 4), `!set rounds N` (1..20, saved by hibernation), `make iso ROUNDS=N`; after
   the last round one more generation with the `<tool_call>` token blocked in the sampler, so every question ends
   with an answer.
3. run_code's description and one system prompt sentence say it is a MicroPython sandbox without os, files,
   network or imports from disk, and point at ls, cat, write_file and run_wasm.
4. A read-only /proc generated at read time (cpuinfo, meminfo, uptime, version, devices, bus/pci/devices), served
   by fs.rs to the shell, the agent's ls and cat tools and WASI programs.

Setup: 4B q4 edition, QEMU/KVM q35, 8 GiB, 8 cores, a fresh copy of the beta data disk for each run, temperature 0,
memory slot 200 tokens. The boot prefix is now 1571 tokens (system prompt + 18 tools).

## /proc

```
ringzero[main]> !ls /proc
cpuinfo  1751 B
meminfo  180 B
uptime  7 B
version  104 B
devices  449 B
bus/
ringzero[main]> !cat /proc/meminfo
MemTotal:          8388072 kB
MemFree:           1623375 kB
MemAvailable:      1623375 kB
ModelWeights:      3911071 kB
SnapshotPool:      1048576 kB
KernelHeap:         196608 kB
ringzero[main]> !cat /proc/devices
Active drivers:
serial       16550 UART: COM1 at I/O port 0x3f8, the console
keyboard     PS/2 (i8042): layout it
framebuffer  Limine framebuffer: 1280x800 32 bpp, text console
block        virtio-blk: PCI 00:03.0, 512 MiB
filesystem   fatfs: Fat32 'RINGZERO', 510 MiB, used 166664 KiB, free 348 MiB (cluster 4 KiB) on virtio-blk, mounted at /
filesystem   proc: this directory, read-only, generated at read time

PCI devices: /proc/bus/pci/devices
ringzero[main]> !cat /proc/bus/pci/devices
00:00.0 8086:29c0 host bridge (class 0600), no driver
00:01.0 1234:1111 display controller (class 0300), no driver
00:02.0 8086:10d3 network controller (class 0200), no driver
00:03.0 1af4:1001 storage controller (class 0100), virtio-blk
00:1f.0 8086:2918 ISA bridge (class 0601), no driver
00:1f.2 8086:2922 SATA controller (class 0106), no driver
00:1f.3 8086:2930 SMBus controller (class 0c05), no driver
```

`/proc/cpuinfo` has one block per core (processor, model name, cpu MHz from the TSC, cpu cores, role, flags);
on the test host the flags line is `fpu tsc sse sse2 pni ssse3 fma sse4_1 sse4_2 popcnt avx f16c rdrand avx2 bmi2
avx512f`. `/proc/version`: `ringzero version 0.1.0 beta (ringzero Qwen35@4b.q4-Qwen3@0.6b (QweQwe)), Rust kernel,
x86_64, not Linux`.

| Check | Result |
|---|---|
| `!run cat /proc/meminfo`, `!run ls /proc`, `!run head -n 8 /proc/cpuinfo` | same content as `!cat` (exit 0, 55-146 ms) |
| `!run python3 -c "print(open('/proc/uptime').read())"` | `136.88` |
| python3 `open('/proc/uptime','w')` | `OSError: [Errno 69] Read-only file system: '/proc/uptime'` |
| `!write /proc/version x`, `!mkdir /proc/x`, `!rm /proc/uptime` | `[fs] write: read-only file system` (and mkdir, rm) |
| `!cat /proc/nope` | `[fs] cat: not found` |
| `!ls /` | lists `proc/` after the volume's directories |
| boot self-test | `proc: /proc path table self-test PASS` |

## Agent questions (one fresh conversation each)

| Question | Tool calls | Answer |
|---|---|---|
| which drivers do we have here? | `ls(/proc/devices)` (not a directory), `cat(/proc/devices)` | the driver list: serial, keyboard, framebuffer, block, two filesystems |
| how much memory is free? | `meminfo()` | "1585 MiB of memory is free." |
| what CPU flags does this machine have? | `cpuinfo()` | AVX2, FMA, F16C (the engine features the cpuinfo tool names) |
| quanti core ha la CPU e a che frequenza? | `cpuinfo()` | "La CPU ha 8 core e funziona a 1997 MHz." |
| what PCI devices does this machine have? | `cat(/proc/devices)`, `cat(/proc/bus/pci/devices)` | the 7 functions with their classes, virtio-blk marked as the one with a driver |
| what is the GPU temperature? | `cat(/proc/device-tree/thermal/zones/gpu/temperature)` (not found) | "I don't have access to GPU temperature information on this system." |
| che temperatura ha la GPU? controlla bene in /proc e sul disco | 10: ls /proc, /proc/bus, /proc/bus/pci, cat pci devices, ls /, ls /bin, run_code, 3 run_wasm | `[agent] tool round limit (10) reached, answering without tools`, then "Non riesco a trovare informazioni sulla temperatura GPU. Ho controllato: ..." |
| how many lines does /demo/stats.py have? | `run_wasm(wc, -l /demo/stats.py)` | "/demo/stats.py has 24 lines." |
| cosa c'e nel file /notes/todo.txt? | `cat(/notes/todo.txt)` (not found) | "Il file /notes/todo.txt non esiste." |

No file question went to run_code. run_code with `import os` still came up in exploratory requests that have no
direct tool ("find which file on the disk contains the word password", the GPU temperature on the disk; in tooleval
"set the hostname" and "copy /etc/motd"): the model got
`ImportError: no module named 'os'` and moved on to run_wasm or ls.

Confidence: 0 questions over the manual sessions (34 calls, none of a tool that writes); the calls below 0.80 (cat,
ls, run_wasm, run_code, recall) ran with the notice.

## Round limit and the answer round

`!set rounds 0`, `21` and `five` print `error: rounds must be an integer from 1 to 20`. With `!set rounds 2`, "what
is the GPU temperature? check /proc and the disk carefully" called ls /proc and ls /proc/bus, printed the limit
line and answered "There is no GPU temperature information available in /proc or on the disk ...".

With `!set rounds 1` and "read /proc/version, then /proc/uptime, then /proc/meminfo, one tool call each, and
summarize them", the first wording of the line on the last tool result ("No more tools in this turn: answer now
with what you have") gave a summary of /proc/version and then "I'll call uptime and meminfo." repeated to the
512-token cap. With "Tools are now disabled for this turn. Write your final answer to the user from the results
above, do not say that you will call a tool; if something is missing, say what you could not find." the same
request answered with /proc/version and "I was unable to read /proc/uptime or /proc/meminfo as the tools were
disabled before I could make those calls." The second wording is the one built in.

## tooleval (M45 protocol)

`tools/tooleval.py`, `SLOT=200`, constrain on, confidence on, temperature 0, a fresh copy of the beta disk.

| | T28 (docs/T28_RESULT.md) | T54 |
|---|---|---|
| spike: correct / first call right / args exact | 22/30, 22, 9 | **24/30**, 22, 9 |
| in-OS: correct / first call right | 23/24, 20 | **23/24**, 20 |
| malformed / schema errors / grammar lines not valid | 0 / 0 / 0 | 0 / 0 / 0 |
| calls (spike + in-OS) | 39 + 20 | 51 + 20 |
| `[confidence]` questions | 2 | 1 (write_file 0.65, `write` vs `mkdir`) |
| turns that hit the round limit and answered without tools | - | 2 (spike: "set the hostname", 10 calls; "copy /etc/motd", 10 calls) |
| decode tok/s of the call rounds, spike / in-OS | - | 36.72 / 41.49 |

The two extra correct spike prompts and the 12 extra calls come from the longer round limit: turns that stopped at
4 rounds before now go on (and two of them run to 10 and answer). In-OS is unchanged; its only miss is "cosa sai di
me?" answered without recall. Every turn ended with an answer.

## Limits

- The answer round blocks only the `<tool_call>` token: a model that wants to call a tool can still say so in
  text; the wording above stopped it in the case tried, not in general.
- 10 rounds make a confused turn longer: "set the hostname" took 36.7 s for 10 calls before its answer.
- run_code with `import os` is not gone: 4 of about 110 calls in this work, all in requests without a direct tool.
- /proc reports the TSC rate as cpu MHz and the engine's free-memory figure as MemFree and MemAvailable; there is
  no per-process data, no /proc/self, no /sys or /dev yet.
