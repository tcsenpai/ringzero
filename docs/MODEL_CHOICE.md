# Working model choice: 2B q8 vs 4B q4 vs 4B q4all

Build: main after the M7 merge (a07a62e) plus the `/clear` fix (5bfc1e8). Driven over the serial TCP port by
`tools/tooleval.py` (`build/ab.sh`-style run: one guest at a time, each on a fresh copy of the user's
`build/disk.img`, BIOS, 8 vCPU KVM, Ryzen AI 9 HX 370). Settings for every prompt: constrain on, `!set slot 0`,
`!set consolidate 0`, temperature 0, `/clear` before each prompt, y/N answered n.

User criteria: weigh quality, usability and tok/s; decode must stay at about 19 tok/s or more (19.x is fine).

## Result

| | 2B q8 (6G) | **4B q4 (8G)** | 4B q4all (8G) |
|---|---:|---:|---:|
| Weights (module) | 2.0 GB | 3.36 GB | 2.71 GB |
| Spike set (30): malformed calls / schema errors / calls leaving the grammar | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Spike: first call is a right tool (of 25 that need one) | 22 | **22** | 21 |
| Spike: exactly the expected calls (of 25) | 14 | **18** | 17 |
| Spike: exact args (of 15 with rules; 5 of them not expressible with the kernel's tools, so 10 is the max) | 9 | **9** | 8 |
| Spike: no-tool prompts answered without a tool (5) | 5 | 5 | 5 |
| In-OS set (24): malformed / schema / grammar | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| In-OS: first call is a right tool (of 22) | 20 | **21** | 20 |
| In-OS: exactly the expected calls (of 22) | 19 | **22** | 21 |
| In-OS: no-tool prompts answered without a tool (2) | 2 | 2 | 2 |
| Italian questions, correct by hand (10) | 7 | **8** | 8 (and one invented fact) |
| Decode tok/s, `/bench` median of 10 (2 rounds, load 7.8-10.6) | **35.2** (31.2-37.7) | **20.9** (20.4-21.7) | 25.0 (24.1-25.6; 16-20 while another guest ran) |
| Decode tok/s, `/bench` x 5 right after the A/B (host load) | 35.0 (load 11) | 18.7 (load 18) | 25.3 (load 13) |
| Prefill, boot prefix 1411 tok (cache miss) | 6.2 s, 227 tok/s | 16.6 s, 85 tok/s | 17.1 s, 82 tok/s |
| Snapshot pool (`!mem`) / size of a point (R) / points before the pool is full | 1024 MiB / 19.3 MiB / about 43 | 1024 MiB / 50.3 MiB / about 17 | 1024 MiB / 50.3 MiB / about 17 |
| Quality vs HF (M7, KLD mean / top-1) | (2B reference) | 0.028 / 92.7% | 0.061 / 89.8% (fails the M7 gate) |

Host: the box was not quiet. Another agent's 6G-8 vCPU guest (`~/coding/ringzero-m8`) started and stopped by itself
during the runs, and the load average stayed at 7.8 to 18. The user's instance was stopped (`!poweroff`) for the whole
test. The quiet-bench table row is two interleaved rounds of the three builds booted back to back (`tools/boottest.py`,
5 x `/bench` each, the boot bench not counted). The q4all round 1 values (16-20 tok/s) overlapped the other guest.
Round 2 (24.1-25.9) matches the post-A/B bench (25.3).

## Recommendation: 4B q4 (`MODEL=4b QUANT=q4`, 8G guest)

- **Quality**: the best of the three on tool selection (in-OS 22/22 exact, spike 18/25), and as good as or better
  than the 2B on the Italian questions. The 2B misses tool calls the 4B makes ("ricordati che il mio colore
  preferito e il verde" and "cosa sai di me?" got no call; it loops on `set_setting` for "set the hostname" and
  "set the screen brightness"). It also gets 3 of 10 Italian answers wrong (Melbourne, "17 per 23 è 0", giovedì).
- **Speed**: 20.9 tok/s median on a partly loaded host, which meets the 19 tok/s floor. Under heavy load
  (load average 18) it drops to 18-19, which is borderline. q4all is 20% faster, but it fails the M7 quality gate.
  On these sets it is also no better than q4: more wandering calls (`run_wasm /bin/cat`, `run_code` with
  `import os`), one less right first call, one less exact args, and an invented name in an answer. The extra speed
  does not buy anything the user asked for.
- **Usability costs** to accept with the 4B: prefill is about 2.7x slower (a cache-miss boot prefix takes 17 s
  instead of 6 s, and a long tool result takes longer before the answer starts). A boot with a cache hit is
  unaffected. The snapshot pool holds about 17 points instead of about 43. After that a turn is not saved
  ("[snap] pool full") until old points are 32 turns old or dropped with `/drop`; the conversation itself goes on.
  The guest needs 8G (the host needs about 10 GiB free).

The 2B stays the build default (`make iso`, `make run-local`), so a machine with less RAM or a user who wants
speed first still gets it. The remote instance runs the 4B q4.

## Italian questions (set `it` in tools/tooleval.py)

Answers as given (first line), verdict by hand.

| # | Question | Expected | 2B q8 | 4B q4 | 4B q4all |
|---|---|---|---|---|---|
| 1 | Qual è la capitale dell'Australia? Rispondi in una parola. | Canberra | Melbourne. ✗ | Canberra ✓ | Canberra ✓ |
| 2 | Quanto fa 17 per 23? Rispondi solo con il numero, senza usare strumenti. | 391 | "17 per 23 è 0." ✗ | 0.7391304347826087 (17/23) ✗ | 0.7391304347826087 ✗ |
| 3 | Chi ha scritto la Divina Commedia? | Dante Alighieri | Dante Alighieri ✓ | Dante Alighieri ✓ | Dante Alighieri ✓ |
| 4 | Se oggi è martedì, che giorno della settimana sarà fra 10 giorni? | venerdì | giovedì ✗ | martedì ✗ | martedì ("2 settimane") ✗ |
| 5 | Qual è il pianeta più grande del sistema solare? | Giove | Giove ✓ | Giove ✓ | Giove ✓ |
| 6 | Un treno parte alle 14:40 e arriva alle 17:15. Quanto dura il viaggio? | 2 h 35 min | 2 ore e 35 minuti ✓ | 2 ore e 35 minuti ✓ | 2 ore e 35 minuti ✓ |
| 7 | In che anno è caduto il muro di Berlino? | 1989 | 1989 ✓ | 1989 ✓ | 1989 ✓, then invents "Hans-Joachim Fuchs" as the GDR leader |
| 8 | Ho 5 mele, ne mangio 2 e poi ne compro il doppio di quelle che mi sono rimaste. Quante mele ho adesso? | 9 | 9 (worked) ✓ | 9 (worked) ✓ | 9 (worked) ✓ |
| 9 | Qual è il fiume più lungo d'Italia? | Po | Po ✓ | Po (652 km) ✓ | Po (652 km) ✓ |
| 10 | Quanti secondi ci sono in un'ora? | 3600 | 3600 ✓ | 3600 ✓ | 3600 ✓ |
| | **correct** | | **7** | **8** | **8** |

`auto_ok` in the JSON (a word-boundary substring check) agreed with the hand verdict on all 30 answers.

## Where the tool sets fail (all three builds)

- Spike prompts written for the spike's own tool list ask for things the kernel does not have: `language=script|wasm`
  on `run_code`, a hostname or language setting, `recursive=true` on `rm`. All three builds miss those args (5 of the
  15 argument rules). The 4B builds call `run_wasm` for the wasm module, and the 2B loops on it.
- Every build runs extra `ls` calls after a failed `cat` / `rm` on paths the disk does not have (`/tmp`, `/home`).
- Copy `/etc/motd` to `/home/motd.bak`: the 2B makes the right first call (`cat`); the 4B builds go for `run_code`
  or `run_wasm cp`.

## Bug found and fixed on the way (5bfc1e8)

The first 4B q4 run showed the context growing across `/clear` (ctx 1576 -> 1610 -> ... 8524). Cause: when the pool
is full, `snap::commit` refuses the save (`unsaved`) and leaves `live == ROOT` while the engine state is ahead.
`snap::new_line` (`/clear`) then skipped the restore because `live == ROOT`. It now restores when the live state is
unsaved too. Before the fix, every 4B prompt after the pool filled up (from spike prompt 17 on) ran on top of the
previous ones. That run was thrown away and the 4B numbers above come from runs on the fixed build (checked: ctx 1576
-> 1445 after the first "pool full"). The 2B run never filled its pool (954/1024 MiB at the end), so the fix does not
touch its path and the run was kept.

Also found: since the M5 / M5b arena users, the 4B no longer boots on a 6G guest. q4all panics with "snapshot pool:
no room for the prefix" and q4 fails `model_init`. The Makefile now defaults every `MODEL=4b` option to 8G
(M7 measured 6G before those merges).

## Reproduce

```sh
# on fedora, one guest at a time, the user's instance stopped
make iso MODEL=4b QUANT=q4 && cp build/ringzero.iso build/ab-4b-q4.iso   # likewise 2b-q8 (make iso), 4b-q4all
# boot it with a serial TCP port (as make run-remote REMOTE_PORT=4556 ... DISK=<copy of disk.img>), then:
python3 tools/tooleval.py 127.0.0.1 4556 on out.json spike,inos,it
MEM=8G python3 tools/boottest.py build/ab-4b-q4.iso bios build/qb 8 /bench /bench /bench /bench /bench
```

Raw results on fedora: `~/coding/ringzero/build/ab/{2b-q8,4b-q4,4b-q4all}/` (eval.json, serial.log, bench.json),
`build/ab/4b-q4-unfixed/` (the discarded run), and `build/ab/4b-q4all-6G-panic/`.
