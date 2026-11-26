# M1 result (2026-11-26)

All numbers come from real runs on the Fedora build box (`fedoralan`, Ryzen AI 9 HX 370, QEMU 10.2.2/KVM,
`-cpu host -M q35 -m 4G`). The box is shared with a torch job from another agent, so host load moved
between about 1 and 25 during the session. Every speed below was measured next to the asm OS under the
same conditions, and the ranges are kept rather than a single best number.

**Status: M1 PASS.** Golden self-test passes at 8 and at 1 vCPU, on BIOS and UEFI. Decode speed
matches the asm OS. The REPL works over TCP from the Mac, and a remote instance is running.

## How to connect (instance left running on fedoralan)

```sh
nc LAN-ADDRESS0 4555                 # serial console: press Enter to get "aios> "
open vnc://LAN-ADDRESS0:5901         # framebuffer (macOS Screen Sharing), password: aios
```

- The serial console has one TCP client at a time. Output produced while nobody is connected is dropped,
  so press Enter after connecting to get a fresh prompt.
- With plain `nc` the terminal echoes the line locally and the kernel echoes it too, so typed text shows
  twice. For clean input: `stty -echo -icanon; nc LAN-ADDRESS0 4555; stty sane`.
- VNC keyboard input arrives in the guest as PS/2, and the REPL reads PS/2 too, so you can type there.
- The VNC password `aios` is a fixed throwaway. This is a LAN-only test VM.
- Commands: `/clear` (reset chat), `/think`, `/nothink`, `/bench`, `/cores N`,
  `/park off|idle|<usec>`, `/temp X` (0 = greedy), `/stats`, `/help`. `/fault` deliberately triggers a
  page fault: it tests the exception handler and **halts the VM**.
- Restart on Fedora (one guest at a time, the target refuses with under 6 GiB free):
  `cd ~/coding/ringzero && setsid nohup make run-remote > build/remote.log 2>&1 < /dev/null &`.
  Stop it with `echo quit | socat - UNIX-CONNECT:build/remote.mon`.

`make run-web` (browser terminal) was skipped: the coordinator asked for VNC instead.

## Status per goal

| | Goal | Result |
|---|---|---|
| A | IDT + exception report | done. 32 exception stubs print vector, name, error code, RIP, CR2, RSP, CS, RFLAGS and LAPIC id on serial and framebuffer, then halt. Proven with `/fault` on BIOS and UEFI (log below, [m1_fault_uefi.png](m1_fault_uefi.png)). |
| B | Memory | done. Model bound zero-copy (`model_init` on the module). Arena = largest usable region (`0x100000000 + 2048 MiB`), bump `sys_mmap_anon`, zeroed. Both identity-mapped in the lower half with 2 MiB pages. `-m 4G` kept: the arena needs 69 MiB. |
| C | SMP | done. Limine MP (x2APIC) releases 7 APs into `ap_limine_entry` (engine/smp.asm). Each AP gets its own 64 KiB stack, loads the IDT, enables the LAPIC, runs `enable_avx_fpu`, then the smp.asm worker loop unchanged. Parking: spin for a TSC window, then `hlt`. One broadcast IPI on the next dispatch wakes them. |
| D | Float shims | done. `sample_top_p_bits` (engine/cpu.asm) passes temp/top-p as u32 bits. Used by `/temp`. |
| E | Generation + self-test | done. Mirrors kernel.asm `run_inference`. PROMPT_TOKENS/GEN_TOKENS lines, PASS/FAIL against golden_2b.json, prefill/decode tok/s from a TSC calibrated against PIT channel 2. |
| F | REPL | done. `aios> ` on COM1 plus PS/2. Multi-turn KV state kept across turns like the asm OS (closing `<\|im_end\|>\n` fed), `/clear` resets it. |
| G | Remote | done. `make run-remote`: serial on TCP 4555 and VNC :1 (5901, password). Tested from the Mac with `nc` and an RFB handshake. |

## Golden self-test (BIOS, 8 vCPU, `tools/boottest.py`)

```
smp:  8 CPUs reported by Limine MP, BSP LAPIC id 0
smp:  7 of 7 APs online in the worker loop, x2APIC=true, TSC-deadline idle timer=true
mem:  engine arena identity-mapped at 0x100200000 (2048 MiB), page size 2048 KiB (HHDM view: 1048576 KiB)
mem:  model identity-mapped at 0x749f000, page size 2048 KiB (HHDM view: 2048 KiB)
model: model_init ok in 124 ms, max_seq 2048, arena 69 MiB, vocab sample_n 248077
selftest: prompt "Say hi in 5 words." (golden_2b.json), greedy, 8 threads
PROMPT_TOKENS 248045,846,198,44240,15131,303,220,20,4105,13,248046,198,248045,74455,198,248068,271,248069,271
Hello there!
GEN_TOKENS 9419,1017,0
[prefill] 19 tok in 620.07 ms (30.64 tok/s)  [decode] 3 tok in 100.04 ms (19.99 tok/s)  cores 8  ctx 22/2048
SELFTEST PASS (prompt ids match, generated ids match vs golden 9419,1017,0 + <|im_end|>)
```

Same result with `-smp 1` (BIOS) and with UEFI/OVMF at `-smp 8`. The 3-token decode rate of the self-test
is too short to mean anything, so speed comes from `/bench` (the asm OS's own bench: 64 tokens of
"Explain how a CPU cache works in 3 concise sentences.").

## Speed (`/bench`, tok/s, ranges over 2-5 runs)

| Config | AIOS prefill | AIOS decode | asm OS prefill | asm OS decode |
|---|---|---|---|---|
| 8 vCPU, light host load | 44.0 - 49.7 | **32.6 - 36.7** | 28.2 - 49.3 | 30.0 - 35.8 |
| 8 vCPU, loaded host (load ~23) | 32.5 - 41.4 | 27.4 - 28.7 | 28.6 - 29.2 | 17.0 - 27.2 |
| 1 vCPU (`-smp 1`) | 27.3 - 29.2 | **21.3 - 22.0** | 28.7 - 28.8 | 21.4 - 21.9 |

Decode is at asm OS parity, which beats the "within 10%" target. Prefill is the same code path at
1 vCPU. At 8 vCPU the spread is host noise.

### Parking (8 vCPU, one boot, same loaded period, 3 benches each)

| `/park` | Workers during a generation | Decode tok/s | Prefill tok/s | Wake IPIs |
|---|---|---|---|---|
| `idle` (default) | spin, park only between requests | 27.9 - 28.7 | 36.2 - 37.1 | 2 per request |
| `off` | always spin | 27.4 - 28.4 | 32.5 - 41.4 | 0 |
| `100` (park after 100 us idle) | park in gaps > 100 us | 18.3 - 20.9 (-27%) | 22 - 24 | 16858 / 3 benches |
| `0` (park right after each job) | park every job | 15.0 - 15.4 (-46%) | 19.8 - 20.8 | 31878 / 3 benches |

Parking inside a generation costs far more than 5%: a `hlt` exit plus the wake IPI under KVM is slow
next to a dispatch every few microseconds. So the default spins for the whole generation and parks
between requests, with no measurable cost against `off`. When idle, the guest uses **0.6% of one host
CPU** (pidstat over 5 s, 8 vCPUs): workers are in `hlt` and the BSP sleeps in `hlt` on a 1 kHz
TSC-deadline timer while it polls COM1 and PS/2.

### TLB / page size

Model and arena use 2 MiB pages, as in the asm OS (whose identity map was 2 MiB too). The 1-vCPU
parity (21.6 vs 21.7 tok/s) leaves no room for a TLB effect, so 1 GiB pages were not tried. Note that
Limine's HHDM maps the arena region with 1 GiB pages but the model region with 2 MiB pages, because the
module is not 1 GiB aligned.

## REPL over TCP, from the Mac

`nc LAN-ADDRESS0 4555` (input piped), against the `make run-remote` instance:

```
aios> What is the capital of France? Answer in one sentence.
The capital of France is Paris.
[prefill] 24 tok in 516.50 ms (46.46 tok/s)  [decode] 7 tok in 207.36 ms (28.93 tok/s, output 0 ms)  cores 8  ctx 33/2048

aios> And of Italy?
The capital of Italy is Rome.
[prefill] 16 tok in 336.77 ms (47.51 tok/s)  [decode] 7 tok in 191.04 ms (31.40 tok/s, output 0 ms)  cores 8  ctx 58/2048

aios> /clear
[chat] history cleared (ctx=0/2048)

aios> /stats
[stats] threads 8, parked now 7, wake IPIs 4, arena used 69 MiB, chat pos 0
```

The follow-up ("And of Italy?") only makes sense with the previous turn in the KV cache, so multi-turn
works. PS/2 input was tested by injecting keys with QEMU's `sendkey h i ret` while watching serial from
the Mac: `hi` -> `Hello! How can I help you today?`. VNC: the RFB handshake from the Mac answers
`RFB 003.008` with security type 2 (VNC password). Framebuffer: [m1_screen.png](m1_screen.png).

## Exception handler (`/fault`, BIOS serial; the UEFI framebuffer shows the same line)

```
aios> /fault
*** EXCEPTION vec=14 (#PF page fault) err=0x0 rip=0xffffffff8000106c cr2=0xdeadb000 rsp=0xffff80007ff969a0 cs=0x28 rflags=0x10046 lapic=0 -> halt
```

## Deviations and decisions

1. **The engine cannot take HHDM pointers from `sys_mmap_anon`.** transformer.asm:529 treats a negative
   return as an error (`js .err_arena`), and every HHDM address is negative as i64. The first M1 boot
   failed in `model_init` for that reason. The fix: `mem::map_low` identity-maps the arena and the model
   in the lower half with 2 MiB pages, in Limine's page tables, before the APs start (they share CR3).
   That matches the asm OS layout. The page-table pages come from the first 64 KiB of the arena region.
2. **x2APIC is required for parking.** Limine is asked for x2APIC, so IPIs and EOI go through MSRs and no
   LAPIC MMIO mapping is needed. Without x2APIC the workers never park (fallback, untested).
3. smp.asm changes are confined to the AP entry and the idle path. The dispatch and partitioning are
   unchanged. One extra load per dispatch checks `tp_n_parked`. `tp_shutdown` was dropped.
4. TSC rate: PIT channel 2 over 20 ms, ported from drivers.asm. It measures 1997.1 - 1997.9 MHz, same as
   the asm OS banner (1997 MHz).
5. REPL `max_tok` is 512 (the asm OS used 128). `/halt` was left out: on a remote VM it would only kill
   the session.
6. `sys_write_all` now writes raw bytes to serial and reassembles UTF-8 for the framebuffer, because a
   token can end in the middle of a multibyte character. M0 printed `<non-utf8>` there.
7. Framebuffer output time during decode is measured and printed (`output N ms`): 3-4 ms per 64 tokens,
   up to 19 ms once the screen scrolls, because scrolling reads the whole framebuffer back.

## Open risks for M2

- **Shell with `@`**: the REPL sends every non-command line to the model. M2 needs a command parser,
  with `@` routing to the agent. Tool-call capture can hook `sys_write_all` (all model output passes
  through it).
- **No way to interrupt a generation**: Ctrl-C is not handled. The BSP is inside `transformer_forward`
  and polls input only between requests.
- **Framebuffer console**: Noto Sans Mono bitmap has no CJK, so Qwen output in Chinese or Japanese shows
  `?`. No shadow buffer, so a scroll reads 4 MB of VRAM (slow on real hardware with UC/WC mappings).
  Only 32 bpp is handled. No ANSI colours or cursor.
- **Keyboard**: PS/2 polling only, scancode set 1, US layout, no arrows or extended keys. Real UEFI
  machines without PS/2 emulation need USB HID (xHCI). Serial is the fallback.
- **Exceptions**: no IST stacks, so a kernel stack overflow still triple-faults. An exception on an AP
  is untested: it should print, but the BSP would then wait forever on that AP's `tp_done_arr` line.
- **Input is polled** on the BSP with a 1 kHz timer. No UART or IOAPIC interrupts yet.
- **Single serial client** on TCP. Output sent while nobody is connected is lost.
- **Parking granularity**: workers spin for a whole generation. A long generation on all cores competes
  with other host work. `/park 100` trades about 27% decode speed for lower host load if that matters.
- **Golden coverage**: only one 3-token golden sequence. The 64-token bench text was identical across the
  BIOS and UEFI boots, but a longer golden reference would catch subtler drift.
