# M2 + M3 result (2026-11-26)

Shell, kernel log, keyboard layouts (M2) and the system agent with native tool calls and a boot-time
prefix snapshot (M3), minimal versions. All numbers come from real runs on the Fedora box (`fedoralan`,
Ryzen AI 9 HX 370, QEMU 10.2.2/KVM, `-cpu host -M q35 -m 4G -smp 8`, BIOS), host load 1.3 to 5.5. The
acceptance run was driven from the Mac over TCP with `tools/accept.py`. Full transcript:
[m2m3_accept.txt](m2m3_accept.txt). Architecture and diagram: [ARCHITECTURE.md](ARCHITECTURE.md).

## How to connect (instance left running on fedoralan)

```sh
nc LAN-ADDRESS0 4555                 # serial shell: press Enter to get "aios> "
stty -echo -icanon; nc LAN-ADDRESS0 4555; stty sane    # same, without the double local echo
open vnc://LAN-ADDRESS0:5901         # framebuffer + PS/2 keyboard (macOS Screen Sharing), password: aios
```

Then type a builtin (`help`, `mem`, `cpu`, `cores`, `log`, `uptime`, `bench`, `stats`, `kbd`, `set`,
`clear`, ...) or `@ <question>` for the agent. Ctrl-C stops a generation. `clear` starts a new
conversation (the next `@` restores the boot snapshot in about 1.5 ms).

Observability for the main session (Fedora paths, relative to `~/coding/ringzero`):

```sh
# everything the guest printed on serial, including the echoed input of whoever is connected (append mode)
ssh fedoralan 'tail -f ~/coding/ringzero/build/remote-serial.log'
# framebuffer screenshot through the QEMU monitor (read-only for the guest), PPM -> PNG
ssh fedoralan 'cd ~/coding/ringzero && python3 tools/screenshot.py build/remote-monitor.sock build/screen.png' && scp fedoralan:coding/ringzero/build/screen.png .
```

Restart on Fedora (one guest at a time; stop the old one first):

```sh
echo quit | socat - UNIX-CONNECT:build/remote-monitor.sock
cd ~/coding/ringzero && setsid nohup make run-remote > build/remote.log 2>&1 < /dev/null &
```

`poweroff` (or the agent's poweroff after `y`) ends the QEMU process. `reboot` does too, because
`run-remote` keeps `-no-reboot`. Restart it with the command above.

## Status per goal

| | Goal | Result |
|---|---|---|
| 1 | Shell (M2) | done. Builtins: help, clear, mem, cpu, cores [N], log [N], uptime, bench, stats, kbd, set, temp, think, park, poweroff, reboot (M1 `/xxx` still accepted). Unknown command: short error suggesting `@`. Ctrl-C (0x03 on serial, Ctrl+C on PS/2) stops a generation between tokens. Kernel log ring (256 lines, `[sec.ms] L text`) read by `log` and `read_log`. Framebuffer: emoji/pictographs/ZWJ/variation selectors skipped, any other missing glyph prints one `U+FFFD` per run, `€` drawn as an overlay. |
| 1b | Keyboard layouts | done. Data tables `us` and `it` (default), Shift, Ctrl, AltGr (E0 38; left Alt too, for macOS VNC clients). `kbd us|it` and `set_setting keyboard`. pc-keyboard has no Italian layout, so the table is ours. Typed à è é ì ò ù reach the line buffer as UTF-8 and render on the framebuffer. |
| 2 | Agent (M3) | done. 8 tools (meminfo, cpuinfo, list_cores, read_log, uptime, poweroff, reboot, set_setting) in the native template, lenient parser (missing closers OK), schema validation (unknown tool/parameter, integer, enum, required), up to 4 tool rounds per turn, y/N confirmation for poweroff/reboot. |
| 3 | Prefix caching | done. 743-token prefix prefilled once at boot, 36 MiB snapshot (KV rows 0..743 of the 6 full-attention layers + DeltaNet S and conv state). Restore 1.2 - 1.9 ms. |
| 4 | Golden self-test + speed | done. SELFTEST PASS on every boot. Bench decode 35.4 - 36.6 tok/s (M1: 32.6 - 36.7). |
| 5 | Architecture diagram | done. Two mermaid diagrams (components, `@` request flow) + legend in ARCHITECTURE.md, both render with mermaid-cli. README points to it. |
| 6 | Acceptance | done, transcript below. |

## Numbers

| What | Value |
|---|---|
| Boot prefix prefill (743 tok, system prompt + 8 tools) | 15.09 - 15.70 s (47 - 49 tok/s), 5 boots |
| Snapshot size / save | 36 MiB / 31 - 188 ms (first touch of the buffer) |
| Snapshot restore (`clear` then `@`) | 1.15 - 1.9 ms |
| Bench decode (64 tok) | 35.4 - 36.6 tok/s |
| Agent turn decode | 29 - 35 tok/s (short answers include the first-token cost) |
| Agent turn prefill (user turn / tool result) | 41 - 49 tok/s |
| Typical tool question (`@ quanti core hai?`), wall clock | 2.4 - 3.0 s |

Tokenizer check: the prefix the kernel builds (special tokens split out, Qwen pre-tokenizer regex in Rust,
then the engine BPE per chunk) gives exactly the HF tokenizer's ids: 743 tokens, FNV-1a `0xb60be314aa20d0b0`
on both sides (`tools/hf_prefix.py`; the hidden `toks` builtin prints the kernel ids for a full diff).

## Boot (serial log of the running instance)

```
selftest: prompt "Say hi in 5 words." (golden_2b.json), greedy, 8 threads
PROMPT_TOKENS 248045,846,198,44240,15131,303,220,20,4105,13,248046,198,248045,74455,198,248068,271,248069,271
Hello there!
GEN_TOKENS 9419,1017,0
SELFTEST PASS (prompt ids match, generated ids match vs golden 9419,1017,0 + <|im_end|>)
bench: 64-token decode, prompt "Explain how a CPU cache works in 3 concise sentences."
[prefill] 25 tok in 518.85 ms (48.18 tok/s)  [decode] 64 tok in 1777.44 ms (35.44 tok/s, output 4 ms)  cores 8  ctx 88/2048
agent: tool-call parser self-test PASS
kbd:  PS/2 layout it (change with 'kbd us|it')
agent: system prompt + 8 tools = 743 tokens (fnv 0xb60be314aa20d0b0), prefilling once...
agent: prefix prefill 743 tok in 15.699 s (47 tok/s), snapshot 36 MiB saved in 37 ms
```

## Acceptance transcript (Mac -> `LAN-ADDRESS0:4555`, trimmed; speed lines kept where useful)

```
aios> mem
RAM total 4095 MiB, free 2057 MiB, used 2038 MiB (model weights 1922 MiB, LLM engine arena 106 MiB, kernel and firmware 10 MiB)

aios> @ how much free memory?
[agent] fresh conversation: prefix (743 tok) restored in 1582 us
[prefill] 17 tok in 395.92 ms (42.93 tok/s)  [decode] 14 tok in 447.62 ms (29.04 tok/s, output 0 ms)  cores 8  ctx 776/2048
[tool] meminfo()
[tool] -> RAM total 4095 MiB, free 2057 MiB, used 2038 MiB (model weights 1922 MiB, LLM engine arena 106 MiB, kernel and firmware 10 MiB)
La memoria libera è di 2057 MiB.

aios> @ quanti core hai?
[tool] cpuinfo()
[tool] -> AMD Ryzen AI 9 HX 370 w/ Radeon 890M; 8 cores; TSC 1997 MHz; features used by the LLM engine: AVX2 FMA F16C
Ho 8 core.

aios> @ show me the last 5 kernel log lines
[tool] read_log(n=5)
[tool] -> [   48.761] A answer (13 tok): La memoria libera è di 2057 MiB.
[   48.784] A user: @ quanti core hai?
[   49.560] A tool cpuinfo -> AMD Ryzen AI 9 HX 370 w/ Radeon 890M; 8 cores; TSC 1997 MHz; features used by the LLM engine: AVX2
[   51.127] A answer (5 tok): Ho 8 core.
[   51.159] A user: @ show me the last 5 kernel log lines
Le ultime 5 righe del log del kernel sono:
(the same 5 lines in a code block)

aios> @ spegni la macchina
[tool] poweroff()

[agent] wants to POWER OFF the machine. Proceed? [y/N] n
[tool] -> cancelled: the user answered no, the machine stays on
L'utente ha risposto "no", quindi la macchina rimane accesa.

aios> uptime
up 0h 1m 2s (62.948 s)

aios> @ write a long detailed essay about the history of operating systems
La storia degli sistemi operativi è un viaggio affascinante ... (Ctrl-C sent after 3 s)
[prefill] 23 tok in 485.61 ms (47.36 tok/s)  [decode] 84 tok in 2572.41 ms (32.26 tok/s, output 62 ms)  cores 8  ctx 1548/2048
^C [agent] interrupted

aios> frobnicate
unknown command 'frobnicate'. Type 'help', or '@ frobnicate' to ask the agent.

aios> log 6
[   60.959] A user: @ spegni la macchina
[   61.776] A confirmation: no
[   61.776] A tool poweroff -> cancelled: the user answered no, the machine stays on
[   62.847] A answer (15 tok): L'utente ha risposto "no", quindi la macchina rimane accesa.
[   62.990] A user: @ write a long detailed essay about the history of operating systems
[   66.094] A turn interrupted by Ctrl-C

aios> clear
aios> @ reply with only a smiley emoji
[agent] fresh conversation: prefix (743 tok) restored in 1894 us
😊
```

The emoji reaches serial as UTF-8 and is skipped on the framebuffer, no `?` ([m2m3_screen.png](m2m3_screen.png)).
`uptime` after the cancelled poweroff shows the machine stayed up.

From an earlier run of the same build (`tools/boottest.py`), the agent also handled `@ set the temperature
to 0.7` -> `set_setting(key=temperature, value=0.7)`, and a follow-up that needs the previous turn
(`@ how much free memory?` then `@ e quanta ne usa il modello?` -> "Il modello utilizza 1922 MiB di memoria.",
no new tool call).

## Keyboard (IT layout) tests

PS/2 through the QEMU monitor (`sendkey`, physical keys, `tools/kbdtest.py`): typed
`x à ò è é ù ì AltGr+ò AltGr+à AltGr+è AltGr+] AltGr+Shift+è AltGr+Shift++ AltGr+e Shift+3 < Shift+< Shift+2 - _ /`:

```
aios> xàòèéùì@#[]{}€£<>"'?/
unknown command 'xàòèéùì@#[]{}€£<>"'?/'. Type 'help', or '@ xàòèéùì@#[]{}€£<>"'?/' to ask the agent.
```

PS/2 Ctrl+C: `sendkey ctrl-c` 4 s into a long answer -> `^C [agent] interrupted`, back at the prompt.

VNC as a plain keysym client sends it (`tools/vnckeys.py`: one keysym per character, Shift_L held for
shifted characters, Alt_L held for the AltGr ones, which is what a Mac Option key gives):

```
aios> vàèéìòù@#[]{}€£<>|AbC~
```

What I found on the way, and what it means for VNC from a Mac with an Italian keyboard:

- macOS Screen Sharing sends keysyms, not scancodes, so QEMU's `-k` keymap decides the scancodes.
- QEMU's stock `it` keymap has no AltGr level at all (no `@ # [ ] { } €`), and keymap `include` files
  are rejected by QEMU 10. So `make run-remote` builds `build/it-aios` = QEMU's `it` + the 9 AltGr
  entries in `tools/keymaps/it-altgr`, and passes it with `-k`.
- With the stock `it` map and no modifiers sent, `é` came out as `è`, `£` as `3`, `>` as `<`, and the AltGr
  characters were dropped. With modifiers and the extended map, all of them arrived.
- Not verified with the real macOS Screen Sharing app (no GUI here). If Option+ò still gives `ò` instead
  of `@`, the client is not sending Option; use the serial console (the Mac terminal sends UTF-8, any
  layout works) or a VNC client with raw keyboard / QEMU extended key events (TigerVNC, Remmina).
- Serial input is not affected by `kbd`: it is UTF-8 from the terminal.

## Implementation notes

- The engine was not changed. The agent uses `transformer_forward` token by token; tool-call markup is
  hidden by switching the console echo off at the `<tool_call>` token (248058), and generation stops at
  `</tool_call>` (248059). The emitted-but-unfed token plus `<|im_end|>\n` are always fed so the KV
  cache matches the chat template for the next turn.
- The tool-response turn follows the Qwen3.5 template exactly:
  `<|im_start|>user\n<tool_response>\nRESULT\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
  The template lives in `tokenizer_config.json` (there is no `chat_template.jinja` in `hf2b`).
- Snapshot restore only copies KV rows `0..P`: rows past P are always written before they are read, so a
  conversation never needs them cleared. The bench and self-test run from position 0 and overwrite the
  prefix rows, so `bench` marks the conversation fresh.
- Context is 2048 tokens and the prefix takes 743. When a conversation gets within 300 tokens of the end,
  the next `@` starts a fresh one (printed).
- Poweroff writes S5 to the ICH9 PM1a_CNT port (0x604), reboot pulses the 8042 reset line, then a triple
  fault. Both are QEMU-q35 shortcuts, no ACPI table parsing.
- System prompt language line: three wordings tried. "the user's language" answered `quanti core hai?`
  in Hindi ("hai" is a Hindi word), "the language the user writes in" kept English after English turns,
  "English or Italian, matching the user's latest message" is the one kept. It still sometimes answers an
  English question in Italian after Italian turns (2B model).

## Open risks

### For M4 (virtio-blk + filesystem tools)

- **Context budget**: 2048 tokens, 743 already used by the prefix. Each new tool costs about 40 - 90 prefix
  tokens (and about 1.3 s of boot prefill per 60 tokens); file contents fed back as tool results will fill
  the context in a few turns. Needs `max_seq` 4096+ (arena has room: KV is 6 layers only) and truncated
  tool results.
- **Tool result buffer** is 1 KiB (`SBuf<1024>`); `cat` needs a bigger buffer or paging.
- **No interrupts for devices**: input is polled, UART/IOAPIC are not set up. virtio-blk can poll too, but
  needs PCI enumeration, BAR mapping (MMIO is outside the HHDM) and DMA-safe buffers (identity map exists).
- **No heap**: everything is fixed buffers or the bump arena (never freed). A filesystem wants an allocator.
- **Confirmation** exists for poweroff/reboot only; `rm`/`write_file` need the same gate, and the validator
  should check paths (the spike saw relative paths and loose values).
- **Language drift** and markdown in long answers (2B model); the spike's fine-tune list still applies.

### For M5 (code execution)

- Everything runs in ring 0 with one address space: any interpreter bug is a kernel bug. Needs a sandbox
  (wasm interpreter with fuel limits, or a ring-3 task with its own page tables).
- The BSP is busy for the whole generation and the APs spin; a long-running program needs a time limit
  and a way to stop it (Ctrl-C reaches only the generation loop today).
- Code passed as a tool parameter is plain text in the template: the lenient parser trims values, which
  is wrong for code (indentation, trailing newlines). Needs an exact-value mode per parameter.
- Exceptions halt the machine (no recovery, no IST stacks), so a faulting guest program would take the
  OS down.
