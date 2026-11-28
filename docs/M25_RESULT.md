# M2.5 result (2026-11-26)

Shell UX inversion, 32768-token context with an FP16 KV cache, 8 KiB tool results, and native context
compaction by the agent. Edition name: **AI OS Qwen35@2b** (boot banner). All numbers come from real runs
on the Fedora box (Ryzen AI 9 HX 370, QEMU 10.2.2/KVM, `-cpu host -M q35 -m 4G -smp 8`,
BIOS). The tests ran on a second guest (`make run-remote REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test`)
next to the user's instance. That one was in use, so host load was 4 to 11 during the measurements (noted
per row). Architecture and diagrams: [ARCHITECTURE.md](ARCHITECTURE.md).

## How to connect (instance left running on the build box, new build)

```sh
nc LAN-ADDRESS0 4555                 # serial shell: press Enter to get "aios> "
stty -echo -icanon; nc LAN-ADDRESS0 4555; stty sane    # same, without the double local echo (Tab works too)
open vnc://LAN-ADDRESS0:5901         # framebuffer + PS/2 keyboard (macOS Screen Sharing), password: aios
```

Type a message to talk to the agent, `!mem` / `!cores` / `!log 10` / `!kbd it` for the machine, `/help`,
`/ctx`, `/history`, `/compact`, `/clear` for the session. Tab completes, a bare `/` or `!` (pause or Enter)
lists the commands, up/down recall history, Ctrl-C stops a generation.

Observability (Fedora paths, relative to `~/coding/ringzero`):

```sh
ssh BUILD-BOX 'tail -f ~/coding/ringzero/build/remote-serial.log'
ssh BUILD-BOX 'cd ~/coding/ringzero && python3 tools/screenshot.py build/remote-monitor.sock build/screen.png' && scp BUILD-BOX:coding/ringzero/build/screen.png .
```

Restart on Fedora (stop the old guest first):

```sh
echo quit | socat - UNIX-CONNECT:build/remote-monitor.sock
cd ~/coding/ringzero && setsid nohup make run-remote > build/remote.log 2>&1 < /dev/null &
```

`~/coding/ringzero/models/qwen35-2b-q8_0-v3.asmbin` there is a symlink to the existing export
(`~/coding/qwenbaremetal/models/`), as the Makefile now uses the repo-relative `models/` path.

## Status per request

| | Request | Result |
|---|---|---|
| 1 | Shell UX inversion | done. Plain text -> agent (`@ text` still accepted). `!` = machine commands: help, mem, cpu, cores [N], log [N], uptime, kbd [us\|it], set <key> <value>, park, stats, poweroff, reboot. `/` = session commands: help, stats, clear, ctx, compact, history [N], think, temp, bench, fill <tok>, gen <n> <text>. help and stats answer to both; the wrong prefix gets "'mem' is a machine command: try !mem". Tab completes names and first arguments (`kbd us\|it`, `set` keys, `think on\|off`, `park off\|idle`); a longest common prefix first, a list on the second Tab. Suggestion line on a bare `/` or `!` after a 400 ms pause, or on Enter. History: 16 lines, up/down (ESC [ A/B on serial, E0 48/50 on PS/2). Ctrl-C, layouts (it default) and y/N confirmations unchanged. Help, banner hint and README updated. |
| 2 | Context 32768, FP16 KV | done. `engine/transformer.asm` (changes marked AIOS): with `-DKV_F16` (default) K/V rows are stored with `vcvtps2ph` and read with `vcvtph2ps` in two small helpers (`dot_f16_avx2`, `saxpy_f16_avx2`) used by `gqa_attn_range`; `AIOS_KV=f32 make iso` builds the upstream FP32 cache. `model_init` lets the 32768 override exceed the .asmbin header's 4096 (Qwen3.5-2B `config.json`: `max_position_embeddings` 262144). Arena 406 MiB (FP16) / 790 MiB (FP32), QEMU stays at `-m 4G` (1728 MiB still free with FP16). The prefix snapshot copies rows 0..P with the element size the engine exports (`kv_elt_bytes`): 27 MiB (FP16) vs 36 MiB (FP32). |
| 3 | Tool result cap | done. 8192 bytes, then `[... truncated by the kernel: the result was longer than 8192 bytes]`. `read_log` accepts up to 256 lines so the cap can be reached. Verified with `read_log(n=200)` on a log of long lines. |
| + | Context management (coordinator) | mechanism done, recall not reliable (see below). Fill shown by `/ctx` and every stats line; compaction at 75% (`!set compact N`) or `/compact`; raw transcript in a 64 KiB ring (`/history`); every compaction logged. |
| + | Model fetch (coordinator) | done. `make model` -> `tools/fetch_model.sh` (uvx `hf download`, curl fallback, skips files already there, `HF_DIR=` reuse) + `tools/export_weights.py` (vendored, credit alesha-pro/llm.asm) via `uv run --with numpy`. `MODEL ?= models/qwen35-2b-q8_0-v3.asmbin`, `models/` gitignored. |
| + | VNC Option key (coordinator) | done. `Meta_L` / `Meta_R` -> right Alt (E0 38, the guest's AltGr) in `tools/keymaps/it-altgr`. |

## Verification

### Golden self-test

SELFTEST PASS on every boot of both builds (tokens 9419,1017,0 + `<|im_end|>`, prompt ids match). `/gen 64 Say hi in 5 words.` gives the same 3 tokens on FP16 and FP32.

### FP16 vs FP32 KV: generated tokens (greedy, 64 tokens, `/gen`)

Prompts: the golden prompt plus three real texts from this repo flattened to one line (docs/M1_RESULT.md,
README.md, docs/ARCHITECTURE.md as they were before this milestone), each with a short instruction. The
margin is top1 - top2 logit at that step, printed by the new `/gen` debug command.

| Prompt (tokens in) | Result | Margin at the divergence (FP16 / FP32) | Smallest margin before it (FP16 / FP32) |
|---|---|---|---|
| golden, 19 | identical (3 tokens + EOS) | - | 0.471 / 0.320 |
| M1_RESULT summary, 402 | first divergence at generated token 41 (FP16 id 561, FP32 id 20025), 41 tokens identical | **0.008 / 0.035** (near tie) | - |
| README main points, 1279 | identical, 64 tokens | - | 0.137 / 0.037 |
| ARCHITECTURE explained, 1795 | identical, 64 tokens | - | 0.029 / 0.010 |

The logits move by about 0.02 to 0.15 between the two caches (e.g. first-step margins 3.055 vs 3.209 and
0.370 vs 0.333), so a divergence shows up only where the top two candidates are almost tied, as at token
41 above.

### Speed (tok/s; `/bench` = 64-token decode at ctx 88; `/fill N` = N-token prefill from position 0, then 64 decode steps)

| Fill | KV | Prefill | Decode | Host load (1 min) |
|---|---|---|---|---|
| ~90 (`/bench`) | FP16 | 44 - 50 | 34.4 - 36.0 (8 runs; one more at 24.7 while the other guest generated) | 4 - 11 |
| ~90 (`/bench`) | FP32 | 47 - 48 | 35.7 - 36.6 (2 runs) | 8 - 11 |
| 105 (`/fill 100`) | FP16 | 46.9 | 34.95 | 7 |
| 105 | FP32 | 47.6 | 34.14 | 11 |
| 8192 | FP16 | 44.0 (186 s) | 30.75 | 7 - 10 |
| 8192 | FP32 | 39.5 (207 s) | 26.84 | 11 |
| 30000 | FP16 | **35.1** (856 s) | **23.75** | 10 - 11 |
| 30000 | FP32 | 28.3 (1061 s) | 17.78 | 11 |

M2+M3 bench at 2048 context: 35.4 - 36.6 tok/s, so the short-fill decode did not regress. At 30k tokens
FP16 is 24% faster in prefill and 34% faster in decode than FP32: attention reads half the bytes. The
prefill is token by token (no batched prefill), so filling 30k tokens takes 14 minutes.

### UX over serial (scripted from the Mac, `tools/uxtest.py LAN-ADDRESS0 4556`, keys sent one by one)

```
aios> /he<Tab>              -> echoed as "/help " (completed), Enter prints the help
aios> !mem
RAM total 4095 MiB, free 1728 MiB, used 2367 MiB (model weights 1922 MiB, LLM engine arena 434 MiB, kernel and firmware 11 MiB)
aios> how much free memory?
[agent] fresh conversation: prefix (743 tok) restored in 53876 us
[prefill] 17 tok in 367.27 ms (46.28 tok/s)  [decode] 14 tok in 421.35 ms (30.85 tok/s, output 0 ms)  cores 8  ctx 776/32768 (2%)
[tool] meminfo()
[tool] -> RAM total 4095 MiB, free 1728 MiB, used 2367 MiB (model weights 1922 MiB, LLM engine arena 434 MiB, kernel and firmware 11 MiB)
La memoria libera è di 1728 MiB.
aios> !kbd <Tab>
  us it
aios> !se<Tab>tem<Tab>0.7   -> "!set temperature 0.7"
temperature = 0.7 (top-p 0.95 sampling)
aios> /hi<Tab>5             -> "/history 5"
user: how much free memory?
tool: meminfo(...) -> RAM total 4095 MiB, free 1728 MiB, ...
assistant: La memoria libera è di 1728 MiB.
aios> <Up><Up>              -> "!set temperature 0.7" recalled
aios> /                     (Enter on a bare prefix)
commands: /help /stats /clear /ctx /compact /history /think /temp /bench /fill /gen
aios> !
commands: !help !mem !cpu !cores !log !uptime !kbd !set !park !stats !poweroff !reboot
aios> /mem
'mem' is a machine command: try !mem
aios> quanti core hai?
[tool] cpuinfo()
[tool] -> AMD Ryzen AI 9 HX 370 w/ Radeon 890M; 8 cores; TSC 1997 MHz; features used by the LLM engine: AVX2 FMA F16C
Hai 8 core.
aios> /ctx
[ctx] 972/32768 tokens (2%), boot prefix 741 tok, KV fp16, compaction at 75%, 0 compaction(s) so far, 2 exchange(s) kept for re-prefill
```

### UX over PS/2 (QEMU monitor `sendkey`, layout it, `tools/ps2ux.py`)

Keys: `! m e m Ret`, `/ h e Tab Ret`, `/` + 1.5 s pause + Backspace, `! k b Tab Tab Ret`, `Up Up Ret`,
`quanti core hai?` + Ret. Serial echo:

```
aios> !mem
RAM total 4095 MiB, free 1728 MiB, used 2367 MiB (...)
aios> /help                 (completed from /he, help printed)
aios> /
  /help /stats /clear /ctx /compact /history /think /temp /bench /fill /gen      <- suggestion after the pause
aios> !kbd                  (completed from !kb, second Tab lists "us it")
keyboard layout it (available: us, it)
aios> /help                 (Up Up: history)
aios> quanti core hai?      (to the agent)
```

### VNC Option key

`tools/vnckeys.py` as a keysym-only client holding `Meta_L` (0xFFE7, what macOS Screen Sharing sends for
Option) and typing `ò à è +` (`HOLD=0xffe7`): the guest got `@#[]`. No "no scancode found for keysym
65511" warning in the QEMU log of that guest (the user's old instance logged it repeatedly). Not tried
with the real macOS app.

### Tool result cap

`Call read_log with n=200 ...` on a log padded with 150 long lines: `[tool] read_log(n=200)`, the result ends
with `[... truncated by the kernel: the result was longer than 8192 bytes]`, and the tool-response prefill
was 6493 tokens (the padded lines are mostly digits, one token each) in 149 s.

### Compaction (scripted over TCP, threshold lowered with `!set compact 3` = 983 tokens)

Conversation: "My name is Cris and my lucky number is 4711. Please remember both.", then 9 short questions,
then "What is my name and what is my lucky number?". Kernel log lines (`!log`):

```
compact #1 auto: 1038->943 tok (3%->2%), summary 25 tok 2984 ms, re-prefill 202 tok 4302 ms, total 7286 ms
compact #2 auto: 1189->968 tok (3%->2%), summary 25 tok 2912 ms, re-prefill 227 tok 4592 ms, total 7504 ms
```

Summary written by the model both times: `User: Cris, 4711, no preference. / Tasks: None. / Tool results: None.`
`/history` keeps the full raw transcript (31 lines) across both compactions, summaries included.

**Recall after compaction failed** in all three variants I tried. The model answered "I don't have
access to personal information like your name or lucky number." Without compaction, the same question
gets "Your name is Cris and your lucky number is 4711."

| Attempt | Where the summary goes | Summary kept the facts? | Answer |
|---|---|---|---|
| 1 | user turn "Summary ... written by you" + assistant "OK, noted." | 1st yes ("Cris, lucky number 4711"), 2nd dropped the name | refused |
| 2 (kept in the build) | appended to the system prompt (snapshot ends before its `<|im_end|>`), "User:" line first | yes, both times | refused |
| 3 | same, facts as "what: value" | yes but noisy ("User: kernel", "User: 8 cores") | refused |

Per the 3-attempt rule I stopped there. The mechanism (fill tracking, summary, rebuild from the snapshot,
re-prefill of the last 2 exchanges, transcript, log, no compaction loop) works; making the 2B model use
the summary does not, yet. Next options: re-prefill verbatim the user turns that state facts ("remember",
"my name is"), keep the summary as the model's own assistant turn, or test with a bigger model.

Timing: about 3 s for the summary (25 tokens) and 47 tok/s re-prefill. At the default 75% (24.5k tokens)
the rebuild re-prefills prefix + summary + up to 3000 tokens of kept exchanges, about 70 s at most, plus
up to 300 summary tokens at about 24 tok/s (decode at that fill). Not measured at 75%: a natural crossing
needs about 24k tokens of conversation (around 10 minutes of prefill).

### Model fetch

- `HF_FILES=config.json SKIP_EXPORT=1 tools/fetch_model.sh`: downloaded config.json from Qwen/Qwen3.5-2B
  with `uvx --from huggingface_hub hf download`, a second run printed `have models/hf/qwen3.5-2b/config.json`.
  The default file list matches the repo listing (`model.safetensors-00001-of-00001.safetensors` + index).
- `HF_DIR=~/coding/qwenbaremetal-mirror/hf2b tools/fetch_model.sh /tmp/aios-export-test.asmbin`: export in
  31 s, sha256 `9cc4697b605aa5dcea5bb2a9cf1be22eee13eccfc12f87c00d897589e8b44e94` (expected value).

## Implementation notes

- A long pasted line used to lose its tail: `poll_input` drained the UART into the 512-byte type-ahead
  ring with no room check, so the Enter could be dropped. Now it stops reading when the ring is nearly
  full and QEMU holds the rest. The line buffer is 16 KiB, so a 6 KB prompt can be pasted.
- `make run-remote` depends on `iso`, so `AIOS_KV=f32` must be set on that command too, or it silently
  rebuilds FP16 (found the hard way).
- A fresh conversation now restores the snapshot and runs 2 forward passes (the system turn's closing
  `<|im_end|>\n`), so it costs 45 - 55 ms instead of 1 - 3 ms.
- Auto-compaction needs more than 2 exchanges since the previous one; without it, a low threshold
  compacted on every turn.

## Open risks

- **Compaction recall** (above): the summary is kept but the 2B model refuses to use it for personal facts.
- **Prefill is token by token**: 30k tokens take 14 minutes, and an 8 KiB tool result can take more than
  2 minutes. Batched prefill (matrix-matrix matmul in the engine) is the fix.
- **8 KiB cap is bytes, not tokens**: digit-heavy text came to about 0.8 tokens per byte (6.5k tokens).
  A token cap (or a cap based on the remaining context) would bound the cost.
- **RoPE above 4096**: positions up to 30k gave coherent text, but no golden reference exists past the
  header's 4096; only FP16 vs FP32 were compared, not against HF at long context.
- **Model behavior**: "how much free memory do you have?" was once answered from memory with an invented
  number (no tool call); it answers English questions in Italian after Italian turns; in a repetitive
  conversation (the same question for the third time, greedy) it closed the turn with no text.
- **Suggestion line**: on serial over TCP it sometimes shows while scripted keys are typed 30 ms apart
  (packet timing); a human typing is not affected in practice.
- Host load from the user's instance (load 4 - 11) is in all numbers; FP16 and FP32 ran under similar load.
