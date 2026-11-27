# T28 result: uncertainty-aware agent (2026-11-27)

The kernel reads, at the decision tokens of every tool call, how sure the model was, and asks the human before
running a low-confidence call of a tool that changes the machine. The method comes from ProbeLogits (D. Son, 2026,
[arXiv:2604.11943](https://arxiv.org/abs/2604.11943)): a softmax restricted to the tokens the grammar allows, read
from logits the kernel already has, so zero extra forward passes. 4B q4, Fedora box, QEMU/KVM 8G test guest next to
the user's running instance, driven from the Mac over the serial port.

## How (kernel/src/gram.rs `conf`, `margin`, `outcome`; agent.rs `conf_line`)

- Decision tokens: inside a call, a pick whose bytes make one of these choices: the tool name, a parameter name when
  an optional parameter is among the candidates (with only required ones left, their order is not a decision), an
  optional parameter or the end of the call, an enum value (`set_setting key`), an integer digit or the end of the
  number (`read_log n`). Not decisions: layout whitespace, a byte the grammar leaves only one way to write (the rest of
  a name once one fits), free string values (paths, file contents, code).
- Margin: P(best outcome) - P(second outcome) of the softmax over the tokens the grammar allows from the state before
  the token, raw logits (temperature 1, whatever the sampler uses). Tokens are grouped by outcome: tokens that leave
  the same set of names open are summed (`cat` and `ca`), tokens that narrow nothing (`=` before a tool name, `<`
  before `<parameter=` or `</function>`) are left out; integers by the digits written and whether the token ends the
  number; the optional-parameter decision by "another parameter" or "end", however the token spells it.
- Where the logits come from: every pick inside a call goes through `gram::fix` (llm.rs `pick_fixed`), both in a
  single decode step and for each row of the M4.8b/T39 verify pass, where the row's logits have been copied to the
  logits buffer. Decision tokens are never drafted by the T39 grammar (it stops at the model's choice), but the row
  that picks them can be any row of a verify pass; the margin is computed there, from that row. The logits are only
  read: no write, no extra pass, the sampler sees the same buffer.
- Per call: the smallest margin, where it was and the best token of the two outcomes, the first 8 margins and the
  time spent. Printed after the `[tool]` line and written to the kernel log:
  `[confidence] min margin 0.02 at 'tool name' ('run' vs 'cat'), 3 decision tokens: 0.02 0.99 0.99, 512 us -> ...`
- Bands as measured (agent.rs `CONF_LO = 0.25`, `CONF_HI = 0.80`, picked below; the beta asks below 0.80, see
  "Beta default 0.80"): margin >= 0.80 `-> run`; 0.25 to 0.80
  `-> run (middle band, logged)`; below 0.25: a tool that changes the machine (set_setting, remember, write_file,
  mkdir, run_code, run_wasm) asks `[confidence] the agent is not sure about this call. Run it? [y/N]` first, and on
  no the model gets `error: not run: the agent was unsure ...`; rm, forget, poweroff and reboot ask y/N themselves,
  so they get no second question (`-> ask (the tool asks y/N itself)`); a read-only tool runs with
  `-> run (read-only tool, low-confidence notice: check the result)`. The notice was chosen over an automatic recall
  as the simplest behaviour that helps: a read-only call cannot change anything, and the line tells the user which
  other tool the model weighed. The signal only adds questions: the existing y/N of rm, forget, poweroff, reboot and
  the M4.7a rule for steered calls stay as they were.
- `!set confidence on|off` (default on). Off: no margin computed, no line, no question. Not saved in the
  hibernation image (it starts on at every boot).
- Self-test (gram.rs, at boot, part of `tool-call grammar self-test PASS`): which next byte is a decision from 10
  states, and e^x.

## Eval set and thresholds (tools/confeval.py)

240 tasks with a known right first call: the 30 spike prompts (labels rewritten for the kernel's 18 tools), the 24
in-OS and 10 Italian prompts of tools/tooleval.py, and 176 new ones: every tool in Italian and English, settings that
do not exist (hostname, volume, dark mode, resolution), near-ties between two tools (cat or read_log for a log file,
forget or rm, recall or meminfo for "memoria", write_file or mkdir, ls or `run_wasm ls`) and vague requests ("check
the system", "go to sleep"). A call is wrong when its tool is not one of the accepted ones, an argument rule fails
(exact path, integer, enum value, or a substring of a value), or no tool was the only right answer. Split by
position: even index tune (120), odd held-out (120). Protocol: `/clear` per prompt, temperature 0, constrain on,
`SLOT=200`, a fresh copy of the user's disk without `/sys`; Ctrl-C after the first tool result (only the first call
is scored). 240 tasks in 11 minutes.

193 of the 240 tasks made a call (95 tune, 98 held-out); 33 answered without a tool where that was right, 14 made
no call where one was needed (no call, so no margin: the signal cannot see these).

Threshold rule, applied to the tune split only (confeval.py `analyze`): LO = the value on a 0.05 grid with the
largest Youden J (share of wrong calls below LO minus share of right calls below LO), with LO >= 0.2 (numeric noise,
below) and at most 10% of the right calls below it; HI = the largest value >= LO + 0.1 that puts at most 15% of the
right calls between LO and HI. Result: LO 0.25 (tune J 0.30), HI 0.80.

| | tune (95 calls) | held-out (98 calls) |
|---|---|---|
| wrong calls (wrong tool) | 9 (9) | 8 (7) |
| AUROC, margin vs wrong call | 0.84 | **0.80** |
| AUROC, margin vs wrong tool | 0.84 | **0.89** |
| low band (< 0.25): right / wrong | 3 / 3 | **0 / 0** |
| middle band (0.25-0.80): right / wrong | 10 / 3 | **12 / 5** |
| high band (>= 0.80): right / wrong | 73 / 3 | **78 / 3** |
| wrong calls caught before running (low) | 3 of 9 | **0 of 8** |
| false escalations (right calls asked) | 3 (3.2%) | **0** |
| calls logged in the middle band | 13.7% | 17.3% |

On the held-out split the chosen LO catches none of the 8 wrong calls: the wrong held-out calls sit at 0.46 to
1.00. The margin does rank them (AUROC 0.80, wrong tool 0.89), and 5 of the 8 are in the logged band. Held-out
counts for other LO values, for comparison only (not picked on this split):

| LO | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|
| wrong calls below | 0 | 1 | 2 | 2 | 5 | 6 |
| right calls below | 3 | 4 | 7 | 8 | 12 | 16 |

Catching most wrong calls on this set costs asking on about 15% of all calls (LO 0.80); the 10% budget of the rule
keeps LO at 0.25. The wrong calls the model makes with a high margin are confident mistakes the signal cannot see:
`run_wasm(path=/bin/ls)` for "list /demo" (0.94), `recall` for "ricordati che lavoro come insegnante" (0.88),
`set_setting(temperature)` for "slow down the inference" (1.00), `meminfo` for "is the disk full?" (0.86).

Where the smallest margin was: tool name 150 calls, enum value 16, parameter 12, optional parameter 9, integer 6.

Numeric noise: FP16 against FP32 KV moves logits by about 0.02-0.15 (docs/M25_RESULT.md). For two outcomes the margin is
tanh(d/2) of the logit gap d, whose slope is at most 1/2, so a 0.15 shift of each of the two logits moves the margin
by at most 0.15, and by at most 0.075 near a tie. LO 0.25 is over three times that; a true near-tie (margin about 0)
stays in the low band under that noise.

### First version: token-level margin (not kept)

The first build took p1 - p2 over single tokens. On the same 240 tasks it gave AUROC 0.74 (tune) and 0.84 (held-out),
but a third of the low margins were tokenization, not doubt: `=m` against `=` for mkdir (0.10-0.17, 5 calls),
`=re` against `=` for reboot, and `\n` against ` ` after the digits of `read_log n` (0.00 on every read_log in the
second build, which grouped names but not integers). Grouping tokens by outcome removed them; the numbers above are
from the third build.

## Beta default 0.80 (user decision, 2026-11-27)

The beta asks below 0.80 instead of 0.25: catching most wrong calls is worth more questions. From the held-out
table above, a margin below 0.80 holds 5 of the 8 wrong calls and 12 of the 90 right ones (13%). Below that line a
tool that changes the machine asks y/N first and a read-only tool runs with the notice, as before; only the line
moved. Build time: `make iso CONF_ASK=N` (percent integer, clamped to 0..100, env `RINGZERO_CONF_ASK`, agent.rs
`CONF_LO`); `CONF_ASK=25` builds the tune-split pick. `CONF_HI` is max(0.80, `CONF_LO`): at the default both are
0.80, so the logged middle band is empty and every call either runs or asks; with `CONF_ASK=25` the bands are the
ones measured above. One variable instead of two keeps the build option simple; the middle band only logs, so
nothing is lost when it is empty. `!set confidence on` prints the line in use (`a margin below 0.80 asks first`).

## tooleval (M45 protocol, final build)

`tools/tooleval.py`, `SLOT=200`, constrain on, temperature 0, a fresh copy of the user's disk without `/sys` for
each run; the harness answers y to a `[confidence]` question (so the same calls run), n to every other y/N.

| | T39 build (docs/T39_RESULT.md) | T28, confidence on | T28, confidence off |
|---|---|---|---|
| spike: correct / first call right / args exact | 22/30, 22, 9 | **22/30**, 22, 9 | 22/30, 22, 9 |
| in-OS: correct / first call right | 23/24, 20 | **23/24**, 20 | 23/24, 20 |
| malformed / schema errors / grammar lines not valid | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| calls / `[confidence]` lines | 39 + 20 | 59 / 59 | 59 / 0 |

Escalations during the eval: **2** questions, both answered y by the harness. "show me /etc/hostname": after `cat`
and `ls` failed (not found, both in the middle band at 0.39 and 0.35), the third call `run_wasm(path=ls, args=/)` had
0.02 at the tool name (`run` against `ls`). The spike's WebAssembly prompt: the fourth call, `run_code` with 0.18
(`_code` against `_w`). 10 more calls were in the middle band (0.27 to 0.65: `ls`, `cat` or `run_code` against
`run_wasm`, `rm` against `ls`), logged and run.

Live check of the question (same build, memory slot off): "crea il file /notes/x.md con scritto # titolo" ->
`write_file(path=/notes/x.md, content=# titolo)`, `[confidence] min margin 0.16 at 'tool name' ('write' vs 'mkdir')
-> ask y/N`, answered n -> `error: not run: the agent was unsure ...`; the file was not written. The 4B then answered
"Ho creato il file /notes/x.md", which is false (see the limits). The same prompts give other margins with the memory
slot on: the slot text changes the context, so a near-tie in one context can be clear in another (the eval's
"show me /etc/hostname" at 0.02 was 0.76 in a later live run).

## Equality and speed

Equality: the two tooleval runs above (confidence on, then off, each on a fresh disk). Compared: the FNV-1a of the
emitted ids of every round (`ids` in the `[decode]` line).

| Check | Result |
|---|---|
| prompts with identical ids in every round, on vs off | **49 of 54** |
| the 5 others | read_log (4 prompts) and uptime: the call round is identical, the answer round differs because the tool result differs on every run (new log lines, a later uptime), as in T39 |
| calls and arguments, on vs off | identical in all 54 prompts |

The margin is computed from the logits buffer without writing it, so the picks cannot change; the 2 escalated calls
were answered y and ran as with the feature off.

Speed (host load average 7.5-8.5, the user's instance running):

| Measure | confidence on | confidence off |
|---|---|---|
| decode tok/s of the call rounds, tooleval (spike + in-OS) | 34.86 (3357 tokens) | 34.71 (3326 tokens) |
| `/bench` (64 tokens, no call), alternated off, on, off, on, off, on | 21.22, 21.31, 20.33 | 21.57, 21.39, 20.46 |
| margin cost per call (confeval, 193 calls) | 0.74 ms mean, 2.3 ms max, 2.02 decision tokens per call | - |

Outside a call the feature costs nothing (the check runs only inside `gram::fix`); inside a call it adds one or two
vocab-trie walks per decision token, about 0.5 ms in a call round of 1-3 s.

## Limits

- The margin sees only how the model chose among the options the grammar allows. It does not see a free value (a
  wrong path, wrong file content, wrong code), a missing call, or a confident wrong choice. On this set 3 of the 8
  held-out wrong calls had a margin of 0.94 or more.
- 17 wrong calls in 193 is a small sample: the AUROC and the band counts would move by several points with another
  240 tasks. The thresholds fit this model (4B q4), this tool list and this prompt; another model needs a new tune run.
- Partly decided tokens (`re`, which keeps read_log, reboot, recall and remember open) are their own outcome, so
  the model splitting between `re` and `read` shows as a lower margin than its real doubt between tools.
- The margin is computed with a soft-float exp in the kernel: 0.74 ms per call on average (2.3 ms at most), next
  to 1-3 s for a call round.
- A declined call returns an error to the model, but the 4B can still claim it did the thing ("Ho creato il file"
  after a declined write_file). The kernel line and the question are what the user can trust, not the answer text.
- The margin depends on the whole context (memory slot, earlier turns): the same request can be a near-tie in one
  conversation and clear in another.
