# T39 result: the tool-call grammar as a speculative draft source (2026-11-27)

Inside a tool call most tokens are not the model's choice: once the name is clear, `>`, `\n`, `<parameter=`, the
parameter name, `</parameter>`, `</function>` follow from the format and the tool schema. M4.8b decoded them one
pass each (the prompt lookup rarely drafts there). T39 drafts them from the M4.7a grammar and checks them in the
M4.8b verify pass. 4B q4, Fedora box, QEMU/KVM 8G test guest next to the user's running instance.

## How (kernel/src/gram.rs `canon`, `draft`; llm.rs `generate`)

- From the automaton's state after the last emitted token, `canon` returns the next piece of the canonical layout
  `<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n` when the grammar leaves one way on:
  the rest of `<function=` / `</parameter>` / `</function>` / `<parameter=`, the rest of a tool or parameter name
  once only one name fits what the model wrote (a single unused parameter fits the empty prefix), a unique enum
  completion, one `\n` between parts (none when the model already wrote whitespace there). It stops at the model's
  next choice: which tool, a free or integer value, whether another optional parameter follows.
- `draft` chains up to 8 pieces (stepping a copy of the automaton through each), tokenizes the bytes with the
  model's BPE (`llm::encode_plain`) and returns at most 15 tokens. `</tool_call>` is not drafted: it comes as the
  pick of the row after a fully accepted draft, so the accepted-stop path (state copy back + re-feed) is not hit.
- In `generate`, inside a call (`gram::active()`), the grammar draft goes first; the prompt lookup runs only when
  it is empty. The verify pass, the per-row picks (sampler, `gram::fix`, stop checks) and the rollback are the
  M4.8b ones, so the output does not depend on the draft. Grammar drafts do not move the lookup's draft length and
  back-off gate.
- `!set gramdraft on|off` (default on; needs `!set spec on`). The `[decode]` line adds `grammar A/D` (grammar
  tokens accepted / drafted).
- Self-test (gram.rs, at boot): the pieces from 5 states, for example after `\n<function=ca` ->
  `t>\n<parameter=path>\n`, after `...<parameter=key>\ncor` -> `es\n</parameter>\n<parameter=value>\n`.

## Equality and speed (interleaved)

12 tool prompts (cat, set_setting x3, ls x2, write_file, read_log, run_code, cpuinfo, recall), each on a fresh
conversation (`/clear`), memory slot and consolidation off. Modes, run in the order A, B, C for every prompt:
A `!set spec off`, B spec on + `!set gramdraft off` (M4.8b), C spec on + gramdraft on. Temperature 0: 3 rounds of
A, B, C; temperature 0.7 with `!set seed 4242` before each prompt: 1 round. Compared: the FNV-1a of the emitted
ids of every round (`ids` in the `[decode]` line).

| Check | Result |
|---|---|
| call rounds (the round that writes the tool call), 144 runs | **identical ids in every mode and repetition**, temperature 0 and 0.7 |
| answer rounds (after the tool result) | identical, except where the tool result itself differed: `read_log` returns new log lines on every run (every run differs, in every mode), and `ls /notes` saw `/notes/t39.txt` only after the write prompt of the first round had run (the first A run differs, the 8 later runs of all modes are identical) |

Decode tok/s of the call rounds, temperature 0, mean over the 3 repetitions:

| Prompt | call tokens | A spec off | B lookup only | C lookup + grammar | C / B | grammar accepted / drafted |
|---|---|---|---|---|---|---|
| leggi /notes/todo.txt | 27 | 19.07 | 18.70 | 33.86 | 1.81x | 18/18 |
| imposta la temperatura a 0 | 35 | 19.58 | 19.61 | 34.72 | 1.77x | 27/29 |
| elenca i file in /notes | 25 | 17.59 | 19.33 | 39.61 | 2.05x | 18/18 |
| scrivi in /notes/t39.txt la parola prova | 43 | 19.71 | 20.82 | 35.83 | 1.72x | 27/28 |
| mostra le ultime 5 righe del log | 24 | 18.22 | 18.92 | 33.66 | 1.78x | 17/18 |
| set the keyboard layout to us | 35 | 19.68 | 20.77 | 34.91 | 1.68x | 27/29 |
| usa 4 core per l'inferenza | 35 | 19.62 | 20.33 | 35.65 | 1.75x | 27/29 |
| quanto fa 2**100? usa python | 45 | 19.84 | 23.41 | 33.70 | 1.44x | 38/39 |
| leggi /demo/data.json | 26 | 19.64 | 19.29 | 36.68 | 1.90x | 18/18 |
| quanti core ha la CPU? | 14 | 18.16 | 18.06 | 37.77 | 2.09x | 11/11 |
| cerca nei file qualcosa sul latte | 25 | 18.79 | 19.85 | 42.20 | 2.13x | 18/18 |
| what is on the disk root? | 24 | 19.02 | 19.71 | 40.63 | 2.06x | 18/18 |
| all call rounds | | 19.26 | 20.37 | **35.83** | **1.76x** | 264/275 (96%) |

11 of 275 grammar tokens were rejected (the model wrote something else there; which tokens was not logged), each
rejection costing one DeltaNet state copy back. Answer rounds have no call, so T39 does not change them.

## Tool-call eval (M45 protocol)

`tools/tooleval.py` unchanged, `SLOT=200`, constrain on, temperature 0, fresh copy of the user's disk without
`/sys`, gramdraft on (default):

| | T22/T38 build (docs/T22_MANUAL_RESULT.md) | T39 build |
|---|---|---|
| spike: correct / first call right / args exact | 22/30, 22, 9 | **22/30**, 22, 9 |
| in-OS: correct / first call right | 23/24, 20 | **23/24**, 20 |
| malformed / schema errors / grammar lines not valid | 0 / 0 / 0 | 0 / 0 / 0 |
| prompts with the same calls and arguments | | 54/54 |
| decode tok/s of the rounds that made a call (spike / in-OS) | 26.13 / 28.74 | 35.41 / 41.51 |

## Limits

- Only the structural text is drafted. Free values (paths, file contents, code) still decode one token per pass
  unless the prompt lookup finds them in the conversation, so long `write_file` / `run_code` calls gain less
  (run_code: 1.44x).
- The canonical layout assumes Qwen's one `\n` between parts; when the model writes something else the draft is
  cut there by the verify (a rollback), the output stays the same.
- The tool name itself is not drafted until only one name fits what the model wrote.
