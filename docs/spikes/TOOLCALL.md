# Spike: Qwen3.5-2B tool calling (base model, no fine-tuning)

Run on the Fedora box with Hugging Face transformers on CPU, float32, greedy, 2026-11-26. Data and scripts in `toolcall/`.

## Result

The base model is good enough for a prototype: 24 of 25 prompts that need a tool are fully correct (96%), and it never invented a tool name.

| Set (non-thinking, 30 prompts) | Correct |
|---|---|
| single tool call | 21/22 |
| two-step chains | 3/3 |
| no tool needed | 4/5 |
| arguments valid against the schema | all calls |

Thinking mode on 6 prompts: 5/6 by hand check, about 4 times the output tokens (mean 110 vs 26), no gain. Use non-thinking.

## Format

Tools are listed as JSON lines inside `<tools>` in the system turn. Calls come back as
`<tool_call><function=NAME><parameter=KEY>\nVALUE\n</parameter></function></tool_call>`, several per turn allowed.

## Failures seen

- Missing closing `</tool_call>` once (otherwise a valid call). A lenient parser fixes it.
- Called `run_code` for a translation that needed no tool.
- Answered from the code instead of the tool result (twice, one partly caused by the test's fake results).
- Loose argument values: `Italiano` instead of `it`, relative path `notes.txt`.

## Prefill cost of the system prompt (at about 37 tok/s)

| Option | Tokens | Seconds |
|---|---|---|
| native template, 13 tools | 942 | 25.5 |
| native template, 7 tools | 647 | 17.5 |
| one-line signatures (not tested on the model) | 391 | 10.6 |
| system+tools prefix cached once at boot | user turn only | under 1 |

## Decisions for M3

- Keep the native call format; cache the system+tools state at boot.
- Lenient parser for a missing closer; validate every call against the schema in the kernel.
- Confirmation before poweroff, reboot and recursive rm.
- Fine-tune later on: no tool for chat/math/translation, answering from real tool results, canonical argument values, exact code copy into run_code.

## Limits

30 prompts (6 for thinking), short and clean, fake tool results that ignore arguments, substring-based argument check. Wall-clock times are not meaningful (slow reference attention path in transformers).
