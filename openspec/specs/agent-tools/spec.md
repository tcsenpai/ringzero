# agent-tools Specification

## Purpose
The system agent's turn loop and its typed kernel tools: which tools exist, their parameters, how a call is parsed, validated and run, which calls ask the human y/N first, and the caps on rounds, answers and results. The tool list is `TOOLS` and `SPECS` in kernel/src/agent.rs.

## Requirements

### Requirement: Tool list
The agent SHALL offer exactly 18 tools to the model, in this order: meminfo, cpuinfo, list_cores, read_log, uptime, poweroff, reboot, set_setting, remember, forget, recall, ls, cat, write_file, mkdir, rm, run_code, run_wasm; the JSON schemas in the system prompt and the validator table SHALL list the same tools and parameters.
Source: kernel/src/agent.rs TOOLS, SPECS

#### Scenario: Boot prefix
- **WHEN** the kernel boots
- **THEN** the log line `agent: system prompt + 18 tools = N tokens` is written

### Requirement: Tool parameters
The tools SHALL take these parameters (required unless marked optional): read_log n (integer, optional); set_setting key (one of temperature, cores, keyboard) and value; remember key and value; forget key; recall query; ls, cat, mkdir and rm path; write_file path and content; run_code code; run_wasm path and args (optional); the other tools take none.
Source: kernel/src/agent.rs SPECS

#### Scenario: Agent setting keys are limited
- **WHEN** the model calls set_setting with key `spec`
- **THEN** the call is refused with `error: bad value 'spec' for set_setting.key` and nothing changes

### Requirement: Lenient call parsing
The agent SHALL parse the first `<tool_call>` after the last `</think>` in the model's output, accept missing closing `</parameter>`, `</function>` and `</tool_call>` tags, accept at most 4 parameters, and report `missing <function=NAME>`, `unterminated <function=NAME>`, `unterminated <parameter=KEY>` or `too many parameters` as a malformed call fed back to the model.
Source: kernel/src/agent.rs Call, parse, ask

#### Scenario: Missing closing tags
- **WHEN** the model writes `<tool_call>\n<function=meminfo>\n</function>` and stops
- **THEN** meminfo runs

### Requirement: Schema validation
Before a tool runs the agent SHALL check the call against the tool table and feed back, without running anything, `error: unknown tool '<name>'`, `error: <tool> has no parameter '<p>'`, `error: bad value '<v>' for <tool>.<p>` (an integer parameter that does not parse, or a value outside an enum) or `error: <tool> needs parameter '<p>'`.
Source: kernel/src/agent.rs parse, validate

#### Scenario: Missing required parameter
- **WHEN** the model calls set_setting with only key `cores`
- **THEN** the tool response is `error: set_setting needs parameter 'value'`

### Requirement: Boot self test of parser and validator
The kernel SHALL run a self test of the parser and the validator at boot, without the model, covering a full call, a call without closing tags, text without a call, unterminated parameters and two invalid set_setting calls.
Source: kernel/src/agent.rs ask, selftest

#### Scenario: Self test passes
- **WHEN** the kernel boots with an unchanged agent.rs
- **THEN** agent::selftest returns true

### Requirement: Round and length caps
Each generation SHALL stop after 512 tokens, and a tool result longer than 8192 bytes SHALL be cut with the marker `[... truncated by the kernel: the result was longer than 8192 bytes]`; the number of tool rounds is set by the tool round limit.
Source: kernel/src/agent.rs MAX_ANSWER, RES_CAP, ask

#### Scenario: Long result
- **WHEN** a tool returns 20000 bytes
- **THEN** the model sees the first part and the truncation marker

### Requirement: Read-only tools
meminfo, cpuinfo, list_cores, uptime, read_log, recall, ls and cat SHALL run without a question; read_log SHALL return the last n log lines with n clamped to 1..256 (10 by default); cat SHALL refuse a file that is not text (`not a text file`) and return at most 7936 bytes with a truncation note.
Source: kernel/src/agent.rs exec, fs_tool, FX

#### Scenario: read_log clamp
- **WHEN** the model calls read_log with n 1000
- **THEN** 256 lines are returned

### Requirement: Tools that change the machine
set_setting, remember, forget, write_file, mkdir, rm, poweroff, reboot, run_code and run_wasm SHALL be treated as tools that change the machine, and each such call that ran SHALL be listed in the turn's effects, which `/undo` and an interrupted turn report but cannot revert.
Source: kernel/src/agent.rs arg, fs_tool, FX, ask

#### Scenario: Interrupted turn keeps effects
- **WHEN** the agent wrote a file and the user presses Ctrl-C before the answer
- **THEN** the turn is rolled back and the shell prints `[agent] note: this turn ran write_file /PATH; the machine keeps it.`

### Requirement: Tools that always ask
rm, poweroff, reboot and forget SHALL ask the human y/N themselves before acting; rm SHALL remove only files and empty directories, and forget of a key that is not saved SHALL return the error listing the saved keys without asking.
Source: kernel/src/agent.rs exec, fs_tool, ASKS

#### Scenario: Agent poweroff
- **WHEN** the model calls poweroff
- **THEN** the shell asks `[agent] wants to POWER OFF the machine. Proceed? [y/N]` and the machine powers off only on yes

#### Scenario: Agent rm of a full directory
- **WHEN** the model calls rm on a directory with entries
- **THEN** the result is `error: directory is not empty; the agent removes only files and empty directories` and no question is asked

### Requirement: Declined call result
When the human answers no to a tool question, the tool response SHALL start with `NOT RUN: the user declined this call. Nothing was changed. Do not call it again; tell the user it was not done.`
Source: kernel/src/agent.rs fs_tool, DECLINED, exec, ask

#### Scenario: Declined rm
- **WHEN** the model calls rm on /a.txt and the user answers `n`
- **THEN** the tool response is the NOT RUN text followed by `/a.txt is kept.`

### Requirement: Ask again after a decline
After a declined call in a turn, every later call in the same turn of a tool that changes the machine and does not ask by itself SHALL ask `[agent] you declined a call in this turn; the agent calls a tool again. Run it?` before it runs.
Source: kernel/src/agent.rs ask

#### Scenario: Retried mkdir after a no
- **WHEN** the user declined a write_file and the model then calls mkdir in the same turn
- **THEN** the shell asks y/N before mkdir runs

### Requirement: Steered calls do not run
When the grammar had to replace a tool name, parameter or value that was not in the schema, a call of a tool that changes the machine SHALL NOT run and SHALL return `error: not run: a tool name, parameter or value was not in the schema and the kernel had to replace it. ...`; read-only tools SHALL run.
Source: kernel/src/agent.rs ask; kernel/src/gram.rs last_steered

#### Scenario: Steered write
- **WHEN** constrain is on and the grammar replaced an invalid parameter of write_file
- **THEN** no file is written and the model gets the not run error

### Requirement: Low-confidence calls ask
When confidence is on and a call's smallest decision margin is below CONF_LO, a call of write_file, mkdir, set_setting or remember SHALL ask `[confidence] the agent is not sure about this call. Run it?`; run_code, run_wasm and the read-only tools SHALL run with a low-confidence notice; rm, forget, poweroff and reboot ask by themselves; thresholds are specified by the decoding capability.
Source: kernel/src/agent.rs ASK_LOW, conf_line, ask

#### Scenario: Unsure write
- **WHEN** a write_file call has a margin of 0.40 with the default build
- **THEN** the `[confidence]` line ends with `-> ask y/N` and the shell asks before writing

#### Scenario: Unsure program run
- **WHEN** a run_wasm call has a margin of 0.60 with the default build
- **THEN** the `[confidence]` line ends with `-> run (low-confidence notice: check the result)` and the program runs without a question

### Requirement: Agent remember is grounded
The remember tool SHALL save a fact with source `agent` only when the value is something the user said in this session, and SHALL return `error: '<value>' is not something the user said in this session` otherwise; `/remember` at the shell SHALL not need grounding.
Source: kernel/src/agent.rs exec; kernel/src/agentd.rs remember; kernel/src/shell.rs run

#### Scenario: Invented value
- **WHEN** the model calls remember user_city = Rome and the user never wrote Rome
- **THEN** no fact is saved and the error is fed back

### Requirement: Program tool budgets
run_code SHALL run MicroPython with a 5 s budget and run_wasm a /bin program with a 10 s budget, both returning what the program printed; when run_wasm's path holds a whole command line, the words after the first SHALL be used as arguments, and a repeated program name at the start of args SHALL be dropped.
Source: kernel/src/upy.rs SHELL_MS, TOOL_MS, tool; kernel/src/wasm.rs SHELL_MS, TOOL_MS, tool

#### Scenario: Whole command in path
- **WHEN** the model calls run_wasm with path `seq 1 3`
- **THEN** /bin/seq runs with arguments 1 and 3

### Requirement: Interrupted turn is atomic
Ctrl-C during a turn SHALL roll the conversation back to the point the turn started from and print `^C [agent] interrupted, turn rolled back (ctx N)`.
Source: kernel/src/agent.rs ask

#### Scenario: Ctrl-C mid answer
- **WHEN** the user presses Ctrl-C while the agent writes
- **THEN** the context returns to the position before the question

### Requirement: Context compaction
When a turn starts with the fill at or above COMPACT_PCT (75 by default) and more than 2 exchanges, the agent SHALL compact: the model writes a summary of at most 300 tokens, the boot prefix is restored, and the summary plus the last exchanges that fit in 3000 tokens (at most 2) are prefilled again; when there is no room to summarize it SHALL start a fresh conversation.
Source: kernel/src/agent.rs SUMMARY_MAX, COMPACT_PCT, compact, ask

#### Scenario: Auto compaction
- **WHEN** the context is 80% full and the user asks a question
- **THEN** the shell prints `[compact] context ... (80%, auto): summarizing...` before the answer

### Requirement: Answer language
The system prompt SHALL tell the model to always answer in the language the user wrote their latest message in, never in the language of tool output or memory notes, briefly, in plain text without markdown headers, and never to invent numbers.
Source: kernel/src/agent.rs SYSTEM

#### Scenario: Italian question
- **WHEN** the user writes in Italian
- **THEN** the system prompt asks for the language of the user's message, without naming one

### Requirement: Answer language after a tool result
Every tool-result turn SHALL end with "(Answer in the language of my message.)", naming no language, and consolidation SHALL write episodes in the language the user wrote in.
Source: kernel/src/agent.rs tool-result turn in the tool loop; kernel/src/agentd.rs CONS_ASK

#### Scenario: English question answered after a tool result
- **WHEN** the user writes in English and the agent calls a tool
- **THEN** the turn with the tool result asks for the language of the user's message

### Requirement: Tool round limit
One user turn SHALL run at most ROUNDS tool rounds (default 10, `!set rounds N`, build option ROUNDS), then one more generation with the `<tool_call>` token blocked in the sampler, so the turn ends with an answer.
Source: kernel/src/agent.rs ROUNDS, NO_MORE_TOOLS, ask; kernel/src/llm.rs NO_TOOLS, pick_next

#### Scenario: Limit reached
- **WHEN** rounds is 2 and the model calls a tool in each of the first two rounds
- **THEN** the shell prints `[agent] tool round limit (2) reached, answering without tools`, the second tool result ends with `(No more tools in this turn: answer now with what you have; if you could not find out, say so.)` and the third generation is an answer, not a tool call

### Requirement: run_code is described as a sandbox
The run_code tool description SHALL say that the code runs in an isolated MicroPython sandbox with no os module, no filesystem, no network and no imports from disk, and that files and programs go through ls, cat, write_file or run_wasm; the system prompt SHALL say that run_code is a MicroPython sandbox without files and name ls, cat and run_wasm for files and programs.
Source: kernel/src/agent.rs SYSTEM, TOOLS

#### Scenario: File question
- **WHEN** the user asks what is in a file on the disk
- **THEN** the boot prefix tells the model that run_code cannot read files and that cat can
