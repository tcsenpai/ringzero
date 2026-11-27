# decoding Specification

## Purpose
Baseline of how the ringzero kernel decodes Qwen3.5 output for the ring 0 agent: context limits, sampling, speculative decoding (prompt lookup and grammar drafts), the tool-call grammar, and the confidence bands that decide whether a tool call runs, runs with a notice, or asks the human first.

## Requirements

### Requirement: Context window
The system SHALL request a 32768-token context from the engine (`CTX_LEN` as max_seq_override, which replaces the model header's max_seq_len when nonzero). The agent SHALL skip a round with `[agent] context full, type 'clear' to start over` when the next prompt would leave fewer than 64 positions.
Source: kernel/src/llm.rs CTX_LEN, init; engine/transformer.asm .store_scale_mode, .use_hdr_seq; kernel/src/agent.rs ask

#### Scenario: near-full context mid-turn
- **WHEN** the position plus the next prompt's tokens plus 64 reaches max_seq
- **THEN** the round is not run and the shell prints `[agent] context full, type 'clear' to start over`

#### Scenario: new question with no room
- **WHEN** a question arrives and the position plus its length plus 300 reaches max_seq
- **THEN** the agent prints `[agent] context full (P/M), starting a fresh conversation` and answers in a fresh conversation (kernel/src/agent.rs:647-650)

### Requirement: Sampling and temperature
The system SHALL decode greedily (argmax) by default, SHALL accept a temperature from 0 to 2 (`!set temperature T`, `/temp T`, or the set_setting tool), and above 0 SHALL sample top-p 0.95 at that temperature. The sampler RNG SHALL start at 42; `!set seed N` sets it.
Source: kernel/src/llm.rs TEMP_BITS, TOPP_BITS, rate, pick_next; kernel/src/shell.rs set_setting

#### Scenario: temperature 0.7
- **WHEN** the user runs `/temp 0.7`
- **THEN** the shell prints `temperature = 0.7 (top-p 0.95 sampling)` and the next tokens are sampled

#### Scenario: out of range
- **WHEN** the user runs `!set temperature 3`
- **THEN** the setting is unchanged and `error: temperature must be a number from 0 to 2` is printed

### Requirement: Speculative decoding availability
The system SHALL decode with speculation on by default (`SPEC_ON`), off with `!set spec off`, and only when the batched prefill buffers, 16 logits rows and a DeltaNet state copy were taken from the arena at boot; otherwise it SHALL log `model: speculative decoding off (arena full)` and decode one token per pass.
Source: kernel/src/llm.rs SPEC_ROWS, SPEC_OK, st, spec_init, generate; kernel/src/shell.rs set_setting

#### Scenario: spec off
- **WHEN** the user runs `!set spec off`
- **THEN** the shell prints `spec = off (one token per pass)` and the next speed line says `spec off`

### Requirement: Prompt-lookup draft source
Outside a tool call the system SHALL draft from the earlier place in the conversation (after the boot prefix, plus emitted tokens not fed yet) that agrees longest with the end of the context, with matches of 3 to 32 tokens (latest on a tie), drafting at most the match length minus 1 and at most the current draft length.
Source: kernel/src/llm.rs fwd_verify, lookup, generate

#### Scenario: two-token match
- **WHEN** the end of the context agrees with an earlier place for only 2 tokens
- **THEN** no draft is made and the token is decoded alone

#### Scenario: boot prefix excluded
- **WHEN** the end of the context matches only text in the system prompt or tool JSON
- **THEN** no draft is taken from it

### Requirement: Draft length and back-off
The system SHALL use a longest draft of `SPEC_K` tokens (default 8; `!set spec N`, N in 1..15, sets it and turns speculation on). The lookup draft length SHALL grow by 2 after a full accept and shrink by 1 otherwise, and lookup drafting SHALL pause for 2, 4, 8, 16, then 32 tokens after consecutive verifies with nothing accepted.
Source: kernel/src/llm.rs SPEC_K, generate; kernel/src/shell.rs set_setting

#### Scenario: set draft length
- **WHEN** the user runs `!set spec 4`
- **THEN** the shell prints `spec = on (prompt lookup, drafts up to 4 tokens)`

### Requirement: Draft verification and acceptance
The system SHALL verify the pending tokens plus a draft in one batched pass with logits per row, SHALL pick each row as the per-token loop does (sampler, grammar inside a call, stop checks), and SHALL accept draft tokens while the pick equals them. On a partial accept it SHALL restore the saved DeltaNet state and feed the accepted tokens again in the next pass.
Source: kernel/src/llm.rs Stop, generate

#### Scenario: draft rejected at the third token
- **WHEN** a 5-token draft matches the picks for 2 tokens and differs at the third
- **THEN** 2 tokens are accepted, the third row's pick is the next token, and the rollback counter grows by 1

#### Scenario: speed line
- **WHEN** a turn decoded with speculation on
- **THEN** its speed line shows `spec A/D accepted (P%) in V verify, R rollback, verify T ms, grammar GA/GD, ids 0x...` (kernel/src/llm.rs:284-293)

### Requirement: Grammar draft inside tool calls
Inside a tool call, with `GRAM_DRAFT` on (default; `!set gramdraft on|off`), the system SHALL draft first the text the grammar implies, chaining at most 8 pieces and at most 15 tokens, never `</tool_call>`, and SHALL use the prompt lookup only when that draft is empty. With the grammar in check-only mode it SHALL draft nothing.
Source: kernel/src/llm.rs Stop, GRAM_DRAFT, generate; kernel/src/gram.rs unique, draft

#### Scenario: tool name prefix
- **WHEN** the call so far is `\n<function=ca`
- **THEN** the grammar draft is the tokens of `t>\n<parameter=path>\n`

#### Scenario: model's choice
- **WHEN** the next part is a free string value
- **THEN** the grammar drafts nothing there

### Requirement: Tool-call grammar scope
The system SHALL start the tool-call grammar when the model emits the `<tool_call>` token (248058), hide the call's text from the console from that token on, end the grammar at the end of the round, and log one `[grammar] ...` line per call ending in `valid call`, `valid call, steered`, `INVALID call` or `unfinished call`.
Source: kernel/src/llm.rs TOK_TOOL_CALL, emit, generate; kernel/src/gram.rs active, begin, draft, finish

#### Scenario: plain answer
- **WHEN** the model answers without `<tool_call>`
- **THEN** the grammar never runs and no `[grammar]` line is printed

### Requirement: Tool-call grammar structure
The system SHALL allow inside a call only `<function=NAME>`, then `<parameter=KEY>` VALUE `</parameter>` blocks, then `</function>` and the `</tool_call>` token, where NAME is one of the 18 schema tools, each KEY is a parameter of that tool used at most once, every required parameter comes before `</function>`, and at most 4 whitespace bytes separate parts.
Source: kernel/src/gram.rs WS_MAX, push, step; kernel/src/agent.rs SPECS

#### Scenario: unknown tool
- **WHEN** the model writes `<function=mem_info>`
- **THEN** the grammar rejects it and the pick is replaced by an allowed token

#### Scenario: whitespace run
- **WHEN** the model writes 5 newlines between `<function=cat>` and `<parameter=`
- **THEN** the grammar rejects the fifth one

### Requirement: Tool-call grammar values
The system SHALL accept an enum value only from its listed values, an integer only as `-?digits` of at most 18 characters, and a free string value that does not contain `</function>` or `<parameter=`; inside a call no token longer than 128 bytes and no special token except `</tool_call>` after `</function>` SHALL be allowed.
Source: kernel/src/gram.rs SPECIAL_MIN, MAXLEN, step, accept

#### Scenario: enum value outside the schema
- **WHEN** the model writes `colour` as the key of set_setting
- **THEN** the grammar rejects it; only `temperature`, `cores`, `keyboard` are allowed

### Requirement: Grammar masking
The system SHALL keep the sampler's pick when the grammar accepts it; otherwise it SHALL set every rejected token's logit to -inf and pick again with the same sampler, forcing the only allowed token when there is one. With `!set constrain off` the grammar SHALL only report where the call leaves it, without changing any pick.
Source: kernel/src/gram.rs walk, fix; kernel/src/shell.rs set_setting

#### Scenario: constrain off
- **WHEN** constrain is off and the model writes an invalid name
- **THEN** the token is kept and the `[grammar] off (check only)` line reports `call leaves the grammar at token N`

### Requirement: Confidence decision tokens
With confidence on (default; `!set confidence on|off`) the system SHALL treat as decision tokens of a call the tool name, a parameter name when an optional one is a candidate, another optional parameter or the end, an enum value, and an integer digit or its end; never a free string value, layout whitespace or a byte with only one allowed way on.
Source: kernel/src/gram.rs fix, CONF_ON, last_conf, decision; kernel/src/shell.rs set_setting

#### Scenario: free value
- **WHEN** the model writes the path of a cat call
- **THEN** no margin is computed for those tokens

#### Scenario: confidence off
- **WHEN** confidence is off
- **THEN** no `[confidence]` line is printed and no low-confidence question is asked

### Requirement: Confidence margin
At each decision token the system SHALL compute P(best outcome) minus P(second) of a softmax at temperature 1 over the tokens the grammar allows, summing tokens with the same outcome, from the logits the pick used (no extra forward pass), and SHALL take the smallest margin of the call as its confidence.
Source: kernel/src/gram.rs outcome, conf, Conf, last_conf; kernel/src/agent.rs Band, conf_line

#### Scenario: confidence line
- **WHEN** the model calls `read_log` with n=5
- **THEN** the shell and the kernel log show `[confidence] min margin M at 'KIND' ('A' vs 'B'), N decision tokens: ... -> ACTION`

### Requirement: Confidence bands
The system SHALL put a call in the High band when its smallest margin is at or above `CONF_HI` = max(0.80, `CONF_ASK`/100), in the Mid band when it is at or above `CONF_ASK`/100, and in the Low band below that; at the default `CONF_ASK` the Mid band is empty.
Source: kernel/src/agent.rs fs_tool, CONF_HI, conf_ask, conf_line

#### Scenario: margin 0.50 with CONF_ASK=25
- **WHEN** a call's smallest margin is 0.50 and `CONF_ASK` is 25
- **THEN** the call is in the Mid band and its `[confidence]` line ends with `run (middle band, logged)`

#### Scenario: margin 0.50 at the default
- **WHEN** a call's smallest margin is 0.50 and `CONF_ASK` is unset
- **THEN** the call is in the Low band

### Requirement: Agent action per band
The system SHALL run High and Mid band calls. For a Low band call it SHALL ask `[confidence] the agent is not sure about this call. Run it? [y/N]` before set_setting, remember, write_file and mkdir, add no second question for rm, poweroff, reboot and forget (they ask themselves), and run every other tool (run_code, run_wasm and the read-only tools) with a notice in the `[confidence]` line.
Source: kernel/src/agent.rs ASK_LOW, ASKS, conf_line, ask

#### Scenario: low-confidence write
- **WHEN** a write_file call is in the Low band and the user answers `n`
- **THEN** nothing is written and the tool result is the declined text

#### Scenario: low-confidence read
- **WHEN** a cat call is in the Low band
- **THEN** it runs without a question and the line ends `run (low-confidence notice: check the result)`

#### Scenario: low-confidence code run
- **WHEN** a run_code call is in the Low band
- **THEN** it runs without a question and the line ends `run (low-confidence notice: check the result)`

### Requirement: Multi-stream batched decode
The system SHALL be able to run rows of several streams in one engine pass, each row at its own position on its own KV cache and DeltaNet state, so the weights are read once for all of them, and every row's logits SHALL be bit-identical to decoding that stream alone.
Source: engine/prefill.asm transformer_prefill, pf_ms, MS_ROW; kernel/src/multi.rs step

#### Scenario: two streams in one pass
- **WHEN** a foreground decode step runs while agentd's consolidation decodes on an extra stream
- **THEN** one engine pass feeds the foreground token and the background token, and the foreground's token ids are the same as without the background row

### Requirement: Extra stream slots
Up to 3 extra streams of 4096 positions SHALL exist only while used, each in a contiguous run of free snapshot pool chunks lent while at least 256 chunks stay free and given back on close. Lent chunks SHALL count as used and SHALL NOT be saved in snapshots or the hibernation image.
Source: kernel/src/multi.rs SLOT_CAP, open, close; kernel/src/snap.rs lend, take_back

#### Scenario: pool too full to lend
- **WHEN** a slot would leave fewer than 256 free chunks in the pool
- **THEN** the slot is not opened: `/multi` runs fewer streams and says so, and a consolidation runs on top of the current point

## Open questions

- Round limits, the tool result cap, steered and declined calls are specified in agent-tools; the CONF_ASK build variable in build-options.
- docs/M48B_RESULT.md gives MATCH_MIN = 4 for the prompt lookup; the code uses 3 (kernel/src/llm.rs:721), with a comment that a 4-token gate was not better. The doc looks stale.
- That speculative output is identical token for token to one token per pass rests on the engine's batched matmuls being bit-identical (engine asm); the kernel checks it only at run time with `/spec`, not by construction in the Rust code read here.
- The held-out numbers behind the 0.80 default (5 of 8 wrong calls asked, 12 of 90 right ones, AUROC 0.80; `CONF_ASK=25` catches 0 of 8) are in comments and docs/T28_RESULT.md only, not confirmable in code.
- With `!set think on` the generation prompt opens a `<think>` block; how thinking text interacts with the 512-token round limit was not specified here.
- The low-confidence notice for read-only tools appears only in the printed and logged `[confidence]` line; the tool result the model sees carries no marker.
- Compaction (`COMPACT_PCT`, default 75) and call parsing/validation affect context length and tool calls but are specified in the agent-tools spec, not here.
