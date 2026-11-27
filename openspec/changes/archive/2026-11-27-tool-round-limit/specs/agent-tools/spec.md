## ADDED Requirements

### Requirement: Tool round limit
One user turn SHALL run at most ROUNDS tool rounds (default 10, `!set rounds N`, build option ROUNDS), then one more generation with the `<tool_call>` token blocked in the sampler, so the turn ends with an answer.
Source: kernel/src/agent.rs ROUNDS, NO_MORE_TOOLS, ask; kernel/src/llm.rs NO_TOOLS, pick_next

#### Scenario: Limit reached
- **WHEN** rounds is 2 and the model calls a tool in each of the first two rounds
- **THEN** the shell prints `[agent] tool round limit (2) reached, answering without tools`, the second tool result ends with `(No more tools in this turn: answer now with what you have; if you could not find out, say so.)` and the third generation is an answer, not a tool call

## MODIFIED Requirements

### Requirement: Round and length caps
Each generation SHALL stop after 512 tokens, and a tool result longer than 8192 bytes SHALL be cut with the marker `[... truncated by the kernel: the result was longer than 8192 bytes]`; the number of tool rounds is set by the tool round limit.
Source: kernel/src/agent.rs MAX_ANSWER, RES_CAP, ask

#### Scenario: Long result
- **WHEN** a tool returns 20000 bytes
- **THEN** the model sees the first part and the truncation marker
