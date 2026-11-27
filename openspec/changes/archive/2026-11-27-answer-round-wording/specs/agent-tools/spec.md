## MODIFIED Requirements

### Requirement: Tool round limit
One user turn SHALL run at most ROUNDS tool rounds (default 10, `!set rounds N`, build option ROUNDS), then one more generation with the `<tool_call>` token blocked in the sampler, so the turn ends with an answer.
Source: kernel/src/agent.rs ROUNDS, NO_MORE_TOOLS, ask; kernel/src/llm.rs NO_TOOLS, pick_next

#### Scenario: Limit reached
- **WHEN** rounds is 2 and the model calls a tool in each of the first two rounds
- **THEN** the shell prints `[agent] tool round limit (2) reached, answering without tools`, the second tool result ends with `(Tools are now disabled for this turn. Write your final answer to the user from the results above, do not say that you will call a tool; if something is missing, say what you could not find.)` and the third generation is an answer, not a tool call
