## Why
Testing the tool round limit on the 4B q4 (T54) with `!set rounds 1` and a request for three reads: after the
first read the answer round wrote "I'll call uptime and meminfo." over and over until the 512-token cap, because
the line on the last tool result ("No more tools in this turn: answer now ...") did not stop it from announcing
calls. With a line that says the tools are disabled and asks for the final answer, the same request ended with
an answer that named what it could not read.

## What Changes
- The line on the last tool result before the answer round becomes: "(Tools are now disabled for this turn. Write
  your final answer to the user from the results above, do not say that you will call a tool; if something is
  missing, say what you could not find.)"

## Impact
- kernel/src/agent.rs NO_MORE_TOOLS. Specs: agent-tools (tool round limit scenario). Results: docs/T54_RESULT.md.
