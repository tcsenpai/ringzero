## ADDED Requirements

### Requirement: Role colors
With color on, the shell SHALL print the prompt bold red (`ESC[1;31m`), `[tool]` lines yellow (`ESC[33m`), `[memory]`, `[grammar]`, `[confidence]` and statistics gray (`ESC[90m`), y/N questions and errors bright red (`ESC[91m`), each closed by `ESC[0m`; user input and agent answers SHALL keep the default color.
Source: kernel/src/console.rs sgr; kernel/src/shell.rs show_prompt, confirm, err; kernel/src/agent.rs ask, conf_line; kernel/src/gram.rs finish; kernel/src/llm.rs print_speed

#### Scenario: A tool turn
- **WHEN** the agent calls a tool
- **THEN** the serial bytes of the call line start with `ESC[33m[tool] ` and the statistics line with `ESC[90m[prefill]`

#### Scenario: Refused setting
- **WHEN** the user types `!set think maybe`
- **THEN** the error line is printed in bright red

### Requirement: Color off is plain text
With color off, the shell's own lines (prompt, tool, memory, statistics, questions, errors, the banner) SHALL contain no escape sequences; the escapes that a WASI program prints itself SHALL still reach both consoles.
Source: kernel/src/console.rs sgr, COLOR; kernel/src/wasm.rs out

#### Scenario: jq with color off
- **WHEN** color is off and the user runs `!run jq -C . /demo/data.json`
- **THEN** the prompt has no escape codes and jq's own colors still show
