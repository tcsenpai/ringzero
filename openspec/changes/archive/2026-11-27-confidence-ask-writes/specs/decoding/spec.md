## MODIFIED Requirements

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
