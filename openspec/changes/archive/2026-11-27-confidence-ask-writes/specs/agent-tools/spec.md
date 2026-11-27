## MODIFIED Requirements

### Requirement: Low-confidence calls ask
When confidence is on and a call's smallest decision margin is below CONF_LO, a call of write_file, mkdir, set_setting or remember SHALL ask `[confidence] the agent is not sure about this call. Run it?`; run_code, run_wasm and the read-only tools SHALL run with a low-confidence notice; rm, forget, poweroff and reboot ask by themselves; thresholds are specified by the decoding capability.
Source: kernel/src/agent.rs ASK_LOW, conf_line, ask

#### Scenario: Unsure write
- **WHEN** a write_file call has a margin of 0.40 with the default build
- **THEN** the `[confidence]` line ends with `-> ask y/N` and the shell asks before writing

#### Scenario: Unsure program run
- **WHEN** a run_wasm call has a margin of 0.60 with the default build
- **THEN** the `[confidence]` line ends with `-> run (low-confidence notice: check the result)` and the program runs without a question
