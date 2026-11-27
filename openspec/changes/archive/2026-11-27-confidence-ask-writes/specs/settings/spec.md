## MODIFIED Requirements

### Requirement: confidence
`!set confidence V` SHALL print a `[confidence]` line per tool call and ask first on low-margin calls of write_file, mkdir, set_setting and remember when on, and do neither when off; values that are not on/off SHALL be refused; the on line SHALL name the ask threshold CONF_LO with two decimals and the tools that ask. Default on. Not saved.
Source: kernel/src/shell.rs set_setting; kernel/src/gram.rs CONF_ON; kernel/src/agent.rs CONF_LO, ASK_LOW

#### Scenario: On line names the threshold
- **WHEN** the kernel was built with the default CONF_ASK and the user types `!set confidence on`
- **THEN** the line says `a margin below 0.80 asks first on write_file, mkdir, set_setting and remember`
