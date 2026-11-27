## ADDED Requirements

### Requirement: On/off values
The on/off settings (think, constrain, confidence, gramdraft) SHALL accept `on`, `1` or `true` for on and `off`, `0` or `false` for off; any other value SHALL be refused with `error: <key> must be on or off`, the setting SHALL keep its value and nothing SHALL be logged.
Source: kernel/src/shell.rs onoff, set_setting

#### Scenario: Typo refused
- **WHEN** think is on and the user types `!set think of`
- **THEN** the shell prints `error: think must be on or off` and think stays on

#### Scenario: Zero means off
- **WHEN** the user types `!set gramdraft 0`
- **THEN** the shell prints `gramdraft = off (inside a tool call the grammar's continuation is the speculative draft)`

### Requirement: think values
`!set think V` SHALL turn model thinking on or off with an on/off value; any other value SHALL be refused. Default off. Saved in the hibernation image.
Source: kernel/src/shell.rs set_setting; kernel/src/llm.rs THINK; kernel/src/hib.rs extra, apply

#### Scenario: Another word is refused
- **WHEN** the user types `!set think yes`
- **THEN** the shell prints `error: think must be on or off`

### Requirement: spec values
`!set spec V` SHALL turn speculative decoding on for `on` or `true`, on with the longest draft N for an integer N from 1 to 15, and off for `off` or `false`; any other value (0, 16, a word) SHALL be refused and change nothing. Default on, drafts up to 8 tokens. Without a speculative path the line SHALL say `on, but not available on this build`. Not saved.
Source: kernel/src/shell.rs set_setting; kernel/src/llm.rs SPEC_ROWS, SPEC_K

#### Scenario: Longest draft
- **WHEN** the user types `!set spec 4`
- **THEN** the shell prints `spec = on (prompt lookup, drafts up to 4 tokens)`

#### Scenario: Out of range number is refused
- **WHEN** the user types `!set spec 16`
- **THEN** the shell prints `error: spec must be on, off or the longest draft from 1 to 15` and speculation stays as it was

## MODIFIED Requirements

### Requirement: Setting keys
`!set` SHALL accept exactly these keys: temperature (alias temp), cores, keyboard (alias kbd), think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed; Tab SHALL complete the 13 main names; any other key SHALL be refused with `error: unknown setting '<key>' (...)` listing every main name, and nothing changes.
Source: kernel/src/shell.rs SETTINGS, set_setting

#### Scenario: Unknown key
- **WHEN** the user sets the key colour, which does not exist
- **THEN** the shell prints `error: unknown setting 'colour' (temperature, cores, keyboard, think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed)`

### Requirement: constrain
`!set constrain V` SHALL make tool calls follow the tool schemas when on, and let the grammar only check tool calls when off; values that are not on/off SHALL be refused. Default on. Saved in the hibernation image.
Source: kernel/src/shell.rs set_setting; kernel/src/gram.rs ON; kernel/src/hib.rs extra, apply

#### Scenario: Off
- **WHEN** the user types `!set constrain off`
- **THEN** the shell prints `constrain = off (the grammar only checks tool calls)`

### Requirement: gramdraft
`!set gramdraft V` SHALL use the grammar's forced continuation inside a tool call as the speculative draft when on and stop when off; values that are not on/off SHALL be refused. Default on. Not saved.
Source: kernel/src/shell.rs set_setting; kernel/src/llm.rs GRAM_DRAFT

#### Scenario: Off
- **WHEN** the user types `!set gramdraft off`
- **THEN** the shell prints `gramdraft = off (inside a tool call the grammar's continuation is the speculative draft)`

### Requirement: confidence
`!set confidence V` SHALL print a `[confidence]` line per tool call and ask first on low-margin calls of tools that change the machine when on, and do neither when off; values that are not on/off SHALL be refused; the on line SHALL name the ask threshold CONF_LO with two decimals. Default on. Not saved.
Source: kernel/src/shell.rs set_setting; kernel/src/gram.rs CONF_ON; kernel/src/agent.rs CONF_LO

#### Scenario: On line names the threshold
- **WHEN** the kernel was built with the default CONF_ASK and the user types `!set confidence on`
- **THEN** the line says `a margin below 0.80 asks first on tools that change the machine`

## REMOVED Requirements

### Requirement: think
**Reason**: any other value no longer turns thinking off; replaced by "think values".
**Migration**: use `!set think off`.

### Requirement: spec
**Reason**: 0 and out-of-range numbers are refused now instead of turning speculation off; replaced by "spec values".
**Migration**: use `!set spec off`.
