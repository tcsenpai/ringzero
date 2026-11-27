# settings Specification

## Purpose
The runtime settings changed with `!set <key> <value>` (and the shortcuts `!kbd`, `!cores`, `/temp`, `/think`, plus the agent's set_setting tool): every key, the values it accepts, its default at boot, and whether it survives a poweroff. One function, shell::set_setting, applies all of them.

## Requirements

### Requirement: Setting keys
`!set` SHALL accept exactly these keys: temperature (alias temp), cores, keyboard (alias kbd), think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed; Tab SHALL complete the 13 main names; any other key SHALL be refused with `error: unknown setting '<key>' (...)` and nothing changes.
Source: kernel/src/shell.rs:103, kernel/src/shell.rs:399-530

#### Scenario: Unknown key
- **WHEN** the user sets the key colour, which does not exist
- **THEN** the shell prints `error: unknown setting 'colour'` and the list in parentheses

### Requirement: Applied settings are logged
Every setting that was applied SHALL write its result line to the kernel log; a refused value SHALL not.
Source: kernel/src/shell.rs:531-534

#### Scenario: Log line
- **WHEN** the user types `!set slot 100`
- **THEN** `slot = 100 tokens for the memory slot of a new conversation` appears in `!log`

### Requirement: temperature
`!set temperature T` SHALL accept a number from 0 to 2: 0 selects greedy decoding, above 0 top-p 0.95 sampling at T; other values SHALL be refused with `error: temperature must be a number from 0 to 2`. Default 0 (greedy). Saved in the hibernation image.
Source: kernel/src/shell.rs:401-411, kernel/src/llm.rs:43, kernel/src/hib.rs:26, kernel/src/hib.rs:39

#### Scenario: Out of range
- **WHEN** the user types `!set temperature 3`
- **THEN** the temperature stays as it was and the error is printed

### Requirement: cores
`!set cores N` SHALL accept 1 up to the number of worker threads online and set the threads used for inference; other values SHALL be refused with `error: cores must be 1..<max>`. Default: every core online. Not saved: each boot uses every core.
Source: kernel/src/shell.rs:412-422, kernel/src/smp.rs:180-182

#### Scenario: Too many cores
- **WHEN** 8 threads are online and the user types `!set cores 9`
- **THEN** the shell prints `error: cores must be 1..8`

### Requirement: keyboard
`!set keyboard L` SHALL accept the layouts us and it; another name SHALL be refused with `error: keyboard layout must be one of: us it`. Default it. Not saved.
Source: kernel/src/shell.rs:423-432, kernel/src/kbd.rs:15-22, kernel/src/kbd.rs:49

#### Scenario: US layout
- **WHEN** the user types `!set keyboard us`
- **THEN** the shell prints `keyboard layout = us`

### Requirement: think
`!set think V` SHALL turn model thinking on for `on`, `1` or `true` and off for any other value. Default off. Saved in the hibernation image.
Source: kernel/src/shell.rs:477-482, kernel/src/llm.rs:42, kernel/src/hib.rs:27, kernel/src/hib.rs:40

#### Scenario: Any other word turns it off
- **WHEN** the user types `!set think yes`
- **THEN** the shell prints `think = off`

### Requirement: compact
`!set compact P` SHALL accept a percent from 1 to 95, the context fill at which a turn compacts the context; other values SHALL be refused with `error: compact must be a percent from 1 to 95`. Default 75. Saved in the hibernation image.
Source: kernel/src/shell.rs:515-525, kernel/src/agent.rs:92, kernel/src/hib.rs:27, kernel/src/hib.rs:42

#### Scenario: Zero refused
- **WHEN** the user types `!set compact 0`
- **THEN** the error is printed and the threshold stays 75

### Requirement: snappool
`!set snappool M` SHALL accept 64 MiB up to the pool size and cap the memory usable for snapshots, never below what is in use; other values SHALL be refused with `error: snappool must be 64..<max> MiB`. Default: the whole pool. Saved with the snapshot tree in the hibernation image.
Source: kernel/src/shell.rs:483-492, kernel/src/snap.rs:546, kernel/src/snap.rs:705, kernel/src/snap.rs:908, kernel/src/snap.rs:1379-1391

#### Scenario: Below the minimum
- **WHEN** the user types `!set snappool 32`
- **THEN** the cap does not change and the error is printed

### Requirement: consolidate
`!set consolidate S` SHALL accept any non-negative integer of idle seconds at the prompt before an automatic memory consolidation, 0 meaning only `/consolidate`; a value that is not an integer SHALL be refused. Default 60. Not saved.
Source: kernel/src/shell.rs:504-514, kernel/src/agentd.rs:48

#### Scenario: Off
- **WHEN** the user types `!set consolidate 0`
- **THEN** the shell prints `consolidate = 0 s (off: /consolidate only)`

### Requirement: slot
`!set slot N` SHALL accept a token budget from 0 to 1000 for the memory slot put in front of the first message of a conversation, 0 turning the slot off; other values SHALL be refused. Default 200. Not saved.
Source: kernel/src/shell.rs:493-503, kernel/src/agentd.rs:32

#### Scenario: Slot off
- **WHEN** the user types `!set slot 0`
- **THEN** the shell prints `slot = 0 tokens (off: no memory slot)`

### Requirement: constrain
`!set constrain V` SHALL make tool calls follow the tool schemas for `on`, `1` or `true`, and for any other value let the grammar only check tool calls. Default on. Saved in the hibernation image.
Source: kernel/src/shell.rs:433-438, kernel/src/gram.rs:259, kernel/src/hib.rs:27, kernel/src/hib.rs:41

#### Scenario: Off
- **WHEN** the user types `!set constrain off`
- **THEN** the shell prints `constrain = off (the grammar only checks tool calls)`

### Requirement: spec
`!set spec V` SHALL turn speculative decoding on for `on` or `true`, set the longest draft to N and turn it on for an integer N from 1 to 15, and turn it off for any other value, including 0 and integers of 16 or more. Default on with drafts up to 8 tokens. When the build has no speculative path the line SHALL say `on, but not available on this build`. Not saved.
Source: kernel/src/shell.rs:460-470, kernel/src/llm.rs:638-643

#### Scenario: Longest draft
- **WHEN** the user types `!set spec 4`
- **THEN** the shell prints `spec = on (prompt lookup, drafts up to 4 tokens)`

#### Scenario: Out of range number turns it off
- **WHEN** the user types `!set spec 16`
- **THEN** the shell prints `spec = off (one token per pass)`

### Requirement: gramdraft
`!set gramdraft V` SHALL use the grammar's forced continuation inside a tool call as the speculative draft for `on`, `1` or `true`, and stop for any other value. Default on. Not saved.
Source: kernel/src/shell.rs:471-476, kernel/src/llm.rs:645

#### Scenario: Off
- **WHEN** the user types `!set gramdraft off`
- **THEN** the shell prints `gramdraft = off (inside a tool call the grammar's continuation is the speculative draft)`

### Requirement: confidence
`!set confidence V` SHALL print a `[confidence]` line per tool call and ask first on low-margin calls of tools that change the machine for `on`, `1` or `true`, and do neither for any other value; the on line SHALL name the ask threshold CONF_LO with two decimals. Default on. Not saved.
Source: kernel/src/shell.rs:439-448, kernel/src/gram.rs:425, kernel/src/agent.rs:587

#### Scenario: On line names the threshold
- **WHEN** the kernel was built with the default CONF_ASK and the user types `!set confidence on`
- **THEN** the line says `a margin below 0.80 asks first on tools that change the machine`

### Requirement: seed
`!set seed R` SHALL set the sampler RNG state to a positive integer R; 0 and non-integers SHALL be refused with `error: seed must be a positive integer`. Default 42. The RNG state is stored with every snapshot point, so `/undo` and `/switch` bring it back.
Source: kernel/src/shell.rs:449-459, kernel/src/llm.rs:44, kernel/src/snap.rs:427

#### Scenario: Zero refused
- **WHEN** the user types `!set seed 0`
- **THEN** the error is printed

### Requirement: Agent can change three settings
The agent's set_setting tool SHALL reach only temperature, cores and keyboard; the other keys SHALL be changeable only by the user at the shell.
Source: kernel/src/agent.rs:30, kernel/src/agent.rs:57, kernel/src/agent.rs:507-512

#### Scenario: Agent tries confidence
- **WHEN** the model calls set_setting with key `confidence`
- **THEN** validation refuses the call and confidence does not change

### Requirement: Settings survive hibernation only in part
A resumed hibernation image SHALL restore temperature, think, constrain, compact, the compaction count, the transcript and the snapshot pool cap; every other setting SHALL start at its default on each boot.
Source: kernel/src/hib.rs:20-46, kernel/src/snap.rs:705, kernel/src/snap.rs:908

#### Scenario: Keyboard after resume
- **WHEN** the user sets `!kbd us`, powers off and boots again into the resumed conversation
- **THEN** the keyboard layout is it again
