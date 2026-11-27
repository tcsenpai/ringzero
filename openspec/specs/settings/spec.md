# settings Specification

## Purpose
The runtime settings changed with `!set <key> <value>` (and the shortcuts `!kbd`, `!cores`, `/temp`, `/think`, plus the agent's set_setting tool): every key, the values it accepts, its default at boot, and whether it survives a poweroff. One function, shell::set_setting, applies all of them.

## Requirements

### Requirement: Setting keys
`!set` SHALL accept exactly these keys: temperature (alias temp), cores, keyboard (alias kbd), think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed; Tab SHALL complete the 13 main names; any other key SHALL be refused with `error: unknown setting '<key>' (...)` listing every main name, and nothing changes.
Source: kernel/src/shell.rs SETTINGS, set_setting

#### Scenario: Unknown key
- **WHEN** the user sets the key colour, which does not exist
- **THEN** the shell prints `error: unknown setting 'colour' (temperature, cores, keyboard, think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed)`

### Requirement: Applied settings are logged
Every setting that was applied SHALL write its result line to the kernel log; a refused value SHALL not.
Source: kernel/src/shell.rs set_setting

#### Scenario: Log line
- **WHEN** the user types `!set slot 100`
- **THEN** `slot = 100 tokens for the memory slot of a new conversation` appears in `!log`

### Requirement: temperature
`!set temperature T` SHALL accept a number from 0 to 2: 0 selects greedy decoding, above 0 top-p 0.95 sampling at T; other values SHALL be refused with `error: temperature must be a number from 0 to 2`. Default 0 (greedy). Saved in the hibernation image.
Source: kernel/src/shell.rs set_setting; kernel/src/llm.rs TEMP_BITS; kernel/src/hib.rs extra, apply

#### Scenario: Out of range
- **WHEN** the user types `!set temperature 3`
- **THEN** the temperature stays as it was and the error is printed

### Requirement: cores
`!set cores N` SHALL accept 1 up to the number of worker threads online and set the threads used for inference; other values SHALL be refused with `error: cores must be 1..<max>`. Default: every core online. Not saved: each boot uses every core.
Source: kernel/src/shell.rs set_setting; kernel/src/smp.rs start

#### Scenario: Too many cores
- **WHEN** 8 threads are online and the user types `!set cores 9`
- **THEN** the shell prints `error: cores must be 1..8`

### Requirement: keyboard
`!set keyboard L` SHALL accept the layouts us and it; another name SHALL be refused with `error: keyboard layout must be one of: us it`. Default it. Not saved.
Source: kernel/src/shell.rs set_setting; kernel/src/kbd.rs LAYOUTS, LAYOUT

#### Scenario: US layout
- **WHEN** the user types `!set keyboard us`
- **THEN** the shell prints `keyboard layout = us`

### Requirement: compact
`!set compact P` SHALL accept a percent from 1 to 95, the context fill at which a turn compacts the context; other values SHALL be refused with `error: compact must be a percent from 1 to 95`. Default 75. Saved in the hibernation image.
Source: kernel/src/shell.rs set_setting; kernel/src/agent.rs COMPACT_PCT; kernel/src/hib.rs extra, apply

#### Scenario: Zero refused
- **WHEN** the user types `!set compact 0`
- **THEN** the error is printed and the threshold stays 75

### Requirement: snappool
`!set snappool M` SHALL accept 64 MiB up to the pool size and cap the memory usable for snapshots, never below what is in use; other values SHALL be refused with `error: snappool must be 64..<max> MiB`. Default: the whole pool. Saved with the snapshot tree in the hibernation image.
Source: kernel/src/shell.rs set_setting; kernel/src/snap.rs init_root, hib_save, txt, set_cap, max_mib

#### Scenario: Below the minimum
- **WHEN** the user types `!set snappool 32`
- **THEN** the cap does not change and the error is printed

### Requirement: consolidate
`!set consolidate S` SHALL accept any non-negative integer of idle seconds at the prompt before an automatic memory consolidation, 0 meaning only `/consolidate`; a value that is not an integer SHALL be refused. Default 60. Not saved.
Source: kernel/src/shell.rs set_setting; kernel/src/agentd.rs CONS_IDLE_S

#### Scenario: Off
- **WHEN** the user types `!set consolidate 0`
- **THEN** the shell prints `consolidate = 0 s (off: /consolidate only)`

### Requirement: slot
`!set slot N` SHALL accept a token budget from 0 to 1000 for the memory slot put in front of the first message of a conversation, 0 turning the slot off; other values SHALL be refused. Default 200. Not saved.
Source: kernel/src/shell.rs set_setting; kernel/src/agentd.rs SLOT_BUDGET

#### Scenario: Slot off
- **WHEN** the user types `!set slot 0`
- **THEN** the shell prints `slot = 0 tokens (off: no memory slot)`

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

### Requirement: seed
`!set seed R` SHALL set the sampler RNG state to a positive integer R; 0 and non-integers SHALL be refused with `error: seed must be a positive integer`. Default 42. The RNG state is stored with every snapshot point, so `/undo` and `/switch` bring it back.
Source: kernel/src/shell.rs set_setting; kernel/src/llm.rs RNG; kernel/src/snap.rs save

#### Scenario: Zero refused
- **WHEN** the user types `!set seed 0`
- **THEN** the error is printed

### Requirement: Agent can change three settings
The agent's set_setting tool SHALL reach only temperature, cores and keyboard; the other keys SHALL be changeable only by the user at the shell.
Source: kernel/src/agent.rs TOOLS, SPECS, exec

#### Scenario: Agent tries confidence
- **WHEN** the model calls set_setting with key `confidence`
- **THEN** validation refuses the call and confidence does not change

### Requirement: Settings survive hibernation only in part
A resumed hibernation image SHALL restore temperature, think, constrain, compact, the compaction count, the transcript and the snapshot pool cap; every other setting SHALL start at its default on each boot.
Source: kernel/src/hib.rs key, apply; kernel/src/snap.rs hib_save, txt

#### Scenario: Keyboard after resume
- **WHEN** the user sets `!kbd us`, powers off and boots again into the resumed conversation
- **THEN** the keyboard layout is it again

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
