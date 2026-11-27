## ADDED Requirements

### Requirement: rounds
`!set rounds N` SHALL set the tool rounds per question to an integer N from 1 to 20 and print `rounds = N (tool calls per question, then the agent answers without tools)`; any other value SHALL be refused with `error: rounds must be an integer from 1 to 20` and change nothing. Default 10, or the build option ROUNDS. Saved in the hibernation image.
Source: kernel/src/shell.rs set_setting; kernel/src/agent.rs ROUNDS, ROUNDS_MAX; kernel/src/hib.rs extra, apply

#### Scenario: Set
- **WHEN** the user types `!set rounds 4`
- **THEN** the shell prints `rounds = 4 (tool calls per question, then the agent answers without tools)`

#### Scenario: Out of range
- **WHEN** the user types `!set rounds 0` or `!set rounds 21` or `!set rounds five`
- **THEN** the shell prints `error: rounds must be an integer from 1 to 20` and the limit stays as it was

## MODIFIED Requirements

### Requirement: Setting keys
`!set` SHALL accept exactly these keys: temperature (alias temp), cores, keyboard (alias kbd), think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed, color, rounds; Tab SHALL complete the 15 main names; any other key SHALL be refused with `error: unknown setting '<key>' (...)` listing every main name, and nothing changes.
Source: kernel/src/shell.rs SETTINGS, set_setting

#### Scenario: Unknown key
- **WHEN** the user sets the key colour, which does not exist
- **THEN** the shell prints `error: unknown setting 'colour' (temperature, cores, keyboard, think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed, color, rounds)`

### Requirement: Settings survive hibernation only in part
A resumed hibernation image SHALL restore temperature, think, constrain, compact, color, rounds, the compaction count, the transcript and the snapshot pool cap; every other setting SHALL start at its default on each boot. An image written before color was saved (extra version 1) SHALL still resume, with color on, and one written before rounds was saved (extra version 2) SHALL resume with the build default of rounds.
Source: kernel/src/hib.rs key, apply, XVER; kernel/src/snap.rs hib_save, txt

#### Scenario: Keyboard after resume
- **WHEN** the user sets `!kbd us`, powers off and boots again into the resumed conversation
- **THEN** the keyboard layout is it again

#### Scenario: Color after resume
- **WHEN** the user sets `!set color off`, powers off and boots again into the resumed conversation
- **THEN** the prompt has no escape codes

#### Scenario: Rounds after resume
- **WHEN** the user sets `!set rounds 3`, powers off and boots again into the resumed conversation
- **THEN** the tool round limit is 3
