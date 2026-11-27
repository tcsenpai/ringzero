## ADDED Requirements

### Requirement: color
`!set color V` SHALL turn the console role colors on or off with an on/off value; any other value SHALL be refused with `error: color must be on or off`. Default on. Saved in the hibernation image.
Source: kernel/src/shell.rs set_setting; kernel/src/console.rs COLOR; kernel/src/hib.rs extra, apply

#### Scenario: Off
- **WHEN** the user types `!set color off`
- **THEN** the shell prints `color = off (plain text)` and the next prompt has no escape codes

## MODIFIED Requirements

### Requirement: Setting keys
`!set` SHALL accept exactly these keys: temperature (alias temp), cores, keyboard (alias kbd), think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed, color; Tab SHALL complete the 14 main names; any other key SHALL be refused with `error: unknown setting '<key>' (...)` listing every main name, and nothing changes.
Source: kernel/src/shell.rs SETTINGS, set_setting

#### Scenario: Unknown key
- **WHEN** the user sets the key colour, which does not exist
- **THEN** the shell prints `error: unknown setting 'colour' (temperature, cores, keyboard, think, compact, snappool, consolidate, slot, constrain, spec, gramdraft, confidence, seed, color)`

### Requirement: On/off values
The on/off settings (think, constrain, confidence, gramdraft, color) SHALL accept `on`, `1` or `true` for on and `off`, `0` or `false` for off; any other value SHALL be refused with `error: <key> must be on or off`, the setting SHALL keep its value and nothing SHALL be logged.
Source: kernel/src/shell.rs onoff, set_setting

#### Scenario: Typo refused
- **WHEN** think is on and the user types `!set think of`
- **THEN** the shell prints `error: think must be on or off` and think stays on

#### Scenario: Zero means off
- **WHEN** the user types `!set gramdraft 0`
- **THEN** the shell prints `gramdraft = off (inside a tool call the grammar's continuation is the speculative draft)`

### Requirement: Settings survive hibernation only in part
A resumed hibernation image SHALL restore temperature, think, constrain, compact, color, the compaction count, the transcript and the snapshot pool cap; every other setting SHALL start at its default on each boot. An image written before color was saved (extra version 1) SHALL still resume, with color on.
Source: kernel/src/hib.rs key, apply, XVER; kernel/src/snap.rs hib_save, txt

#### Scenario: Keyboard after resume
- **WHEN** the user sets `!kbd us`, powers off and boots again into the resumed conversation
- **THEN** the keyboard layout is it again

#### Scenario: Color after resume
- **WHEN** the user sets `!set color off`, powers off and boots again into the resumed conversation
- **THEN** the prompt has no escape codes
