# shell-commands Specification

## Purpose
The ringzero shell on COM1 and PS/2: how a typed line is routed to the agent, to a machine command (`!name`) or to a session command (`/name`), and what every command takes and does. Baseline of the code as of the OpenSpec adoption; the command table is `CMDS` in kernel/src/shell.rs and `/help` prints it.

## Requirements

### Requirement: Line routing
The shell SHALL send a line that does not start with `!` or `/` to the agent as one user turn, with a leading `@` and the spaces after it removed, and SHALL treat a line starting with `!` as a machine command and a line starting with `/` as a session command.
Source: kernel/src/shell.rs:713-719, kernel/src/shell.rs:160-166

#### Scenario: Plain text goes to the agent
- **WHEN** the user types `how much free memory do I have`
- **THEN** agent::ask receives that text and no command runs

#### Scenario: The old @ prefix still works
- **WHEN** the user types `@ hello`
- **THEN** the agent receives `hello`

### Requirement: Wrong prefix and unknown command
The shell MUST NOT run a command typed with the wrong prefix; it SHALL print that the name is a machine or a session command and suggest the other prefix, and for a name that is in neither table it SHALL print `unknown command '<line>'. Type /help, or type without <prefix> to ask the agent.`
Source: kernel/src/shell.rs:734-741

#### Scenario: Session command typed with !
- **WHEN** the user types undo with the ! prefix
- **THEN** the shell prints `'undo' is a session command: try /undo` and nothing is undone

#### Scenario: Unknown name
- **WHEN** the user types a slash and a name that is in neither table
- **THEN** the shell prints the unknown command line and the agent is not called

### Requirement: Bare prefix lists commands
A line that is only `!` or `/` SHALL print `commands:` followed by every command of that prefix; at the prompt, a bare `!` or `/` left alone for 400 ms SHALL print the same list as a suggestion.
Source: kernel/src/shell.rs:720-725, kernel/src/shell.rs:286-292, kernel/src/shell.rs:378-380

#### Scenario: Bare slash
- **WHEN** the user types `/` and Enter
- **THEN** the shell prints every session command name with its `/` prefix

### Requirement: Line history and limits
The line editor SHALL recall the last 16 distinct lines with up and down, SHALL accept lines up to 16384 bytes at the prompt (history keeps lines up to 1024 bytes), and SHALL cancel the line on Ctrl-C.
Source: kernel/src/shell.rs:192-194, kernel/src/shell.rs:280-285, kernel/src/shell.rs:315-336, kernel/src/shell.rs:344-356, kernel/src/shell.rs:702

#### Scenario: Ctrl-C cancels the line
- **WHEN** the user types half a line and presses Ctrl-C
- **THEN** the shell prints `^C` and nothing runs

### Requirement: Tab completion
Tab SHALL complete command names and the first argument from the command table: the longest common prefix, a trailing space when one candidate is left, a list when several are.
Source: kernel/src/shell.rs:209-232, kernel/src/shell.rs:240-270

#### Scenario: Tab completes a unique command
- **WHEN** the user types /hib (a prefix of hibernate) and Tab
- **THEN** the line becomes `/hibernate ` with a trailing space

#### Scenario: Tab completes an argument
- **WHEN** the user types `!kbd u` and Tab
- **THEN** the line becomes `!kbd us `

### Requirement: Confirmation questions
A y/N question SHALL confirm only on an answer of `y` or `yes` (any case, spaces trimmed); every other answer, an empty line and Ctrl-C SHALL mean no, and each answer SHALL be recorded in the kernel log as `confirmation: yes` or `confirmation: no`.
Source: kernel/src/shell.rs:387-396

#### Scenario: Enter means no
- **WHEN** a command asks `[y/N]` and the user presses Enter
- **THEN** the answer is no and the action does not run

### Requirement: Hidden test commands
The shell SHALL accept two commands that are not in the table or in `/help`, with either prefix: `toks` prints the boot prefix token ids, and `fault` reads an unmapped address to test the page fault handler (the machine halts).
Source: kernel/src/shell.rs:726-733

#### Scenario: Prefix token dump
- **WHEN** the user types `/toks`
- **THEN** the boot prefix token ids are printed comma separated

### Requirement: Help command
`!help` and `/help` SHALL print the introduction, every machine command under `!  machine commands:` and every session command under `/  session commands:`, each with its help text from the command table, then the editing and snapshot notes.
Source: kernel/src/shell.rs:105, kernel/src/shell.rs:172-188, kernel/src/shell.rs:745

#### Scenario: Help lists both tables
- **WHEN** the user types `/help`
- **THEN** every entry of CMDS is printed once per prefix it has, `/stats` and `!stats` included

### Requirement: Machine information commands
`!mem` SHALL print total and free memory, the model and the engine arena; `!cpu` the CPU model, clock and features; `!uptime` the time since boot; `!cores [N]` the cores online after setting the inference cores to N when N is given; `!log [N]` the last N kernel log lines, 20 when N is missing or not a number.
Source: kernel/src/shell.rs:106-110, kernel/src/shell.rs:751-764

#### Scenario: Default log length
- **WHEN** the user types `!log`
- **THEN** the last 20 kernel log lines are printed

#### Scenario: Cores with an argument
- **WHEN** the user types `!cores 2`
- **THEN** the cores setting is applied as `!set cores 2` and the core list is printed

### Requirement: Keyboard command
`!kbd` without an argument SHALL print the current layout and `(available: us, it)`; `!kbd us` or `!kbd it` SHALL set the layout like `!set keyboard`.
Source: kernel/src/shell.rs:111, kernel/src/shell.rs:807-818

#### Scenario: Show the layout
- **WHEN** the user types `!kbd`
- **THEN** the shell prints `keyboard layout it (available: us, it)` on a fresh boot

### Requirement: Set command
`!set <key> <value>` SHALL apply one setting through the same function the agent's set_setting tool uses and print its result line; the keys are specified by the settings capability.
Source: kernel/src/shell.rs:103, kernel/src/shell.rs:112, kernel/src/shell.rs:807-818

#### Scenario: Set a key
- **WHEN** the user types `!set temperature 0.7`
- **THEN** the shell prints `temperature = 0.7 (top-p 0.95 sampling)`

### Requirement: Park command
`!park off|idle|<usec>` SHALL set worker core parking to off (always spin), idle (spin during a generation, park between requests) or always with that spin window, and SHALL print the mode; without a valid argument it SHALL only print the current mode. The default is idle.
Source: kernel/src/shell.rs:113, kernel/src/shell.rs:819-835, kernel/src/smp.rs:112

#### Scenario: Query parking
- **WHEN** the user types `!park` on a fresh boot
- **THEN** the shell prints `[park] idle (spin during a generation, park between requests)`

### Requirement: Stats command
`!stats` and `/stats` SHALL print one `[stats]` line with threads, parked cores, wake IPIs, arena and heap use, agent context fill against the maximum, KV type and keyboard layout.
Source: kernel/src/shell.rs:114, kernel/src/shell.rs:802-806

#### Scenario: Stats line
- **WHEN** the user types `!stats`
- **THEN** one line starting with `[stats] threads` is printed

### Requirement: File write commands
`!write PATH TEXT` SHALL create or replace the file with TEXT trimmed and `\n` turned into new lines, and queue it for the memory index; `!mkdir PATH` SHALL create the directory and missing parents; `!sync` SHALL flush the disk. Errors SHALL print as `[fs] <cmd>: <error>`.
Source: kernel/src/shell.rs:117-118, kernel/src/shell.rs:121, kernel/src/shell.rs:644-655, kernel/src/shell.rs:682-691

#### Scenario: Write with new lines
- **WHEN** the user types `!write /a.txt one\ntwo`
- **THEN** /a.txt holds two lines and the shell prints `[fs] wrote 7 bytes to /a.txt`

### Requirement: File read commands
`!ls [PATH]` SHALL list a directory (default `/`, `(empty)` for none); `!cat PATH` SHALL print at most the first 64 KiB and say how many bytes were shown; `!df` SHALL print disk size and free space.
Source: kernel/src/shell.rs:115-116, kernel/src/shell.rs:120, kernel/src/shell.rs:631-643, kernel/src/shell.rs:677-681

#### Scenario: Long file
- **WHEN** the user types `!cat` on a 100 KiB file
- **THEN** 65536 bytes are printed and `[fs] first 65536 of 102400 bytes shown`

### Requirement: Delete command asks
`!rm PATH` SHALL refuse a non-empty directory without `-r` (`directory is not empty (use !rm -r)`), SHALL ask y/N naming the file and its size or the directory and its entry count, and SHALL delete only on yes, dropping the file's memory index chunks; on no it SHALL print `[fs] kept PATH`.
Source: kernel/src/shell.rs:119, kernel/src/shell.rs:656-676

#### Scenario: Declined delete
- **WHEN** the user types `!rm /a.txt` and answers `n`
- **THEN** the file stays and the shell prints `[fs] kept /a.txt`

### Requirement: Program commands
`!py CODE` and `!py -f PATH` SHALL run MicroPython; `!wasm PATH|NAME [ARGS] [< FILE]` SHALL run a WASI program, NAME meaning /bin/NAME.wasm, and `!run` SHALL be the same command as `!wasm`. Limits are specified by the programs capability.
Source: kernel/src/shell.rs:122-124, kernel/src/shell.rs:884-885

#### Scenario: run is wasm
- **WHEN** the user types `!run wc -l /demo/notes.txt`
- **THEN** the same code path as `!wasm wc -l /demo/notes.txt` runs

### Requirement: Power commands
`!poweroff` and `!reboot` SHALL act at once, without a question, when typed by the user at the shell; both SHALL shut agentd down and sync the disk first, and `!poweroff` SHALL also write the hibernation image unless `/hibernate discard` turned that off.
Source: kernel/src/shell.rs:125-126, kernel/src/shell.rs:886-887, kernel/src/sys.rs:108-127

#### Scenario: Poweroff from the shell
- **WHEN** the user types `!poweroff`
- **THEN** the machine powers off without a y/N question

### Requirement: Conversation commands
`/clear` SHALL clear the screen and start a new conversation from the boot prefix, keeping the old one in `/tree`; `/ctx` SHALL print the context fill out of the maximum; `/compact` SHALL summarize and rebuild the context now and then print the fill; `/history [N]` SHALL print the last N raw transcript lines, 40 by default, kept across compactions.
Source: kernel/src/shell.rs:127-130, kernel/src/shell.rs:746-750, kernel/src/shell.rs:765-770, kernel/src/agent.rs:188-203

#### Scenario: Clear keeps the old line
- **WHEN** the user types `/clear`
- **THEN** the shell prints `[agent] new conversation from the boot prefix (the old one stays in /tree)`

### Requirement: Session setting shortcuts
`/think on|off` SHALL set the think setting and `/temp <0..2>` SHALL set the temperature setting, both through the settings function.
Source: kernel/src/shell.rs:131-132, kernel/src/shell.rs:807-818

#### Scenario: Greedy shortcut
- **WHEN** the user types `/temp 0`
- **THEN** the shell prints `temperature = 0 (greedy)`

### Requirement: Snapshot commands
The shell SHALL offer `/undo [N]` and `/redo [N]` (N defaults to 1), `/retry`, `/tree [all]`, `/branch NAME`, `/switch NAME|#ID`, `/save NAME`, `/load NAME` (the same code as `/switch`), `/drop NAME|#ID`, `/snap` and `/hibernate [discard|status]`; `/branch`, `/switch`, `/save`, `/load` and `/drop` without an argument SHALL print `usage: /<cmd> NAME`. Their behaviour is specified by the snapshots-hibernation capability.
Source: kernel/src/shell.rs:133-143, kernel/src/shell.rs:872-882

#### Scenario: Missing name
- **WHEN** the user types `/save`
- **THEN** the shell prints `usage: /save NAME`

### Requirement: Retry command
`/retry` SHALL undo the last turn and ask its question again, reseeding the sampler when the temperature is above 0; with no question at the current point it SHALL print `[retry] nothing to retry here (no question at this point)`.
Source: kernel/src/shell.rs:135, kernel/src/agent.rs:135-148

#### Scenario: Retry at temperature 0
- **WHEN** the temperature is 0 and the user types `/retry`
- **THEN** the same question is asked again from the previous point

### Requirement: Memory commands
The shell SHALL offer `/agentd [log [N]]` (status and the last 10 decisions, or the last N decisions, 30 by default), `/facts`, `/remember KEY VALUE` (source `user`), `/forget KEY` (asks y/N, prints `[facts] kept` on no), `/consolidate`, `/embed [-q] [-v] TEXT`, `/index add TEXT|stats|clear|dim N` and `/search QUERY` (top 5). Their behaviour is specified by the memory capability.
Source: kernel/src/shell.rs:144-151, kernel/src/shell.rs:836-866, kernel/src/shell.rs:537-624

#### Scenario: Forget asks
- **WHEN** the user types `/forget user_name` and answers `n`
- **THEN** the fact stays and the shell prints `[facts] kept`

#### Scenario: Remember without a value
- **WHEN** the user types `/remember name`
- **THEN** the shell prints `usage: /remember KEY VALUE`

### Requirement: Manual command
`/man [TOPIC|check]` SHALL search the built-in manual; its behaviour is specified by the manual capability.
Source: kernel/src/shell.rs:152, kernel/src/shell.rs:867-871

#### Scenario: Manual lookup
- **WHEN** the user types `/man undo`
- **THEN** agentd::man runs with the argument `undo`

### Requirement: Benchmark and debug commands
`/bench` SHALL run a 64-token decode bench, `/fill [tokens]` a prefill and decode at that fill (1024 by default), `/prefill [B] [avx2|vnni]` a timed boot prefix prefill, `/spec [N]` a check of N speculative verifier positions (16 by default) and `/gen <n> <text>` greedy ids with logit margins.
Source: kernel/src/shell.rs:153-157, kernel/src/shell.rs:771-801

#### Scenario: gen usage
- **WHEN** the user types `/gen 5`
- **THEN** the shell prints `usage: /gen <max tokens> <prompt text>`

### Requirement: Benches restore the conversation
Each bench and debug command SHALL run from position 0 and put the conversation back afterwards.
Source: kernel/src/shell.rs:771-801, kernel/src/agent.rs:127-133

#### Scenario: Conversation restored after a bench
- **WHEN** the user types `/bench` in the middle of a conversation
- **THEN** after the bench the shell prints `[snap] conversation restored (ctx N)` with the same N as before
