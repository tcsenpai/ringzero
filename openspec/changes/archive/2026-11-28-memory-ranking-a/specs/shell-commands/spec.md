## MODIFIED Requirements

### Requirement: Memory commands
The shell SHALL offer `/agentd [log [N]]` (status and the last 10 decisions, or the last N decisions, 30 by default), `/facts`, `/remember KEY VALUE` (source `user`), `/forget KEY` (asks y/N, prints `[facts] kept` on no), `/consolidate`, `/embed [-q] [-v] TEXT`, `/index add [-d YYYY-MM-DD] TEXT|stats|clear|dim N` and `/search QUERY` (top 5 of the ranked answer). Their behaviour is specified by the memory capability.
Source: kernel/src/shell.rs CMDS, run, set_setting, index_cmd, search_cmd

#### Scenario: Forget asks
- **WHEN** the user types `/forget user_name` and answers `n`
- **THEN** the fact stays and the shell prints `[facts] kept`

#### Scenario: Remember without a value
- **WHEN** the user types `/remember name`
- **THEN** the shell prints `usage: /remember KEY VALUE`
