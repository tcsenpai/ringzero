## ADDED Requirements

### Requirement: Hidden aliases for the information tools
The validator and the dispatcher SHALL still accept calls to meminfo, cpuinfo, list_cores and uptime, which are not in the prompt, and SHALL answer them with the content of /proc/meminfo, /proc/cpuinfo, the /sys/devices/system/cpu summary and /proc/uptime; each such call SHALL be logged as `[tool] NAME (hidden alias of FILE)`.
Source: kernel/src/agent.rs SPECS, exec

#### Scenario: Old habit
- **WHEN** the model calls meminfo
- **THEN** the tool result is the text of /proc/meminfo and the hidden-alias line is logged

## MODIFIED Requirements

### Requirement: Tool list
The agent SHALL offer exactly 14 tools to the model, in this order: read_log, poweroff, reboot, set_setting, remember, forget, recall, ls, cat, write_file, mkdir, rm, run_code, run_wasm; the JSON schemas in the system prompt and the visible entries of the validator table SHALL list the same tools and parameters.
Source: kernel/src/agent.rs TOOLS, SPECS

#### Scenario: Boot prefix
- **WHEN** the kernel boots
- **THEN** the log line `agent: system prompt + 14 tools = N tokens` is written

### Requirement: Read-only tools
read_log, recall, ls, cat and the hidden aliases SHALL run without a question; read_log SHALL return the last n log lines with n clamped to 1..256 (10 by default); cat SHALL refuse a file that is not text (`not a text file`) and a device other than /dev/null (`is a device`), and return at most 7936 bytes with a truncation note.
Source: kernel/src/agent.rs exec, fs_tool, FX

#### Scenario: read_log clamp
- **WHEN** the model calls read_log with n 1000
- **THEN** 256 lines are returned

#### Scenario: System state from a file
- **WHEN** the user asks how much memory is free and the model calls cat with path /proc/meminfo
- **THEN** the tool result holds the MemFree line
