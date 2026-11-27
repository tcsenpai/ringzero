## ADDED Requirements

### Requirement: Information tools read the files
meminfo, cpuinfo and uptime SHALL answer with the content of /proc/meminfo, /proc/cpuinfo and /proc/uptime, and list_cores with the `online` and `inference` lines of /sys/devices/system/cpu and one `cpuN: ROLE` line per core; their descriptions in the prompt SHALL name the file.
Source: kernel/src/agent.rs TOOLS, FILES, exec; kernel/src/sysfs.rs read, role

#### Scenario: Memory
- **WHEN** the model calls meminfo
- **THEN** the tool result is the text of /proc/meminfo

## MODIFIED Requirements

### Requirement: Read-only tools
meminfo, cpuinfo, list_cores, read_log, uptime, recall, ls and cat SHALL run without a question; read_log SHALL return the last n log lines with n clamped to 1..256 (10 by default); cat SHALL refuse a file that is not text (`not a text file`) and a device other than /dev/null (`is a device`), and return at most 7936 bytes with a truncation note.
Source: kernel/src/agent.rs exec, fs_tool, FX

#### Scenario: read_log clamp
- **WHEN** the model calls read_log with n 1000
- **THEN** 256 lines are returned

#### Scenario: System state from a file
- **WHEN** the user asks how much memory is free and the model calls cat with path /proc/meminfo
- **THEN** the tool result holds the MemFree line
