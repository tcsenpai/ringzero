## ADDED Requirements

### Requirement: ringzero data is read-only for programs
The system SHALL refuse with EACCES (2) every WASI write, create, truncate, mkdir, unlink, rmdir or rename whose path is under `/var/lib/ringzero` (case-insensitive), while reads there stay allowed; /proc, /sys (but /sys/power/state) and /dev/vda refuse changes with EROFS (69).
Source: kernel/src/wasm.rs data_ro, path_open, path_op, path_rename

#### Scenario: write to the boot cache
- **WHEN** a program opens `var/lib/ringzero/bootsnap.bin` for writing
- **THEN** path_open returns 2 and the file is unchanged

## REMOVED Requirements

### Requirement: /sys is read-only for programs
**Reason**: /sys becomes the virtual sysfs (EROFS for everyone) and ringzero's files move to /var/lib/ringzero.
**Migration**: the same rule applies to /var/lib/ringzero ("ringzero data is read-only for programs").
