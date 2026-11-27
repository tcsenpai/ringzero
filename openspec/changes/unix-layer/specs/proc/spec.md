## ADDED Requirements

### Requirement: /proc kernel log and mounts
/proc/kmsg SHALL hold the kernel log ring, oldest line first, as a snapshot taken at read time; /proc/mounts SHALL list one line per mount: `/dev/vda / vfat rw`, `proc /proc proc ro`, `sysfs /sys sysfs ro` and `devfs /dev devfs rw` (the first line only when a disk is mounted).
Source: kernel/src/procfs.rs ENTRIES, read; kernel/src/log.rs

#### Scenario: Mounts with a disk
- **WHEN** the user types `!cat /proc/mounts` on a machine with a disk
- **THEN** the output has the four lines, the first one `/dev/vda / vfat rw`

#### Scenario: Log through a program
- **WHEN** the user types `!run rg disk /proc/kmsg`
- **THEN** rg prints the kernel log lines that contain `disk`

### Requirement: /proc uptime format
/proc/uptime SHALL start with the seconds since boot as a decimal number, followed by ` seconds (H h M min S s)`.
Source: kernel/src/procfs.rs read

#### Scenario: Uptime for a program
- **WHEN** python3 runs `float(open('/proc/uptime').read().split()[0])`
- **THEN** it gets the seconds since boot

## MODIFIED Requirements

### Requirement: /proc files
The system SHALL serve a virtual directory /proc with the files cpuinfo, meminfo, uptime, version, devices, kmsg, mounts and bus/pci/devices (directories bus and bus/pci), generated when read, and SHALL list `proc/` in the root directory.
Source: kernel/src/procfs.rs ENTRIES, entry, list, read; kernel/src/fs.rs list, mount_of

#### Scenario: Listing
- **WHEN** the user types `!ls /proc`
- **THEN** the shell lists bus/, cpuinfo, devices, kmsg, meminfo, mounts, uptime and version with their sizes

#### Scenario: Unknown file
- **WHEN** a program opens /proc/nope
- **THEN** it gets not found (WASI ENOENT)

### Requirement: /proc CPU and memory
cpuinfo SHALL hold one block per core (processor, model name, cpu MHz, cpu cores, role) with the CPUID flags under their Linux names in processor 0 only, the other blocks saying `flags: as processor 0`; meminfo SHALL hold MemTotal, MemFree, MemAvailable, ModelWeights, SnapshotPool and KernelHeap in kB; version SHALL start with `ringzero version` and the package version.
Source: kernel/src/procfs.rs cpuinfo, flags, meminfo, read

#### Scenario: Memory
- **WHEN** the user types `!cat /proc/meminfo`
- **THEN** the lines `MemTotal:`, `MemFree:` and `MemAvailable:` show kB values

#### Scenario: Engine flags
- **WHEN** the CPU has AVX2, FMA and F16C
- **THEN** the flags line of processor 0 in /proc/cpuinfo contains avx2, fma and f16c and the other blocks say `flags: as processor 0`

