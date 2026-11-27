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


### Requirement: /proc drivers and PCI
devices SHALL list the active drivers (serial, keyboard, framebuffer, block, filesystem), and bus/pci/devices SHALL list every function on PCI bus 0 as `bus:dev.fn vendor:device class (code), driver`.
Source: kernel/src/procfs.rs devices, pci, pci_scan, class; kernel/src/blk.rs location; kernel/src/console.rs devices

#### Scenario: Disk driver
- **WHEN** the system booted with a virtio-blk disk
- **THEN** /proc/devices has a `block        virtio-blk: PCI 00:DD.F, N MiB` line and /proc/bus/pci/devices marks that function `virtio-blk`

### Requirement: /proc is read-only
Every change under /proc (write, append, truncate, mkdir, rm, rename, create) SHALL fail with `read-only file system`, and a WASI program SHALL get EROFS (69) for an open with write rights, a create, mkdir, unlink or rename there.
Source: kernel/src/fs.rs ROFS, mount_of, check_rw; kernel/src/wasm.rs data_ro, ROFS

#### Scenario: Shell write
- **WHEN** the user types `!write /proc/version x`
- **THEN** the shell prints an error with `read-only file system` and nothing changes

#### Scenario: Program write
- **WHEN** python3 opens /proc/uptime for writing
- **THEN** the open fails with EROFS

### Requirement: One /proc for every reader
The shell (`!ls`, `!cat`), the agent's ls and cat tools and WASI programs SHALL read /proc through the same fs.rs functions; a WASI open of a /proc file SHALL generate the content once and serve read, seek and fstat from those bytes.
Source: kernel/src/fs.rs read, ls, entry, list, virt_read, tree_entry, tree_list; kernel/src/wasm.rs path_open, fd_read, fd_seek, OpenFile

#### Scenario: Program read
- **WHEN** the user types `!run cat /proc/cpuinfo`
- **THEN** cat prints the same blocks as `!cat /proc/cpuinfo`

#### Scenario: Agent read
- **WHEN** the model calls cat with path /proc/devices
- **THEN** the tool result is the driver list
