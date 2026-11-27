# proc Specification

## Purpose

The read-only virtual /proc directory (T54, the Unix design principle): which files it has, what they hold, that nothing under it can be changed, and that the shell, the agent tools and WASI programs read it through the same fs.rs path layer.

## Requirements

### Requirement: /proc files
The system SHALL serve a virtual directory /proc with the files cpuinfo, meminfo, uptime, version, devices and bus/pci/devices (directories bus and bus/pci), generated when read, and SHALL list `proc/` in the root directory.
Source: kernel/src/procfs.rs ENTRIES, entry, list, read; kernel/src/fs.rs list

#### Scenario: Listing
- **WHEN** the user types `!ls /proc`
- **THEN** the shell lists bus/, cpuinfo, devices, meminfo, uptime and version with their sizes

#### Scenario: Unknown file
- **WHEN** a program opens /proc/nope
- **THEN** it gets not found (WASI ENOENT)

### Requirement: /proc CPU and memory
cpuinfo SHALL hold one block per core with processor, model name, cpu MHz, cpu cores, role and the CPUID flags under their Linux names; meminfo SHALL hold MemTotal, MemFree, MemAvailable, ModelWeights, SnapshotPool and KernelHeap in kB; uptime SHALL hold the seconds since boot; version SHALL start with `ringzero version` and the package version.
Source: kernel/src/procfs.rs cpuinfo, flags, meminfo, read

#### Scenario: Memory
- **WHEN** the user types `!cat /proc/meminfo`
- **THEN** the lines `MemTotal:`, `MemFree:` and `MemAvailable:` show kB values

#### Scenario: Engine flags
- **WHEN** the CPU has AVX2, FMA and F16C
- **THEN** every flags line of /proc/cpuinfo contains avx2, fma and f16c

### Requirement: /proc drivers and PCI
devices SHALL list the active drivers (serial, keyboard, framebuffer, block, filesystem), and bus/pci/devices SHALL list every function on PCI bus 0 as `bus:dev.fn vendor:device class (code), driver`.
Source: kernel/src/procfs.rs devices, pci, class; kernel/src/blk.rs location; kernel/src/console.rs devices

#### Scenario: Disk driver
- **WHEN** the system booted with a virtio-blk disk
- **THEN** /proc/devices has a `block        virtio-blk: PCI 00:DD.F, N MiB` line and /proc/bus/pci/devices marks that function `virtio-blk`

### Requirement: /proc is read-only
Every change under /proc (write, append, truncate, mkdir, rm, rename, create) SHALL fail with `read-only file system`, and a WASI program SHALL get EROFS (69) for an open with write rights, a create, mkdir, unlink or rename there.
Source: kernel/src/fs.rs ROFS, check_rw; kernel/src/wasm.rs sys_ro, ROFS

#### Scenario: Shell write
- **WHEN** the user types `!write /proc/version x`
- **THEN** the shell prints an error with `read-only file system` and nothing changes

#### Scenario: Program write
- **WHEN** python3 opens /proc/uptime for writing
- **THEN** the open fails with EROFS

### Requirement: One /proc for every reader
The shell (`!ls`, `!cat`), the agent's ls and cat tools and WASI programs SHALL read /proc through the same fs.rs functions; a WASI open of a /proc file SHALL generate the content once and serve read, seek and fstat from those bytes.
Source: kernel/src/fs.rs read, ls, entry, list, proc_read; kernel/src/wasm.rs path_open, fd_read, fd_seek, OpenFile

#### Scenario: Program read
- **WHEN** the user types `!run cat /proc/cpuinfo`
- **THEN** cat prints the same blocks as `!cat /proc/cpuinfo`

#### Scenario: Agent read
- **WHEN** the model calls cat with path /proc/devices
- **THEN** the tool result is the driver list
