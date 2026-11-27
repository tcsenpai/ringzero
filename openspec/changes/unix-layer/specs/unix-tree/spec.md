## ADDED Requirements

### Requirement: One file tree for every reader
The system SHALL route every path through one mount table in fs.rs (/proc, /sys and /dev virtual, /etc/os-release and /etc/passwd virtual, everything else the FAT volume), so the shell (`!ls`, `!cat`), the agent's ls and cat tools and WASI programs see the same entries and content, and `ls /` SHALL list proc/, sys/ and dev/ next to the FAT entries.
Source: kernel/src/fs.rs mount_of, read, entry, list, ls, open, check_rw; kernel/src/wasm.rs path_open

#### Scenario: Same content
- **WHEN** the user types `!cat /sys/devices/system/cpu/online` and `!run cat /sys/devices/system/cpu/online`
- **THEN** both print the same line, e.g. `0-7`

### Requirement: /sys tree
The system SHALL serve a virtual read-only /sys with class/block/vda/{size (512-byte sectors), ro, device (PCI address)}, bus/pci/devices/<0000:BB:DD.F>/{vendor, device, class, driver} for every function on PCI bus 0, devices/system/cpu/{online, possible, inference, cpuN/role} and power/state (`disk`), one value per file.
Source: kernel/src/sysfs.rs nodes, entry, list, read; kernel/src/procfs.rs pci_scan; kernel/src/blk.rs location, size; kernel/src/smp.rs

#### Scenario: Disk size
- **WHEN** the disk is 512 MiB and the user types `!cat /sys/class/block/vda/size`
- **THEN** the shell prints 1048576

#### Scenario: Inference cores
- **WHEN** the user types `!set cores 4` and then `!cat /sys/devices/system/cpu/inference`
- **THEN** the shell prints `0-3`

### Requirement: Virtual trees are read-only
Every write, append, truncate, create, mkdir, rm or rename under /proc, /sys (but a write of /sys/power/state), /dev (but a write of the data devices and the console), /dev/vda, /etc/os-release or /etc/passwd SHALL fail with `read-only file system` in the shell and the agent tools and with EROFS (69) for a WASI program.
Source: kernel/src/fs.rs ROFS, mount_of, check_rw, write; kernel/src/wasm.rs data_ro, path_open

#### Scenario: Program writes sysfs
- **WHEN** python3 opens /sys/devices/system/cpu/online for writing
- **THEN** the open fails with EROFS and nothing changes

### Requirement: /sys/power/state
A write of `disk` to /sys/power/state SHALL hibernate and power off (the image of `/hibernate`, then `!poweroff`); `mem`, `freeze` and `standby` SHALL fail as not supported (no suspend to RAM) and any other value as an invalid argument (EINVAL 28 for a program).
Source: kernel/src/sysfs.rs POWER, power_write; kernel/src/hib.rs power_disk; kernel/src/fs.rs write; kernel/src/wasm.rs path_open, fd_write

#### Scenario: Hibernate from a program
- **WHEN** the user types `!run python3 -c "open('/sys/power/state','w').write('disk')"`
- **THEN** the log says "hib: disk written to /sys/power/state", the image is saved, the machine powers off and the next boot resumes the conversation

#### Scenario: Suspend to RAM
- **WHEN** the user types `!write /sys/power/state mem`
- **THEN** the shell prints the not-supported error and the machine stays on

### Requirement: Who may write /sys/power/state
The shell's `!write` and a program run from the shell SHALL act at once (the human typed them); the agent's write_file and a program the agent runs SHALL ask y/N first, as the poweroff tool does, and a no SHALL change nothing.
Source: kernel/src/agent.rs fs_tool; kernel/src/wasm.rs fd_write

#### Scenario: The agent asks
- **WHEN** the model calls write_file with path /sys/power/state and content disk
- **THEN** the user is asked y/N, and on no the tool result says the call was not run

### Requirement: /dev data devices
The system SHALL serve /dev/null (reads end at once), /dev/zero (zero bytes), /dev/random and /dev/urandom (RDRAND bytes, else the TSC-seeded generator), all four discarding writes (also the shell's `!write` and the agent's write_file), and a WASI fstat SHALL report them as character devices.
Source: kernel/src/devfs.rs NAMES, writable; kernel/src/fs.rs write; kernel/src/wasm.rs path_open, dev_read, fd_write, fd_filestat_get, Fd

#### Scenario: Random bytes
- **WHEN** the user types `!run head -c 16 /dev/urandom`
- **THEN** 16 bytes are written to the console and the run ends with exit code 0

#### Scenario: Discard
- **WHEN** a program writes 1 MiB to /dev/null
- **THEN** every write returns the full length and nothing is stored

### Requirement: /dev terminal and disk
/dev/tty and /dev/console SHALL be the console for a WASI run from the shell (raw input, writes printed) and not found for an agent run; /dev/vda SHALL be the raw disk, readable by programs as a block device and never writable.
Source: kernel/src/devfs.rs; kernel/src/wasm.rs path_open, tty_read, dev_read; kernel/src/blk.rs read_raw

#### Scenario: Boot sector
- **WHEN** python3 reads 512 bytes from /dev/vda
- **THEN** it gets the FAT boot sector of the disk

### Requirement: Devices are not read by cat
The shell's `!cat` and the agent's cat SHALL refuse a device file other than /dev/null with `is a device: read it with a program, e.g. head -c 16 /dev/urandom`, and SHALL print nothing for /dev/null.
Source: kernel/src/fs.rs DEVICE, virt_read, read; kernel/src/agent.rs fs_tool; kernel/src/shell.rs fs_cmd

#### Scenario: Endless device
- **WHEN** the model calls cat with path /dev/zero
- **THEN** the tool result is the `is a device` error

### Requirement: /etc
The system SHALL keep /etc as a FAT directory the shell, the agent and programs may change, SHALL create /etc/hostname with `ringzero` at boot when it is missing and log its first line as `host: NAME (/etc/hostname)` right after the disk mounts.
Source: kernel/src/layout.rs init, hostname; kernel/src/fs.rs mount_of

#### Scenario: Hostname change
- **WHEN** the user types `!write /etc/hostname lab` and reboots
- **THEN** the boot log has the line `host: lab (/etc/hostname)`

### Requirement: Virtual /etc files
The system SHALL serve the read-only virtual files /etc/os-release (NAME, ID, ID_LIKE=unix, VERSION, VERSION_ID, PRETTY_NAME with the edition) and /etc/passwd (`root:x:0:0:root:/:/bin/sh`).
Source: kernel/src/layout.rs etc_virtual; kernel/src/fs.rs mount_of

#### Scenario: os-release
- **WHEN** the user types `!cat /etc/os-release`
- **THEN** the lines include `ID=ringzero` and `ID_LIKE=unix`

### Requirement: /tmp
The system SHALL keep /tmp as a FAT directory, create it at boot when missing, empty it at a boot that does not resume a hibernation image, keep it when an image resumes, and the file indexer SHALL not walk it.
Source: kernel/src/layout.rs init, tmp_clean; kernel/src/hib.rs boot; kernel/src/fs.rs indexed, walk

#### Scenario: Fresh boot
- **WHEN** /tmp/a.txt exists and the user types `/hibernate discard` then `!poweroff` and boots again
- **THEN** /tmp is empty

### Requirement: ringzero data under /var/lib/ringzero
The system SHALL keep its own files under /var/lib/ringzero (mem/journal.log, mem/facts.log, mem/episodes.log, mem/index.bin, hib/image.bin, bootsnap.bin, layout), and the file indexer SHALL not walk that tree.
Source: kernel/src/layout.rs JOURNAL, FACTS, EPISODES, INDEX, HIB, BOOTSNAP; kernel/src/fs.rs indexed, walk

#### Scenario: Paths in the status
- **WHEN** the user types `/agentd` with a disk mounted
- **THEN** the stores are reported in /var/lib/ringzero/mem

### Requirement: ringzero data is protected from the agent
The agent's write_file, mkdir and rm SHALL fail under /var/lib/ringzero with `permission denied: ringzero's own data (use remember and forget)`; reads stay allowed and the shell keeps full access (`!rm` still asks).
Source: kernel/src/agent.rs fs_tool

#### Scenario: Agent overwrites a store
- **WHEN** the model calls write_file with path /var/lib/ringzero/mem/facts.log
- **THEN** the call fails with `permission denied` and the file is unchanged

### Requirement: Migration of an old disk
At mount, when FAT /sys exists or /var/lib/ringzero/layout is missing, the system SHALL copy /sys/mem/{journal,facts,episodes}.log and index.bin to /var/lib/ringzero/mem/, and any other file under FAT /sys, /proc or /dev to /var/lib/ringzero/legacy/, and verify each copy by size and FNV-1a 64 read back from the disk, never using rename.
Source: kernel/src/layout.rs MOVES, migrate, same; kernel/src/fs.rs RAW, write_parts, read

#### Scenario: Old disk
- **WHEN** a disk with five facts in /sys/mem/facts.log boots the new kernel
- **THEN** `/facts` lists the five facts, FAT /sys is gone and a `layout: migrated` summary line is logged

### Requirement: Migration commit and cleanup
After every copy verifies, the system SHALL write /var/lib/ringzero/layout (`ringzero-layout 2`) and sync, and only then remove the sources, /sys/bootsnap.bin and /sys/hib/image.bin (neither is copied: an older image cannot match the new boot key) and the emptied directories, then sync again.
Source: kernel/src/layout.rs DROPS, migrate; kernel/src/fs.rs rm, sync

#### Scenario: Old hibernation image
- **WHEN** an old disk holds /sys/hib/image.bin
- **THEN** it is removed after the commit marker and the boot starts a fresh conversation

### Requirement: Migration is idempotent and crash-safe
A migration interrupted at any step SHALL end, at a later boot, in the same state as an uninterrupted one: before the commit marker every source stays and verified copies are skipped, after it the removal resumes.
Source: kernel/src/layout.rs migrate, same, halt_at

#### Scenario: Killed after the commit
- **WHEN** the machine stops after the commit marker and before the removals
- **THEN** the next boot removes the sources it finds verified and copies nothing again

### Requirement: Migration that cannot finish
When a migration step fails (disk full, I/O error) the system SHALL keep every source, run agentd without disk stores for this boot with `/agentd` showing `off (layout migration pending: REASON)`, write no boot cache and no hibernation image, and retry at the next boot.
Source: kernel/src/layout.rs init, migrate, pending; kernel/src/agentd.rs init, status; kernel/src/hib.rs save; kernel/src/agent.rs init

#### Scenario: Power cut during the copies
- **WHEN** the machine stops after index.bin was half written to /var/lib/ringzero/mem
- **THEN** the next boot copies it again, verifies it and completes the migration with no record lost

### Requirement: Data written by an older kernel
When the commit marker exists and FAT /sys holds files again (an older kernel booted the disk), the system SHALL copy them to /var/lib/ringzero/legacy/sys-<unix time>/ (the time of the FAT /sys directory, so a retry after a crash uses the same folder), verify and remove them, leave /var/lib/ringzero/mem unchanged and log a warning naming the folder; it SHALL not merge them into the stores on its own (`/agentd merge-legacy` does, on request).
Source: kernel/src/layout.rs migrate

#### Scenario: Back from an older kernel
- **WHEN** an older kernel wrote /sys/mem/journal.log after the migration and the new kernel boots
- **THEN** that journal is in /var/lib/ringzero/legacy/sys-<time>/mem/ and a warning says so
