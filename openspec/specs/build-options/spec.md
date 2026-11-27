# build-options Specification

## Purpose
The build-time options of ringzero: Makefile variables (MODEL, QUANT, EMBED, GUEST_MEM, MANUAL_VECTORS, CONF_ASK, DISK_MB and the run targets' limits) and the RINGZERO_* environment variables the kernel build reads, with their accepted values and defaults.

## Requirements

### Requirement: MODEL and QUANT select the edition
`MODEL` SHALL accept 2b, 4b or a path to an .asmbin file, default 2b; for 2b and 4b, `QUANT` SHALL accept q8 (every matrix Q8_0), q4 (MLP Q4_0, the rest Q8_0) or q4all (every projection Q4_0, embed and lm_head Q8_0), default q8, and any other QUANT SHALL stop the build with `QUANT must be q8, q4 or q4all`.
Source: Makefile MODEL, QUANT, MODEL_FILE

#### Scenario: Bad QUANT
- **WHEN** the user runs `make iso QUANT=q2`
- **THEN** make stops with the QUANT error

#### Scenario: Custom model file
- **WHEN** the user runs `make iso MODEL=models/x.asmbin`
- **THEN** that file is the boot model and the edition is `ringzero (custom model)`

### Requirement: Edition name
The build SHALL pass the edition name to the kernel in RINGZERO_EDITION: `ringzero Qwen35@<size>-Qwen3@0.6b (QweQwe)` for q8, with `.q4` or `.q4all` after the size for the Q4 builds; a kernel built without it SHALL use `ringzero Qwen35@2b-Qwen3@0.6b (QweQwe)`.
Source: Makefile EDITION, target kernel; kernel/src/main.rs EDITION; kernel/build.rs main

#### Scenario: 4B q4
- **WHEN** the user runs `make iso MODEL=4b QUANT=q4`
- **THEN** the boot banner names `ringzero Qwen35@4b.q4-Qwen3@0.6b (QweQwe)`

### Requirement: EMBED
`EMBED` SHALL name the embedder file, default models/qwen3-embed-0.6b-q8_0-v3.asmbin, copied into the ISO as boot/embed.asmbin.
Source: Makefile EMBED, target iso

#### Scenario: Default embedder
- **WHEN** the user runs `make iso` without EMBED
- **THEN** the ISO holds the Qwen3-Embedding-0.6B Q8_0 file

### Requirement: GUEST_MEM and MIN_AVAIL_MB
`GUEST_MEM` SHALL default to 6G, and to 8G for every 4b build; run-remote and run-local SHALL refuse to start when the build box has less available memory than MIN_AVAIL_MB (8192 by default, 10240 for 4b, 11264 for 4b q8).
Source: Makefile GUEST_MEM, MIN_AVAIL_MB, target run-remote, target run-local

#### Scenario: Not enough memory
- **WHEN** the box has 6000 MiB available and the user runs `make run-remote`
- **THEN** make prints `refusing: only 6000 MiB available (want > 8192 for a 6G guest)` and no guest starts

### Requirement: MANUAL_VECTORS
`MANUAL_VECTORS` SHALL default to 1, which builds a first ISO without vectors, boots it headless to write build/manvec.bin, and links that file into the kernel through RINGZERO_MANVEC; 0 SHALL build without vectors (the manual is embedded at the first boot).
Source: Makefile target model, target kernel; kernel/build.rs main

#### Scenario: No vectors
- **WHEN** the user runs `make iso MANUAL_VECTORS=0`
- **THEN** no headless pass runs and `/man check` reports `not built in (MANUAL_VECTORS=0)`

### Requirement: CONF_ASK
`CONF_ASK` SHALL set, through RINGZERO_CONF_ASK, the percent below which a low-confidence call of a tool that changes the machine asks first: an integer clamped to 0..100, 80 when unset or empty; a value with a character that is not a digit SHALL fail the kernel build. A change of the value SHALL rebuild the kernel.
Source: Makefile target kernel; kernel/src/agent.rs fs_tool, conf_ask; kernel/build.rs main

#### Scenario: Default
- **WHEN** the user runs `make iso`
- **THEN** CONF_LO is 0.80

#### Scenario: Clamp
- **WHEN** the user runs `make iso CONF_ASK=2500`
- **THEN** CONF_LO is 1.00

#### Scenario: Not a number
- **WHEN** the user runs `make iso CONF_ASK=0.8`
- **THEN** the kernel build fails with `RINGZERO_CONF_ASK: a percent integer, 0..100`

### Requirement: Data disk
`make disk` SHALL create build/disk.img as a whole-volume FAT32 of DISK_MB MiB (512 by default) labelled RINGZERO, and SHALL never overwrite an existing image.
Source: Makefile DISK_MB, LOCAL_DISK, target iso, target disk

#### Scenario: Existing disk
- **WHEN** build/disk.img exists and the user runs `make disk`
- **THEN** make prints `build/disk.img exists, not touching it`

### Requirement: WASI apps on the disk
`make wasm-apps` SHALL refuse while a guest has the disk open, and otherwise copy the built programs to /bin, CPython's library to /lib/python3.15 and tools/demo to /demo, replacing same-name files and keeping every other file.
Source: Makefile target disk, target wasm-apps

#### Scenario: Guest running
- **WHEN** a guest has build/disk.img open
- **THEN** make prints `refusing: a guest has build/disk.img open`

### Requirement: RINGZERO_KV
The kernel build SHALL use an FP16 KV cache by default and an FP32 cache when RINGZERO_KV=f32, and the KV choice SHALL be part of the engine hash that keys the boot snapshot cache.
Source: kernel/build.rs main

#### Scenario: FP32 cache
- **WHEN** the kernel is built with RINGZERO_KV=f32
- **THEN** nasm runs without -DKV_F16 and the boot snapshot cache is rebuilt

### Requirement: Heap and program memory variables
The kernel SHALL read RINGZERO_HEAP_MIB (kernel heap, default 192 MiB), RINGZERO_WASM_MIB (WASI program memory, default 256, clamped to 16..2048) and RINGZERO_UPY_HEAP_MIB (MicroPython heap, default 8, clamped to 1..256) at build time.
Source: kernel/src/mem.rs init, HEAP_BYTES; kernel/src/wasm.rs init; kernel/src/upy.rs init; kernel/build.rs upy

#### Scenario: Default heap
- **WHEN** the kernel is built without RINGZERO_HEAP_MIB
- **THEN** the heap is 192 MiB

### Requirement: Beta build
`make beta` SHALL build ringzero-beta-2b-q8.iso, then ringzero-beta-4b-q4.iso, then a fresh data disk with the apps in build/beta, and write SHA256SUMS; SOURCE_DATE_EPOCH SHALL default to the last commit time so two runs on the same tree give the same bytes.
Source: Makefile target beta

#### Scenario: Checksums
- **WHEN** `make beta` finishes
- **THEN** build/beta/SHA256SUMS lists the two ISOs and disk.img

### Requirement: VNC_PASS
`make run-remote` SHALL protect its VNC framebuffer with the password VNC_PASS, default `ringzero`.
Source: Makefile VNC_PASS, target run-remote

#### Scenario: Default password
- **WHEN** the user runs `make run-remote` without VNC_PASS and connects a VNC client to port 5901
- **THEN** the client asks for a password and `ringzero` opens the framebuffer

### Requirement: Build variables rebuild the kernel
kernel/build.rs SHALL declare rerun-if-env-changed for every RINGZERO_* variable the kernel reads at build time (KV, EDITION, MANVEC, CONF_ASK, HEAP_MIB, WASM_MIB, UPY_HEAP_MIB), so changing one of them alone rebuilds the kernel.
Source: kernel/build.rs main, upy

#### Scenario: Heap changed alone
- **WHEN** only RINGZERO_HEAP_MIB changes between two builds
- **THEN** cargo reruns the build script and compiles the kernel again
