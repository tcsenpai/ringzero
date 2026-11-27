## ADDED Requirements

### Requirement: ROUNDS
`ROUNDS` SHALL set, through RINGZERO_ROUNDS, the default tool rounds per question: an integer clamped to 1..20, 10 when unset or empty; a value with a character that is not a digit SHALL fail the kernel build. A change of the value SHALL rebuild the kernel. `!set rounds` changes it at run time.
Source: Makefile target kernel; kernel/src/agent.rs ROUNDS, env_num; kernel/build.rs main

#### Scenario: Default
- **WHEN** the user runs `make iso`
- **THEN** the agent allows 10 tool rounds per question

#### Scenario: Clamp
- **WHEN** the user runs `make iso ROUNDS=0`
- **THEN** the agent allows 1 tool round per question

## MODIFIED Requirements

### Requirement: CONF_ASK
`CONF_ASK` SHALL set, through RINGZERO_CONF_ASK, the percent below which a low-confidence call of a tool that writes asks first: an integer clamped to 0..100, 80 when unset or empty; a value with a character that is not a digit SHALL fail the kernel build. A change of the value SHALL rebuild the kernel.
Source: Makefile target kernel; kernel/src/agent.rs CONF_LO, env_num; kernel/build.rs main

#### Scenario: Default
- **WHEN** the user runs `make iso`
- **THEN** CONF_LO is 0.80

#### Scenario: Clamp
- **WHEN** the user runs `make iso CONF_ASK=2500`
- **THEN** CONF_LO is 1.00

#### Scenario: Not a number
- **WHEN** the user runs `make iso CONF_ASK=0.8`
- **THEN** the kernel build fails with `RINGZERO_CONF_ASK / RINGZERO_ROUNDS: an integer (percent 0..100 / rounds 1..20)`

### Requirement: Build variables rebuild the kernel
kernel/build.rs SHALL declare rerun-if-env-changed for every RINGZERO_* variable the kernel reads at build time (KV, EDITION, MANVEC, CONF_ASK, ROUNDS, HEAP_MIB, WASM_MIB, UPY_HEAP_MIB), so changing one of them alone rebuilds the kernel.
Source: kernel/build.rs main, upy

#### Scenario: Heap changed alone
- **WHEN** only RINGZERO_HEAP_MIB changes between two builds
- **THEN** cargo reruns the build script and compiles the kernel again
