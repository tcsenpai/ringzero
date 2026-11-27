## ADDED Requirements

### Requirement: Build variables rebuild the kernel
kernel/build.rs SHALL declare rerun-if-env-changed for every RINGZERO_* variable the kernel reads at build time (KV, EDITION, MANVEC, CONF_ASK, HEAP_MIB, WASM_MIB, UPY_HEAP_MIB), so changing one of them alone rebuilds the kernel.
Source: kernel/build.rs main, upy

#### Scenario: Heap changed alone
- **WHEN** only RINGZERO_HEAP_MIB changes between two builds
- **THEN** cargo reruns the build script and compiles the kernel again

## REMOVED Requirements

### Requirement: OPEN: rebuild on heap and wasm memory changes
**Reason**: resolved: kernel/build.rs now declares both variables (requirement "Build variables rebuild the kernel").
**Migration**: none.
