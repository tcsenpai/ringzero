## ADDED Requirements

### Requirement: LAYOUT_STOP
`LAYOUT_STOP` SHALL set, through RINGZERO_LAYOUT_STOP, a crash test build: the migration of an old disk halts the machine after its Nth step (a verified copy, the commit marker, a removal), logging `layout: crash test build: halting after migration step N`; unset or empty means never. A change of the value SHALL rebuild the kernel.
Source: Makefile target kernel; kernel/src/layout.rs STOP, halt_at; kernel/build.rs main

#### Scenario: Stop after the commit
- **WHEN** an ISO built with `make iso LAYOUT_STOP=5` boots a 0.1 disk with four stores
- **THEN** it copies the four stores, writes the commit marker and halts before removing any source
