## ADDED Requirements

### Requirement: run_code is described as a sandbox
The run_code tool description SHALL say that the code runs in an isolated MicroPython sandbox with no os module, no filesystem, no network and no imports from disk, and that files and programs go through ls, cat, write_file or run_wasm; the system prompt SHALL say that run_code is a MicroPython sandbox without files and name ls, cat and run_wasm for files and programs.
Source: kernel/src/agent.rs SYSTEM, TOOLS

#### Scenario: File question
- **WHEN** the user asks what is in a file on the disk
- **THEN** the boot prefix tells the model that run_code cannot read files and that cat can
