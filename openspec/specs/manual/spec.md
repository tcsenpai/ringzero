# manual Specification

## Purpose
The built-in manual: docs/MANUAL.md compiled into the kernel, indexed one entry per section, searched with `/man`, and handed to the model by the recall tool and the memory slot when the user asks how to do something.

## Requirements

### Requirement: Manual compiled in
The kernel SHALL include docs/MANUAL.md at build time and index every `## ` section as one entry `[manual: TITLE] body`; the text before the first section SHALL not be indexed.
Source: kernel/src/agentd.rs:50-52, kernel/src/agentd.rs:431-437

#### Scenario: Section count
- **WHEN** docs/MANUAL.md has 26 `## ` sections
- **THEN** `/man` lists 26 section titles

### Requirement: Manual index stays in sync
At boot and on an index rebuild the kernel SHALL drop indexed manual entries whose text is no longer in its manual and queue only the sections not yet in the index, so a changed section is embedded again and the others are kept.
Source: kernel/src/agentd.rs:439-460, kernel/src/agentd.rs:325, kernel/src/agentd.rs:372

#### Scenario: One section edited
- **WHEN** one section of the manual changes and the kernel boots on a disk indexed by the old kernel
- **THEN** the old entry of that section is dropped and only that section is queued

### Requirement: Baked manual vectors
With MANUAL_VECTORS=1 the kernel SHALL carry the sections' Q8_0 vectors and store them in the index without embedder work when their header key matches the manual text, the embedder file, the engine build, the chunker version and the index dim; on any mismatch, or when built with MANUAL_VECTORS=0, the sections SHALL be queued for embedding in idle time instead.
Source: kernel/src/agentd.rs:462-507, kernel/src/agentd.rs:441-460, Makefile:79-92

#### Scenario: Key mismatch
- **WHEN** the index dim was changed with `/index dim 256`
- **THEN** the baked vectors are not used and the sections are embedded at the prompt

### Requirement: man lists sections
`/man` without an argument SHALL print `manual sections (N): ` with every section title, then `/man TOPIC searches them, for example /man undo`.
Source: kernel/src/agentd.rs:545-554

#### Scenario: Bare man
- **WHEN** the user types `/man`
- **THEN** the section titles are printed comma separated

### Requirement: man searches
`/man TOPIC` SHALL print at most the two best manual sections for TOPIC, each whole with its score and without its `Asked as:` line, and `[man] nothing found for '<TOPIC>'` when none matches; without an embedder it SHALL print `[man] no embedder: the manual cannot be searched`.
Source: kernel/src/agentd.rs:555-569, kernel/src/agentd.rs:744-752

#### Scenario: Italian wording
- **WHEN** the user types `/man ibernare`
- **THEN** the Hibernation section is among the printed sections

### Requirement: man check and dump
`/man check` SHALL embed every section now and print how many rows are byte-identical to the baked vectors; `/man dump` SHALL print the header and every row as `MV <hex>` lines and a final `MV end <fnv>` line for tools/manvec.py.
Source: kernel/src/agentd.rs:507-543, kernel/src/agentd.rs:546-548

#### Scenario: Check on a matching build
- **WHEN** the kernel was built with MANUAL_VECTORS=1 and the user types `/man check`
- **THEN** the line reads `[man] vectors in the kernel: 26/26 rows byte-identical to a fresh embedding`

### Requirement: How-to detection
A message SHALL count as a how-to question when it contains one of: how do i, how can i, how to, how do you, how should i, how does, what command, which command, is there a command, quale comando, che comando, un comando per, in che modo, come si fa, come si usa, come posso; or the word `come` followed by a word of 3 or more letters ending in `o` other than `sono`.
Source: kernel/src/agentd.rs:803-811

#### Scenario: Italian first person verb
- **WHEN** the user writes `come creo un file`
- **THEN** the message is a how-to question

### Requirement: Manual line for a how-to message
In the middle of a conversation, a how-to message SHALL get the best manual section in front of it under `[from the ringzero manual, kept by the kernel; use it to answer]` when it scores 0.55 or more and fits the slot token budget; with `!set slot 0` it SHALL get none.
Source: kernel/src/agentd.rs:38, kernel/src/agentd.rs:53, kernel/src/agentd.rs:832-858, kernel/src/agent.rs:672-677

#### Scenario: Slot off
- **WHEN** the slot is 0 and the user asks `how do I undo`
- **THEN** no manual line is added to the message

### Requirement: Recall returns a manual section
The recall tool SHALL append the best manual section to its top 3 hits when it scores 0.55 or more and the user's message or the query is a how-to question.
Source: kernel/src/agentd.rs:813-830

#### Scenario: How-to query
- **WHEN** the model calls recall with query `how to hibernate`
- **THEN** the result includes a line starting with `ringzero manual, Hibernation:`
