## MODIFIED Requirements

### Requirement: Memory slot for a new conversation
The system SHALL put in front of the first user message of a conversation, and of the first one after a compaction, a block headed "[memory from earlier conversations, kept by the kernel; use it to answer]" with facts newest first within 120 tokens, then at most 3 notes from the top 4 hits, all within `!set slot N` tokens (default 200, 0 = off).
Source: kernel/src/agentd.rs SLICE_MS, SLOT_HEAD, NOTE_MIN, FILE_NOTE_MIN, tokens, slot; kernel/src/agent.rs ask; kernel/src/shell.rs set_setting

#### Scenario: Note filters
- **WHEN** a hit is a fact, has cosine under 0.55 (under 0.40 for a BM25 hit), a body under 20 characters (under 8 for a file chunk `[/path] ...`), or is the message itself
- **THEN** it is not added as a note and its verdict is logged as a "slot cand" line

#### Scenario: Slot off
- **WHEN** the user types `!set slot 0` and starts a new conversation
- **THEN** no memory block is added and "slot (new conversation): off (!set slot 0)" is logged

#### Scenario: Weak note dropped
- **WHEN** the best embedding-only hit for the first message has cosine 0.50
- **THEN** it is logged as "below the threshold" and not put in the slot

#### Scenario: Short file note
- **WHEN** `/notes/todo.txt` holds "buy milk", its chunk is indexed and the first message of a new conversation is "what do I need to buy?" with the chunk at cosine 0.55 or more
- **THEN** the chunk is kept as a note ("file /notes/todo.txt: buy milk") while a `[said]` note of the same length would be logged as "too short"
