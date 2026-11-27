# memory Specification

## Purpose
Baseline of the agent memory owned by the kernel subsystem agentd: on-disk record stores, validated facts, the embedded memory index with hybrid search, the memory slot put in front of a new conversation, and idle-time consolidation of a session into an episode and facts.

## Requirements

### Requirement: Append-only record stores on disk
The system SHALL keep three append-only stores, `/var/lib/ringzero/mem/journal.log`, `/var/lib/ringzero/mem/facts.log` and `/var/lib/ringzero/mem/episodes.log`, where each record is an 18-byte header (`'R'`, kind u8, payload length u32, Unix-seconds timestamp u64, folded FNV-1a checksum u32 over kind, timestamp and payload, little endian) followed by a UTF-8 payload of at most 16384 bytes (longer payloads are cut at a char boundary).
Source: kernel/src/mstore.rs FNV0, sum, append

#### Scenario: Oversized payload is cut
- **WHEN** a record with a 20000-byte payload is appended
- **THEN** the stored payload is at most 16384 bytes and ends on a UTF-8 char boundary

### Requirement: Torn tail recovery at boot
The system SHALL, when loading a store, stop at the first record with a bad magic byte, a length over 16384, a length past the end of the file, a checksum mismatch or invalid UTF-8, truncate the file to the end of the last good record, and log the number of good records and the bytes cut.
Source: kernel/src/mstore.rs Rec, load

#### Scenario: Append cut short by a crash
- **WHEN** the last record of `/var/lib/ringzero/mem/facts.log` is missing its final 3 bytes at boot
- **THEN** every earlier record is replayed, the file is truncated at the end of the previous record, and a decision line "recover ... torn tail of N bytes ... truncated" is logged

### Requirement: RAM-only mode without a disk
The system SHALL, when no FAT volume is mounted, keep memory stores and index persistence off (nothing is appended, nothing is saved) and log "no disk: memory stores and index persistence are off (RAM only)".
Source: kernel/src/agentd.rs init, append

#### Scenario: Boot without a disk
- **WHEN** the kernel boots with no mounted volume
- **THEN** `/agentd` reports the stores as "off (no disk)" and no file under `/var/lib/ringzero/mem` is written

### Requirement: Journal of turns
The system SHALL append to the journal a `B` record "boot" at boot, a `U` record per user message, a `T` record "name -> result" per tool call with the result cut to 1024 bytes, an `A` record per non-empty answer, a `C` record per compaction summary and a `P` record "poweroff" at shutdown or reboot when the stores are on and the embedder is ready.
Source: kernel/src/agentd.rs init, save_index, shutdown, said_end, on_compact

#### Scenario: A tool call is journaled
- **WHEN** the agent calls a tool whose result is 5000 bytes long
- **THEN** a `T` record "<tool name> -> <first 1024 bytes of the result>" is appended to `/var/lib/ringzero/mem/journal.log`

### Requirement: Fact key validation
The system SHALL normalize a fact key by trimming it, mapping space, `-` and `.` to `_` and lowercasing ASCII, and SHALL reject the fact unless the result is 2 to 32 characters of `a-z`, `0-9` and `_` starting with a letter.
Source: kernel/src/agentd.rs take_after_compact, norm_key, remember

#### Scenario: Key normalization
- **WHEN** a fact is stored with key "User Name"
- **THEN** it is stored under key `user_name`

#### Scenario: Invalid key
- **WHEN** a fact is stored with key "1abc" or "x"
- **THEN** it is rejected with "error: key '...' must be snake_case, 2..32 characters"

### Requirement: Fact value validation
The system SHALL trim a fact value (whitespace, a trailing `.`, surrounding quotes) and reject it when it is empty or longer than 120 bytes, contains `<`, a newline, a tab or `|`, is a placeholder (unknown, none, n/a, null, ?, -, nessuno, sconosciuto, case-insensitive), or equals the key with `_` read as spaces.
Source: kernel/src/agentd.rs norm_value, remember

#### Scenario: Placeholder value
- **WHEN** the agent calls `remember` with key `user_city` and value "unknown"
- **THEN** the fact is rejected with "error: placeholder value" and nothing is written

### Requirement: Fact dedup, update and capacity
The system SHALL answer "already saved: KEY = VALUE" without writing when the key already holds the same value, replace the value of an existing key (one `S` record in `/var/lib/ringzero/mem/facts.log`, the old value logged), and refuse a new key once 64 facts are stored with "the fact store is full (64 facts); forget one first".
Source: kernel/src/agentd.rs MAX_FACTS, remember

#### Scenario: Store full
- **WHEN** 64 facts exist and a fact with a new key is remembered
- **THEN** it is rejected, while a new value for one of the 64 existing keys is still accepted

### Requirement: Grounding of model-proposed facts
The system SHALL accept a fact proposed by the model (the agent's `remember` tool or consolidation) only if every number in the value appears in the user text of the current session and at least half (and at least one) of its words that contain a digit or have 3 or more characters appear there; facts typed with `/remember KEY VALUE` are not grounded.
Source: kernel/src/agentd.rs norm_value, grounded, remember; kernel/src/agent.rs exec; kernel/src/shell.rs run

#### Scenario: Value not said by the user
- **WHEN** the user said "mi chiamo Anna e il mio numero preferito e 4711" and the model proposes the value "4712"
- **THEN** the fact is rejected as "not something the user said in this session"

### Requirement: Forgetting a fact
The system SHALL delete a fact by appending an `X` record with its normalized key and removing its index entry; for an unknown key it SHALL make no change and return an error listing the saved keys (or "none"). `/forget KEY` and the agent's `forget` tool on an existing key SHALL ask the user for confirmation first.
Source: kernel/src/agentd.rs forget; kernel/src/shell.rs run; kernel/src/agent.rs exec

#### Scenario: Declined forget
- **WHEN** the user types `/forget user_name` and answers no
- **THEN** "[facts] kept" is printed and the fact stays

### Requirement: Which user turns are remembered as statements
The system SHALL queue a user message for the index as "[said] TEXT" (TEXT cut to 600 bytes) only when it does not end with `?`, contains a first-person word (io, mi, mio, mia, miei, mie, me, sono, ho, ricordati, ricorda, i, my, mine, remember) and its turn ran no tool other than `remember`, `forget` or `recall`; the same rule applies when the journal is replayed.
Source: kernel/src/agentd.rs man, said_end, apply

#### Scenario: A request carried out with a tool is not remembered
- **WHEN** the user says "my notes go in /notes, create /notes/todo.txt" and the turn calls `write_file`
- **THEN** no "[said]" entry is indexed for it and "said: not indexed (the turn ran a tool)" is logged

#### Scenario: A question is not remembered
- **WHEN** the user asks "come mi chiamo?"
- **THEN** no "[said]" entry is queued

### Requirement: Memory index limits
The memory index SHALL hold at most 4096 entries and 1 MiB of text, cut each text to 2048 bytes at a char boundary, refuse an empty text, and report "memory index full" when an entry does not fit; a queued memory text whose exact text is already indexed SHALL be skipped.
Source: kernel/src/memidx.rs CAP, TEXT_MAX, add, room, store_row; kernel/src/agentd.rs embed_slice

#### Scenario: Index full
- **WHEN** 4096 entries are stored and another text is added with `/index add TEXT`
- **THEN** "[index] memory index full" is printed

### Requirement: Index dimension
The system SHALL accept `/index dim N` only for N a multiple of 128 from 128 to 1024, and only when the index is empty or N equals the current dimension.
Source: kernel/src/memidx.rs clear, set_dim; kernel/src/shell.rs index_cmd

#### Scenario: Dim change on a non-empty index
- **WHEN** the index has entries at dim 1024 and the user types `/index dim 512`
- **THEN** "[index] the index is not empty: /index clear first" is printed and the dim stays 1024

### Requirement: Hybrid retrieval with a BM25 gate
The system SHALL rank index entries by embedding cosine and SHALL fuse BM25 (k1 1.5, b 0.75) into that order by reciprocal rank fusion with k = 60 only when the query is identifier-like or its top BM25 hit holds every query term and scores at least 1.5 times the second. `/search QUERY` SHALL print the top 5 with cos, bm25 and via.
Source: kernel/src/memidx.rs ROW_MAX, RRF_K, bm25, identifier_like, Found, search; kernel/src/shell.rs search_cmd

#### Scenario: Identifier query
- **WHEN** the user types `/search HL-L2350DW` (an inner `-` between alphanumerics; paths, letters mixed with digits, all caps of 2+ letters and a leading `/` count too)
- **THEN** the summary line says "BM25 identifier-like query"

#### Scenario: Plain question
- **WHEN** the user types `/search how are my files backed up?` and no BM25 hit is strong
- **THEN** the summary line says "BM25 not used (embedding order)"

### Requirement: Index persistence and rebuild
The system SHALL save the index to `/var/lib/ringzero/mem/index.bin` with a 72-byte `RZIX` version 1 header (embedder hash, dim, layout 0x5108, chunker version, counts, checksum) and the store sizes it covers, and at boot SHALL load it only if header, size and checksum match, else rebuild it by embedding every record, file and manual section again.
Source: kernel/src/memidx.rs search, load; kernel/src/agentd.rs append, replay, scan_files, index_extra

#### Scenario: Records written after the last save
- **WHEN** the index loads and the journal is longer than the size the index covers
- **THEN** only the records past the covered offset are queued for embedding

#### Scenario: Embedder changed
- **WHEN** the kernel boots with a different embedder file than the one that saved `/var/lib/ringzero/mem/index.bin`
- **THEN** "index: not loaded (embedder changed): rebuilding" is logged and every memory record and file is queued for embedding

### Requirement: When the index is saved
The system SHALL save the index from the idle scheduler when it changed since the last save and nothing is queued, and at poweroff or reboot SHALL save it only if it changed and the embedding queue is empty (otherwise the next boot re-embeds from the last saved index).
Source: kernel/src/agentd.rs save_index, shutdown, pick

#### Scenario: Poweroff with work still queued
- **WHEN** `!poweroff` runs while memory texts are still queued
- **THEN** "shutdown: index not saved, N texts and files still queued" is logged and `/var/lib/ringzero/mem/index.bin` is left as it was

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

### Requirement: Recall tool
The agent's `recall` tool SHALL return the top 3 index hits that are not manual sections, each as "N. (score X.XX) LINE" with notes cut to 300 bytes, plus the best manual section when the message or query asks how to do something and that section scores at least 0.55, and "nothing found in memory" when there are no hits.
Source: kernel/src/agentd.rs howto, recall; kernel/src/agent.rs exec

#### Scenario: Empty memory
- **WHEN** the index and the queue are empty and the agent calls `recall`
- **THEN** the tool result is "nothing found in memory"

### Requirement: Idle scheduler priorities
The system SHALL run background memory work only while the shell waits for input, in slices of 100 ms that stop on any input, picking the lowest priority number among: 1 embed queued memory texts, 2 embed queued file chunks, 3 save the changed index, 4 consolidate (lowered by 1 for every 50 slices it waited, floor 0); an embedding in flight is finished first.
Source: kernel/src/agentd.rs SLICE_MS, cons_due, idle; kernel/src/shell.rs readline

#### Scenario: Input preempts a slice
- **WHEN** the user presses a key while a file chunk is being embedded
- **THEN** the slice stops between engine steps and the prompt gets the engine back

### Requirement: Consolidation trigger
The system SHALL start a consolidation when the stores are on, the session has a new user turn, the live state is saved and the prompt has been idle for `!set consolidate N` seconds (default 60; 0 = only `/consolidate`), and SHALL skip it with a logged reason at the boot prefix ("fresh conversation") or when the context lacks room for the instruction plus 220 tokens.
Source: kernel/src/agentd.rs CONS_MAX, CONS_IDLE_S, cons_due, cons_begin; kernel/src/shell.rs set_setting

#### Scenario: Nothing to consolidate
- **WHEN** the user types `/consolidate` with no new turns since the last one
- **THEN** "[agentd] nothing to consolidate: no new turns since the last consolidation" is printed

### Requirement: Consolidation leaves the conversation untouched
The system SHALL run consolidation greedily (temperature forced to 0 and restored after), generating at most 220 tokens on top of the current snapshot point into a capture buffer that is not shown, and SHALL restore the current point after it finishes, when input arrives at the prompt, or on Ctrl-C during `/consolidate`.
Source: kernel/src/agentd.rs CONS_MAX, cons_begin, cons_abort, consolidate_now

#### Scenario: Interrupted by the user
- **WHEN** the user presses Ctrl-C during `/consolidate`
- **THEN** "^C [agentd] consolidation stopped, conversation restored (ctx N)" is printed and the context position is the one before the consolidation

### Requirement: Consolidation output becomes an episode and facts
The system SHALL store the first non-empty "EPISODE:" line of the consolidation output (cut to 600 bytes) as an `E` record in `/var/lib/ringzero/mem/episodes.log` and index it as "[episode N] TEXT", SHALL pass at most 8 "FACT: key = value" lines through the grounded fact validation with source "consolidation (episode N)", and SHALL then reset the session's turns and grounding text.
Source: kernel/src/agentd.rs cons_finish

#### Scenario: No episode line
- **WHEN** the model output has FACT lines but no EPISODE line
- **THEN** "no EPISODE line in the output, no episode saved" is logged and valid facts are still stored

### Requirement: Memory shell commands
The system SHALL provide `/facts`, `/remember KEY VALUE`, `/forget KEY`, `/consolidate`, `/agentd` (status plus the last 10 decisions), `/agentd log N` (last N decisions, default 30, from a ring of 128), `/agentd merge-legacy [DIR]`, `/index add TEXT`, `/index stats`, `/index clear` (rebuild), `/index dim N` and `/search QUERY`.
Source: kernel/src/shell.rs CMDS, index_cmd, run; kernel/src/agentd.rs DL, log_last, list_facts, merge_legacy

#### Scenario: Clearing the index
- **WHEN** the user types `/index clear`
- **THEN** "[index] cleared: every file and memory record is queued again (idle time at the prompt)" is printed and facts in `/var/lib/ringzero/mem/facts.log` are kept

### Requirement: Merge of stores an older kernel wrote
`/agentd merge-legacy [DIR]` SHALL merge legacy/DIR/mem (empty DIR: every folder not merged yet): facts through `remember` (source `merge-legacy`) unless the same or a later value is held, journal and episode records not held yet appended with their own time, all queued for the index, then a `merged` file in the folder.
Source: kernel/src/agentd.rs merge_legacy, merge_one; kernel/src/shell.rs run

#### Scenario: Merge after a rollback
- **WHEN** an older kernel saved the fact pet_name = Pixel after the migration and the user types `/agentd merge-legacy`
- **THEN** `/facts` lists pet_name = Pixel with source merge-legacy and the summary says one fact merged

### Requirement: Merge refuses an incompatible store
`/agentd merge-legacy` SHALL refuse a folder, changing no store, when a store present there does not parse from its first byte with this kernel's record codec (`not in this kernel's record format (not compatible)`), and SHALL answer `already merged` for a folder that has a `merged` file.
Source: kernel/src/agentd.rs merge_one; kernel/src/mstore.rs parse

#### Scenario: Incompatible store
- **WHEN** a legacy facts.log does not start with a valid record
- **THEN** the folder is refused with the not-compatible message and no store changes

#### Scenario: Twice
- **WHEN** the user types `/agentd merge-legacy` again for the same folder
- **THEN** nothing is merged

## Open questions
- `/forget KEY` and the agent's `forget` tool append an `X` record even when the store is off; with no disk `append` returns early, so a forget is RAM only. Intended, but not stated anywhere in the docs.
- The `remember` tool of the agent runs grounding against the session text since the last consolidation, so a fact stated before an idle consolidation cannot be saved by the model afterwards (only by `/remember`). Not confirmed whether this is intended.
