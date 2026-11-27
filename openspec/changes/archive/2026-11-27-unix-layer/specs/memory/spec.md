## ADDED Requirements

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

## MODIFIED Requirements

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

### Requirement: Fact dedup, update and capacity
The system SHALL answer "already saved: KEY = VALUE" without writing when the key already holds the same value, replace the value of an existing key (one `S` record in `/var/lib/ringzero/mem/facts.log`, the old value logged), and refuse a new key once 64 facts are stored with "the fact store is full (64 facts); forget one first".
Source: kernel/src/agentd.rs MAX_FACTS, remember

#### Scenario: Store full
- **WHEN** 64 facts exist and a fact with a new key is remembered
- **THEN** it is rejected, while a new value for one of the 64 existing keys is still accepted

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
