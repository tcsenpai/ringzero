## MODIFIED Requirements

### Requirement: Hybrid retrieval with a BM25 gate
The system SHALL rank entries by score (cosine minus the penalties) and fuse BM25 (k1 1.5, b 0.75) in by RRF (k = 60) only for an identifier-like query or a top BM25 hit holding every query term at 1.5 times the second. `/search QUERY` SHALL print the top 5 of the ranked answer (cos, bm25, via, score, date), then `#OLD superseded by #NEW` lines.
Source: kernel/src/memidx.rs bm25, identifier_like, search; kernel/src/agentd.rs ranked; kernel/src/shell.rs search_cmd

#### Scenario: Identifier query
- **WHEN** the user types `/search HL-L2350DW` (an inner `-` between alphanumerics; paths, letters mixed with digits, all caps of 2+ letters and a leading `/` count too)
- **THEN** the summary line says "BM25 identifier-like query"

#### Scenario: Plain question
- **WHEN** the user types `/search how are my files backed up?` and no BM25 hit is strong
- **THEN** the summary line says "BM25 not used (embedding order)"

### Requirement: Index persistence and rebuild
The system SHALL save the index to `/var/lib/ringzero/mem/index.bin` (72-byte `RZIX` version 2 header: embedder hash, dim, layout 0x5108, chunker version, counts, checksum; the store sizes it covers; the dates) and at boot SHALL load it only if header, size and checksum match (version 1, chunker 1 too), else rebuild it by embedding every record, file and manual section.
Source: kernel/src/memidx.rs VER, save, load; kernel/src/agentd.rs replay, scan_files, migrate

#### Scenario: Records written after the last save
- **WHEN** the index loads and the journal is longer than the size the index covers
- **THEN** only the records past the covered offset are queued for embedding

#### Scenario: Embedder changed
- **WHEN** the kernel boots with a different embedder file than the one that saved `/var/lib/ringzero/mem/index.bin`
- **THEN** "index: not loaded (embedder changed): rebuilding" is logged and every memory record and file is queued for embedding

### Requirement: Memory slot for a new conversation
Before the first user message of a conversation, and the first after a compaction, the system SHALL put a block headed "[memory from earlier conversations, kept by the kernel; use it to answer]": facts newest first within 120 tokens, the manual section when it goes first, then at most 3 notes from the top 4 ranked hits, within `!set slot N` tokens (default 200, 0 = off).
Source: kernel/src/agentd.rs SLOT_HEAD, NOTE_MIN, FILE_NOTE_MIN, slot, man_line; kernel/src/agent.rs ask

#### Scenario: Note filters
- **WHEN** a hit is a fact, has a raw cosine under 0.55 (under 0.40 for a BM25 hit), a body under 20 characters (under 8 for a file chunk `[/path] ...`), or is the message itself
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
The agent's `recall` tool SHALL return the best manual section first when it goes first and its raw cosine is at least 0.55, then the top 3 ranked hits that are not manual sections (superseded ones left out), each as "N. (score X.XX) LINE" with the raw cosine and notes cut to 300 bytes, and "nothing found in memory" when there are no hits.
Source: kernel/src/agentd.rs howto, recall, hits_split, ranked; kernel/src/agent.rs exec

#### Scenario: Empty memory
- **WHEN** the index and the queue are empty and the agent calls `recall`
- **THEN** the tool result is "nothing found in memory"

### Requirement: Memory shell commands
The system SHALL provide `/facts`, `/remember KEY VALUE`, `/forget KEY`, `/consolidate`, `/agentd` (status, last 10 decisions), `/agentd log N` (default 30, ring of 128), `/agentd merge-legacy [DIR]`, `/index add [-d YYYY-MM-DD] TEXT` (that day 00:00 UTC, else now; `\n` a line break, `\\` a backslash), `/index stats`, `/index clear`, `/index dim N` and `/search QUERY`.
Source: kernel/src/shell.rs CMDS, index_cmd, parse_date, unescape; kernel/src/agentd.rs log_last, merge_legacy

#### Scenario: Clearing the index
- **WHEN** the user types `/index clear`
- **THEN** "[index] cleared: every file and memory record is queued again (idle time at the prompt)" is printed and facts in `/var/lib/ringzero/mem/facts.log` are kept

#### Scenario: A dated entry
- **WHEN** the user types `/index add -d 2026-11-02 ho cambiato telefono`
- **THEN** `/search telefono` shows the entry with the date `2026-11-02 00:00`

## ADDED Requirements

### Requirement: Dated index entries
Every index entry SHALL carry a date in Unix seconds: a memory record's own time (journal, fact or episode record), a file chunk's file mtime (the current time when the file has none), and the current time for a manual section, a live turn or `/index add` without `-d`.
Source: kernel/src/memidx.rs TS, ts, store, store_row; kernel/src/agentd.rs apply, said_begin, file_ts, embed_slice, queue_manual

#### Scenario: Rebuild keeps the record times
- **WHEN** the index is rebuilt from the stores
- **THEN** each `[fact]`, `[said]` and `[episode N]` entry has the time of the record it came from

### Requirement: Migration of an older index
A version 1 index SHALL load with file chunks dated by their file's mtime (indexed-file table), fact entries by their fact record and the rest by the boot date, counts logged; an index cut with chunker 1 SHALL drop its `.csv` and `.tsv` chunks and queue those files again. Both change the index, so the next idle save writes version 2.
Source: kernel/src/agentd.rs init, migrate; kernel/src/memidx.rs load, set_ts

#### Scenario: Upgrade from 0.1.1
- **WHEN** a disk written by ringzero 0.1.1 boots on this kernel
- **THEN** the index loads without re-embedding, "index: version 1 had no dates: N file chunks dated by their file's mtime, F facts by their record, M other entries by the boot date" is logged, and after the next idle save index.bin is version 2

### Requirement: Hub penalty
The system SHALL give each entry a hub score, the mean cosine to its 10 nearest of 54 compiled-in generic probe questions (embedded as queries, baked with the manual vectors or embedded at the first search), and subtract half of it from the entry's cosine when ranking.
Source: kernel/src/memrank.rs PROBES, HUB_W, hub_of, penalties; kernel/src/memidx.rs set_probes, hubs; kernel/src/agentd.rs ensure_probes, queue_manual

#### Scenario: Probes built in
- **WHEN** the kernel boots with matching baked vectors
- **THEN** "hub: 54 probe questions from the vectors in the kernel" is logged and no probe is embedded

### Requirement: Owner penalty
A memory entry (not a file or manual entry) SHALL lose 0.03 when it has an owner the query does not name and 0.06 when the query names an owner it lacks. Owners are the relations (sister, brother, mother, father, colleague, friend, partner; Italian and English words, fact keys too) named in a text, plus those of names in an alias table (the capitalized word nearest a relation word, within 5 words, in a memory).
Source: kernel/src/memrank.rs REL, CAPSTOP, rel_words, aliases, owner, penalties

#### Scenario: Someone else's cat
- **WHEN** the index holds "il gatto di Giulia si chiama Mochi" and "la mia ragazza si chiama Giulia" and the query is "come si chiama il mio gatto?"
- **THEN** the Mochi entry loses 0.03 (its owner, the partner, is not in the query)

### Requirement: Supersession
Of the first 8 non-manual hits, the first and those within 0.2 of it whose vector has a cosine of 0.6 or more with its vector (not chunks of its own file) SHALL form a group; the newest (date, then the later entry) SHALL go first and the others SHALL leave the answer of recall, the slot and `/search`, staying in the index and on disk. Each supersession SHALL be logged.
Source: kernel/src/memrank.rs SUP_DD, SUP_MARGIN, SUP_TOP; kernel/src/agentd.rs ranked, hits_split; kernel/src/memidx.rs dd

#### Scenario: A newer phone
- **WHEN** the index holds "[fact] phone: Motorola g54" dated 2026-02-10 and "[said] ho cambiato telefono, adesso ho un iPhone 15" dated 2026-11-02, and the query is "che telefono ho?"
- **THEN** the iPhone entry is first and `/search` prints the Motorola entry as superseded by it

### Requirement: Manual section ranked apart
The best manual section SHALL go before the other hits for any query when its score is at least 0.06 above the best other hit, and for a how-to message when its raw cosine is at least 0.55.
Source: kernel/src/memrank.rs MAN_MARGIN; kernel/src/agentd.rs ranked, RECALL_MIN

#### Scenario: Not a how-to wording
- **WHEN** the query is "the screen is too small" and the manual's display section scores 0.06 over every memory
- **THEN** that section is first in `/search`

### Requirement: Tables indexed by rows
The file indexer SHALL cut a `.csv` or `.tsv` file into one entry per non-blank data row, `[/path] HEADER` then a line break and the row, where HEADER is the first non-blank line; other text files keep the 1000-byte chunks.
Source: kernel/src/fidx.rs VERSION, table_file, next_line, next_chunk

#### Scenario: A small table
- **WHEN** `/demo/people.csv` holds a header and 5 rows
- **THEN** 5 entries are indexed, each starting with `[/demo/people.csv]` and the header line

### Requirement: Index listing
With the stores on, every index save SHALL also write `/var/lib/ringzero/mem/index.tsv`: a comment line, then one line per entry with its id, date (`YYYY-MM-DD HH:MM` UTC), source tag and the first 80 bytes of its text.
Source: kernel/src/agentd.rs INDEX_LIST, index_list, date, save_index

#### Scenario: Reading the dates
- **WHEN** the user types `!cat /var/lib/ringzero/mem/index.tsv` after the index was saved
- **THEN** each entry is listed with its date
