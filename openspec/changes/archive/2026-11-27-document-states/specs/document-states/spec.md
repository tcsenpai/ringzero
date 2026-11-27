## ADDED Requirements

### Requirement: Read a file into the conversation
`/read PATH` SHALL put a text file into the conversation: a user turn `Here is the file PATH (N bytes). Read it, my questions about it follow.` followed by the file's text, encoded without special tokens, and an assistant turn with the fixed text `I have read PATH.`; a fresh conversation SHALL first get the 2 tokens that close the system turn. The result SHALL be saved as a snapshot point labelled `/read PATH`.
Source: kernel/src/docstate.rs cmd, read; kernel/src/shell.rs CMDS

#### Scenario: Read and ask
- **WHEN** the user types `/read /docs/handbook.txt` and then a question about the file
- **THEN** the agent answers from the file's text, and `/tree` shows a point labelled `/read /docs/handbook.txt`

### Requirement: Read output and refusals
After a read the shell SHALL print `[read] PATH is in the context (N tokens, ctx P), S s in all (snapshot M ms); ask about it`. A file larger than 65536 bytes, not text, missing, or that does not fit in the context with 600 tokens to spare SHALL be refused with a `[read]` line, and the conversation SHALL not change.
Source: kernel/src/docstate.rs read, MAX_BYTES

#### Scenario: Too large
- **WHEN** the user types `/read` with a file of 100000 bytes
- **THEN** the shell prints `[read] PATH has 100000 bytes; /read takes files up to 65536 bytes` and nothing is added

### Requirement: Document state cache
For a file of 512 tokens or more, `/read` SHALL store the engine state its tokens leave (token ids, KV rows of their positions, the DeltaNet block R) in `/var/lib/ringzero/cache/docs/<16 hex digits>.bin`, keyed by the boot key (model, prefix, engine, KV geometry), the conversation's token ids before the file and the file's token ids.
Source: kernel/src/docstate.rs read, save, Key, MIN_TOK; kernel/src/snap.rs state_parts, geometry

#### Scenario: First read
- **WHEN** the user reads a 4000-token file for the first time
- **THEN** the file is prefilled and the log line ends with `cached as /var/lib/ringzero/cache/docs/...bin (N MiB, written in S s)`

### Requirement: Cache hit is bit-exact
A later `/read` with the same key SHALL load the stored state instead of prefilling, and the live state SHALL then be bit-identical to the one the prefill leaves, so the next greedy answer is the same. The kernel log SHALL get one `doc:` line per read: a hit with the bytes and the load time, or a prefill with its time and where it was cached, or why it was not.
Source: kernel/src/docstate.rs read, load

#### Scenario: Second read after a fresh start
- **WHEN** the user reads the same 4000-token file twice, each time as the first thing of a fresh conversation (after boot or `/clear`)
- **THEN** the second read logs `cache hit`, `/snap` prints the same state hash as after the first read, and the same question at temperature 0 gets the same answer

#### Scenario: Other context
- **WHEN** the same file is read after a different conversation
- **THEN** the key differs, the file is prefilled, and a new entry is written

### Requirement: Invalidation of a changed file
A file whose content changed SHALL get a new key, and writing its new entry SHALL delete the older entry of the same path, context and boot key, with the log line `doc: PATH changed, old cache entry FILE removed`.
Source: kernel/src/docstate.rs make_room, entries

#### Scenario: Edited file
- **WHEN** the user changes /docs/handbook.txt and reads it again in a fresh conversation
- **THEN** the file is prefilled, the old entry is removed and `/read` lists one entry for the path

### Requirement: Size cap and eviction
The cache SHALL stay within 512 MiB and SHALL leave 64 MiB free on the disk: before a write it evicts entries of another model or engine first, then the least recently used ones (a hit refreshes the entry's time). An entry larger than the cap, or one that does not fit after eviction, is not written, and the read still succeeds from the prefill.
Source: kernel/src/docstate.rs make_room, CAP, RESERVE, load

#### Scenario: Cap reached
- **WHEN** a new entry would take the cache over 512 MiB
- **THEN** the least recently used entries are deleted first, each with a `doc: evicted FILE (PATH, N MiB)` log line

### Requirement: Corrupt or stale entries are refused
`/read` SHALL refuse a cache entry whose magic, version, keys, KV geometry, size, token ids or checksum do not match, SHALL delete it, log `doc: cache entry FILE for PATH refused (REASON), deleted; prefilling`, restore the current snapshot point when the DeltaNet state was already overwritten, and prefill the file; the conversation SHALL then be the same as without the cache.
Source: kernel/src/docstate.rs load, read

#### Scenario: Flipped byte
- **WHEN** one byte inside a cache entry's state is changed on the disk and the file is read again in a fresh conversation
- **THEN** the log says `refused (checksum mismatch)`, the file is prefilled, and `/snap` shows the same state hash as a prefill

### Requirement: Cache listing and clearing
`/read` without an argument SHALL print the cache directory, the number of entries, their size against the 512 MiB cap and the free space on the disk, and one line per entry (file name, path, tokens, position, MiB, and `(another model or engine: evicted first)` for an entry of another boot key); `/read clear` SHALL delete the directory and print `[read] cache cleared: N entries, M MiB`.
Source: kernel/src/docstate.rs cmd, list, entries

#### Scenario: List
- **WHEN** the user types `/read` after one cached read
- **THEN** the shell prints `[read] document cache /var/lib/ringzero/cache/docs: 1 entries, ...` and the entry's line
