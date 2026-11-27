# snapshots-hibernation Specification

## Purpose
Baseline of the engine state snapshots (a tree of points kept in a pool of 1 MiB chunks, with undo, redo, branches, saves and eviction), the boot snapshot cache, and hibernation of the whole conversation to the disk with resume at the next boot.

## Requirements

### Requirement: Snapshot pool and root point
The system SHALL allocate at boot a snapshot pool of 1024 chunks of 1 MiB, halving the request down to 64 MiB while the arena cannot provide it, and SHALL save the prefilled boot prefix as point #0 (the root) on the branch `main`.
Source: kernel/src/snap.rs CHUNK, POOL_MIB, goto, init_root; kernel/src/agent.rs init

#### Scenario: Fresh boot
- **WHEN** the kernel finishes booting
- **THEN** `/tree` shows `#0` labelled "boot prefix" with the name `main` and the marker "<- here"

### Requirement: Every finished turn is a point
The system SHALL save each finished turn as a new point with the next id, as a child of the node holding the live rows, storing the KV rows from the parent's position to the new position, their token ids and the whole DeltaNet block R, and SHALL make it the current point and the tip of the current branch; a turn that added no tokens is not saved.
Source: kernel/src/snap.rs gc, save, restore_root, commit

#### Scenario: Two turns
- **WHEN** the user asks two questions after boot
- **THEN** `/tree all` lists `#1` and `#2`, `#2` carries the name `main` and is marked "<- here"

### Requirement: Table limits and names
The system SHALL keep at most 256 points and 64 names, and SHALL accept a branch or save name only if it is 1 to 16 characters of `a-z`, `0-9`, `_` and `-` and not already used.
Source: kernel/src/snap.rs MAX_NODES, MAX_NAMES, valid_name, branch, save_cmd

#### Scenario: Invalid name
- **WHEN** the user types `/save My-Point`
- **THEN** "usage: /save NAME  (1-16 of a-z 0-9 _ -)" is printed and no name is added

### Requirement: Pool eviction keeps a reserve
The system SHALL, before saving a point, free space until 64 chunks beyond the need are free: first by thinning (R freed, rows kept) the least recently visited unnamed point that is not current and not among the last 16 points of the current history, then by dropping the oldest anonymous leaf off the current history last visited 32 or more turns ago, else the oldest one of any age.
Source: kernel/src/snap.rs DROP_AGE, is_leaf, free_slot

#### Scenario: Anonymous means unnamed or kernel-named
- **WHEN** a leaf off the current history carries only an unpinned kernel name such as `main@5`
- **THEN** it is a drop candidate like an unnamed leaf

#### Scenario: Many fresh conversations fill the pool
- **WHEN** the pool is full of anonymous leaves of `/clear` conversations all visited fewer than 32 turns ago
- **THEN** the oldest of them is dropped ("snap: dropped #N" in the kernel log) and the new turn is saved

### Requirement: Names chosen by the user are never evicted
The system SHALL never thin or drop, during eviction, a point that carries a branch name, a `/save` name, or a pinned kernel name.
Source: kernel/src/snap.rs thin_candidate, drop_candidate

#### Scenario: Saved point under pressure
- **WHEN** the user ran `/save keep1` on an old point and the pool fills up
- **THEN** that point is neither thinned nor dropped

### Requirement: Pool full refuses the save
The system SHALL, when eviction cannot make room, leave the turn unsaved, print "[snap] pool full: this turn is not saved and Ctrl-C would lose it. /drop something or /save less", and show `*` in the prompt until a point is restored or saved.
Source: kernel/src/snap.rs commit, last_question, prompt, status

#### Scenario: Unsaved marker
- **WHEN** a turn's save is refused
- **THEN** the prompt reads "ringzero[main*]> " and `/snap` says "live state NOT saved (pool full)"

### Requirement: Restoring a thinned point replays tokens
The system SHALL restore a thinned point by copying R from its nearest ancestor that still has one and replaying the stored token ids from there, printing whether the resulting R hash matches the one stored at thinning; Ctrl-C during the replay SHALL keep the previous current point; after a successful replay R is stored again when there is room.
Source: kernel/src/snap.rs save, restore

#### Scenario: Replay
- **WHEN** the user runs `/switch #2` and `#2` is thinned
- **THEN** "[snap] #2 was thinned: replaying N tokens from #M" is printed, then "[snap] replay done, R hash matches the stored one"

### Requirement: Undo and redo move only the model state
The system SHALL move `/undo [N]` back N points (default 1) along the current history and `/redo [N]` forward N points (default 1) toward the branch tip, and for each point stepped over by an undo whose turn had tool side effects SHALL print "[undo] note: #ID ran ...; the machine keeps it."
Source: kernel/src/snap.rs valid_name, redo; kernel/src/shell.rs run

#### Scenario: Nothing to undo or redo
- **WHEN** the user types `/undo` at the root, or `/redo` at the branch tip
- **THEN** "[undo] nothing to undo (at the boot prefix)" or "[redo] nothing to redo (at the tip of BRANCH)" is printed

#### Scenario: Undo over a file write
- **WHEN** the last turn called `write_file` and the user types `/undo`
- **THEN** the current point is the previous one and a note says the machine keeps the write

### Requirement: Prompt shows branch and distance from the tip
The system SHALL show the prompt as "ringzero[BRANCH]> ", with "~K" after the branch name when the current point is K steps behind the branch tip and `*` when the live state is unsaved.
Source: kernel/src/snap.rs last_question, prompt

#### Scenario: After two undos
- **WHEN** the user types `/undo 2` on `main`
- **THEN** the prompt is "ringzero[main~2]> "

### Requirement: A new turn after an undo keeps the old line
The system SHALL, when a turn is committed while the current point is behind the tip, or when the branch moves off its history, give the old tip the kernel name "BRANCH@ID" if no other name points at it, so it stays reachable.
Source: kernel/src/snap.rs steps, keep, restore, goto, commit

#### Scenario: Side line after undo
- **WHEN** on `main` at `#3` the user types `/undo` then asks a new question
- **THEN** `/tree` shows `#3` with the name `main@3` and the new point is the tip of `main`

### Requirement: Branches, saves and switching
The system SHALL provide `/branch NAME` (a new branch at the current point that becomes current), `/save NAME` (a save name on the current point, "kept until /drop NAME"), and `/switch NAME|#ID` or `/load NAME|#ID`, where a branch name switches branch and a save name or point id moves the current branch there; switching to a kernel-named branch on purpose makes it a user branch that is never dropped.
Source: kernel/src/snap.rs branch, switch; kernel/src/shell.rs run

#### Scenario: Unknown target
- **WHEN** the user types `/switch nosuch`
- **THEN** "[switch] no branch, save or point 'nosuch' (/tree lists them)" is printed

### Requirement: Dropping names and points
The system SHALL let `/drop NAME` remove any name except the current branch and then drop every unnamed leaf left unreferenced, and SHALL let `/drop #ID` remove a point only if it is not the root, not current, not on the current history, has no later points and has no names, printing the MiB freed.
Source: kernel/src/snap.rs free_slot, gc, drop_cmd

#### Scenario: Dropping a point with children
- **WHEN** the user types `/drop #1` while `#2` is a child of `#1`
- **THEN** "[drop] #1 is on the current line, has later points or has names: drop those first" is printed

### Requirement: New conversation keeps the old line
The system SHALL, on `/clear`, name the old tip "BRANCH@ID" (unless the tip is the root or already named) and move the current branch and point to the root.
Source: kernel/src/snap.rs commit, new_line; kernel/src/agent.rs reset

#### Scenario: Clear
- **WHEN** the user types `/clear` on `main` at `#4`
- **THEN** the current point is `#0` and `#4` carries the name `main@4`

### Requirement: Compaction keeps the uncompacted point
The system SHALL, when a compaction is committed as point N, add the pinned save name "pre-compact-N" on the point before it and unpin every earlier pre-compact name, so only the latest uncompacted line is protected from eviction.
Source: kernel/src/snap.rs commit

#### Scenario: Second compaction
- **WHEN** the second compaction of a session is committed
- **THEN** `pre-compact-2` is pinned and `pre-compact-1` becomes evictable

### Requirement: Snapshot inspection and pool cap
The system SHALL provide `/tree` (compact tree, inner points of linear runs collapsed to "... N more"), `/tree all` (every point), `/snap` (current point, pool use, last save and restore timings, and the state hash when the live state is the saved point), and `!set snappool MIB` accepting 64 up to the allocated pool size and never set below the chunks in use.
Source: kernel/src/snap.rs drop_cmd, max_mib; kernel/src/shell.rs set_setting, run

#### Scenario: Cap out of range
- **WHEN** the user types `!set snappool 10`
- **THEN** "error: snappool must be 64..N MiB" is printed

### Requirement: Boot snapshot cache
The system SHALL write the prefilled boot prefix (token ids, every K/V plane's rows, R) to `/sys/bootsnap.bin` with a 48-byte `RZBS` header keyed by the prefix token ids, the model hash, the engine build hash and the KV geometry, and SHALL load it at the next boot instead of prefilling only when magic, version, key, geometry, size, checksum and token count all match.
Source: kernel/src/snap.rs init_root, load_prefix; kernel/src/agent.rs build_prefix, init

#### Scenario: Engine rebuilt
- **WHEN** the kernel boots with a different engine build than the one that wrote `/sys/bootsnap.bin`
- **THEN** "boot cache: miss (key differs (model, prefix or engine changed)), prefilling N tokens" is logged and the cache is rewritten

### Requirement: Hibernation commands
The system SHALL write the hibernation image `/sys/hib/image.bin` on `/hibernate` and on every `!poweroff` (not on `!reboot`), SHALL on `/hibernate discard` delete the image and stop the next poweroffs of this session from writing one until a plain `/hibernate`, and SHALL on `/hibernate status` print the image size or "[hib] no image: the next boot starts fresh".
Source: kernel/src/hib.rs SKIP, boot, cmd; kernel/src/sys.rs poweroff, reboot; kernel/src/shell.rs run

#### Scenario: Discard then poweroff
- **WHEN** the user types `/hibernate discard` then `!poweroff`
- **THEN** the image is deleted, the log says "hib: poweroff without an image (/hibernate discard)", and the next boot starts a fresh conversation

### Requirement: What the image holds
The hibernation image SHALL hold a 64-byte `RZHB` version 1 header (key, state hash, sizes, checksum, pool chunk count), the metadata of every non-root point (with its question, answer and side-effect texts) and every name, the KV row chunks of every non-root point, R of the current point only, and the agent settings (temperature, think, constrain, compaction percent and count, color) and transcript.
Source: kernel/src/snap.rs load_prefix, hib_save; kernel/src/hib.rs key, extra

#### Scenario: Mid-turn poweroff
- **WHEN** `!poweroff` runs while the live state is ahead of the current point
- **THEN** the image holds the current point, not the unsaved live state

### Requirement: Resume at boot
The system SHALL at boot resume the image only if magic, version, key (boot snapshot key mixed with "ringzero hibernate v1"), pool chunk count, boot prefix length, sizes and checksum match, restoring tree, names, settings and transcript with every point but the current one thinned, and logging the state hash as "same as saved" or "DIFFERS from the saved one"; else it SHALL start fresh.
Source: kernel/src/hib.rs key, apply, save, boot; kernel/src/snap.rs hdr_u, txt; kernel/src/main.rs kmain

#### Scenario: Model changed
- **WHEN** the kernel boots with a different model than the one that wrote `/sys/hib/image.bin`
- **THEN** "hib: image /sys/hib/image.bin not resumed (key differs ...), fresh conversation; /hibernate discard deletes it" is logged

#### Scenario: Same model
- **WHEN** the kernel boots with the same model, prefix, engine and pool size
- **THEN** "hib: resumed #N on BRANCH (ctx P), K points ... same as saved" is logged and `/tree` shows the hibernated tree

### Requirement: No partial image
The system SHALL delete `/sys/hib/image.bin` when writing it fails, and SHALL write nothing when no disk is mounted or the boot has not finished.
Source: kernel/src/hib.rs apply, save

#### Scenario: Write error
- **WHEN** the disk write of the image fails during `/hibernate`
- **THEN** the file is removed and "hib: hibernate not saved (...), the next boot starts fresh" is logged

### Requirement: OPEN: size of R per point
OPEN: the docs state that R is 19.27 MiB (about 20 chunks) per point and KV rows cost about 12 KiB per token; the code takes both from engine constants at run time (`CTX_DN_ZERO_BYTES`, head dim, KV heads, full-attention layers), so the system SHALL be checked against `/snap`, which prints "R N KiB per point, N KV rows per MiB chunk", rather than against a fixed number.
Source: kernel/src/snap.rs r_bytes, kv_chunks; docs/SNAPSHOTS_V1_RESULT.md section "Times and memory per point"

#### Scenario: Checking the size
- **WHEN** the user types `/snap` on the shipped Qwen3.5 2B edition
- **THEN** the printed R size per point is compared with 19 728 KiB from the docs

## Open questions
- `/load` is an alias of `/switch` in code (both call the same function); the help text describes `/load NAME` as "go to a saved point" only. Intended behaviour for `/load main` (a branch name) is not documented.
- Resume restores R only for the current point; whether a visit to another point after resume always matches the stored R hash is measured in docs, not checked by code beyond the printed comparison.
