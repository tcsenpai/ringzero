## Why
A long text file costs a long prefill every time it enters the conversation: a 4k-token file takes tens of seconds
on the 2B and about a minute on the 4B. The engine state those tokens leave can be stored on the disk like the boot
snapshot and loaded instead (T27, milestone M4.8c). On a hybrid model the state of a token depends on every token
before it (the DeltaNet recurrent state, and the attention KV through the layers below), so a stored document state
is valid only at the same position after the same context.

## What Changes
- New session command `/read PATH`: puts a text file (64 KiB at most, encoded without special tokens) into the
  conversation as a user turn with a fixed kernel reply (`I have read PATH.`) and saves it as a snapshot point. A
  fresh conversation first closes the system turn, as a question does.
- Document states: for a file of 512 tokens or more, the KV rows of its positions, its token ids and the DeltaNet
  block R are written to `/var/lib/ringzero/cache/docs/<key>.bin`, keyed by the boot key (model, boot prefix,
  engine, KV geometry), the token ids of the conversation before it and the file's token ids. The next `/read` of
  the same file after the same context loads the state instead of prefilling it; the result is the same state, bit
  for bit, as the prefill.
- A changed file gets another key; the older entry of the same path and context is deleted when the new one is
  written. Size cap 512 MiB with least recently used eviction (entries of another model or engine first) and a
  64 MiB free-space reserve on the disk. An entry with a bad header, size, token ids or checksum is refused,
  deleted, and the file is prefilled.
- `/read` alone lists the cache, `/read clear` deletes it. Log lines `doc: ...` for every hit, prefill, write,
  refusal and eviction.

## Impact
- New kernel/src/docstate.rs; kernel/src/snap.rs state_parts, geometry; kernel/src/fs.rs free_bytes;
  kernel/src/shell.rs CMDS and the dispatch; kernel/src/main.rs.
- Specs: new capability document-states.
- Docs: README.md, INFO.md, docs/MANUAL.md, docs/ARCHITECTURE.md (diagram), docs/ROADMAP.md, site/index.html,
  docs/M48C_RESULT.md.
- Out of scope: the agent's `cat` tool. Its result enters mid-turn after the model's own call, so the context before
  it differs on almost every turn and a cached state would almost never apply.
