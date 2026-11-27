## Why
Follow-up F8, found on the 0.1.0 beta (4B q4): `!write /notes/todo.txt buy milk`, a short idle, `/clear`, then
"what do I need to buy?" answered "no information". The file is indexed as `[/notes/todo.txt] buy milk`; the memory
slot takes the body after `] ` ("buy milk", 8 characters) and drops every note body under 20 characters (T35). That
rule was tuned against greetings and short requests in `[said]` entries ("ciao", "hello"), but a file the user
wrote is the user's own note, and a short one is still the whole answer.

## What Changes
- The 20-character minimum stays for `[said]` and `[episode N]` notes. A file chunk (`[/path] ...`) needs a body of
  8 characters or more (FILE_NOTE_MIN): "buy milk" passes, one-word files ("ciao", "hello", "prova") stay
  out. A first try with 3 let those in: they scored 0.55 or more against unrelated requests and one in-OS eval
  prompt ("crea la cartella /notes/m47", with /notes/m47.txt "prova" in the slot) called `rm` instead of `mkdir`. The cosine floors (0.55, 0.40 for a BM25 hit) still apply to every note.
- The `recall` tool is unchanged: it never applied the length minimum.

## Impact
- kernel/src/agentd.rs FILE_NOTE_MIN, slot.
- Specs: memory (memory slot, note filters).
- Docs: INFO.md, docs/ARCHITECTURE.md, docs/BETA.md, release/NOTES.md (the 0.1.0 known limit is fixed after the
  release), docs/M45_RESULT.md (new section with the A/B).
