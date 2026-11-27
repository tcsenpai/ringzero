# ringzero manual

This is the manual the system agent reads. The kernel ships it inside the kernel image and indexes each `##`
section as one entry of the memory index (source `[manual: TITLE]`); `/man QUERY` searches it, the `recall`
tool and the memory slot return it to the model. Rules for editing: one topic per section, each section answers
a question alone (name the command in it, do not refer to other sections), at most about 900 bytes per section,
English. A line that starts with `Asked as:` helps retrieval (Italian and English wordings) and is not shown to
the model. Changing a section re-embeds only that section at the next boot.

## Talking to the system agent
Asked as: how do I use the shell, come si usa, how do I talk to the agent, cosa posso fare
Type a plain sentence at the `ringzero[main]>` prompt to talk to the system agent: it answers and can call kernel tools (files, memory, CPU, settings, Python, WASI programs). A line that starts with `!` is a machine command (`!ls /`, `!mem`), a line that starts with `/` is a session command (`/clear`, `/undo`). Ctrl-C stops the agent or a running program. `/help` lists every command.

## Command list and help
Asked as: what commands are there, quali comandi ci sono, list of commands, elenco comandi, help
`/help` (or `!help`) prints every command with a one-line description. A bare `/` or `!` followed by a pause or Enter lists the commands of that kind. Tab completes command names and their arguments, the up and down arrows recall earlier lines. `/man TOPIC` searches this manual, for example `/man undo`.

## Undo and redo a turn
Asked as: how do I go back one turn, come torno indietro di un turno, annulla l'ultima risposta, undo, take back the last answer
`/undo` goes back one turn of the conversation, `/undo N` goes back N turns; `/redo` (or `/redo N`) steps forward again. After an undo the prompt shows `ringzero[main~1]>`. `/retry` asks the last question again. Only the model's state goes back: files written and settings changed by tools stay as they are. Going back costs about 1 ms.

## Branches and saved points
Asked as: how do I save a conversation, salvare la conversazione, branches, rami, punti
Every finished turn is saved as a point (`#7`). `/tree` shows points, branches and saves (`/tree all` shows every point). `/save NAME` keeps the current point, `/load NAME` goes back to it. `/branch NAME` starts a new line here, `/switch NAME|#ID` goes to a branch, a save or a point, `/drop NAME|#ID` deletes one. Typing after an `/undo` starts a side line; the old one stays in `/tree`. `/snap` prints the snapshot pool and the state hash.

## New conversation
Asked as: how do I start over, ricominciare da capo, nuova conversazione, clear the chat, pulire lo schermo
`/clear` clears the screen and starts a new conversation from the boot prompt. Saved facts and the memory index stay; the first message of the new conversation gets the memory slot again.

## Context size and compaction
Asked as: how full is the context, quanto contesto resta, compact, riassumere la conversazione
The context holds 32768 tokens. `/ctx` shows how many are in use. Past 75% fill the agent summarizes the conversation and rebuilds the context (compaction); `/compact` does it now, `!set compact N` changes the percentage. `/history [N]` prints the raw transcript, kept across compactions. `/undo` after a compaction returns to the uncompacted context.

## Hibernation
Asked as: how do I hibernate, come faccio a ibernare, ibernazione, save the session before power off, resume after reboot
Type `/hibernate` (a session command: it starts with a slash, not with `!`) to save the current conversation and all its snapshot points to the disk (`/sys/hib/image.bin`); every `!poweroff` does the same. At the next boot the conversation is resumed exactly where it was. `/hibernate status` says whether an image exists; `/hibernate discard` deletes it, so the next boot starts a fresh conversation.

## Power off and reboot
Asked as: how do I shut down, come spengo, spegnere, riavviare, reboot
`!poweroff` powers the machine off and `!reboot` restarts it; both flush the disk and save the conversation for hibernation first. You can also ask the agent ("spegni il computer"); it asks y/N before doing it. `!sync` flushes the disk without powering off.

## Listing and reading files
Asked as: how do I list files, come vedo i file, leggere un file, show a file, disk space, spazio libero
`!ls PATH` lists a directory on the disk (`!ls /` for the root), `!cat PATH` prints a file, `!df` shows the disk size and free space. Paths are absolute (start with `/`), without `.` or `..`. The agent can do the same with its `ls` and `cat` tools ("leggi /notes/todo.txt").

## Writing and deleting files
Asked as: how do I create a file, come creo un file, scrivere un file, delete a file, cancellare, make a folder, creare una cartella
`!write PATH TEXT` creates or replaces a file (`\n` in TEXT is a new line; missing parent folders are created). `!mkdir PATH` makes a folder, `!rm PATH` deletes a file and `!rm -r PATH` a folder with its content; `!rm` always asks y/N. Files are at most 64 KiB. Text files are added to the memory index in idle time, so the agent can find them later.

## Running Python
Asked as: how do I run python, come eseguo python, usare python, run a python script, eseguire uno script
Two Pythons. MicroPython is built in: `!py CODE` runs one line (`!py 17*23`), `!py -f PATH` runs a script file; 10 s budget, Ctrl-C stops it. CPython 3.15 runs as a WASI program: `!run python3 /demo/stats.py` runs a script, `!run python3` alone opens the REPL (Ctrl-D ends it). You can also ask the agent "quanto fa 2**100? usa python": it uses its `run_code` tool (MicroPython).

## Running WASI programs
Asked as: how do I run a program, come lancio un programma, eseguire un wasm, run a command, coreutils
`!run NAME [ARGS] [< FILE]` (the same as `!wasm`) runs a WASI program from the disk: NAME is `/bin/NAME.wasm`, otherwise a coreutils tool (`!run ls -l /`, `!run wc -l < /notes/todo.txt`). `"..."` keeps spaces in an argument, `< FILE` is the input. 60 s budget, Ctrl-C stops it. The agent has the same programs through its `run_wasm` tool.

## Programs in /bin
Asked as: how do I list the programs in /bin, quali programmi ci sono, which programs are installed, programmi installati
`!ls /bin` lists the installed programs. They are WASI programs put there by `make wasm-apps`: coreutils (ls, cat, wc, sort, ...), rg (ripgrep), jq, sqlite3, lua, python3 (CPython 3.15), qjs (QuickJS JavaScript) and kibi (a text editor), plus rztest (runtime checks). Run one with `!run NAME`, for example `!run rg -n latte /notes`.

## SQLite, Lua, JavaScript, jq and the editor
Asked as: how do I use sqlite, come uso lua, javascript, edit a file, modificare un file, editor
`!run sqlite3 FILE "SQL"` runs a query (`!run sqlite3 /demo/shop.db "select 1+1;"`), `!run sqlite3 FILE` alone opens its REPL. `!run lua FILE` or `!run lua` (REPL), `!run qjs FILE.js` for JavaScript, `!run jq .version /demo/data.json` for JSON. `!run kibi FILE` opens a text editor (serial console only). REPLs end with Ctrl-D.

## Keyboard layout
Asked as: how do I change the keyboard layout, come cambio layout tastiera, tastiera italiana, US keyboard
`!kbd it` switches the PS/2 keyboard to the Italian layout, `!kbd us` to the US layout; `!kbd` alone shows the current one. `!set keyboard it` does the same, and the agent can do it ("set the keyboard layout to us"). The serial console is not affected: it sends characters, not key codes.

## Settings
Asked as: how do I change a setting, come cambio le impostazioni, settings list, configurazione
`!set KEY VALUE` changes a setting: temperature (0 = greedy), cores (cores used for inference), keyboard (us or it), think (on or off), compact (percent of context before compaction), snappool (MiB), consolidate (idle seconds, 0 = off), slot (memory slot tokens, 0 = off), constrain (on or off: tool-call grammar), spec (on, off or N from 1 to 15: speculative decoding), gramdraft (on or off), confidence (on or off), seed (sampler RNG), color (on or off: console colors). On/off settings take on, off, 1, 0, true or false; anything else is refused with an error and the setting keeps its value. A hibernation image keeps temperature, think, constrain, compact and color; the others go back to their defaults at boot.

## Temperature and thinking
Asked as: how do I make answers deterministic, temperatura, more creative answers, thinking mode, ragionamento
`/temp 0.7` or `!set temperature 0.7` sets the sampling temperature; 0 is greedy (the same question gives the same answer). `/think on` lets the model think before it answers (slower), `/think off` turns it off.

## CPU, cores, memory and uptime
Asked as: how many cores, quanti core, how much RAM, quanta memoria, uptime, da quanto e acceso
`!cpu` shows the CPU model, clock and features; `!cores` the cores online and `!cores N` sets how many are used for inference; `!mem` total and free memory, the model and the engine arena; `!uptime` the time since boot; `!stats` threads, parking, arena and context. The agent can answer the same questions with its tools.

## Kernel log
Asked as: how do I see the log, vedere il log, errori nel log, kernel messages
`!log` prints the last 20 kernel log lines, `!log N` the last N. The agent reads it with its `read_log` tool ("mostra le ultime 5 righe del log").

## Saved facts
Asked as: how do I make it remember something, come faccio a fargli ricordare, ricordati, forget a fact, dimenticare
The agent keeps facts about you across reboots: say "ricordati che il mio colore preferito e il verde" and it calls `remember`. `/facts` lists the saved facts (key = value, source, time), `/remember KEY VALUE` saves one yourself, `/forget KEY` deletes one (asks y/N). Saved facts are put in front of the first message of every conversation.

## What the memory system decided
Asked as: how do I see what the memory system decided, cosa ha deciso la memoria, agentd log, why did it remember, memory decisions
`/agentd` shows the memory subsystem (agentd): stores, index, idle queue, consolidation state and the last 10 decisions. `/agentd log N` prints the last N decisions: what went into the memory slot and why (each candidate with its score), facts accepted or rejected with the reason, index saves, file scans and boot cache hits.

## Memory slot and consolidation
Asked as: memory slot, consolidare, consolidation, riassunto della sessione, turn off memory
The first message of every conversation gets a memory slot (`[memory]` lines): saved facts and notes recalled for that message, at most 200 tokens; `!set slot 0` turns it off. After 60 s idle with new turns the agent summarizes the session into an episode and facts (consolidation); `/consolidate` does it now, `!set consolidate N` sets the idle seconds (0 = only by hand).

## Memory index and search
Asked as: how do I search my notes, cercare nei file, search memory, indice, find a file by meaning
`/search QUERY` returns the top 5 entries of the memory index (files, facts, past statements, episodes, this manual) by meaning. `/index stats` shows its size and queue, `/index add TEXT` stores a note, `/index clear` rebuilds it. The agent uses the same index through its `recall` tool ("cerca nei file qualcosa sul latte").

## The manual
Asked as: is there a manual, c'e un manuale, documentation, documentazione, man page
`/man TOPIC` searches this manual and prints the best sections, for example `/man ibernare` or `/man python`; `/man` alone lists the section titles. The agent finds the same sections with its `recall` tool.

## Confidence of tool calls
Asked as: what is the confidence line, cosa vuol dire confidence, why does it ask Run it, perche chiede conferma, turn off the confidence question
Every tool call prints a `[confidence]` line: the smallest margin (0 to 1) between the model's best and second choice at the call's decisions (tool name, parameters, values), where it was, and what the kernel does. Below 0.80 a tool that writes (write_file, mkdir, remember, set_setting) asks `Run it? [y/N]` first; only y runs it, any other answer tells the model the call was not run. Any other tool, run_code and run_wasm included, runs with a notice to check its result. `!set confidence off` turns the line and the question off, `!set confidence on` back on (on at every boot). The 0.80 line is set when the system is built (`make iso CONF_ASK=N`, a percent).

## Speed and benchmarks
Asked as: how fast is it, quanto e veloce, benchmark, tokens per second, speculative decoding
`/bench` runs a 64-token decode benchmark (the conversation is restored after). `/prefill` times the boot prompt prefill. Every answer ends with a `[decode]` line with its tokens per second. `!set spec off` turns speculative decoding off (it speeds up answers that copy text from the conversation), `!set spec N` sets the longest draft.

## What ringzero is
Asked as: what is ringzero, cos'e ringzero, which model, quale modello, chi sei
ringzero is an experimental operating system: its own Rust kernel (not Linux), booted by Limine, with a Qwen3.5 language model running inside the kernel (ring 0) as the system agent, and Qwen3-Embedding-0.6B for the memory index. The banner at boot shows the edition (for example `ringzero Qwen35@4b.q4-Qwen3@0.6b`).
