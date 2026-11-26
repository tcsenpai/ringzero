
### mode=nothink  (system+tools = 942 tok, 25.5s prefill @37 tok/s)

| id | kind | prompt | calls | parsed | right tool | schema ok | args correct | halluc | unneeded call | prose instead | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | single | quanta RAM libera? | meminfo | Y | Y | Y | Y | - | - | - | **Y** |
| 2 | single | list files in /home | ls | Y | Y | Y | Y | - | - | - | **Y** |
| 3 | single | create notes.txt with hello | write_file | Y | Y | Y | Y | - | - | - | **Y** |
| 4 | single | spegni la macchina | poweroff | Y | Y | Y | Y | - | - | - | **Y** |
| 5 | single | run this: print 2+2 | run_code | Y | Y | Y | Y | - | - | - | **Y** |
| 6 | single | show me /etc/hostname | cat | Y | Y | Y | Y | - | - | - | **Y** |
| 7 | single | crea la cartella /home/progetti | mkdir | Y | Y | Y | Y | - | - | - | **Y** |
| 8 | single | delete /tmp/old.log | rm | Y | Y | Y | Y | - | - | - | **Y** |
| 9 | single | how many cores do I have? | list_cores | Y | Y | Y | Y | - | - | - | **Y** |
| 10 | single | che processore ho? | cpuinfo | Y | Y | Y | Y | - | - | - | **Y** |
| 11 | single | riavvia | reboot | Y | Y | Y | Y | - | - | - | **Y** |
| 12 | single | show the last 20 log lines | read_log | Y | Y | Y | Y | - | - | - | **Y** |
| 13 | single | set the hostname to aios-dev | set_setting | Y | Y | Y | Y | - | - | - | **Y** |
| 14 | single | imposta la lingua su italiano | set_setting | Y | Y | Y | Y | - | - | - | **Y** |
| 15 | single | cosa c'è nella root? | ls | Y | Y | Y | Y | - | - | - | **Y** |
| 16 | single | memory usage please | (none) | - | - | - | - | - | - | Y | **-** |
| 17 | single | rimuovi la directory /tmp/cache e tutto il co | rm | Y | Y | Y | Y | - | - | - | **Y** |
| 18 | single | reboot the system now | reboot | Y | Y | Y | Y | - | - | - | **Y** |
| 19 | single | esegui questo modulo wasm: (module (func (exp | run_code | Y | Y | Y | Y | - | - | - | **Y** |
| 20 | single | ci sono errori nel log? | read_log | Y | Y | Y | Y | - | - | - | **Y** |
| 21 | single | shut down | poweroff | Y | Y | Y | Y | - | - | - | **Y** |
| 22 | single | leggi il file /home/todo.md | cat | Y | Y | Y | Y | - | - | - | **Y** |
| 23 | none | ciao, chi sei? | (none) | Y | n/a | n/a | n/a | - | - | - | **Y** |
| 24 | none | explain what a kernel is in one sentence | (none) | Y | n/a | n/a | n/a | - | - | - | **Y** |
| 25 | none | grazie mille! | (none) | Y | n/a | n/a | n/a | - | - | - | **Y** |
| 26 | none | traduci 'good morning' in italiano | run_code | Y | n/a | n/a | n/a | - | Y | - | **-** |
| 27 | none | what's the capital of France? | (none) | Y | n/a | n/a | n/a | - | - | - | **Y** |
| 28 | multi | crea la cartella /home/demo e poi mettici den | mkdir,write_file | Y | Y | Y | Y | - | - | - | **Y** |
| 29 | multi | how much free RAM and how many cores? | meminfo,list_cores | Y | Y | Y | Y | - | - | - | **Y** |
| 30 | multi | copy /etc/motd to /home/motd.bak | cat,write_file | Y | Y | Y | Y | - | - | - | **Y** |

- tool-needed prompts: 25; PASS (well-formed + right tool + schema-valid + correct args): **24/25**
- schema-valid call rate (right tool + schema ok, ignoring arg semantics): 24/25
  - single: 21/22
  - multi: 3/3
- no-tool prompts answered without a call: 4/5
- malformed outputs: 1, hallucinated tool names: 0, prose instead of call: 1, hit max tokens: 0
- first-turn generated tokens: mean 26, max 61

### mode=think  (system+tools = 942 tok, 25.5s prefill @37 tok/s)

| id | kind | prompt | calls | parsed | right tool | schema ok | args correct | halluc | unneeded call | prose instead | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | single | quanta RAM libera? | meminfo | Y | Y | Y | Y | - | - | - | **Y** |
| 3 | single | create notes.txt with hello | write_file | Y | Y | Y | Y | - | - | - | **Y** |
| 5 | single | run this: print 2+2 | run_code | Y | Y | Y | Y | - | - | - | **Y** |
| 17 | single | rimuovi la directory /tmp/cache e tutto il co | rm | Y | Y | Y | Y | - | - | - | **Y** |
| 19 | single | esegui questo modulo wasm: (module (func (exp | run_code | Y | Y | Y | Y | - | - | - | **Y** |
| 28 | multi | crea la cartella /home/demo e poi mettici den | mkdir,write_file | Y | Y | Y | Y | - | - | - | **Y** |

- tool-needed prompts: 6; PASS (well-formed + right tool + schema-valid + correct args): **6/6**
- schema-valid call rate (right tool + schema ok, ignoring arg semantics): 6/6
  - single: 5/5
  - multi: 1/1
- no-tool prompts answered without a call: 0/0
- malformed outputs: 0, hallucinated tool names: 0, prose instead of call: 0, hit max tokens: 0
- first-turn generated tokens: mean 110, max 192
