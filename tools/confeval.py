#!/usr/bin/env python3
"""confeval.py (T28): does the decision-token margin (gram.rs, `[confidence]` lines) separate right from wrong tool calls?

  confeval.py run HOST PORT OUT.json   run the 240 tasks below (tooleval.drive: /clear per prompt, temperature 0,
                                       SLOT env as tooleval, a [confidence] y/N answered y, other y/N n; Ctrl-C after
                                       the first tool result: only the first call is scored)
  confeval.py analyze OUT.json         thresholds picked on the tune split, the held-out report

Each task: (text, expected first call, argument rules). Expected: tool names joined by '|', '-' = answering without a
tool is right; '' = no tool is the only right answer. Rules (first call only): '=x' exact, '~x' contains (case
insensitive). A call is wrong when its tool is not expected, or a rule fails, or no tool was right. Tasks reuse the
spike prompts (docs/spikes/toolcall, labels rewritten for the kernel's 18 tools), the in-OS and Italian sets of
tools/tooleval.py, then new ones: every tool in Italian and English, settings that do not exist, near-ties between two
tools (cat vs read_log, forget vs rm, recall vs meminfo) and vague requests. Split: even index = tune, odd = held-out."""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

T = [
    # spike prompts, relabelled for the kernel's tools
    ("quanta RAM libera?", "meminfo", {}),
    ("list files in /home", "ls", {"path": "=/home"}),
    ("create notes.txt with hello", "write_file", {"path": "~notes.txt", "content": "~hello"}),
    ("spegni la macchina", "poweroff", {}),
    ("run this: print 2+2", "run_code|run_wasm", {}),
    ("show me /etc/hostname", "cat", {"path": "=/etc/hostname"}),
    ("crea la cartella /home/progetti", "mkdir", {"path": "=/home/progetti"}),
    ("delete /tmp/old.log", "rm", {"path": "=/tmp/old.log"}),
    ("how many cores do I have?", "list_cores|cpuinfo", {}),
    ("che processore ho?", "cpuinfo", {}),
    ("riavvia", "reboot", {}),
    ("show the last 20 log lines", "read_log", {"n": "=20"}),
    ("set the hostname to aios-dev", "", {}),
    ("imposta la lingua su italiano", "-|set_setting", {"key": "=keyboard", "value": "=it"}),
    ("cosa c'è nella root?", "ls", {"path": "=/"}),
    ("memory usage please", "meminfo", {}),
    ("rimuovi la directory /tmp/cache e tutto il contenuto", "rm|ls", {"path": "=/tmp/cache"}),
    ("reboot the system now", "reboot", {}),
    ("esegui questo modulo wasm: (module (func (export \"main\") (result i32) i32.const 42))", "-|run_wasm|run_code", {}),
    ("ci sono errori nel log?", "read_log", {}),
    ("shut down", "poweroff", {}),
    ("leggi il file /home/todo.md", "cat", {"path": "=/home/todo.md"}),
    ("ciao, chi sei?", "", {}),
    ("explain what a kernel is in one sentence", "", {}),
    ("grazie mille!", "", {}),
    ("traduci 'good morning' in italiano", "", {}),
    ("what's the capital of France?", "", {}),
    ("crea la cartella /home/demo e poi mettici dentro un file readme.txt con scritto ciao", "mkdir", {"path": "=/home/demo"}),
    ("how much free RAM and how many cores?", "meminfo|cpuinfo|list_cores", {}),
    ("copy /etc/motd to /home/motd.bak", "cat", {"path": "=/etc/motd"}),
    # tools/tooleval.py in-OS set
    ("ricordati che il mio colore preferito e il verde", "remember", {"value": "~verde"}),
    ("cosa sai di me?", "recall", {}),
    ("dimentica il mio colore preferito", "forget", {"key": "~colore"}),
    ("quanto fa 2**100? usa python", "run_code", {"code": "~2**100"}),
    ("calcola la media di 3, 7 e 11 con python", "run_code", {}),
    ("what's 17 times 23? use python", "run_code", {"code": "~17"}),
    ("elenca i file in /notes", "ls", {"path": "=/notes"}),
    ("leggi /notes/todo.txt", "cat", {"path": "=/notes/todo.txt"}),
    ("scrivi in /notes/m47.txt la parola prova", "write_file", {"path": "=/notes/m47.txt", "content": "~prova"}),
    ("crea la cartella /notes/m47", "mkdir", {"path": "=/notes/m47"}),
    ("cancella il file /notes/m47.txt", "rm", {"path": "=/notes/m47.txt"}),
    ("imposta la temperatura a 0", "set_setting", {"key": "=temperature", "value": "=0"}),
    ("usa 4 core per l'inferenza", "set_setting", {"key": "=cores", "value": "=4"}),
    ("set the keyboard layout to us", "set_setting", {"key": "=keyboard", "value": "=us"}),
    ("set the screen brightness to 80", "", {}),
    ("da quanto tempo e acceso il sistema?", "uptime", {}),
    ("mostra le ultime 5 righe del log", "read_log", {"n": "=5"}),
    ("show the last ten log lines", "read_log", {"n": "=10"}),
    ("quanti core ha la CPU?", "cpuinfo|list_cores", {}),
    ("what is on the disk root?", "ls", {"path": "=/"}),
    ("cerca nei file qualcosa sul latte", "recall", {"query": "~latte"}),
    ("spegni il computer", "poweroff", {}),
    ("hi, how are you?", "", {}),
    ("che cos'e un kernel? una frase", "", {}),
    # tools/tooleval.py Italian knowledge set (no tool)
    ("Qual è la capitale dell'Australia? Rispondi in una parola.", "", {}),
    ("Quanto fa 17 per 23? Rispondi solo con il numero, senza usare strumenti.", "", {}),
    ("Chi ha scritto la Divina Commedia?", "", {}),
    ("Se oggi è martedì, che giorno della settimana sarà fra 10 giorni?", "-|run_code", {}),
    ("Un treno parte alle 14:40 e arriva alle 17:15. Quanto dura il viaggio?", "-|run_code", {}),
    ("In che anno è caduto il muro di Berlino?", "", {}),
    ("Ho 5 mele, ne mangio 2 e poi ne compro il doppio di quelle che mi sono rimaste. Quante mele ho adesso?", "-|run_code", {}),
    ("Quanti secondi ci sono in un'ora?", "-|run_code", {}),
    # meminfo
    ("how much memory is free?", "meminfo", {}),
    ("quanta memoria sto usando?", "meminfo", {}),
    ("is the machine low on memory?", "meminfo", {}),
    ("dimmi la memoria libera in MiB", "meminfo", {}),
    ("quanta RAM c'è in tutto?", "meminfo", {}),
    # cpuinfo
    ("what CPU is this?", "cpuinfo", {}),
    ("che modello di CPU ha questa macchina?", "cpuinfo", {}),
    ("does the CPU support AVX2?", "cpuinfo", {}),
    ("la CPU supporta AVX-512?", "cpuinfo", {}),
    ("what's the clock speed?", "cpuinfo", {}),
    # list_cores
    ("list the cores and what each one does", "list_cores", {}),
    ("cosa sta facendo ogni core?", "list_cores", {}),
    ("which cores run the inference?", "list_cores", {}),
    ("elenca i core", "list_cores|cpuinfo", {}),
    ("show me the role of every core", "list_cores", {}),
    ("quali core sono liberi?", "list_cores", {}),
    # uptime
    ("how long has it been running?", "uptime", {}),
    ("da quanto è acceso?", "uptime", {}),
    ("uptime?", "uptime", {}),
    ("how long ago did the system boot?", "uptime", {}),
    ("quanto tempo è passato dall'avvio?", "uptime", {}),
    ("has it been up for more than an hour?", "uptime", {}),
    # read_log
    ("show the last 3 log lines", "read_log", {"n": "=3"}),
    ("ultime 15 righe del log", "read_log", {"n": "=15"}),
    ("mostrami il log del kernel", "read_log", {}),
    ("any warnings in the kernel log?", "read_log", {}),
    ("fammi vedere le ultime 50 righe del log", "read_log", {"n": "=50"}),
    ("print 100 lines of the log", "read_log", {"n": "=100"}),
    ("leggi le ultime due righe del log", "read_log", {"n": "=2"}),
    ("check the log for errors", "read_log", {}),
    ("dammi le ultime 30 righe del registro di sistema", "read_log", {"n": "=30"}),
    # poweroff / reboot
    ("power off the machine", "poweroff", {}),
    ("spegniti", "poweroff", {}),
    ("turn the computer off please", "poweroff", {}),
    ("chiudi tutto e spegni", "poweroff", {}),
    ("restart the machine", "reboot", {}),
    ("riavvia il sistema", "reboot", {}),
    ("please reboot", "reboot", {}),
    ("fai un riavvio", "reboot", {}),
    # set_setting
    ("set the temperature to 0.7", "set_setting", {"key": "=temperature", "value": "=0.7"}),
    ("metti la temperatura a 1", "set_setting", {"key": "=temperature", "value": "=1"}),
    ("make the sampling greedy", "set_setting", {"key": "=temperature", "value": "=0"}),
    ("imposta temperature 0.2", "set_setting", {"key": "=temperature", "value": "=0.2"}),
    ("raise the temperature to 1.5", "set_setting", {"key": "=temperature", "value": "=1.5"}),
    ("use 2 cores", "set_setting", {"key": "=cores", "value": "=2"}),
    ("usa tutti gli 8 core", "set_setting", {"key": "=cores", "value": "=8"}),
    ("imposta i core a 6", "set_setting", {"key": "=cores", "value": "=6"}),
    ("limit the inference to one core", "set_setting", {"key": "=cores", "value": "=1"}),
    ("use 3 CPU cores for inference", "set_setting", {"key": "=cores", "value": "=3"}),
    ("switch the keyboard to italian", "set_setting", {"key": "=keyboard", "value": "=it"}),
    ("passa alla tastiera americana", "set_setting", {"key": "=keyboard", "value": "=us"}),
    ("cambia il layout della tastiera in it", "set_setting", {"key": "=keyboard", "value": "=it"}),
    ("keyboard layout: us", "set_setting", {"key": "=keyboard", "value": "=us"}),
    ("set the volume to 50", "", {}),
    ("imposta il fuso orario su Roma", "", {}),
    ("change the system language to English", "", {}),
    ("set the screen resolution to 1920x1080", "", {}),
    ("attiva la modalità scura", "", {}),
    ("set the wifi password to hunter2", "", {}),
    # remember
    ("remember that my dog is called Rex", "remember", {"value": "~rex"}),
    ("ricordati che lavoro come insegnante", "remember", {"value": "~insegnante"}),
    ("save the fact that I live in Turin", "remember", {"value": "~turin"}),
    ("tieni a mente che il mio compleanno è il 3 maggio", "remember", {"value": "~maggio"}),
    ("remember my favourite number is 7", "remember", {"value": "~7"}),
    ("salva che il mio gatto si chiama Pippo", "remember", {"value": "~pippo"}),
    ("memorizza che il mio colore preferito è il blu", "remember", {"value": "~blu"}),
    # forget
    ("forget my favourite number", "forget", {"key": "~number"}),
    ("dimentica il nome del mio cane", "forget", {"key": "~dog|cane"}),
    ("delete the fact about my job", "forget", {}),
    ("scordati dove abito", "forget", {}),
    ("forget that I use vim", "forget", {"key": "~editor|vim"}),
    ("rimuovi il fatto sul mio compleanno", "forget", {}),
    # recall
    ("what's my dog's name?", "recall", {}),
    ("come si chiama il mio gatto?", "recall", {}),
    ("what do you remember about me?", "recall", {}),
    ("search my files for milk", "recall", {}),
    ("cerca nei miei appunti la parola latte", "recall", {"query": "~latte"}),
    ("hai qualcosa salvato sul dentista?", "recall", {"query": "~dentist"}),
    ("find notes about the project deadline", "recall", {"query": "~deadline"}),
    ("cosa ti ho detto della casa?", "recall", {"query": "~casa"}),
    ("trova nei file qualcosa su javascript", "recall", {"query": "~javascript"}),
    # ls
    ("what's in /notes?", "ls", {"path": "=/notes"}),
    ("elenca /bin", "ls", {"path": "=/bin"}),
    ("list /demo", "ls", {"path": "=/demo"}),
    ("mostra il contenuto della cartella /lib", "ls", {"path": "=/lib"}),
    ("ls /", "ls", {"path": "=/"}),
    ("which files are in /notes/m47?", "ls", {"path": "=/notes/m47"}),
    ("che file ci sono in /demo?", "ls", {"path": "=/demo"}),
    ("elenca i programmi disponibili in /bin", "ls", {"path": "=/bin"}),
    ("list the root directory", "ls", {"path": "=/"}),
    # cat
    ("read /demo/data.json", "cat", {"path": "=/demo/data.json"}),
    ("mostra /notes/todo.txt", "cat", {"path": "=/notes/todo.txt"}),
    ("open /notes/casa.txt", "cat", {"path": "=/notes/casa.txt"}),
    ("cat /demo/hello.js", "cat", {"path": "=/demo/hello.js"}),
    ("leggi il file /demo/data.json", "cat", {"path": "=/demo/data.json"}),
    ("fammi vedere il contenuto di /notes/idea.txt", "cat", {"path": "=/notes/idea.txt"}),
    ("show me what is written in /notes/boot2.txt", "cat", {"path": "=/notes/boot2.txt"}),
    ("apri /notes/misto.txt", "cat", {"path": "=/notes/misto.txt"}),
    ("read the todo list in /notes/todo.txt", "cat", {"path": "=/notes/todo.txt"}),
    # write_file
    ("write hello world to /notes/hello.txt", "write_file", {"path": "=/notes/hello.txt", "content": "~hello"}),
    ("scrivi 'ciao' nel file /notes/ciao.txt", "write_file", {"path": "=/notes/ciao.txt", "content": "~ciao"}),
    ("create /notes/list.txt containing milk, eggs", "write_file", {"path": "=/notes/list.txt", "content": "~milk"}),
    ("salva in /notes/idea2.txt: comprare il pane", "write_file", {"path": "=/notes/idea2.txt", "content": "~pane"}),
    ("make a file /notes/a.txt with the text abc", "write_file", {"path": "=/notes/a.txt", "content": "~abc"}),
    ("metti 'test' in /notes/test.txt", "write_file", {"path": "=/notes/test.txt", "content": "~test"}),
    ("overwrite /notes/todo2.txt with done", "write_file", {"path": "=/notes/todo2.txt", "content": "~done"}),
    ("crea il file /notes/x.md con scritto # titolo", "write_file", {"path": "=/notes/x.md", "content": "~titolo"}),
    ("save the text 42 into /notes/answer.txt", "write_file", {"path": "=/notes/answer.txt", "content": "~42"}),
    ("scrivi un file /notes/poesia.txt con una riga di poesia", "write_file", {"path": "=/notes/poesia.txt"}),
    # mkdir
    ("make a folder /notes/work", "mkdir", {"path": "=/notes/work"}),
    ("crea la directory /notes/foto", "mkdir", {"path": "=/notes/foto"}),
    ("create the directory /demo/out", "mkdir", {"path": "=/demo/out"}),
    ("nuova cartella /tmp2", "mkdir", {"path": "=/tmp2"}),
    ("mkdir /notes/2026", "mkdir", {"path": "=/notes/2026"}),
    ("fai una cartella chiamata /archivio", "mkdir", {"path": "=/archivio"}),
    # rm
    ("delete /notes/hello.txt", "rm", {"path": "=/notes/hello.txt"}),
    ("cancella /notes/ciao.txt", "rm", {"path": "=/notes/ciao.txt"}),
    ("remove the file /notes/a.txt", "rm", {"path": "=/notes/a.txt"}),
    ("elimina la cartella /notes/foto", "rm", {"path": "=/notes/foto"}),
    ("rm /notes/test.txt", "rm", {"path": "=/notes/test.txt"}),
    ("butta via /notes/idea2.txt", "rm", {"path": "=/notes/idea2.txt"}),
    # run_code
    ("compute 2**64 with python", "run_code", {"code": "~2**64"}),
    ("calcola il fattoriale di 20 in python", "run_code", {"code": "~20"}),
    ("use python to sort [3, 1, 2]", "run_code", {"code": "~sort"}),
    ("quanto fa 123456*789? usa python", "run_code", {"code": "~789"}),
    ("esegui in python sum(range(101))", "run_code", {"code": "~sum(range(101))"}),
    ("what's 3.7 squared? run it in python", "run_code", {"code": "~3.7"}),
    ("converti 255 in esadecimale con python", "run_code", {"code": "~255"}),
    ("run python code that prints hello", "run_code", {"code": "~print"}),
    ("conta le lettere di 'ringzero' con python", "run_code", {"code": "~ringzero"}),
    # run_wasm
    ("count the words in /notes/todo.txt with wc", "run_wasm", {"path": "~wc"}),
    ("usa jq per leggere /demo/data.json", "run_wasm", {"path": "~jq"}),
    ("run lua to print 1+1", "run_wasm", {"path": "~lua"}),
    ("esegui sqlite3 con 'select 1'", "run_wasm", {"path": "~sqlite3"}),
    ("use rg to search 'latte' in /notes", "run_wasm", {"path": "~rg"}),
    ("sort the lines of /notes/misto.txt with the sort program", "run_wasm", {"path": "~sort"}),
    ("run qjs: console.log(6*7)", "run_wasm", {"path": "~qjs"}),
    ("run head on /notes/casa.txt", "run_wasm", {"path": "~head"}),
    ("usa CPython (python3 in /bin) per stampare 2+2", "run_wasm", {"path": "~python3"}),
    ("esegui /demo/hello.js con qjs", "run_wasm", {"path": "~qjs"}),
    # no tool
    ("hello!", "", {}),
    ("tell me a joke", "", {}),
    ("scrivi un haiku sull'autunno", "", {}),
    ("spiegami cos'è la RAM in due frasi", "", {}),
    ("qual è la differenza tra un processo e un thread?", "", {}),
    ("what does ring 0 mean?", "", {}),
    ("translate 'gatto' to English", "", {}),
    ("che lingua parli?", "", {}),
    # near-ties and vague requests
    ("read the log file /notes/log.txt", "cat", {"path": "=/notes/log.txt"}),
    ("cancella il fatto 'colore preferito'", "forget", {}),
    ("quanta memoria hai di me?", "recall", {}),
    ("tell me about the memory of our past chats", "recall", {}),
    ("write the free memory into /notes/mem.txt", "meminfo", {}),
    ("leggi il log e salvalo in /notes/log.txt", "read_log", {}),
    ("how many lines does /notes/todo.txt have?", "cat|run_wasm", {}),
    ("save this: buy bread", "remember|write_file", {}),
    ("prendi nota: chiamare la mamma", "remember|write_file", {}),
    ("what time is it?", "-|uptime", {}),
    ("check the system", "meminfo|cpuinfo|uptime|read_log|list_cores", {}),
    ("come sta la macchina?", "meminfo|cpuinfo|uptime|read_log|list_cores", {}),
    ("trova il file todo", "recall|ls", {}),
    ("is there anything about milk in my notes?", "recall|run_wasm", {}),
    ("compute 17*23", "-|run_code", {}),
    ("slow down the inference", "-|set_setting", {"key": "=cores"}),
    ("be more creative in your answers", "-|set_setting", {"key": "=temperature"}),
    ("go to sleep", "-|poweroff", {}),
    ("free up some memory", "-|meminfo", {}),
    ("delete the saved fact about my sister", "forget", {}),
    ("delete the note file /notes/answer.txt", "rm", {"path": "=/notes/answer.txt"}),
    ("show the 12 most recent kernel messages", "read_log", {"n": "=12"}),
    ("list my saved facts", "recall", {}),
    ("cosa c'è scritto nelle mie note?", "recall|ls|cat", {}),
    ("run 'print(1)'", "run_code|run_wasm", {}),
    ("quanti file ci sono in /notes?", "ls", {"path": "=/notes"}),
    ("is the disk full?", "-|ls", {}),
    ("ricordami quanta RAM ho", "meminfo", {}),
    ("dimentica il file /notes/a.txt", "rm|forget", {}),
    ("use the default number of cores", "-|set_setting", {"key": "=cores"}),
]

SPLIT = ("tune", "held-out")


def tasks():
    assert len(T) == 240, len(T)
    return [{"i": i, "split": SPLIT[i % 2], "text": t, "expect": e, "args": a} for i, (t, e, a) in enumerate(T)]


CONF = re.compile(r"^\[confidence\] min margin ([\d.]+) at '([^']+)' \('(.*)' vs '(.*)'\), (\d+) decision tokens:([\d. ]*), (\d+) us -> (.*)$", re.M)
CALL = re.compile(r"^\[tool\] ([a-z_]+)\((.*)\)$", re.M)


def judge(t, out):
    """(first call name or None, args, margin row or None, wrong: True/False/None (None = no call))"""
    m = CALL.search(out)
    if not m:
        return None, {}, None, None
    name = m.group(1)
    args = dict(kv.split("=", 1) for kv in re.split(r", (?=[a-z_]+=)", m.group(2)) if "=" in kv)
    c = CONF.search(out, m.end())
    conf = None
    if c:
        conf = {"min": float(c.group(1)), "kind": c.group(2), "top": [c.group(3), c.group(4)], "n": int(c.group(5)),
                "m": [float(x) for x in c.group(6).split()], "us": int(c.group(7)), "act": c.group(8)}
    ok_tools = [x for x in t["expect"].split("|") if x and x != "-"]
    wrong = name not in ok_tools
    for k, rule in t["args"].items():
        v = args.get(k, "").strip().lower()
        alts = rule[1:].lower().split("|")
        if rule[0] == "=" and not any(v.rstrip("/") == w.rstrip("/") for w in alts):
            wrong = True
        if rule[0] == "~" and not any(w in v for w in alts):
            wrong = True
    return name, args, conf, wrong


def auroc(pos, neg):
    """P(score of a wrong call < score of a right call), ties 1/2: low margin should mean wrong."""
    if not pos or not neg:
        return None
    s = sum((1.0 if p < n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def bands(rows, lo, hi):
    b = {"high": [0, 0], "mid": [0, 0], "low": [0, 0]}  # [right, wrong]
    for r in rows:
        k = "high" if r["margin"] >= hi else "mid" if r["margin"] >= lo else "low"
        b[k][r["wrong"]] += 1
    return b


def analyze(path):
    raw = json.load(open(path))
    res = raw["results"]
    rows, nocall = [], {"tune": [0, 0], "held-out": [0, 0]}  # [no call and that was right, missed a needed call]
    for t, r in zip(tasks(), res):
        name, args, conf, wrong = judge(t, r["out"])
        if name is None:
            nocall[t["split"]][0 if (t["expect"] == "" or "-" in t["expect"].split("|")) else 1] += 1
            continue
        tool_wrong = name not in [x for x in t["expect"].split("|") if x and x != "-"]
        rows.append({**t, "call": name, "args": args, "margin": conf["min"] if conf else 1.0, "conf": conf,
                     "wrong": int(wrong), "tool_wrong": int(tool_wrong)})
    tune = [r for r in rows if r["split"] == "tune"]
    held = [r for r in rows if r["split"] == "held-out"]
    grid = [round(0.05 * k, 2) for k in range(1, 20)]
    # Picked on the tune split only. LO: the largest Youden J (share of wrong calls below LO minus share of right calls
    # below LO), with LO >= NOISE_LO (numeric noise, docs/T28_RESULT.md) and at most ESC_MAX of the right calls below
    # it (asked for nothing). HI: the largest value >= LO + 0.1 that puts at most LOG_MAX of the right calls in
    # [LO, HI) (logged only).
    NOISE_LO, ESC_MAX, LOG_MAX = 0.2, 0.10, 0.15
    tw = [r for r in tune if r["wrong"]]; tr = [r for r in tune if not r["wrong"]]
    below = lambda rs, g: sum(r["margin"] < g for r in rs) / max(len(rs), 1)

    def j(lo):
        return below(tw, lo) - below(tr, lo)
    cand = [g for g in grid if g >= NOISE_LO and below(tr, g) <= ESC_MAX]
    lo = max(cand, key=lambda g: (round(j(g), 9), -g)) if cand else NOISE_LO
    hs = [g for g in grid if g >= lo + 0.1 - 1e-9 and below(tr, g) - below(tr, lo) <= LOG_MAX]
    hi = max(hs) if hs else round(lo + 0.1, 2)
    rep = {"tasks": len(res), "calls": len(rows), "no_call": nocall, "lo": lo, "hi": hi, "tune_j": round(j(lo), 3)}
    for name, rs in (("tune", tune), ("held-out", held)):
        w = [r["margin"] for r in rs if r["wrong"]]; ok = [r["margin"] for r in rs if not r["wrong"]]
        tw_ = [r["margin"] for r in rs if r["tool_wrong"]]; tok = [r["margin"] for r in rs if not r["tool_wrong"]]
        b = bands(rs, lo, hi)
        rep[name] = {"calls": len(rs), "wrong": len(w), "wrong_tool": len(tw_),
                     "auroc_wrong_call": auroc(w, ok), "auroc_wrong_tool": auroc(tw_, tok),
                     "bands_right_wrong": b,
                     "wrong_caught_low": b["low"][1], "wrong_in_mid": b["mid"][1], "wrong_missed_high": b["high"][1],
                     "false_escalations_low": b["low"][0],
                     "escalation_rate_low": round((b["low"][0] + b["low"][1]) / max(len(rs), 1), 3),
                     "logged_rate_mid": round((b["mid"][0] + b["mid"][1]) / max(len(rs), 1), 3)}
    rep["held-out_operating_points"] = {f"{g:.2f}": {"wrong_below": sum(r["margin"] < g for r in held if r["wrong"]),
                                                     "right_below": sum(r["margin"] < g for r in held if not r["wrong"])}
                                        for g in (0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    print(json.dumps(rep, indent=1))
    print("\nwrong calls and low/middle margins (all splits):")
    for r in sorted(rows, key=lambda r: r["margin"]):
        if r["wrong"] or r["margin"] < hi:
            c = r["conf"] or {}
            print(f"  {r['split']:8} {r['margin']:.2f} {'WRONG' if r['wrong'] else 'right'} {r['call']}({', '.join(f'{k}={v}' for k, v in r['args'].items())[:60]}) "
                  f"at {c.get('kind')} {c.get('top')} expect {r['expect'] or 'no tool'} | {r['text'][:60]}")
    us = [r["conf"]["us"] for r in rows if r["conf"]]
    print(f"\nmargin cost per call: mean {sum(us) / max(len(us), 1):.0f} us, max {max(us, default=0)} us over {len(us)} calls; "
          f"decision tokens per call: mean {sum(r['conf']['n'] for r in rows if r['conf']) / max(len(us), 1):.2f}")
    return rep


def main():
    if sys.argv[1] == "run":
        from tooleval import drive
        host, port, path = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        ts = tasks()
        out = drive(host, port, "on", [(t["text"], []) for t in ts], first=True)
        json.dump({"results": out}, open(path, "w"), indent=1, default=sorted)
        analyze(path)
    else:
        analyze(sys.argv[2])


if __name__ == "__main__":
    main()
