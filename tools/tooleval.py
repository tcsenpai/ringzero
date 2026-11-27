#!/usr/bin/env python3
"""tooleval.py HOST PORT on|off OUT.json [SETS] : run the tool-call prompt sets through the ringzero agent over the
serial TCP port (M4.7a A/B of constrained decoding) and score them.

Sets: the spike prompts (docs/spikes/toolcall/prompts.json, written for the spike's own tool list) and
in-OS prompts (below) for the kernel's tools. Each prompt runs on a fresh conversation (/clear); y/N
questions are answered n; temperature 0 and 8 cores are set again after every prompt. agentd's memory slot and
idle consolidation are off (`!set slot 0`, `!set consolidate 0`): the journal holds the earlier eval prompts.
Counted per prompt: tool calls ([tool] lines), parser errors ([tool] malformed call), schema errors
(unknown tool, unknown parameter, bad value, missing parameter), the grammar line ([grammar] ...: valid
call / INVALID / leaves the grammar), decode speed of the rounds that made a call, whether the first call is a
right tool, and (spike set) whether the arguments match the spike's rules (docs/spikes/toolcall/score.py).
SLOT=N (env) sets `!set slot N` instead of 0 and keeps `/agentd log 12` (the slot's candidates) per prompt.
T28: a [confidence] y/N question (a low-margin call of a tool that changes the machine) is answered y, so the eval
runs the same calls as without it; every other y/N is answered n. CONF=on|off (env) sends `!set confidence ...`.
T54: `round_limit` counts the turns that hit the tool round limit and answered without tools.
T55: an expected entry "cat /proc/uptime" is the tool cat (or ls) with a path that starts with that path, so reading
the right file counts as a right call; `alias` counts the calls of the hidden aliases (meminfo, cpuinfo, list_cores,
uptime: not in the prompt since T55); INFO is the 8 system-information prompts of the T55 gate (also in inos).
SETS: comma list of spike, inos, it (default spike,inos). `it`: 10 short Italian knowledge / reasoning questions
(docs/MODEL_CHOICE.md); `auto` is a substring pre-check of the answer, the verdict is by hand."""
import json, os, re, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = json.load(open(f"{HERE}/../docs/spikes/toolcall/prompts.json"))
sys.path.insert(0, f"{HERE}/../docs/spikes/toolcall")
from score import args_match  # the spike's "=exact" / "~contains" argument rules
# (text, expected tool names in order: each a set of acceptable names; [] = no tool)
INOS = [
    ("ricordati che il mio colore preferito e il verde", [{"remember"}]),
    ("cosa sai di me?", [{"recall"}]),
    ("dimentica il mio colore preferito", [{"forget"}]),
    ("quanto fa 2**100? usa python", [{"run_code"}]),
    ("calcola la media di 3, 7 e 11 con python", [{"run_code"}]),
    ("what's 17 times 23? use python", [{"run_code"}]),
    ("elenca i file in /notes", [{"ls"}]),
    ("leggi /notes/todo.txt", [{"cat"}]),
    ("scrivi in /notes/m47.txt la parola prova", [{"write_file"}]),
    ("crea la cartella /notes/m47", [{"mkdir"}]),
    ("cancella il file /notes/m47.txt", [{"rm"}]),
    ("imposta la temperatura a 0", [{"set_setting"}]),
    ("usa 4 core per l'inferenza", [{"set_setting"}]),
    ("set the keyboard layout to us", [{"set_setting"}]),
    ("set the screen brightness to 80", [{"set_setting"}, set()]),  # no such setting: a call or an answer
    ("da quanto tempo e acceso il sistema?", [{"uptime", "cat /proc/uptime"}]),
    ("mostra le ultime 5 righe del log", [{"read_log"}]),
    ("show the last ten log lines", [{"read_log"}]),
    ("quanti core ha la CPU?", [{"cpuinfo", "list_cores", "cat /proc/cpuinfo", "cat /sys/devices/system/cpu", "ls /sys/devices/system/cpu"}]),
    ("what is on the disk root?", [{"ls"}]),
    ("cerca nei file qualcosa sul latte", [{"recall"}]),
    ("spegni il computer", [{"poweroff"}]),
    ("hi, how are you?", []),
    ("che cos'e un kernel? una frase", []),
]
# T55: system information that files answer (the D5 gate): memory, cores, uptime, cpu model, drivers / PCI, disk size,
# hostname, version
CORES = {"cpuinfo", "list_cores", "cat /proc/cpuinfo", "cat /sys/devices/system/cpu", "ls /sys/devices/system/cpu"}
DRIVERS = {"cat /proc/devices", "cat /proc/bus/pci", "ls /proc/bus/pci", "ls /sys/bus/pci", "cat /sys/bus/pci", "ls /proc", "ls /sys"}
INFO = [
    ("quanta RAM libera c'e?", [{"meminfo", "cat /proc/meminfo"}]),
    ("how many CPU cores does this machine have?", [CORES]),
    ("quanto uptime?", [{"uptime", "cat /proc/uptime"}]),
    ("what CPU model is this?", [{"cpuinfo", "cat /proc/cpuinfo"}]),
    ("che driver abbiamo?", [DRIVERS]),
    ("how big is the disk?", [{"cat /sys/class/block/vda", "ls /sys/class/block", "cat /proc/devices", "ls /sys/block"}]),
    ("come si chiama questa macchina?", [{"cat /etc/hostname"}]),
    ("which version of ringzero is running?", [{"cat /proc/version", "cat /etc/os-release"}]),
]
INOS += INFO
# T55: reading the file a hidden alias stands for is the same right call (the spike prompts name the old tools)
EQUIV = {"meminfo": {"cat /proc/meminfo"}, "cpuinfo": {"cat /proc/cpuinfo"}, "list_cores": CORES, "uptime": {"cat /proc/uptime"}}
# (question, accepted answers: lowercase substrings, any one); no tool expected
IT = [
    ("Qual è la capitale dell'Australia? Rispondi in una parola.", ["canberra"]),
    ("Quanto fa 17 per 23? Rispondi solo con il numero, senza usare strumenti.", ["391"]),
    ("Chi ha scritto la Divina Commedia?", ["dante"]),
    ("Se oggi è martedì, che giorno della settimana sarà fra 10 giorni?", ["venerdì", "venerdi"]),
    ("Qual è il pianeta più grande del sistema solare?", ["giove"]),
    ("Un treno parte alle 14:40 e arriva alle 17:15. Quanto dura il viaggio?", ["2 ore e 35", "2h35", "2 h 35", "155 minuti"]),
    ("In che anno è caduto il muro di Berlino?", ["1989"]),
    ("Ho 5 mele, ne mangio 2 e poi ne compro il doppio di quelle che mi sono rimaste. Quante mele ho adesso?", ["9"]),
    ("Qual è il fiume più lungo d'Italia?", ["po"]),
    ("Quanti secondi ci sono in un'ora?", ["3600", "3.600"]),
]
PROMPT = re.compile(rb"ringzero(?:\[[^\]\n]*\])?> ")
YN = b"[y/N] "
SGR = re.compile(rb"\x1b\[[0-9;]*m")  # T46: the shell's role colors, stripped before matching


def drive(host, port, mode, prompts, first=False):
    """first=True (tools/confeval.py): Ctrl-C once the first tool result is printed (the rest of the turn is not
    needed; the turn is rolled back)."""
    s = socket.create_connection((host, port)); s.settimeout(0.2)
    buf = b""
    intr = [False]

    def until(timeout=600):
        nonlocal buf
        t = time.time()
        while time.time() - t < timeout:
            m = PROMPT.search(buf); y = buf.find(YN)
            if y >= 0 and (not m or y < m.start()):
                seen, buf = buf[:y], buf[y + len(YN):]
                a = "y" if b"[confidence]" in seen[seen.rfind(b"\n", 0, max(len(seen) - 1, 0)) + 1:] else "n"
                s.sendall(a.encode() + b"\r")
                return seen.decode(errors="replace") + f"[y/N] {a}\n" + until(timeout)
            if first and not intr[0] and re.search(rb"^\[tool\] -> ", buf, re.M):
                s.sendall(b"\x03"); intr[0] = True
            if m:
                seen, buf = buf[:m.start()], buf[m.end():]
                return seen.decode(errors="replace")
            try: buf = SGR.sub(b"", buf + s.recv(65536))
            except socket.timeout: pass
        return buf.decode(errors="replace") + "\n<<timeout>>"

    def cmd(line):
        s.sendall(line.encode() + b"\r")
        return until()

    s.sendall(b"\r"); until(20)
    cmd(f"!set constrain {mode}")
    if os.environ.get("CONF"):
        cmd(f"!set confidence {os.environ['CONF']}")
    slot = os.environ.get("SLOT", "0")  # SLOT=200: memory slot on (T35 A/B); its decisions are kept per prompt
    cmd(f"!set slot {slot}"); cmd("!set consolidate 0")  # the slot carries earlier eval prompts in (one session)
    out = []
    for text, exp in prompts:
        cmd("/clear")
        intr[0] = False
        t0 = time.time()
        txt = cmd(text)
        out.append({"text": text, "expect": exp, "out": txt.replace("\r", ""), "secs": round(time.time() - t0, 1)})
        if slot != "0":
            out[-1]["agentd"] = cmd("/agentd log 12").replace("\r", "")
        print(f"[{mode}] {len(out)}/{len(prompts)} {time.time() - t0:5.1f}s {text}", flush=True)
        cmd("!set temperature 0"); cmd("!set cores 8")
    s.close()  # the serial chardev takes one client at a time
    return out


SCHEMA = re.compile(r"\[tool\] -> error: (unknown tool|\S+ has no parameter|bad value|\S+ needs parameter)")


def hit(call, args, e):
    """T55: "cat /path" = that tool with a path under /path; else the tool name."""
    name, _, path = e.partition(" ")
    return call == name and (not path or args.get("path", "").strip('"').startswith(path))


def score(r):
    o = r["out"]
    calls = re.findall(r"^\[tool\] ([a-z_]+)\(", o, re.M)
    # shortcut: args split on ", " (a value containing ", k=" would split wrong; the spike values do not)
    cargs = [dict(kv.split("=", 1) for kv in re.split(r", (?=[a-z_]+=)", a) if "=" in kv)
             for a in re.findall(r"^\[tool\] [a-z_]+\((.*)\)$", o, re.M)]
    inset = lambda i, e: any(hit(calls[i], cargs[i] if i < len(cargs) else {}, x) for x in set(e).union(*(EQUIV.get(n, set()) for n in e)))
    gl = re.findall(r"^\[grammar\] .*$", o, re.M)
    exp = r["expect"]
    if exp and exp[-1] == set():  # optional call
        ok = len(calls) <= 1 and (not calls or inset(0, exp[0]))
    else:
        ok = len(calls) == len(exp) and all(inset(i, e) for i, e in enumerate(exp))
    first = bool(calls) and bool(exp) and bool(exp[0]) and inset(0, exp[0])
    args = None
    if r.get("spike_expect") is not None:
        se = r["spike_expect"]
        args = len(calls) >= len(se) > 0 and all(c in e["tool"] for c, e in zip(calls, se)) and \
            all(args_match(e, a) for e, a in zip(se, cargs))
    auto = None
    if r.get("accept"):
        ans = re.sub(r"^\[.*$", "", o, flags=re.M).lower()  # the answer text without [memory]/[prefill] lines
        auto = not calls and any(re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", ans) for a in r["accept"])
    dec = re.findall(r"\[grammar\][^\n]*\n(?:[^\n]*\n)*?\[prefill\][^\n]*\[decode\] (\d+) tok in ([\d.]+) ms", o)
    return {
        "calls": calls, "args": cargs, "correct": ok, "first_right": first, "args_exact": args, "auto_ok": auto,
        "malformed": len(re.findall(r"^\[tool\] malformed call", o, re.M)),
        "schema": len(SCHEMA.findall(o)),
        "grammar_bad": sum(1 for g in gl if "valid call" not in g or "INVALID" in g or "leaves" in g),
        "grammar": gl, "decode": [(int(a), float(b)) for a, b in dec],
        "masked": sum(int(m) for m in re.findall(r"(\d+) masked", " ".join(gl))),
        "conf": re.findall(r"^\[confidence\] min.*$", o, re.M),
        "conf_ask": len(re.findall(r"^\[confidence\] the agent is not sure.*\[y/N\] y$", o, re.M)),
        "round_limit": len(re.findall(r"^\[agent\] tool round limit", o, re.M)),  # T54: answered without tools
        "alias": len(re.findall(r"^\[tool\] [a-z_]+ \(hidden alias of ", o, re.M)),  # T55
    }


def main():
    if sys.argv[1] == "--rescore":  # T55: score a saved run again with the current rules: tooleval.py --rescore IN.json OUT.json
        d = json.load(open(sys.argv[2]))
        res, mode, path = d["results"], d["mode"], sys.argv[3]
        for rs in res.values():
            for r in rs:
                r["expect"] = [set(e) for e in r["expect"]]
    else:
        res = None
    if res is None:
        host, port, mode, path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
        res = run(host, port, mode, (sys.argv[5] if len(sys.argv) > 5 else "spike,inos").split(","))
    summarize(res, mode, path)


def run(host, port, mode, sets):
    sp = [(p["text"], [set(e["tool"]) for e in p["expect"]]) for p in SPIKE]
    res = {}
    if "spike" in sets:
        res["spike"] = drive(host, port, mode, sp)
        for r, p in zip(res["spike"], SPIKE):
            r["spike_expect"] = p["expect"] if p["expect"] and not p.get("unordered") and any("args" in e for e in p["expect"]) else None
    if "inos" in sets:
        res["inos"] = drive(host, port, mode, INOS)
    if "it" in sets:
        res["it"] = drive(host, port, mode, [(q, []) for q, _ in IT])
        for r, (_, acc) in zip(res["it"], IT):
            r["accept"] = acc
    return res


def summarize(res, mode, path):
    summ = {}
    for k, rs in res.items():
        sc = [score(r) for r in rs]
        for r, x in zip(rs, sc):
            r.update(x); r["expect"] = [sorted(e) for e in r["expect"]]
        ntok = sum(a for x in sc for a, _ in x["decode"]); ms = sum(b for x in sc for _, b in x["decode"])
        summ[k] = {"prompts": len(rs), "correct": sum(x["correct"] for x in sc), "calls": sum(len(x["calls"]) for x in sc),
                   "malformed": sum(x["malformed"] for x in sc), "schema_errors": sum(x["schema"] for x in sc),
                   "grammar_bad": sum(x["grammar_bad"] for x in sc), "masked_tokens": sum(x["masked"] for x in sc),
                   "need_tool": sum(1 for r in rs if r["expect"]), "first_right": sum(x["first_right"] for x in sc),
                   "args_rule_prompts": sum(x["args_exact"] is not None for x in sc), "args_exact": sum(bool(x["args_exact"]) for x in sc),
                   "no_tool_ok": sum(1 for r, x in zip(rs, sc) if not r["expect"] and not x["calls"]),
                   "auto_ok": sum(bool(x["auto_ok"]) for x in sc),
                   "conf_lines": sum(len(x["conf"]) for x in sc), "conf_asked_y": sum(x["conf_ask"] for x in sc),
                   "round_limit": sum(x["round_limit"] for x in sc), "alias_hits": sum(x["alias"] for x in sc),
                   "info_correct": sum(x["correct"] for r, x in zip(rs, sc) if any(r["text"] == t for t, _ in INFO)),
                   "info_first_right": sum(x["first_right"] for r, x in zip(rs, sc) if any(r["text"] == t for t, _ in INFO)),
                   "conf_low_mid": [c for x in sc for c in x["conf"] if not c.endswith("-> run")],
                   "call_round_decode_tok_s": round(ntok / (ms / 1000), 2) if ms else None}
    print(json.dumps(summ, indent=1))
    json.dump({"mode": mode, "summary": summ, "results": res}, open(path, "w"), indent=1, default=sorted)


if __name__ == "__main__":
    main()
