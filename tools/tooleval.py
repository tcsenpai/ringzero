#!/usr/bin/env python3
"""tooleval.py HOST PORT on|off OUT.json : run the tool-call prompt sets through the ringzero agent over the
serial TCP port (M4.7a A/B of constrained decoding) and score them.

Sets: the spike prompts (docs/spikes/toolcall/prompts.json, written for the spike's own tool list) and
in-OS prompts (below) for the kernel's tools. Each prompt runs on a fresh conversation (/clear); y/N
questions are answered n; temperature 0 and 8 cores are set again after every prompt. agentd's memory slot and
idle consolidation are off (`!set slot 0`, `!set consolidate 0`): the journal holds the earlier eval prompts.
Counted per prompt: tool calls ([tool] lines), parser errors ([tool] malformed call), schema errors
(unknown tool, unknown parameter, bad value, missing parameter), the grammar line ([grammar] ...: valid
call / INVALID / leaves the grammar), decode speed of the rounds that made a call."""
import json, os, re, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = json.load(open(f"{HERE}/../docs/spikes/toolcall/prompts.json"))
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
    ("da quanto tempo e acceso il sistema?", [{"uptime"}]),
    ("mostra le ultime 5 righe del log", [{"read_log"}]),
    ("show the last ten log lines", [{"read_log"}]),
    ("quanti core ha la CPU?", [{"cpuinfo", "list_cores"}]),
    ("what is on the disk root?", [{"ls"}]),
    ("cerca nei file qualcosa sul latte", [{"recall"}]),
    ("spegni il computer", [{"poweroff"}]),
    ("hi, how are you?", []),
    ("che cos'e un kernel? una frase", []),
]
PROMPT = re.compile(rb"ringzero(?:\[[^\]\n]*\])?> ")
YN = b"[y/N] "


def drive(host, port, mode, prompts):
    s = socket.create_connection((host, port)); s.settimeout(0.2)
    buf = b""

    def until(timeout=600):
        nonlocal buf
        t = time.time()
        while time.time() - t < timeout:
            m = PROMPT.search(buf); y = buf.find(YN)
            if y >= 0 and (not m or y < m.start()):
                seen, buf = buf[:y], buf[y + len(YN):]
                s.sendall(b"n\r")
                return seen.decode(errors="replace") + "[y/N] n\n" + until(timeout)
            if m:
                seen, buf = buf[:m.start()], buf[m.end():]
                return seen.decode(errors="replace")
            try: buf += s.recv(65536)
            except socket.timeout: pass
        return buf.decode(errors="replace") + "\n<<timeout>>"

    def cmd(line):
        s.sendall(line.encode() + b"\r")
        return until()

    s.sendall(b"\r"); until(20)
    cmd(f"!set constrain {mode}")
    cmd("!set slot 0"); cmd("!set consolidate 0")  # agentd's memory slot would carry earlier eval prompts in
    out = []
    for text, exp in prompts:
        cmd("/clear")
        t0 = time.time()
        txt = cmd(text)
        out.append({"text": text, "expect": exp, "out": txt.replace("\r", ""), "secs": round(time.time() - t0, 1)})
        print(f"[{mode}] {len(out)}/{len(prompts)} {time.time() - t0:5.1f}s {text}", flush=True)
        cmd("!set temperature 0"); cmd("!set cores 8")
    s.close()  # the serial chardev takes one client at a time
    return out


SCHEMA = re.compile(r"\[tool\] -> error: (unknown tool|\S+ has no parameter|bad value|\S+ needs parameter)")


def score(r):
    o = r["out"]
    calls = re.findall(r"^\[tool\] ([a-z_]+)\(", o, re.M)
    gl = re.findall(r"^\[grammar\] .*$", o, re.M)
    exp = r["expect"]
    if exp and exp[-1] == set():  # optional call
        ok = len(calls) <= 1 and (not calls or calls[0] in exp[0])
    else:
        ok = len(calls) == len(exp) and all(c in e for c, e in zip(calls, exp))
    dec = re.findall(r"\[grammar\][^\n]*\n(?:[^\n]*\n)*?\[prefill\][^\n]*\[decode\] (\d+) tok in ([\d.]+) ms", o)
    return {
        "calls": calls, "correct": ok,
        "malformed": len(re.findall(r"^\[tool\] malformed call", o, re.M)),
        "schema": len(SCHEMA.findall(o)),
        "grammar_bad": sum(1 for g in gl if "valid call" not in g or "INVALID" in g or "leaves" in g),
        "grammar": gl, "decode": [(int(a), float(b)) for a, b in dec],
        "masked": sum(int(m) for m in re.findall(r"(\d+) masked", " ".join(gl))),
    }


def main():
    host, port, mode, path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    sp = [(p["text"], [set(e["tool"]) for e in p["expect"]]) for p in SPIKE]
    res = {"spike": drive(host, port, mode, sp), "inos": drive(host, port, mode, INOS)}
    summ = {}
    for k, rs in res.items():
        sc = [score(r) for r in rs]
        for r, x in zip(rs, sc):
            r.update(x); r["expect"] = [sorted(e) for e in r["expect"]]
        ntok = sum(a for x in sc for a, _ in x["decode"]); ms = sum(b for x in sc for _, b in x["decode"])
        summ[k] = {"prompts": len(rs), "correct": sum(x["correct"] for x in sc), "calls": sum(len(x["calls"]) for x in sc),
                   "malformed": sum(x["malformed"] for x in sc), "schema_errors": sum(x["schema"] for x in sc),
                   "grammar_bad": sum(x["grammar_bad"] for x in sc), "masked_tokens": sum(x["masked"] for x in sc),
                   "call_round_decode_tok_s": round(ntok / (ms / 1000), 2) if ms else None}
    print(json.dumps(summ, indent=1))
    json.dump({"mode": mode, "summary": summ, "results": res}, open(path, "w"), indent=1, default=sorted)


if __name__ == "__main__":
    main()
