#!/usr/bin/env python3
"""T74: prove the fine-tune data is disjoint from the T69 eval set (tools/memeval.py), no model.

  check_disjoint.py DATA_DIR      report on DATA_DIR/train.jsonl and dev.jsonl; exit 1 on any hard leak

Hard leaks (any one fails the check):
  query_exact   a training question equals an eval question (lowercased, words only)
  query_6gram   a training question shares a word 6-gram with an eval question
  entry_exact   a training text (positive or negative) equals an eval index entry
  entry_8gram   a training text shares a word 8-gram with an eval index entry (manual, memories, files)
  path          a training text names a file path of the eval set
  entity        a training text uses a name, place or brand of the eval set (its capitalized words, minus the
                generic ones in GENERIC)
Reported, not failing: per class the share of training questions whose word 3-grams and 4-grams also occur in
an eval question (template phrasing such as "come si chiama" is expected to overlap at that length).
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import memeval as M  # noqa: E402
import rerankeval as rr  # noqa: E402

GENERIC = set("""The User User's Il La Lo Le Gli Un Una Asked Tool Task Past Default After Any At Below Do Every Only On Run
Read Type What When You They Two New Date Text Listing Reading Running Saved Settings Speed Writing Typing Going
Checking Command Context Counter Memory Kernel Power System Files Paths Programs Devices Disk Enter Tab Ctrl
NAME FILE KEY VALUE TEXT PATH QUERY TOPIC VERSION ARGS CODE
Linux Unix Python Rust JavaScript Lua CSV JSON IP PIN ID RAM CPU GB GiB MiB KiB OS US TODO CET CLI SQL TLS NAS PCI
January February March April May June July August September October November December
Monday Tuesday Wednesday Thursday Friday Saturday Sunday""".split())


def words(s):
    return re.findall(r"\w+", s.lower())


def grams(s, n):
    w = words(s)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def eval_texts():
    return [t for _, _, t in rr.corpus() + M.corpus_rows()]


def eval_caps():
    txt = " ".join(eval_texts() + [r["q"] for r in M.queries()])
    return {w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", txt)} - GENERIC


class EvalSide:
    def __init__(self):
        self.q = [r["q"] for r in M.queries()]
        self.qn = {" ".join(words(q)) for q in self.q}
        self.q6 = set().union(*(grams(q, 6) for q in self.q))
        self.q3 = set().union(*(grams(q, 3) for q in self.q))
        self.q4 = set().union(*(grams(q, 4) for q in self.q))
        T = eval_texts()
        self.tn = {" ".join(words(t)) for t in T}
        self.t8 = set().union(*(grams(t, 8) for t in T))
        self.paths = {p for p in re.findall(r"(?<![\w/])/[\w.-]+/[\w./-]*\w", " ".join(T + self.q))}
        self.caps = eval_caps()

    def leak(self, x):
        """The first hard leak of record x, or None."""
        q = " ".join(words(x["q"]))
        if q in self.qn:
            return "query_exact"
        if grams(x["q"], 6) & self.q6:
            return "query_6gram"
        texts = [x["q"], x["pos"]] + x["neg"]
        for t in [x["pos"]] + x["neg"]:
            if " ".join(words(t)) in self.tn:
                return "entry_exact"
            if grams(t, 8) & self.t8:
                return "entry_8gram"
        allt = " ".join(texts)
        if any(p in allt for p in self.paths):
            return "path"
        if set(re.findall(r"\b[A-Z][a-zA-Z]+\b", allt)) & self.caps:
            return "entity"
        return None


def main(d):
    ev = EvalSide()
    bad, rep = {}, {}
    for f in ("train.jsonl", "dev.jsonl"):
        for line in open(os.path.join(d, f)):
            x = json.loads(line)
            why = ev.leak(x)
            if why:
                bad.setdefault(why, []).append(x["q"])
            c = rep.setdefault(x["cls"], {"n": 0, "q3": 0, "q4": 0})
            c["n"] += 1
            c["q3"] += bool(grams(x["q"], 3) & ev.q3)
            c["q4"] += bool(grams(x["q"], 4) & ev.q4)
    print(f"eval side: {len(ev.q)} questions, {len(ev.tn)} entries, {len(ev.paths)} paths, {len(ev.caps)} entity words")
    print("| class | records | shares a 3-gram with an eval question | shares a 4-gram |")
    print("|---|---|---|---|")
    for c, v in rep.items():
        print(f"| {c} | {v['n']} | {v['q3']} ({100 * v['q3'] / v['n']:.0f}%) | {v['q4']} ({100 * v['q4'] / v['n']:.0f}%) |")
    n = sum(v["n"] for v in rep.values())
    print(f"hard leaks: {sum(map(len, bad.values()))} of {n} records " + json.dumps({k: len(v) for k, v in bad.items()}))
    for k, v in bad.items():
        print(f"  {k}: " + " | ".join(v[:5]))
    json.dump({"records": n, "hard_leaks": {k: len(v) for k, v in bad.items()}, "per_class": rep},
              open(os.path.join(d, "disjoint.json"), "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
