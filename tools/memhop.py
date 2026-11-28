#!/usr/bin/env python3
"""T73: multi-hop recall on top of design A (T70's frozen config), no model writing anything.

  memhop.py embed VEC2.json   GPU host (torch, transformers): second-step query vectors (the query plus the
                              entities or the text of A's first hits) against the rows view; checks that its own
                              first-step cosines match docs/spikes/memab/vec.json
  memhop.py tune VEC2.json CFG.json        each variant's grid on the TUNE split only; writes the frozen configs
  memhop.py ab VEC2.json CFG.json OUT.json the frozen variants scored once (held-out and all), per class, Wilson CIs

Variants, each applied after A's ranking (memeval.rank with A's frozen config), all promote at most one entry
to the top and keep A's first hit second:
  budget  A with a larger hit window (HITS 24 -> 48 / all); no hop rule, a control
  join    entity-key join: the names in A's first hit(s) (not in the query) and the alias names of the relations
          the query names (my sister -> Chiara); entries holding one of them, among the top B per pool, score
          adjusted cosine + bonus; the best is promoted when it beats the first hit
  prf     two-step in vector space: adjusted cosine + w * cos(first hit row, entry row), promoted over a threshold
  req     two-step retrieval: the query re-embedded with the first hit's names (req_names) or its index text
          (req_key); fused score adjusted cosine + w * second cosine, promoted when it beats the first hit's
"""
import json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memeval as M  # noqa: E402
import rerankeval as rr  # noqa: E402

RZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC = RZ + "/docs/spikes/memab/vec.json"
CTX = RZ + "/docs/spikes/memab/ctx.json"
CFG_A = RZ + "/docs/spikes/memab/config.json"


def frozen_a():
    F = json.load(open(CFG_A))["A"]
    return {k: tuple(v) if isinstance(v, list) else v for k, v in F["cfg"].items()}, F["floor"]


def qwords(q):
    return set(re.findall(r"[a-z]+", q.lower()))


def ents(view, d, q):
    """Entity keys of entry d not already in the query: its capitalized names (memeval.names)."""
    return M.names(view.text[d]) - qwords(q)


def alias_ents(view, q):
    """Names the alias table gives for the relations the query names (my best friend -> davide)."""
    rel = M.rel_words(q)
    return {n for n, rs in view.al_text.items() if rs & rel} - qwords(q)


def first_hits(view, rows, cfg):
    """Per query: A's answer list (non-manual part) as entry indices, and the adjusted scores."""
    out = []
    for qi, r in rows:
        lst = M.rank(view, r, qi, cfg)
        out.append(([x[3] for x in lst if view.pool[x[3]] != "manual"], lst))
    return out


def req_texts(view, r, own):
    """The second-step queries for one query: per source rank 0 and 1, by names and by index text."""
    out = {}
    for j, d in enumerate(own[:2]):
        nm = sorted(ents(view, d, r["q"]))
        out[f"names{j}"] = r["q"] + (" " + " ".join(n.capitalize() for n in nm) if nm else "")
        out[f"key{j}"] = r["q"] + "\n" + view.text[d]
    return out


def embed(out):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    et = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    em = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.float32).to(dev).eval()

    def emb(text):  # rerankeval.embedder, on the GPU: kernel instruction format, last-token pooling
        ids = et(text, add_special_tokens=False)["input_ids"] + [rr.EOT]
        with torch.no_grad():
            h = em(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0, -1]
        return torch.nn.functional.normalize(h, dim=-1).float().cpu(), len(ids)

    Vs = M.load_views(VEC, CTX)
    v = Vs["rows"]
    cfg, _ = frozen_a()
    Q = M.queries()
    rows = list(enumerate(Q))
    D = torch.stack([emb(e)[0] for e in v.etext])
    qv = lambda q: emb(f"Instruct: {rr.TASK}\nQuery:{q}")
    # proof the vectors match T70's: first-step cosines recomputed here vs the stored ones
    Q1 = [qv(r["q"]) for r in Q]
    c1 = torch.stack([x[0] for x in Q1]) @ D.T
    diff = (c1 - torch.tensor(v.cos)).abs().max().item()
    agree = sum(int(c1[i].argmax()) == max(range(v.n), key=lambda d: v.cos[i][d]) for i in range(len(Q)))
    print(f"device {dev}; max |cos - stored| {diff:.4f}; same top-1 entry {agree}/{len(Q)}", flush=True)
    R = {"keys": v.keys, "check": {"max_abs_diff": round(diff, 5), "same_top1": agree, "n": len(Q)}, "rows": []}
    for (qi, r), (own, _) in zip(rows, first_hits(v, rows, cfg)):
        row = {"q": r["q"], "q_tokens": Q1[qi][1], "src": [v.keys[d] for d in own[:2]], "texts": {}, "cos": {}, "tokens": {}}
        for name, t in req_texts(v, r, own).items():
            x, n = qv(t)
            row["texts"][name], row["tokens"][name] = t, n
            row["cos"][name] = [round(c, 4) for c in (D @ x).tolist()]
        R["rows"].append(row)
        if qi % 50 == 0:
            print(qi, r["q"][:50], flush=True)
    json.dump(R, open(out, "w"))


# ---- the hop rules ----

def budget(view, s, own, B):
    """Candidate set: the top B non-manual entries per pool (memory, files) by adjusted score; B None = all."""
    out = set()
    for pool in ("memory", "files"):
        ds = sorted((d for d in range(view.n) if view.pool[d] == pool), key=lambda d: -s[d])
        out |= set(ds if B is None else ds[:B])
    return out


# A definition record: a name tied to a role ("il mio migliore amico e' Davide", "Marta e' la mia collega",
# "si chiama Rocco", "is called Giulia"). A query that asks for the name itself is answered by it: no hop.
DEF = re.compile(r"(?:\b(?:e'|is|si chiama|is called|sono)\s+(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?\s*(?:$|[;.,]|\s+(?:e'|is|ed|and|vive|lives))"
                 r"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:e'|is)\s+(?:il|la|lo|the|my|mio|mia|our)\b)")
NAMEQ = re.compile(r"\b(?:come si chiama|come si chiamano|chi e'|chi è|who is|who's|what's the name|what is the name|"
                   r"name of|nome|whose)\b", re.I)
CONTENT = lambda t: {w.decode() for w in M.words(t) if len(w) >= 4 and w.isalpha()}


def is_def(view, d):
    return view.pool[d] == "memory" and bool(DEF.search(view.text[d].split("] ", 1)[-1]))


def hop(view, r, qi, own, s, cfg, E2):
    """The entry to promote above A's first hit, or None."""
    g = cfg.get
    var = g("var")
    if not own or var in (None, "budget"):
        return None
    src = own[:g("src", 1)]
    e1 = own[0]
    if g("gate") == "def" and (not is_def(view, e1) or NAMEQ.search(r["q"])):
        return None
    cand = budget(view, s, own, g("B")) - set(src)
    if var == "join":
        nm = set()
        if g("ent") in ("words",):  # the other side of the definition: e1's content words the query lacks
            nm = CONTENT(view.text[e1].split("] ", 1)[-1]) - CONTENT(r["q"]) - qwords(r["q"])
        if g("ent") in ("hit", "both"):
            for d in src:
                nm |= ents(view, d, r["q"])
        if g("ent") in ("alias", "both"):
            nm |= alias_ents(view, r["q"])
        if not nm:
            return None
        cand = [d for d in cand if nm & qwords(view.text[d])]
        if not cand:
            return None
        c = max(cand, key=lambda d: s[d])
        return c if s[c] + g("bonus") > s[e1] else None
    if var == "prf":
        if not cand:
            return None
        c = max(cand, key=lambda d: s[d] + g("w") * view.dd[e1][d])
        return c if s[c] + g("w") * view.dd[e1][c] > s[e1] + g("t") else None
    if var == "req":
        x = E2["rows"][qi]["cos"][f"{g('by')}0"]
        if not cand:
            return None
        f = lambda d: s[d] + g("w") * x[d]
        c = max(cand, key=f)
        return c if f(c) > f(e1) - g("m") else None
    raise ValueError(var)


def evaluate(view, rows, cfg, floor, E2):
    """memeval.evaluate with the hop rule after A's ranking."""
    a_cfg, _ = frozen_a()
    old = M.HITS
    M.HITS = {None: 24, "all": view.n}.get(cfg.get("hits"), cfg.get("hits") or 24)
    out = []
    try:
        for qi, r in rows:
            lst = M.rank(view, r, qi, a_cfg)
            man = [x for x in lst if view.pool[x[3]] == "manual"]
            own = [x for x in lst if view.pool[x[3]] != "manual"]
            lead = lst[:1] if man and lst[0] is man[0] else []  # A put a manual section first: keep it there
            s = M.adjusted(view, r, qi, a_cfg)
            c = hop(view, r, qi, [x[3] for x in own], s, cfg, E2)
            if c is not None:
                own = [(view.keys[c], s[c], "emb", c)] + [x for x in own if x[3] != c]
            lst = [(k, sc, via) for k, sc, via, _ in lead + own]
            if r["cls"] == "noanswer":
                a = not any(x[1] >= (floor if x[2] == "emb" else M.RECALL_MIN_BM25) for x in lst[:M.RECALL_K])
                out.append((r, (a, a, a), lst, c is not None))
            else:
                t1, t3, _ = M.score(r, lst)
                out.append((r, (t1, t3, t1), lst, c is not None))
    finally:
        M.HITS = old
    return out


BUD = (8, 24, 48, None)
GRIDS = {
    "A": [{}],
    "budget": [{"var": "budget", "hits": h} for h in (48, "all")],
    "join": [{"var": "join", "gate": None, "ent": e, "src": k, "B": b, "bonus": x} for e in ("hit", "alias", "both")
             for k in (1, 2) for b in BUD for x in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3)],
    "prf": [{"var": "prf", "gate": None, "B": b, "w": w, "t": t} for b in BUD for w in (0.1, 0.2, 0.3, 0.5)
            for t in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)],
    "req_names": [{"var": "req", "gate": None, "by": "names", "B": b, "w": w, "m": m} for b in BUD
                  for w in (0.25, 0.5, 1.0, 2.0) for m in (0.0, 0.03, 0.06, 0.1, 0.15)],
    "req_key": [{"var": "req", "gate": None, "by": "key", "B": b, "w": w, "m": m} for b in BUD
                for w in (0.25, 0.5, 1.0, 2.0) for m in (0.0, 0.03, 0.06, 0.1, 0.15)],
}
# the same rules, firing only when A's first hit is a definition record and the query does not ask for the name
for _k in ("join", "prf", "req_names", "req_key"):
    GRIDS[_k + "_def"] = [dict(c, gate="def") for c in GRIDS[_k]]
GRIDS["join_def"] += [dict(c, ent="words") for c in GRIDS["join_def"] if c["ent"] == "hit"]
for _k in GRIDS:
    if _k != "A":
        GRIDS[_k] = [{}] + GRIDS[_k]  # the rule off: a variant can always tie A


def setup(vec2):
    Vs = M.load_views(VEC, CTX)
    E2 = json.load(open(vec2))
    v = Vs["rows"]
    assert E2["keys"] == v.keys and [x["q"] for x in E2["rows"]] == [r["q"] for r in M.queries()], "re-embed"
    return v, E2, list(enumerate(M.queries())), frozen_a()[1]


def tune(vec2, cfg_out):
    """Exhaustive grid per variant on the tune split: answerable top-1, then top-3, then the earlier grid point."""
    v, E2, rows, floor = setup(vec2)
    tr = [x for x in rows if x[1]["split"] == "tune" and x[1]["cls"] != "noanswer"]
    out = {}
    for name, grid in GRIDS.items():
        best, on = None, None  # on = the best grid point with the rule on, kept when "off" wins (for the record)
        for cfg in grid:
            S = evaluate(v, tr, cfg, floor, E2)
            o = (sum(s[1][0] for s in S), sum(s[1][1] for s in S))
            if best is None or o > best[0]:
                best = (o, cfg)
            if cfg and (on is None or o > on[0]):
                on = (o, cfg)
        for nm, b in ((name, best),) + (((name + "@on", on),) if on and not best[1] else ()):
            mh = [s for s in evaluate(v, tr, b[1], floor, E2) if s[0]["cls"] == "multihop"]
            out[nm] = {"cfg": b[1], "tune_top1": b[0][0], "tune_top3": b[0][1], "tune_n": len(tr),
                       "tune_multihop": sum(s[1][0] for s in mh), "grid": len(grid)}
            print(f"{nm}: tune top-1 {b[0][0]}/{len(tr)}, top-3 {b[0][1]}, multihop {out[nm]['tune_multihop']}"
                  f"/{len(mh)}, {b[1]}", flush=True)
    json.dump(out, open(cfg_out, "w"), indent=1)


def ab(vec2, cfg_path, out):
    v, E2, rows, floor = setup(vec2)
    F = json.load(open(cfg_path))
    S = {d: evaluate(v, rows, f["cfg"], floor, E2) for d, f in F.items()}
    D = list(S)
    res = {"configs": F, "summary": {}, "queries": [], "check": E2["check"]}
    for sp in ("held-out", "all"):
        print(f"\n== {sp}: top-1 (top-3) per class ==")
        print("| class | n | " + " | ".join(D) + " |")
        print("|---|---|" + "---|" * len(D))
        for c in M.CLASSES + ("answerable",):
            f = lambda r, c=c: sp in ("all", r["split"]) and (r["cls"] == c or c == "answerable" and r["cls"] != "noanswer")
            cells = []
            for d in D:
                g = [s for s in S[d] if f(s[0])]
                a1, a3, n = sum(s[1][0] for s in g), sum(s[1][1] for s in g), len(g)
                res["summary"][f"{sp}|{c}|{d}"] = {"top1": a1, "top3": a3, "n": n, "ci": M.wilson(a1, n)}
                cells.append(f"{a1}/{n}" + ("" if c == "noanswer" else f" ({a3})"))
            print(f"| {c} | {n} | " + " | ".join(cells) + " |")
        for c in ("multihop", "answerable"):
            print(f"  {c} Wilson 95%: " + ", ".join(f"{d} {res['summary'][f'{sp}|{c}|{d}']['ci']}" for d in D))
        for d in D:
            g = [s for s in S[d] if sp in ("all", s[0]["split"])]
            won = sum(s[1][0] and not a[1][0] for s, a in zip(S[d], S["A"]) if sp in ("all", s[0]["split"]))
            lost = sum(a[1][0] and not s[1][0] for s, a in zip(S[d], S["A"]) if sp in ("all", s[0]["split"]))
            res["summary"][f"{sp}|vsA|{d}"] = {"won": won, "lost": lost, "fired": sum(s[3] for s in g), "n": len(g)}
            print(f"  {d}: vs A +{won} -{lost}; hop fired on {sum(s[3] for s in g)}/{len(g)}")
    for i, r in rows:
        res["queries"].append({"q": r["q"], "cls": r["cls"], "split": r["split"], "gold": sorted(r["gold"]),
                               **{d: {"t1": S[d][i][1][0], "t3": S[d][i][1][1], "hop": S[d][i][3],
                                      "top3": [[k, round(s, 4)] for k, s, _ in S[d][i][2][:3]]} for d in D}})
    tok = {k: sorted(x["tokens"][k] for x in E2["rows"]) for k in ("names0", "key0")}
    tok["query"] = sorted(x["q_tokens"] for x in E2["rows"])
    res["q2_tokens_median"] = {k: t[len(t) // 2] for k, t in tok.items()}
    print("\nsecond-step query tokens, median:", res["q2_tokens_median"])
    print("\nheld-out multihop, per variant (top-1 key):")
    for i, r in rows:
        if r["split"] == "held-out" and r["cls"] == "multihop":
            print(f"  {r['q'][:50]:50} " + " ".join(f"{d}:{int(S[d][i][1][0])}" for d in D))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    {"embed": lambda: embed(a[2]), "tune": lambda: tune(a[2], a[3]), "ab": lambda: ab(a[2], a[3], a[4])}[a[1]]()
