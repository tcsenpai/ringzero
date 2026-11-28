#!/usr/bin/env python3
"""T74: regression gate for a fine-tuned embedder (the M7 idea for the LLM, applied to the embedder): the model
must stay general and keep the kernel's floors calibrated, not only win on the memory classes.

  gate.py --ft DIR [--base Qwen/Qwen3-Embedding-0.6B] [--out GATE.json]     GPU host; exit 0 = pass, 1 = fail

Measured for the base and the fine-tuned model, in the kernel format (train.py encode):
  1. general retrieval: heading -> section over the repo's prose (README, INFO, docs/*.md, without the manual and
     the memory research docs, which are near the eval set): top-1 and MRR@10
  2. agreement: cosine between the base and the fine-tuned vector of the same text (every general text)
  3. calibration on the T69 set (memeval kernel ranking): held-out answerable top-1, right no-answer abstentions
     at the kernel floors, median top-1 cosine (RECALL_MIN = 0.55 was set on the base model's scale)
Pass [E thresholds, to revisit after the first run]: general top-1 drop <= 2 points and MRR drop <= 0.02;
agreement mean >= 0.90; no-answer right >= base; median top-1 cosine within 0.05 of base.
shortcut: the general set is English technical prose from this repo; add a public IT/EN retrieval set (e.g.
MIRACL it and en dev) before trusting the gate on general Italian.
"""
import argparse, glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import memeval as M  # noqa: E402
import rerankeval as rr  # noqa: E402
from train import encode, query_text  # noqa: E402

RZ = os.path.dirname(os.path.dirname(HERE))
SKIP = ("MANUAL.md", "MEMORY_", "spikes/")


def general_set(cap=300):
    pairs, seen = [], set()
    files = [RZ + "/README.md", RZ + "/INFO.md"] + sorted(glob.glob(RZ + "/docs/**/*.md", recursive=True))
    for f in files:
        if any(s in f for s in SKIP):
            continue
        for sec in open(f).read().split("\n## ")[1:]:
            head, body = (sec.split("\n", 1) + [""])[:2]
            head, body = head.strip(), re.sub(r"\s+", " ", body).strip()
            if len(body) < 200 or len(head) < 8 or head.lower() in seen:
                continue
            seen.add(head.lower())
            pairs.append((head, body[:600]))
    return pairs[:cap]


def vectors(path, texts, dev, bs=16):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        out = torch.cat([encode(model, tok, texts[i:i + bs], dev, 512) for i in range(0, len(texts), bs)])
    del model
    return out.cpu()


def measure(path, dev, gen, C, Q):
    import torch
    qs = [query_text(h) for h, _ in gen]
    docs = [b for _, b in gen]
    t69 = [query_text(r["q"]) for r in Q] + [t for _, _, t in C]
    V = vectors(path, qs + docs + t69, dev)
    gq, gd, tq, td = V[:len(qs)], V[len(qs):len(qs) + len(docs)], V[len(qs) + len(docs):len(qs) + len(docs) + len(Q)], \
        V[len(qs) + len(docs) + len(Q):]
    S = gq @ gd.T
    ranks = [(S[i] > S[i, i]).sum().item() for i in range(len(gen))]
    R = {"task": rr.TASK, "keys": [k for k, _, _ in C], "rows": [{"q": r["q"], "cos": (tq[i] @ td.T).tolist()}
                                                                  for i, r in enumerate(Q)]}
    sc = M.scored(R, "kernel")
    ho = [s for s in sc if s[0]["split"] == "held-out" and s[0]["cls"] != "noanswer"]
    na = [s for s in sc if s[0]["cls"] == "noanswer"]
    top = [max(x["cos"]) for x in R["rows"]]
    return {"general_top1": round(100 * sum(r == 0 for r in ranks) / len(ranks), 1),
            "general_mrr10": round(sum(1 / (r + 1) for r in ranks if r < 10) / len(ranks), 4),
            "t69_heldout_top1": sum(s[1][0] for s in ho), "t69_heldout_n": len(ho),
            "t69_noanswer_right": sum(s[1][0] for s in na), "t69_noanswer_n": len(na),
            "t69_median_top_cos": round(statistics.median(top), 4)}, torch.cat([gq, gd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", default="gate.json")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen, C, Q = general_set(), rr.corpus(), M.queries()
    b, vb = measure(a.base, dev, gen, C, Q)
    f, vf = measure(a.ft, dev, gen, C, Q)
    agree = (vb * vf).sum(-1)
    res = {"general_pairs": len(gen), "base": b, "ft": f,
           "agreement_mean": round(agree.mean().item(), 4), "agreement_min": round(agree.min().item(), 4)}
    checks = {"general_top1": f["general_top1"] >= b["general_top1"] - 2,
              "general_mrr10": f["general_mrr10"] >= b["general_mrr10"] - 0.02,
              "agreement": res["agreement_mean"] >= 0.90,
              "noanswer": f["t69_noanswer_right"] >= b["t69_noanswer_right"],
              "calibration": abs(f["t69_median_top_cos"] - b["t69_median_top_cos"]) <= 0.05}
    res["checks"], res["pass"] = checks, all(checks.values())
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
