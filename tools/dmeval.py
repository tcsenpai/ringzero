"""T49 offline prototype: can a small decision model pick ringzero's first tool (or none) from the user's text?

dmeval.py report OUT.json...  the tables of docs/research/DECISION_MODELS.md
dmeval.py MODEL OUT.json   MODEL = kai | rerank | embed (host only: needs torch, transformers >= 5.17)
Tasks and labels: tools/confeval.py (240, even tune / odd held-out). The tool texts are the kernel's own
descriptions (kernel/src/agent.rs TOOLS). CPU only, 8 threads.
"""
import json, os, re, sys, time
import torch

RZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RZ + "/tools")
import confeval  # noqa: E402

torch.set_num_threads(8)
src = open(RZ + "/kernel/src/agent.rs").read()
TOOLS = {}
for m in re.finditer(r'"name": "([a-z_]+)", "description": "((?:[^"\\]|\\.)*)"', src):
    TOOLS.setdefault(m.group(1), m.group(2).replace('\\"', '"'))
assert len(TOOLS) == 18, TOOLS.keys()
NONE = ("No tool: answer directly in words (greetings, thanks, translations, general knowledge, explanations), "
        "or the request asks for something none of the tools can do.")
OPTS = dict(TOOLS, none=NONE)
NEED_Q = ("Does this request to the operating system agent need one of its tools (files on the disk, saved memory, "
          "system information, settings, power off or reboot, running code or programs)?")


def accepted(expect):
    ok = {x for x in expect.split("|") if x and x != "-"}
    if expect == "" or "-" in expect.split("|"):
        ok.add("none")
    return ok


def run_kai():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("vllm-sr/Decision-2.0-Kai-0.6B", trust_remote_code=True)
    qs = {"tool": {"type": "choice", "instructions": "Which tool should the operating system agent call first for this "
                   "user request?", "criteria": OPTS},
          "need": {"type": "noul", "instructions": NEED_Q}}

    def one(text):
        r = m.system_one(state=text, questions=qs)
        a = r["answers"]
        p = a["tool"]["probabilities"]
        return p, a["need"]["noul"], r.get("usage")
    return one


def run_rerank():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "Qwen/Qwen3-Reranker-0.6B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    mod = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
           "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ins = "Given a user request to an operating system agent, judge whether this tool is the one to call first"

    def one(text):
        docs = [f"{k}: {v}" for k, v in OPTS.items()]
        batch = [pre + f"<Instruct>: {ins}\n<Query>: {text}\n<Document>: {d}" + suf for d in docs]
        enc = tok(batch, padding=True, return_tensors="pt")
        with torch.no_grad():
            lg = mod(**enc).logits[:, -1, :]
        py = torch.softmax(torch.stack([lg[:, no], lg[:, yes]], 1), 1)[:, 1].tolist()
        s = sum(py) or 1.0
        p = {k: v / s for k, v in zip(OPTS, py)}
        return p, 1.0 - py[-1], {"tokens": int(enc["attention_mask"].sum())}
    return one


def run_embed():
    from transformers import AutoTokenizer, AutoModel
    name = "Qwen/Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    mod = AutoModel.from_pretrained(name, dtype=torch.float32).eval()

    def emb(texts):
        enc = tok(texts, padding=True, return_tensors="pt")
        with torch.no_grad():
            h = mod(**enc).last_hidden_state[:, -1]
        return torch.nn.functional.normalize(h, dim=-1), int(enc["attention_mask"].sum())
    D, _ = emb([f"{k}: {v}" for k, v in OPTS.items()])
    task = "Given a user request to an operating system agent, retrieve the tool that serves it"

    def one(text):
        q, n = emb([f"Instruct: {task}\nQuery:{text}"])
        c = (q @ D.T)[0]
        p = torch.softmax(c / 0.05, 0).tolist()  # shortcut: fixed temperature, not fitted
        return dict(zip(OPTS, p)), None, {"tokens": n, "cos": c.tolist()}
    return one


CASES = ["list /demo", "ricordati che lavoro come insegnante", "slow down the inference", "is the disk full?"]


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def report(paths):
    import statistics as st
    for name in paths:
        R = json.load(open(name))
        print(f"== {name}")
        for split in ("all", "held-out"):
            r = [x for x in R if split == "all" or x["split"] == split]
            acc = sum(x["right"] for x in r) / len(r)
            t2 = sum(x["top2"] for x in r) / len(r)
            au = auroc([x["p1"] for x in r if x["right"]], [x["p1"] for x in r if not x["right"]])
            tool = [x for x in r if x["need_gold"] is True]
            nont = [x for x in r if x["need_gold"] is False]
            print(f" {split}: n {len(r)} top1 {acc:.3f} top2 {t2:.3f} AUROC(p1 vs right) {au:.3f} | "
                  f"tool-needed top1 {sum(x['right'] for x in tool)}/{len(tool)}, no-tool picked none "
                  f"{sum(x['right'] for x in nont)}/{len(nont)}")
            for th in (0.5, 0.7, 0.9):
                s = [x for x in r if x["p1"] >= th]
                if s:
                    print(f"   p1>={th}: covers {len(s)/len(r):.2f}, accuracy {sum(x['right'] for x in s)/len(s):.3f}")
        nd = [x for x in R if x["need_gold"] is not None and x["need"] is not None]
        if nd:
            print(f" need-a-tool noul: AUROC {auroc([x['need'] for x in nd if x['need_gold']], [x['need'] for x in nd if not x['need_gold']]):.3f}, "
                  f"acc@0.5 {sum((x['need'] >= 0.5) == x['need_gold'] for x in nd)/len(nd):.3f} (n {len(nd)})")
            pn = [x for x in R if x["need_gold"] is not None]
            print(f" need-a-tool from the choice (pick != none): acc {sum((x['pick'] != 'none') == x['need_gold'] for x in pn)/len(pn):.3f}")
        ms = [x["ms"] for x in R]
        tk = [(x["usage"] or {}).get("input_tokens") or (x["usage"] or {}).get("tokens") for x in R]
        print(f" CPU ms per request: median {st.median(ms):.0f}, p90 {sorted(ms)[int(.9*len(ms))]:.0f}; tokens median {st.median(tk)}")
        for c in CASES:
            x = next(y for y in R if y["text"] == c)
            print(f"   T28 confident 4B mistake '{c}': pick {x['pick']} {x['p1']:.2f} (2nd {x['second']} {x['p2']:.2f}), expect {x['expect']}")
        wrong = [x for x in R if not x["right"] and x["split"] == "held-out"]
        print("  held-out wrong:", "; ".join(f"{x['text'][:38]}->{x['pick']}({x['p1']:.2f}) want {x['expect']}" for x in wrong[:25]))


def main():
    if sys.argv[1] == "report":
        return report(sys.argv[2:])
    which, out = sys.argv[1], sys.argv[2]
    one = {"kai": run_kai, "rerank": run_rerank, "embed": run_embed}[which]()
    rows = []
    for t in confeval.tasks():
        t0 = time.perf_counter()
        p, need, usage = one(t["text"])
        ms = (time.perf_counter() - t0) * 1000
        rank = sorted(p, key=p.get, reverse=True)
        ok = accepted(t["expect"])
        rows.append({"i": t["i"], "split": t["split"], "text": t["text"], "expect": t["expect"], "pick": rank[0],
                     "p1": p[rank[0]], "p2": p[rank[1]], "second": rank[1], "right": rank[0] in ok,
                     "top2": bool({rank[0], rank[1]} & ok), "need": need, "need_gold": True if "none" not in ok else
                     (False if t["expect"] == "" else None), "ms": ms, "usage": usage})
        print(t["i"], rows[-1]["right"], rank[0], f"{p[rank[0]]:.2f}", f"{ms:.0f}ms", t["text"][:50], flush=True)
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
