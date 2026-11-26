#!/usr/bin/env python3
"""hf_agree.py (ringzero M7): top-1 agreement of the engine's greedy output with the HF model, from a boot log.
Reads every `PROMPT_TOKENS a,b,..` / `GEN_TOKENS x,y,..` pair of a ringzero serial log (/gen, the self-test, ...),
runs HF once over prompt + generated ids (teacher forcing on the engine's own trajectory) and counts, at each
generated position, whether the engine's token is HF's argmax, plus the first divergence. Distribution
comparisons (KLD) are q4_kld.py's job.
Usage: hf_agree.py --model-dir HF --dtype bf16 LOG [LOG ...]
"""
import argparse, re, sys
import torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("logs", nargs="+")
a = ap.parse_args()
torch.set_num_threads(a.threads)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model_dir)
# /gen prints no PROMPT_TOKENS: rebuild the engine's chat prompt (tok_build_chat_prompt, think off), the same ids as
# the self-test's PROMPT_TOKENS; checked against the "PROMPT n tokens" line
HEAD, TAIL = [248045, 846, 198], [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
pairs = []
for f in a.logs:
    p = None
    for l in open(f, errors="replace"):
        m = re.search(r"PROMPT_TOKENS[ :]([\d,]+)", l)
        if m: p = [int(x) for x in m.group(1).split(",")]
        m = re.search(r"> /gen \d+ (.*)$", l)
        if m: p = HEAD + tok(m.group(1).strip())["input_ids"] + TAIL
        m = re.search(r"PROMPT (\d+) tokens fnv", l)
        if m and p and int(m.group(1)) != len(p): sys.exit(f"{f}: prompt rebuild gave {len(p)} ids, engine {m.group(1)}")
        m = re.search(r"GEN_TOKENS[ :]([\d,]+)", l)
        if m and p: pairs.append((f, p, [int(x) for x in m.group(1).split(",")]))
model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16).eval()
tot = ok = 0
for f, p, g in pairs:
    with torch.no_grad():
        lg = model(torch.tensor([p + g])).logits[0, len(p) - 1:len(p) + len(g) - 1].float()
    am = lg.argmax(-1).tolist()
    hit = [int(x == y) for x, y in zip(am, g)]
    div = next((i for i, h in enumerate(hit) if not h), None)
    tot += len(g); ok += sum(hit)
    pr = torch.softmax(lg, -1)
    pe = [float(pr[i, t]) for i, t in enumerate(g)]
    print(f"{f}: prompt {len(p)} tok, gen {len(g)}: top-1 agree {sum(hit)}/{len(g)}, first divergence {div}, "
          f"min HF prob of the engine token {min(pe):.3f}")
print(f"TOTAL top-1 agreement {ok}/{tot} = {100 * ok / max(1, tot):.1f}%")
