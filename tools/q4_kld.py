#!/usr/bin/env python3
"""q4_kld.py (ringzero M7): quality of the engine's weight formats vs the HF model, on CPU.

Pass 1 runs the HF model (fp32 or bf16) over a fixed corpus and stores the reference log-probs (fp16, on disk).
Pass 2 replaces every weight the engine quantizes by its dequantized value, with the exporter's own rules
(tools/export_weights.py): projection matrices Q4_0 (--plan q4) or Q8_0 (--plan q8), embed/lm_head Q8_0 in both;
norms, conv1d, A_log, dt_bias, in_proj_a/b stay full precision (as in the .asmbin). Then it reports, against pass 1:
mean / p99 KLD(ref || quant), top-1 agreement, and the perplexity of both. Activation Q8 (the engine's A8) is not
simulated here: the asm-vs-mirror check covers it (docs/M7_Q4_RESULT.md).
Corpus: N chunks of LEN tokens from two sources (English prose: repo docs; code: Python stdlib), plus --long chunks of
LONG tokens of which only the last LEN positions are scored (DeltaNet state over a long context).
Usage: q4_kld.py --model-dir HF --dtype bf16 --plans q8,q4 --work DIR [--chunks 16 --len 512 --long 2 --long-len 4096]
"""
import argparse, glob, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
ap.add_argument("--plans", default="q8,q4")
ap.add_argument("--work", required=True)
ap.add_argument("--chunks", type=int, default=16)
ap.add_argument("--len", type=int, default=512)
ap.add_argument("--long", type=int, default=2)
ap.add_argument("--long-len", type=int, default=4096)
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--prose", default=os.path.expanduser("~/coding/qwenbaremetal"))
ap.add_argument("--out")
a = ap.parse_args()

import torch  # noqa: E402
torch.set_num_threads(a.threads)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_weights import quantize_q4_0_blocks  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

os.makedirs(a.work, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model_dir)


def corpus():
    prose = "\n\n".join(open(f, errors="replace").read() for f in sorted(glob.glob(a.prose + "/**/*.md", recursive=True))[:400])
    code = "\n\n".join(open(f, errors="replace").read() for f in sorted(glob.glob(os.path.dirname(os.__file__) + "/*.py"))[:200])
    out = []
    for name, text in (("prose", prose), ("code", code)):
        ids = tok(text)["input_ids"]
        n = a.chunks // 2
        step = max(a.len, (len(ids) - a.long_len) // max(1, n + a.long // 2))
        for i in range(n):
            out.append((name, ids[i * step:i * step + a.len], 0))
        for i in range(a.long // 2):
            s = len(ids) - (i + 1) * a.long_len
            out.append((name + "-long", ids[s:s + a.long_len], a.long_len - a.len))
    return out


chunks = corpus()
print(f"{len(chunks)} chunks: " + ", ".join(f"{n}:{len(t)}" for n, t, _ in chunks), flush=True)
dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt)
model.eval()
print(f"loaded {type(model).__name__} {a.dtype} in {time.time() - t0:.0f}s", flush=True)


def logprobs(ids, skip):
    with torch.no_grad():
        lg = model(torch.tensor([ids])).logits[0, skip:-1].float()
    return torch.log_softmax(lg, -1)


ref_tag = f"ref-{a.dtype}"
for i, (name, ids, skip) in enumerate(chunks):
    f = f"{a.work}/{ref_tag}-{i}.npy"
    if os.path.exists(f):
        continue
    t0 = time.time()
    np.save(f, logprobs(ids, skip).numpy().astype(np.float16))
    print(f"ref chunk {i} {name} {len(ids)} tok {time.time() - t0:.0f}s", flush=True)


def q8_deq(w):
    b = w.reshape(w.shape[0], -1, 32)
    amax = np.abs(b).max(-1)
    se = np.where(amax > 0, amax / np.float32(127), 0).astype(np.float32).astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(se > 0, 1 / se, 0).astype(np.float32)
    return (np.clip(np.rint(b * inv[:, :, None]), -127, 127) * se[:, :, None]).reshape(w.shape)


def q4_deq(w):
    q, d16 = quantize_q4_0_blocks(w)
    return (q.astype(np.float32) * d16.astype(np.float32)[:, :, None]).reshape(w.shape)


MATS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")
res = {}
for pi, plan in enumerate(a.plans.split(",")):
    if pi:  # fresh weights for the next plan (one copy of the model in RAM)
        del model
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt); model.eval()
    t0 = time.time()
    n_q = 0
    for n, p in model.named_parameters():
        if p.dim() != 2:
            continue
        w = p.detach().float().numpy()
        if n.split(".")[-2] in MATS:
            fq = q4_deq if plan == "q4" else q8_deq
        elif "embed_tokens" in n or "lm_head" in n:
            fq = q8_deq
        else:
            continue  # in_proj_a / in_proj_b: f32 in the .asmbin
        out = np.empty_like(w)
        for r in range(0, w.shape[0], 8192):
            out[r:r + 8192] = fq(w[r:r + 8192])
        with torch.no_grad():
            p.copy_(torch.from_numpy(out).to(p.dtype))
        n_q += 1
    print(f"plan {plan}: {n_q} matrices fake-quantized in {time.time() - t0:.0f}s", flush=True)
    kl, agree, nll_r, nll_q, cat = [], [], [], [], []
    for i, (name, ids, skip) in enumerate(chunks):
        lr = torch.from_numpy(np.load(f"{a.work}/{ref_tag}-{i}.npy").astype(np.float32))
        lq = logprobs(ids, skip)
        k = (lr.exp() * (lr - lq)).sum(-1)
        tgt = torch.tensor(ids[skip + 1:])
        kl.append(k.numpy()); agree.append((lr.argmax(-1) == lq.argmax(-1)).numpy())
        nll_r.append(-lr.gather(1, tgt[:, None])[:, 0].numpy()); nll_q.append(-lq.gather(1, tgt[:, None])[:, 0].numpy())
        cat += [name] * len(k)
    kl, agree, nll_r, nll_q, cat = map(np.concatenate, (kl, agree, nll_r, nll_q, [np.array(cat)]))
    r = {}
    for c in ["all"] + sorted(set(cat)):
        m = np.ones_like(kl, bool) if c == "all" else cat == c
        r[c] = dict(n=int(m.sum()), kld_mean=float(kl[m].mean()), kld_p99=float(np.percentile(kl[m], 99)),
                    top1=float(agree[m].mean()), ppl_ref=float(np.exp(nll_r[m].mean())), ppl_q=float(np.exp(nll_q[m].mean())))
        print(f"{plan} {c:10s} n={r[c]['n']:5d} KLD mean {r[c]['kld_mean']:.4f} p99 {r[c]['kld_p99']:.4f} "
              f"top1 {100 * r[c]['top1']:.1f}% PPL ref {r[c]['ppl_ref']:.3f} quant {r[c]['ppl_q']:.3f}", flush=True)
    res[plan] = r
if a.out:
    json.dump(dict(model=a.model_dir, dtype=a.dtype, len=a.len, long_len=a.long_len, results=res), open(a.out, "w"), indent=1)
