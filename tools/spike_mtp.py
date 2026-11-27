#!/usr/bin/env python3
"""spike_mtp.py (T45 spike): is Qwen3.5-4B's own MTP head a good draft for greedy speculative decoding?

Runs the main model greedy (HF, bf16) on ringzero-style prompts (the agent's system prompt + tool list from
kernel/src/agent.rs, thinking off), records the main model's last hidden state at every fed position, then
replays the decode loop with the MTP head (mtp.* tensors of the checkpoint, our own forward) as the drafter and
greedy verify: per verify step the MTP drafts n tokens (chained), the accepted ones are the leading drafts equal
to the main model's greedy tokens, plus the main model's own token. Bit-exact verify (M4.8b) means the output is
the greedy trajectory, so the replay over that trajectory gives the exact acceptance.

MTP forward (vLLM qwen3_next_mtp / llama.cpp draft-mtp): x = fc(cat(norm_e(embed(tok_{t+1})), norm_h(h_t)));
one full-attention decoder layer (gated q, q/k norm, partial RoPE) with its own KV; out = norm(x);
logits = out @ embed^T (tied lm_head) predict tok_{t+2}. Chained steps feed the MTP's own output hidden and draft.
The MTP KV holds one entry per accepted position, computed from the main model's real hidden state (teacher),
as the engine would rebuild it after every verify.

Run on the build box (needs ~10 GB RAM):
  uv run --no-project --with torch --with transformers --with safetensors \
    python3 tools/spike_mtp.py --model-dir DIR --agent kernel/src/agent.rs --out spike_mtp.json
"""
import argparse, json, re, resource, time
import torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--agent", required=True, help="kernel/src/agent.rs (SYSTEM, TOOLS, template strings)")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--limit", type=int, default=0, help="prompts per set (0 = all)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

CHAT = [
    "che cos'e un kernel? spiegalo in 5 frasi",
    "Explain how a CPU cache works in 3 concise sentences.",
    "scrivi una breve poesia sul mare",
    "what are the pros and cons of microkernels?",
    "Describe the difference between a process and a thread.",
    "come funziona la memoria virtuale? rispondi in poche frasi",
    "Give me three tips to write clean Rust code.",
    "raccontami una breve storia su un gatto curioso",
    "Why is the sky blue? Answer briefly.",
    "quali sono i vantaggi di un sistema operativo scritto in Rust?",
]
TOOLS_SET = [  # tools/tooleval.py INOS requests (first round: the call)
    "ricordati che il mio colore preferito e il verde",
    "cosa sai di me?",
    "quanto fa 2**100? usa python",
    "calcola la media di 3, 7 e 11 con python",
    "elenca i file in /notes",
    "leggi /notes/todo.txt",
    "scrivi in /notes/m47.txt la parola prova",
    "imposta la temperatura a 0",
    "mostra le ultime 5 righe del log",
    "quanti core ha la CPU?",
]


def rust_str(src, name):
    s = re.search(r'const %s: &str = "((?:[^"\\]|\\.)*)";' % name, src, re.S).group(1)
    s = re.sub(r"\\\n\s*", "", s)  # Rust line continuation
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s)


src = open(a.agent).read()
tools = re.findall(r'r#"(.*?)"#', src[src.index("const TOOLS"):src.index("];", src.index("const TOOLS"))], re.S)
assert len(tools) == 18, len(tools)
PREFIX = (rust_str(src, "TEMPLATE_TOOLS_HEAD") + "".join("\n" + j for j in tools) + rust_str(src, "TEMPLATE_TOOLS_TAIL")
          + "\n\n" + rust_str(src, "SYSTEM") + "<|im_end|>\n")
turn = lambda q: PREFIX + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

tok = AutoTokenizer.from_pretrained(a.model_dir)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
print(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10} MiB", flush=True)
cfg = lm.config
EMB = lm.embed_tokens.weight
assert model.lm_head.weight.data_ptr() == EMB.data_ptr()  # tied
EPS = cfg.rms_norm_eps
STOP = [tok.convert_tokens_to_ids("<|im_end|>"), cfg.eos_token_id]

idx = json.load(open(f"{a.model_dir}/model.safetensors.index.json"))["weight_map"]
W = {}
for k in [k for k in idx if k.startswith("mtp.")]:
    with safe_open(f"{a.model_dir}/{idx[k]}", "pt") as f:
        W[k[4:]] = f.get_tensor(k).to(torch.bfloat16)
print("mtp tensors:", {k: tuple(v.shape) for k, v in W.items()}, flush=True)
L0 = "layers.0."


def norm(x, w):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w.float())).to(x.dtype)


def mtp_in(h, ids, order):
    e = norm(EMB[ids], W["pre_fc_norm_embedding.weight"])
    hh = norm(h, W["pre_fc_norm_hidden.weight"])
    return torch.cat([e, hh] if order == "eh" else [hh, e], -1) @ W["fc.weight"].T


def rope(pos):
    cos, sin = lm.rotary_emb(EMB[:1], torch.as_tensor(pos).view(1, 1, -1))
    return cos[0], sin[0]  # (N, rotary_dim)


def rot(x, cos, sin):  # x (H, N, D): rotate the first rotary_dim dims (half split)
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2:]
    return torch.cat([xr * cos + torch.cat([-x2, x1], -1) * sin, xp], -1)


NH, NKV, HD = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim


def layer(x, pos, pk, pv):
    """One MTP decoder layer over rows x (N, hidden) at positions pos, after past KV pk/pv (NKV, M, HD)."""
    N = x.shape[0]
    r = x
    y = norm(x, W[L0 + "input_layernorm.weight"])
    q, g = (y @ W[L0 + "self_attn.q_proj.weight"].T).view(N, NH, 2 * HD).chunk(2, -1)
    g = g.reshape(N, -1)
    q = norm(q, W[L0 + "self_attn.q_norm.weight"]).transpose(0, 1)
    k = norm((y @ W[L0 + "self_attn.k_proj.weight"].T).view(N, NKV, HD), W[L0 + "self_attn.k_norm.weight"]).transpose(0, 1)
    v = (y @ W[L0 + "self_attn.v_proj.weight"].T).view(N, NKV, HD).transpose(0, 1)
    cos, sin = rope(pos)
    q, k = rot(q, cos, sin), rot(k, cos, sin)
    K, V = torch.cat([pk, k], 1), torch.cat([pv, v], 1)
    M = pk.shape[1]
    j = torch.arange(M + N)
    mask = (j[None, :] < M) | ((j[None, :] - M) <= torch.arange(N)[:, None])
    o = F.scaled_dot_product_attention(q, K.repeat_interleave(NH // NKV, 0), V.repeat_interleave(NH // NKV, 0), attn_mask=mask)
    x = r + (o.transpose(0, 1).reshape(N, -1) * torch.sigmoid(g)) @ W[L0 + "self_attn.o_proj.weight"].T
    r = x
    y = norm(x, W[L0 + "post_attention_layernorm.weight"])
    x = r + (F.silu(y @ W[L0 + "mlp.gate_proj.weight"].T) * (y @ W[L0 + "mlp.up_proj.weight"].T)) @ W[L0 + "mlp.down_proj.weight"].T
    return x, k, v


def argmax_rows(hn, chunk=256):
    return torch.cat([(hn[i:i + chunk] @ EMB.T).argmax(-1) for i in range(0, hn.shape[0], chunk)])


EMPTY = torch.zeros(NKV, 0, HD, dtype=torch.bfloat16)


def check_layer(x):
    """Our layer vs the transformers decoder layer (full-attention type) on the same rows, causal, no past."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        cfg._attn_implementation = "sdpa"
        ref = Qwen3_5DecoderLayer(cfg, cfg.layer_types.index("full_attention")).to(torch.bfloat16).eval()
        ref.load_state_dict({k[len(L0):]: v for k, v in W.items() if k.startswith(L0)})
        pos = torch.arange(x.shape[0])
        with torch.no_grad():
            want = ref(x[None], position_embeddings=lm.rotary_emb(x[None], pos.view(1, 1, -1)), attention_mask=None)
        got = layer(x, pos, EMPTY, EMPTY)[0]
        return float((got.float() - want[0].float()).abs().max()), float(want.float().abs().max())
    except Exception as e:  # report, do not stop the spike
        return f"skipped: {e!r}"


def run_prompt(q):
    ids = tok(turn(q), add_special_tokens=False, return_tensors="pt").input_ids
    P = ids.shape[1]
    pre, post = [], []
    def grab(m, i, o):  # must return None: a returned value replaces the module output
        pre.append(i[0][0])
        post.append(o[0])
    hk = lm.norm.register_forward_hook(grab)
    t = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=STOP, pad_token_id=STOP[0])
    hk.remove()
    S = out[0]
    L = S.shape[0]
    H = {"pre": torch.cat(pre), "post": torch.cat(post)}
    assert H["pre"].shape[0] == L - 1, (H["pre"].shape, L)
    return S, P, H, time.time() - t


def teacher(S, H, order):
    """Step 1 of the MTP over every position p = 1..L-1 (row p-1): input (h_{p-1}, S[p]) at position p."""
    L = S.shape[0]
    x = mtp_in(H, S[1:], order)
    out, K, V = layer(x, torch.arange(1, L), EMPTY, EMPTY)
    return out, K, V


def simulate(S, P, out1, K1, V1, n, chain, order):
    """Greedy-verify replay with draft length n. chain: 'post' feeds norm(out) to the next step, 'pre' feeds out."""
    L = S.shape[0]
    on1 = norm(out1, W["norm.weight"])
    D1 = argmax_rows(on1[P - 1:])  # drafts of step 1 for j = P.. (row j-1)
    tried, ok = [0] * (n + 1), [0] * (n + 1)
    j, steps, partial, acc_total = P, 0, 0, 0
    while j < L - 1:
        ne = min(n, L - 1 - j)
        acc = 0
        d = int(D1[j - P])
        hprev = (on1 if chain == "post" else out1)[j - 1]
        pk, pv = K1[:, :j], V1[:, :j]
        for k in range(1, ne + 1):
            if k > 1:
                o, kk, vv = layer(mtp_in(hprev[None], torch.tensor([d]), order), torch.tensor([j + k - 1]), pk, pv)
                pk, pv = torch.cat([pk, kk], 1), torch.cat([pv, vv], 1)
                on = norm(o, W["norm.weight"])
                d = int(argmax_rows(on)[0])
                hprev = (on if chain == "post" else o)[0]
            tried[k] += 1
            if d != int(S[j + k]):
                break
            ok[k] += 1
            acc += 1
        steps += 1
        partial += acc < ne
        acc_total += acc
        j += acc + 1
    return {"steps": steps, "tokens": L - 1 - P, "accepted": acc_total, "partial": partial, "tried": tried[1:], "ok": ok[1:]}


res = {"prompts": [], "check": None}
with torch.no_grad():
    for set_name, qs in (("chat", CHAT), ("tools", TOOLS_SET)):
        for q in qs[: a.limit or None]:
            S, P, H, gt = run_prompt(q)
            r = {"set": set_name, "q": q, "prompt_tokens": P, "gen_tokens": int(S.shape[0] - P), "gen_s": round(gt, 1),
                 "text": tok.decode(S[P:]), "step1_acc": {}, "sim": {}}
            for hv in ("post", "pre"):
                for order in ("eh", "he"):
                    out1, K1, V1 = teacher(S, H[hv], order)
                    on1 = norm(out1, W["norm.weight"])
                    # step-1 top-1 over the generated positions: row p-1 predicts S[p+1], p = P..L-2
                    pred = argmax_rows(on1[P - 1:-1])
                    r["step1_acc"][f"{hv}/{order}"] = [int((pred == S[P + 1:]).sum()), int(pred.shape[0])]
                    if res["check"] is None:
                        res["check"] = check_layer(mtp_in(H["post"][-64:], S[-64:], "eh"))
                        print("layer check (max abs diff, max abs ref):", res["check"], flush=True)
                    if (hv, order) == ("post", "eh"):  # the vLLM / llama.cpp convention: simulate it fully
                        for chain in ("post", "pre"):
                            for n in (1, 2, 3):
                                r["sim"][f"{chain}/n{n}"] = simulate(S, P, out1, K1, V1, n, chain, order)
            res["prompts"].append(r)
            s3 = r["sim"]["post/n3"]
            print(f"[{set_name}] P={P} G={r['gen_tokens']} gen {gt:.0f}s step1 {r['step1_acc']} n3 tok/step "
                  f"{s3['tokens'] / s3['steps']:.2f} maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10} MiB :: {q}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

res["maxrss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

# summary
for set_name in ("chat", "tools"):
    ps = [p for p in res["prompts"] if p["set"] == set_name]
    for v in ps[0]["step1_acc"]:
        c = [sum(p["step1_acc"][v][i] for p in ps) for i in (0, 1)]
        print(f"{set_name} step1 top-1 {v}: {c[0]}/{c[1]} = {c[0] / c[1]:.3f}")
    for key in ps[0]["sim"]:
        agg = {f: sum(p["sim"][key][f] for p in ps) for f in ("steps", "tokens", "accepted", "partial")}
        tried = [sum(p["sim"][key]["tried"][i] for p in ps) for i in range(len(ps[0]["sim"][key]["tried"]))]
        ok = [sum(p["sim"][key]["ok"][i] for p in ps) for i in range(len(tried))]
        print(f"{set_name} {key}: tokens/verify {agg['tokens'] / agg['steps']:.3f}, accepted/drafted-pos "
              f"{[f'{o}/{t}={o / t:.3f}' for o, t in zip(ok, tried)]}, partial {agg['partial']}/{agg['steps']}")
print("maxrss MiB", res["maxrss_mib"])
