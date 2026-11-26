#!/usr/bin/env python3
"""mirror_q4.py (ringzero M7): the numpy mirror of the engine (docs/qwen35-2b/mirror/mirror_q8.py in the qwenbaremetal
repo, --mirror PATH) extended to what M7 added:
  * Q4_0 projection matrices: quantize_q4_0_blocks from tools/export_weights.py (the exporter's own rule), so the
    weights are the ones in the .asmbin. The engine's Q4 kernel does the Q8 kernel's integer lane sums and float
    sequence on q - 8 (engine/matmul.asm), so the mirror's Q8W class runs them unchanged with int8 values in [-8, 7].
  * any Qwen3.5 config (config.json), multi-shard safetensors, Gated DeltaNet with Hv = r * Hk (V head h uses K head
    h // r, HF repeat_interleave).
Embeddings / lm_head stay Q8_0, a/b f32, norms 1+w (FORMAT_V3), as in the engine.
Usage: mirror_q4.py --mirror DIR --model-dir HF --prompt-ids 1,2,3 [--max-new N] [--q8] [--taps F] [--out F]
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mirror", required=True, help="dir with mirror_q8.py")
ap.add_argument("--model-dir", required=True)
ap.add_argument("--prompt-ids", required=True)
ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--q8", action="store_true", help="all matrices Q8_0 (the v3 Q8 build)")
ap.add_argument("--cache", help="quantized-weight cache dir")
ap.add_argument("--taps")
ap.add_argument("--out")
a = ap.parse_args()
sys.path.insert(0, a.mirror)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mirror_q8 as M  # noqa: E402
from export_weights import quantize_q4_0_blocks  # noqa: E402

F32, F64 = np.float32, np.float64


def q4w(get_rows, Mr, K, chunk=4096):
    nb = K // 32
    q = np.empty((nb, Mr, 32), np.int8); ws = np.empty((nb, Mr), F32)
    for r0 in range(0, Mr, chunk):
        r1 = min(Mr, r0 + chunk)
        qq, d16 = quantize_q4_0_blocks(get_rows(r0, r1))
        q[:, r0:r1] = qq.transpose(1, 0, 2); ws[:, r0:r1] = d16.astype(F32).T
    return M.Q8W(q, ws)


class STS:
    """multi-shard ST: name -> the shard's ST"""
    def __init__(self, d):
        idx = os.path.join(d, "model.safetensors.index.json")
        files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
        self.sts = [M.ST(os.path.join(d, f)) for f in files]
    def _st(self, n): return next(s for s in self.sts if n in s.hdr)
    def f32(self, n): return self._st(n).f32(n)
    def mat(self, n, cache, tag, four):
        s = self._st(n); m, _ = s.raw(n); Mr, K = m["shape"]
        if four:
            return q4w(s.rows(n), Mr, K)
        return s.q8(n, cache, tag)


class Mirror4(M.Mirror):
    def __init__(self, model_dir, q8_only, cache=None, taps=False):
        cfg = json.load(open(os.path.join(model_dir, "config.json"))); t = cfg.get("text_config", cfg)
        rp = t.get("rope_parameters", {})
        lt = t["layer_types"]
        self.c = c = dict(arch="qwen35", H=t["hidden_size"], FF=t["intermediate_size"], L=t["num_hidden_layers"],
                          NH=t["num_attention_heads"], NKV=t["num_key_value_heads"], HD=t["head_dim"],
                          ROT=int(t["head_dim"] * rp.get("partial_rotary_factor", 1.0)), THETA=float(rp["rope_theta"]),
                          EPS=float(t["rms_norm_eps"]), VOCAB=t["vocab_size"], pref="model.language_model.", gate=True,
                          plus1=True, layer_full=[x == "full_attention" for x in lt], argmax_lim=248077,
                          LK=t["linear_num_key_heads"], LV=t["linear_num_value_heads"], LD=t["linear_key_head_dim"],
                          CK=t["linear_conv_kernel_dim"], LH=t["linear_num_value_heads"])
        self.taps = taps; self.ab_fp32 = True; self.eps = F32(c["EPS"])
        st = STS(model_dir); P = c["pref"]; t0 = time.time(); four = not q8_only
        nrm = lambda n: (st.f32(n) + F32(1.0)).astype(F32)
        self.E = st.mat(P + "embed_tokens.weight", cache, "embed", False)
        self.FN = nrm(P + "norm.weight")
        self.Ls = []
        for i in range(c["L"]):
            p = f"{P}layers.{i}."; d = {"full": c["layer_full"][i]}
            q = lambda n, tg: st.mat(p + n, cache, f"L{i}.{tg}", four)
            d["in"], d["post"] = nrm(p + "input_layernorm.weight"), nrm(p + "post_attention_layernorm.weight")
            d["gate"], d["up"], d["down"] = q("mlp.gate_proj.weight", "gate"), q("mlp.up_proj.weight", "up"), q("mlp.down_proj.weight", "down")
            if d["full"]:
                s = "self_attn."
                d["q"], d["k"], d["v"], d["o"] = q(s + "q_proj.weight", "q"), q(s + "k_proj.weight", "k"), q(s + "v_proj.weight", "v"), q(s + "o_proj.weight", "o")
                d["qn"], d["kn"] = nrm(p + s + "q_norm.weight"), nrm(p + s + "k_norm.weight")
            else:
                s = "linear_attn."
                d["qkv"], d["z"], d["out"] = q(s + "in_proj_qkv.weight", "qkv"), q(s + "in_proj_z.weight", "z"), q(s + "out_proj.weight", "out")
                d["a"], d["b"] = st.f32(p + s + "in_proj_a.weight"), st.f32(p + s + "in_proj_b.weight")
                d["conv"] = st.f32(p + s + "conv1d.weight")[:, 0, :]
                d["negexpA"] = (-np.exp(st.f32(p + s + "A_log").astype(F64))).astype(F32).astype(F64)
                d["dtb"] = st.f32(p + s + "dt_bias").astype(F64); d["nw"] = st.f32(p + s + "norm.weight")
            self.Ls.append(d)
        half = c["ROT"] // 2
        step = -np.log(F64(F32(c["THETA"]))) / F64(c["ROT"])
        self.inv_freq = np.exp((np.arange(half, dtype=F64) * 2.0) * step)
        self.att_scale = F32(1.0) / np.sqrt(F32(c["HD"]))
        print(f"loaded {'Q8' if q8_only else 'Q4'} mirror in {time.time()-t0:.1f}s", file=sys.stderr)
        self.reset()

    def reset(self):
        c = self.c; self.K = {}; self.V = {}; self.S = {}; self.cs = {}
        for i, d in enumerate(self.Ls):
            if d["full"]:
                self.K[i] = np.zeros((c["NKV"], 0, c["HD"]), F32); self.V[i] = np.zeros((c["NKV"], 0, c["HD"]), F32)
            else:
                self.S[i] = np.zeros((c["LV"], c["LD"], c["LD"]), F32)
                self.cs[i] = np.zeros((c["CK"] - 1, (2 * c["LK"] + c["LV"]) * c["LD"]), F32)

    def delta_net(self, i, d, n, xp):
        c = self.c; LK, LV, LD = c["LK"], c["LV"], c["LD"]; r = LV // LK; eps = self.eps
        mixed = d["qkv"].matvec(xp); z = d["z"].matvec(xp).reshape(LV, LD)
        b = M.sumprod32(d["b"], n); aa = M.sumprod32(d["a"], n)
        win = np.concatenate([self.cs[i], mixed[None]], 0); cw = d["conv"]
        y = (((win[0] * cw[:, 0] + win[1] * cw[:, 1]) + win[2] * cw[:, 2]) + win[3] * cw[:, 3]).astype(F32)
        self.cs[i] = win[1:].copy(); y = M.silu(y)
        kd = LK * LD
        q, k, v = y[:kd].reshape(LK, LD), y[kd:2 * kd].reshape(LK, LD), y[2 * kd:].reshape(LV, LD)
        l2n = lambda x: (x * (F32(1.0) / np.sqrt(M.sumprod32(x, x) + F32(1e-6)))[:, None]).astype(F32)
        q = (l2n(q) * F32(1.0 / np.sqrt(LD))).astype(F32); k = l2n(k)
        q, k = np.repeat(q, r, 0), np.repeat(k, r, 0)          # V head h uses K head h // r
        beta = (1.0 / (1.0 + np.exp(-b.astype(F64)))); sp = aa.astype(F64) + d["dtb"]
        g = d["negexpA"] * np.where(sp > 20, sp, np.log1p(np.exp(np.minimum(sp, 20))))
        dec = np.exp(g).astype(F32); beta = beta.astype(F32)
        S = self.S[i]; o = np.empty((LV, LD), F32)
        for h in range(LV):
            Sh = S[h] * dec[h]; kv = Sh.T @ k[h]
            Sh = Sh + np.outer(k[h], (v[h] - kv) * beta[h]); S[h] = Sh; o[h] = Sh.T @ q[h]
        on = M.silu_mul(z, M.rmsnorm(o, d["nw"], eps)).reshape(-1)
        return d["out"].matvec(M.prep(on))


m = Mirror4(a.model_dir, a.q8, a.cache, taps=bool(a.taps))
prompt = [int(v) for v in a.prompt_ids.split(",")]
gen, top5, taps, lt = M.run(m, prompt, a.max_new, set())
print("PROMPT_TOKENS:" + ",".join(map(str, prompt))); print("GEN_TOKENS:" + ",".join(map(str, gen)))
print("first_top5:", top5)
if a.taps: json.dump(dict(prompt=prompt, gen=gen, taps=taps), open(a.taps, "w"))
if a.out: json.dump(dict(prompt=prompt, gen=gen, first_top5=top5, top1_logits=lt), open(a.out, "w"), indent=1)
