#!/usr/bin/env python3
"""T74: contrastive LoRA fine-tune of Qwen3-Embedding-0.6B on the gen_data.py triples (not run yet: the user
decides when). Plain torch + peft, no sentence-transformers, so the text format is exactly the kernel's.

  train.py --data DIR --out DIR [--dry-run] [--check-model] [options]

Format (kernel/src/embed.rs, tools/rerankeval.py embedder): a query is "Instruct: TASK\\nQuery:QUESTION", a stored
entry is its index text; tokens without special tokens, then EOT (151643); the vector is the last token's hidden
state, L2-normalized. Batches pad on the left, so the last position is the EOT of every row.

Loss: InfoNCE, temperature --tau, over each query's positive, its 3 hard negatives and every other text in the
batch (in-batch negatives). LoRA (r --r, alpha 2r, dropout 0.05) on every attention and MLP projection; the
merged model is saved in bf16 as a bare Qwen3Model with the tokenizer, which tools/export_weights.py --v3 reads.
Why LoRA, not full: the gate wants the model to stay general (a low-rank update moves it less), the optimizer
state is a few MB instead of about 10 GB (fits 16 GB with room), and merging restores the kernel's exact format.

--dry-run validates the data (schema, classes, 3 negatives, no positive among negatives, no leak against the
T69 eval set via check_disjoint.py) and prints the config and the planned steps; nothing is trained. With
--check-model it also loads the base model and checks that a left-padded batch gives the same vectors as the
one-at-a-time kernel path (cosine >= 0.9999).
"""
import argparse, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rerankeval as rr  # noqa: E402
import check_disjoint as CD  # noqa: E402

CLASSES = ("para", "exact", "pointer", "sibling", "hub", "owner", "raw", "table", "temporal", "contradiction", "multihop")
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def validate(recs, ev):
    errs = []
    for i, x in enumerate(recs):
        if set(x) < {"cls", "lang", "q", "pos", "neg"}:
            errs.append((i, "schema"))
        elif x["cls"] not in CLASSES or x["lang"] not in ("it", "en"):
            errs.append((i, "class or language"))
        elif len(x["neg"]) != 3 or not x["q"].strip() or not x["pos"].strip() or x["pos"] in x["neg"]:
            errs.append((i, "negatives"))
        elif x["q"].startswith("Instruct:"):
            errs.append((i, "prefix already in the data"))
        elif ev.leak(x):
            errs.append((i, "leak: " + ev.leak(x)))
    return errs


def query_text(q):
    return f"Instruct: {rr.TASK}\nQuery:{q}"


def encode(model, tok, texts, dev, max_len):
    """Kernel format, batched: ids + EOT, left padding, last position pooled, L2-normalized."""
    import torch
    ids = [tok(t, add_special_tokens=False)["input_ids"][:max_len - 1] + [rr.EOT] for t in texts]
    n = max(map(len, ids))
    pad = tok.pad_token_id if tok.pad_token_id is not None else rr.EOT
    x = torch.tensor([[pad] * (n - len(i)) + i for i in ids], device=dev)
    m = torch.tensor([[0] * (n - len(i)) + [1] * len(i) for i in ids], device=dev)
    h = model(input_ids=x, attention_mask=m).last_hidden_state[:, -1]
    return torch.nn.functional.normalize(h.float(), dim=-1)


def check_model(base, dev, recs, max_len):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModel.from_pretrained(base, dtype=torch.float32).to(dev).eval()
    texts = [query_text(recs[0]["q"]), recs[0]["pos"]] + recs[1]["neg"] + [query_text(recs[2]["q"])]
    with torch.no_grad():
        B = encode(model, tok, texts, dev, max_len)
        one = torch.cat([encode(model, tok, [t], dev, max_len) for t in texts])
    agree = (B * one).sum(-1).min().item()
    print(f"batched vs one-at-a-time vectors: min cosine {agree:.6f} over {len(texts)} texts")
    assert agree >= 0.9999, "left padding changes the vectors: fix the attention mask or the pooling"


def train(a, recs, dev_recs):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(a.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModel.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=list(TARGETS)))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(recs) / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) *
                                              max(0.0, (steps - s) / steps))
    rng, step, t0 = random.Random(a.seed), 0, time.time()

    def batch_loss(b):
        q = encode(model, tok, [query_text(x["q"]) for x in b], dev, a.max_len)
        d = encode(model, tok, [x["pos"] for x in b] + [n for x in b for n in x["neg"]], dev, a.max_len)
        logits = q @ d.T / a.tau  # row i: its positive is column i; every other column is a negative
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(b), device=dev))

    log = []
    for ep in range(a.epochs):
        rng.shuffle(recs)
        for i in range(0, len(recs), a.batch):
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = batch_loss(recs[i:i + a.batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            step += 1
            if step % 20 == 0 or step == steps:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    dl = [batch_loss(dev_recs[j:j + a.batch]).item() for j in range(0, len(dev_recs), a.batch)]
                e = {"step": step, "loss": round(loss.item(), 4), "dev_loss": round(sum(dl) / len(dl), 4),
                     "s": round(time.time() - t0), "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                log.append(e)
                print(json.dumps(e), flush=True)
    merged = model.merge_and_unload()
    os.makedirs(a.out, exist_ok=True)
    merged.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(a.out)
    json.dump({"args": vars(a), "log": log}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
    print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-model", action="store_true")
    a = ap.parse_args()
    recs, dev_recs = load(os.path.join(a.data, "train.jsonl")), load(os.path.join(a.data, "dev.jsonl"))
    ev = CD.EvalSide()
    errs = validate(recs + dev_recs, ev)
    by = {c: sum(x["cls"] == c for x in recs) for c in CLASSES}
    steps = a.epochs * math.ceil(len(recs) / a.batch)
    print(json.dumps({"train": len(recs), "dev": len(dev_recs), "per_class": by, "errors": len(errs),
                      "query_format": query_text("QUESTION"), "eot": rr.EOT, "lora": {"r": a.r, "alpha": 2 * a.r,
                      "targets": TARGETS}, "lr": a.lr, "epochs": a.epochs, "batch": a.batch, "tau": a.tau,
                      "texts_per_step": a.batch * 5, "steps": steps}, indent=1))
    for e in errs[:10]:
        print("  bad record", e)
    assert not errs, f"{len(errs)} bad records: fix the data first"
    assert all(by.values()), "a class has no training record"
    if a.check_model:
        import torch
        check_model(a.base, "cuda" if torch.cuda.is_available() else "cpu", recs, a.max_len)
    if a.dry_run:
        print("dry run: data and config valid, nothing trained")
        return
    train(a, recs, dev_recs)


if __name__ == "__main__":
    main()
