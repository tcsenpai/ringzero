#!/usr/bin/env python3
"""embed_ref.py HF_DIR SERIAL_LOG : Qwen3-Embedding-0.6B fp32 reference (transformers, CPU, batch 1) for the strings
in STRINGS, compared with the kernel's `/embed -v` / `/embed -q -v` output (the EMB lines of SERIAL_LOG, in order).
Prints the token counts and the cosine per string. Without SERIAL_LOG: prints the /embed commands to type.
Run on the build box: uv run --no-project --with torch --with transformers python3 tools/embed_ref.py DIR LOG"""
import re, sys

TASK = "Given a question from the user of an operating system agent, retrieve the stored memory that answers it"
# (text, is_query): docs and queries, English and Italian, short and long (the last one ~100 tokens)
STRINGS = [
    ("hello", False),
    ("Il gatto dell'utente si chiama Pixel ed e' allergico al pollo.", False),
    ("Tool result: df -h showed /home at 91% usage, 18 GB free.", False),
    ("The backup job runs nightly at 02:00 with restic to a Backblaze bucket.", False),
    ("La password del Wi-Fi di casa e' salvata in Bitwarden sotto 'casa-wifi'.", False),
    ("how much disk space is left on my home partition?", True),
    ("come si chiama il mio animale domestico?", True),
    ("notes/todo.md", True),
    ("What CPU does this machine have?", True),
    ("ringzero is an experimental AI OS: a small Rust no_std x86-64 kernel, booted by Limine, with an LLM running in "
     "ring 0 as a system agent. The inference engine is NASM assembly linked into the kernel as objects. Every finished "
     "turn is saved as a snapshot point, so the user can undo, redo, branch and save conversations, and the agent calls "
     "kernel tools in its native format.", False),
]

if len(sys.argv) < 3:
    for t, q in STRINGS:
        print(f"/embed {'-q ' if q else ''}-v {t}")
    sys.exit(0)

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(12)
tok = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModel.from_pretrained(sys.argv[1], torch_dtype=torch.float32).eval()
kern = [[float(x) for x in l.split()[1:]] for l in open(sys.argv[2], errors="replace").read().replace("\r", "").split("\n")
        if l.startswith("EMB ")]
ntok_k = [int(m) for m in re.findall(r"\[embed\] (?:doc|query) dim \d+, norm [\d.]+ \(pooled \|x\*w\| [\d.]+\), (\d+) tok", open(sys.argv[2], errors="replace").read())]
assert len(kern) >= len(STRINGS), f"{len(kern)} EMB lines in the log, want {len(STRINGS)}"
kern, ntok_k = kern[-len(STRINGS):], ntok_k[-len(STRINGS):]
cs = []
with torch.no_grad():
    for (t, q), k, nk in zip(STRINGS, kern, ntok_k):
        s = f"Instruct: {TASK}\nQuery:{t}" if q else t
        b = tok([s], return_tensors="pt")
        ids = b["input_ids"][0].tolist()
        assert ids[-1] == 151643, "tokenizer did not append <|endoftext|>"
        h = F.normalize(model(**b).last_hidden_state[0, -1], dim=0)
        c = float(torch.dot(h, F.normalize(torch.tensor(k), dim=0)))
        cs.append(c)
        print(f"cos {c:.5f}  tok hf {len(ids)} kernel {nk}  {'query' if q else 'doc  '}  {t[:60]}")
print(f"min cos {min(cs):.5f}, mean {sum(cs) / len(cs):.5f}, {sum(c >= 0.998 for c in cs)}/{len(cs)} >= 0.998")
