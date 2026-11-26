#!/usr/bin/env python3
"""hf_prefix.py HF_DIR [KERNEL_IDS_FILE]: render the agent's system+tools prefix with the HF chat
template (tools and system prompt read from kernel/src/agent.rs), print its token count and FNV-1a
hash (same as the kernel's boot line), and diff against the ids printed by the kernel `toks` builtin."""
import json, re, sys, os
from tokenizers import Tokenizer
src = open(os.path.join(os.path.dirname(__file__), "../kernel/src/agent.rs")).read()
tools = [json.loads(t) for t in re.findall(r'r#"(\{"type": "function".*?)"#', src)]
system = re.search(r'const SYSTEM: &str = "(.*?)";', src, re.S).group(1).replace("\\\n", "")
system = re.sub(r"\s+", " ", system)
cfg = json.load(open(f"{sys.argv[1]}/tokenizer_config.json"))
from jinja2 import Environment
env = Environment(); env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
txt = env.from_string(cfg["chat_template"]).render(messages=[{"role": "system", "content": system}, {"role": "user", "content": "x"}], tools=tools)
prefix = txt.split("<|im_end|>")[0] + "<|im_end|>\n"
ids = Tokenizer.from_file(f"{sys.argv[1]}/tokenizer.json").encode(prefix, add_special_tokens=False).ids
h = 0xcbf29ce484222325
for x in ids: h = ((h ^ x) * 0x100000001b3) & (2**64 - 1)
print(f"HF prefix: {len(ids)} tokens, fnv {h:#x}")
if len(sys.argv) > 2:
    k = [int(x) for x in re.findall(r"\d+", open(sys.argv[2]).read().strip().splitlines()[-1])]
    d = next((i for i, (a, b) in enumerate(zip(ids, k)) if a != b), None)
    print(f"kernel: {len(k)} tokens, first difference at {d}")
    if d is not None: print("HF", ids[d:d+8], "kernel", k[d:d+8])
