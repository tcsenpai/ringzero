"""Greedy tool-call spike for base Qwen3.5-2B. usage: run.py MODEL_DIR OUT.json MODE(nothink|think) [ids,comma,sep]"""
import json, sys, time, os, torch
# torchvision ABI workaround copied from docs/qwen35-2b/ref-scripts/golden.py (text-only, no vision needed)
import transformers.utils.import_utils as _iu
_orig = _iu._is_package_available
_iu.is_torchvision_available = lambda: False
_iu.is_torchvision_v2_available = lambda: False
import transformers.utils as _u; _u.is_torchvision_available = _iu.is_torchvision_available; _u.is_torchvision_v2_available = _iu.is_torchvision_v2_available
_iu._is_package_available = lambda name, *a, **k: (False, "N/A") if name == "torchvision" else _orig(name, *a, **k)
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import parse_calls, strip_think

P, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
ONLY = set(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else None
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = json.load(open(f"{HERE}/tools.json"))
PROMPTS = [p for p in json.load(open(f"{HERE}/prompts.json")) if ONLY is None or p["id"] in ONLY]
SYSTEM = "You are the system agent of aiOS, an experimental OS. Use the tools to act on this machine. Be brief."
THINK = MODE == "think"
MAX_NEW = 1536 if THINK else 384
MOCK = {  # canned tool results so multi-step prompts can continue
    "ls": "bin etc home tmp", "cat": "Welcome to aiOS\n", "write_file": "ok", "mkdir": "ok", "rm": "ok",
    "meminfo": "total=8192MiB free=6120MiB", "cpuinfo": "Intel Xeon (24 cores, avx2)", "list_cores": "24 cores online",
    "read_log": "[0.01] boot ok\n[2.10] agent ready", "poweroff": "ok", "reboot": "ok", "set_setting": "ok", "run_code": "4",
}

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
tok = AutoTokenizer.from_pretrained(P)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float32).eval()
dt_before = sorted({str(p.dtype) for p in model.parameters()})
model = model.float()
# same exact-f32 reload as golden.py: HF may build in bf16 and round the F32 checkpoint tensors
from safetensors import safe_open
sf = safe_open(P + "/model.safetensors-00001-of-00001.safetensors", "pt")
for name, prm in model.named_parameters():
    if name == "lm_head.weight": continue
    prm.data.copy_(sf.get_tensor("model.language_model." + name[len("model."):]).to(torch.float32))
print(f"loaded in {time.time()-t0:.1f}s dtype_before={dt_before}", file=sys.stderr)

# prefill cost of system+tools block
sys_text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "x"}], tools=TOOLS, tokenize=False)
sys_block = sys_text.split("<|im_end|>")[0] + "<|im_end|>\n"
n_sys = len(tok(sys_block, add_special_tokens=False)["input_ids"])
print(f"system+tools tokens={n_sys}", file=sys.stderr)

results = {"mode": MODE, "dtype_loaded_before_float": dt_before, "system_tokens": n_sys, "system_block": sys_block, "runs": []}
for p in PROMPTS:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["text"]}]
    turns = []
    for _ in range(3):
        text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        t1 = time.time()
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
        gen = out[0, ids.shape[1]:]
        raw = tok.decode(gen, skip_special_tokens=False)
        calls, malformed = parse_calls(raw)
        turns.append({"prompt_tokens": ids.shape[1], "gen_tokens": len(gen), "secs": round(time.time() - t1, 1),
                      "raw": raw, "calls": calls, "malformed": malformed, "hit_max": len(gen) >= MAX_NEW})
        print(f"[{p['id']}] {p['text'][:40]!r} -> {calls} malformed={malformed} {len(gen)}tok {time.time()-t1:.1f}s", file=sys.stderr)
        if not calls:
            break
        content = strip_think(raw).split("<tool_call>")[0].strip()
        msgs.append({"role": "assistant", "content": content,
                     "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]})
        for n, _a in calls:
            msgs.append({"role": "tool", "content": MOCK.get(n, "error: unknown tool")})
    results["runs"].append({"id": p["id"], "text": p["text"], "turns": turns})
    json.dump(results, open(OUT, "w"), indent=1, ensure_ascii=False)
print("done", file=sys.stderr)
