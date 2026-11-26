"""Token cost of the tool block vs alternatives (tokenizer only, no model)."""
import json, sys, os
from tokenizers import Tokenizer
H = os.path.dirname(os.path.abspath(__file__))
tk = Tokenizer.from_file(f"{H}/tok/tokenizer.json")
n = lambda s: len(tk.encode(s, add_special_tokens=False).ids)
d = json.load(open(sys.argv[1])); full = d["system_block"]
tools = json.load(open(f"{H}/tools.json"))
tool_json = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
preamble = full.replace(tool_json, "")
sig = "\n".join(f"{t['function']['name']}({', '.join(t['function']['parameters']['properties'])}) - {t['function']['description']}" for t in tools)
fs = [t for t in tools if t["function"]["name"] in ("ls","cat","write_file","mkdir","rm","meminfo","run_code")]
print(f"full template block: {n(full)} tok = {n(full)/37:.1f}s @37tok/s")
print(f"  of which 13 tool JSON lines: {n(tool_json)} tok; fixed template preamble+system: {n(preamble)} tok")
print(f"compact signatures instead of JSON (same preamble): {n(preamble)+n(sig)} tok = {(n(preamble)+n(sig))/37:.1f}s")
print(f"7-tool subset JSON: {n(preamble)+n(''.join(chr(10)+json.dumps(t) for t in fs))} tok")
