"""Score raw outputs. usage: score.py raw.json [more.json...]  -> prints markdown table + summary."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import coerce

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = {t["function"]["name"]: t["function"]["parameters"] for t in json.load(open(f"{HERE}/tools.json"))}
PROMPTS = {p["id"]: p for p in json.load(open(f"{HERE}/prompts.json"))}


def schema_ok(name, args):
    """Minimal JSON-schema check: required present, no unknown keys, types, enums."""
    s = TOOLS[name]; props = s.get("properties", {}); a = coerce(args, s)
    if any(r not in a for r in s.get("required", [])): return False, "missing required"
    for k, v in a.items():
        if k not in props: return False, f"unknown param {k}"
        t = props[k]["type"]
        py = {"string": str, "integer": int, "boolean": bool}[t]
        if not isinstance(v, py) or (t == "integer" and isinstance(v, bool)): return False, f"{k} not {t}"
        if "enum" in props[k] and v not in props[k]["enum"]: return False, f"{k}={v!r} not in enum"
    return True, ""


def args_match(exp, args):
    for k, rule in exp.get("args", {}).items():
        v = str(args.get(k, "")).strip().lower()
        want = rule[1:].lower()
        if rule[0] == "=" and v.rstrip("/") != want.rstrip("/") and v != want: return False
        if rule[0] == "~" and want not in v: return False
    return True


def score(run):
    p = PROMPTS[run["id"]]; exp = p["expect"]
    calls = [c for t in run["turns"] for c in t["calls"]]
    r = {"id": p["id"], "lang": p["lang"], "text": p["text"], "kind": "none" if not exp else ("multi" if len(exp) > 1 else "single"),
         "malformed": any(t["malformed"] for t in run["turns"]), "hit_max": any(t["hit_max"] for t in run["turns"]),
         "calls": [n for n, _ in calls]}
    r["hallucinated"] = any(n not in TOOLS for n, _ in calls)
    r["parsed_ok"] = not r["malformed"] and (bool(calls) or not exp)
    if not exp:
        r["unneeded_call"] = bool(calls)
        r["pass"] = not calls and not r["malformed"]
        return r
    r["prose_instead"] = not calls
    valid = [schema_ok(n, a) if n in TOOLS else (False, "unknown tool") for n, a in calls]
    r["schema_errs"] = [e for ok, e in valid if not ok]
    need = calls[:len(exp)]
    if p.get("unordered"):
        names = [n for n, _ in need]
        r["right_tool"] = len(need) == len(exp) and all(any(n in e["tool"] for n in names) for e in exp)
        r["args_correct"] = r["right_tool"]
    else:
        r["right_tool"] = len(need) == len(exp) and all(n in e["tool"] for (n, _), e in zip(need, exp))
        r["args_correct"] = r["right_tool"] and all(args_match(e, a) for (_, a), e in zip(need, exp))
    r["args_valid"] = bool(need) and all(ok for ok, _ in valid[:len(exp)])
    r["extra_calls"] = len(calls) - len(exp)
    r["pass"] = r["parsed_ok"] and r["right_tool"] and r["args_valid"] and r["args_correct"] and not r["hallucinated"]
    return r


def report(path):
    d = json.load(open(path))
    rows = [score(run) for run in d["runs"]]
    Y = lambda b: "Y" if b else "-"
    print(f"\n### mode={d['mode']}  (system+tools = {d['system_tokens']} tok, {d['system_tokens']/37:.1f}s prefill @37 tok/s)\n")
    print("| id | kind | prompt | calls | parsed | right tool | schema ok | args correct | halluc | unneeded call | prose instead | PASS |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['id']} | {r['kind']} | {r['text'][:45]} | {','.join(r['calls']) or '(none)'} | {Y(r['parsed_ok'])} | "
              f"{Y(r.get('right_tool', True)) if r['kind']!='none' else 'n/a'} | {Y(r.get('args_valid')) if r['kind']!='none' else 'n/a'} | "
              f"{Y(r.get('args_correct')) if r['kind']!='none' else 'n/a'} | {Y(r['hallucinated'])} | {Y(r.get('unneeded_call'))} | "
              f"{Y(r.get('prose_instead'))} | **{Y(r['pass'])}** |")
    tool_rows = [r for r in rows if r["kind"] != "none"]
    none_rows = [r for r in rows if r["kind"] == "none"]
    s = lambda rs, k: sum(1 for r in rs if r.get(k))
    print(f"\n- tool-needed prompts: {len(tool_rows)}; PASS (well-formed + right tool + schema-valid + correct args): "
          f"**{s(tool_rows,'pass')}/{len(tool_rows)}**")
    print(f"- schema-valid call rate (right tool + schema ok, ignoring arg semantics): "
          f"{sum(1 for r in tool_rows if r['right_tool'] and r['args_valid'])}/{len(tool_rows)}")
    for k in ("single", "multi"):
        rs = [r for r in tool_rows if r["kind"] == k]
        if rs: print(f"  - {k}: {s(rs,'pass')}/{len(rs)}")
    print(f"- no-tool prompts answered without a call: {s(none_rows,'pass')}/{len(none_rows)}")
    print(f"- malformed outputs: {s(rows,'malformed')}, hallucinated tool names: {s(rows,'hallucinated')}, "
          f"prose instead of call: {s(tool_rows,'prose_instead')}, hit max tokens: {s(rows,'hit_max')}")
    errs = [(r['id'], e) for r in tool_rows for e in r.get('schema_errs', [])]
    if errs: print(f"- schema errors: {errs}")
    gen = [t["gen_tokens"] for run in d["runs"] for t in run["turns"][:1]]
    print(f"- first-turn generated tokens: mean {sum(gen)/len(gen):.0f}, max {max(gen)}")
    return rows


if __name__ == "__main__":
    for f in sys.argv[1:]:
        report(f)
