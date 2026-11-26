"""Parser for Qwen3.5 XML-ish tool calls: <tool_call><function=N><parameter=K>\nV\n</parameter></function></tool_call>."""
import re, json

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def strip_think(text):
    return text.split("</think>")[-1] if "</think>" in text else text


def parse_calls(text):
    """Returns (calls, malformed). calls = [(name, {k: raw_str})]. malformed = tool-call markup present but not parseable."""
    body = strip_think(text)
    calls = []
    for name, inner in CALL_RE.findall(body):
        calls.append((name, {k: v for k, v in PARAM_RE.findall(inner)}))
    leftover = CALL_RE.sub("", body)
    malformed = any(t in leftover for t in ("<tool_call>", "<function=", "<parameter=", "</tool_call>"))
    # inner junk that is not a parameter block also counts as malformed
    for _, inner in CALL_RE.findall(body):
        if PARAM_RE.sub("", inner).strip():
            malformed = True
    return calls, malformed


def coerce(args, schema):
    """Parameter values arrive as strings; convert per schema type (the template stringifies non-strings)."""
    props = schema.get("properties", {})
    out = {}
    for k, v in args.items():
        t = props.get(k, {}).get("type")
        if t in ("integer", "number", "boolean", "object", "array"):
            try:
                out[k] = json.loads(v.strip().lower() if t == "boolean" else v.strip())
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


if __name__ == "__main__":  # self-check
    s = "<think>\nx\n</think>\n\n<tool_call>\n<function=ls>\n<parameter=path>\n/home\n</parameter>\n</function>\n</tool_call>"
    c, m = parse_calls(s)
    assert c == [("ls", {"path": "/home"})] and not m, (c, m)
    c, m = parse_calls("<tool_call>\n<function=ls>\n<parameter=path>/x</parameter>\n</tool_call>")
    assert c == [] and m
    c, m = parse_calls('<tool_call>\n{"name": "ls"}\n</tool_call>')
    assert c == [] and m
    assert coerce({"n": "20", "recursive": "True"}, {"properties": {"n": {"type": "integer"}, "recursive": {"type": "boolean"}}}) == {"n": 20, "recursive": True}
    print("parse.py self-check ok")
