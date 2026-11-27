#!/usr/bin/env python3
"""Ties the OpenSpec specs to the docs (called by tools/surfaces_check.sh).

Commands are backticked `!name` / `/name` (and `!set KEY` for settings) in openspec/specs/*/spec.md and INFO.md
(INFO.md also counts commands anywhere in its code blocks). /help is the CMDS table in kernel/src/shell.rs
(help() prints every entry) and the settings are the keys named in the `set` entry's help text.
Fails when: a command or setting in the specs is missing from INFO.md or /help, one in INFO.md is missing from
the specs, or one in the code table is missing from the specs.
"""
import glob, os, re, sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# placeholders and directory names that look like commands; hidden test commands (not in /help by design)
IGNORE = {"name", "cmd", "bin", "lib", "demo", "sys", "home", "notes", "tmp", "boot", "proc", "dev", "etc", "var"}
HIDDEN = {"toks", "fault"}
CMD = re.compile(r"(?:^|(?<=[\s(|]))([!/])([a-z][a-z0-9]*)(?![\w/.])")
SET = re.compile(r"!set ([a-z]+)\b")


def scan(text, blocks):
    cmds, sets = set(), set()
    for span in re.findall(r"`([^`\n]+)`", text):
        cmds |= {p + n for p, n in CMD.findall(span) if n not in IGNORE}
        sets |= set(SET.findall(span))
    if blocks:
        for b in re.findall(r"```[^\n]*\n(.*?)```", text, re.S):
            for line in b.splitlines():
                cmds |= {p + n for p, n in CMD.findall(line) if n not in IGNORE}
                sets |= set(SET.findall(line))
    return cmds, sets


specs = sorted(glob.glob("openspec/specs/*/spec.md"))
if not specs:
    sys.exit("spec_check: no specs in openspec/specs")
s_cmds, s_sets = set(), set()
for f in specs:
    c, s = scan(open(f, encoding="utf-8").read(), False)
    s_cmds |= c
    s_sets |= s

i_cmds, i_sets = scan(open("INFO.md", encoding="utf-8").read(), True)

shell = open("kernel/src/shell.rs", encoding="utf-8").read()
h_cmds, set_help = set(), ""
for name, pre, hlp in re.findall(r'c\("(\w+)", ([A-Z |]+), (?:&\[[^\]]*\]|[A-Z_]+), "((?:[^"\\]|\\.)*)"\)', shell):
    for p, flag in (("!", "BANG"), ("/", "SLASH")):
        if flag in pre:
            h_cmds.add(p + name)
    if name == "set":
        set_help = hlp
m = re.search(r"const SETTINGS: &\[&str\] = &\[([^\]]*)\]", shell)
code_sets = set(re.findall(r'"(\w+)"', m.group(1))) if m else set()
h_sets = {k for k in code_sets if re.search(r"\b%s\b" % k, set_help)}

bad = []
def miss(what, items):
    for x in sorted(items):
        bad.append(f"{what}: {x}")

s_vis = {c for c in s_cmds if c[1:] not in HIDDEN}
miss("in specs, not in INFO.md", s_vis - i_cmds)
miss("in specs, not in /help", s_vis - h_cmds)
miss("in INFO.md, not in specs", i_cmds - s_cmds)
miss("in the shell.rs command table, not in specs", h_cmds - s_cmds)
miss("setting in specs, not in INFO.md", {"!set " + k for k in s_sets - i_sets})
miss("setting in specs, not in the /help text of !set", {"!set " + k for k in s_sets - h_sets})
miss("setting in INFO.md, not in specs", {"!set " + k for k in i_sets - s_sets})
miss("setting in shell.rs SETTINGS, not in specs", {"!set " + k for k in code_sets - s_sets})

for b in bad:
    print("spec_check:", b)
print(f"spec_check: {len(specs)} specs, {len(s_cmds)} commands and {len(s_sets)} settings in specs, "
      f"{len(i_cmds)}/{len(i_sets)} in INFO.md, {len(h_cmds)}/{len(h_sets)} in /help: "
      + ("ok" if not bad else f"{len(bad)} mismatch(es)"))
sys.exit(1 if bad else 0)
