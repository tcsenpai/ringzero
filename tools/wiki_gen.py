#!/usr/bin/env python3
"""wiki_gen.py OUTDIR : write the GitHub wiki pages from the OpenSpec specs (openspec/specs/*/spec.md).
One page per capability, Home.md (index with each spec's purpose), _Sidebar.md and _Footer.md.
The specs are the source of truth: the wiki is regenerated, never edited by hand (tools/publish_wiki.sh)."""
import os, re, subprocess, sys

REPO = "https://github.com/tcsenpai/ringzero"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "openspec", "specs")
# repo paths named in "Source:" lines (kernel/src/x.rs, tools/y.py, docs/MANUAL.md, Makefile ...)
PATH_RE = re.compile(r"\b((?:kernel|engine|tools|docs|release|site|openspec|wasm)/[\w./-]+\.\w+|Makefile)\b")


def title(cap):
    return " ".join(w.capitalize() for w in cap.split("-"))


def page(cap):
    return title(cap).replace(" ", "-")


def link_sources(line):
    seen = set()
    def sub(m):
        p = m.group(1)
        if p in seen or not os.path.exists(os.path.join(ROOT, p)):
            return p
        seen.add(p)
        return f"[{p}]({REPO}/blob/main/{p})"
    return PATH_RE.sub(sub, line)


def convert(cap, text):
    out = []
    for i, line in enumerate(text.splitlines()):
        if i == 0 and line.startswith("# "):
            out.append(f"# {title(cap)}")
            continue
        if line.startswith("Source:"):
            line = link_sources(line)
        out.append(line)
    out += ["", f"Source of this page: [openspec/specs/{cap}/spec.md]({REPO}/blob/main/openspec/specs/{cap}/spec.md)"]
    return "\n".join(out) + "\n"


def purpose(text):
    m = re.search(r"^## Purpose\s*\n+(.+?)(?:\n\n|\n#)", text, re.S | re.M)
    p = " ".join(m.group(1).split()) if m else ""
    return p.split(". ")[0].rstrip(".") + "." if p else ""


def main(out):
    os.makedirs(out, exist_ok=True)
    caps = sorted(d for d in os.listdir(SPECS) if os.path.isfile(os.path.join(SPECS, d, "spec.md")))
    pages = {c: page(c) for c in caps}
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for c in caps:
        text = open(os.path.join(SPECS, c, "spec.md"), encoding="utf-8").read()
        n = len(re.findall(r"^### Requirement:", text, re.M))
        open(os.path.join(out, pages[c] + ".md"), "w", encoding="utf-8").write(convert(c, text))
        rows.append(f"| [{title(c)}]({pages[c]}) | {n} | {purpose(text)} |")
    home = [
        "# ringzero wiki",
        "",
        "ringzero is an experimental x86-64 operating system where a language model and an embedding model run in",
        "ring 0. This wiki is the behaviour reference: what every command, tool, file and setting does, with the code",
        "it comes from. Each page is generated from the OpenSpec specs in the repository, so it always matches them.",
        "",
        f"Start with the [README]({REPO}#readme), the [technical notes]({REPO}/blob/main/INFO.md) and the",
        f"[manual]({REPO}/blob/main/docs/MANUAL.md). To propose a change, see [CONTRIBUTING]({REPO}/blob/main/CONTRIBUTING.md).",
        "",
        "| Capability | Requirements | What it covers |",
        "|---|---|---|",
        *rows,
        "",
    ]
    open(os.path.join(out, "Home.md"), "w", encoding="utf-8").write("\n".join(home))
    side = ["**[Home](Home)**", "", "Specs:", ""] + [f"- [{title(c)}]({pages[c]})" for c in caps]
    open(os.path.join(out, "_Sidebar.md"), "w", encoding="utf-8").write("\n".join(side) + "\n")
    foot = (f"Generated from [openspec/specs]({REPO}/tree/main/openspec/specs) at {rev} by tools/wiki_gen.py. "
            "Do not edit here: change the spec and regenerate.\n")
    open(os.path.join(out, "_Footer.md"), "w", encoding="utf-8").write(foot)
    assert len(caps) > 0 and all(os.path.getsize(os.path.join(out, pages[c] + ".md")) > 100 for c in caps)
    print(f"wiki_gen: {len(caps)} spec pages + Home, _Sidebar, _Footer in {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
