#!/usr/bin/env bash
# Render every mermaid diagram in docs/ARCHITECTURE.md (in order) and merge them into
# docs/ARCHITECTURE_DIAGRAMS.pdf, one page per diagram. Fails if any diagram does not parse,
# so it doubles as a check before committing ARCHITECTURE.md.
# Needs: bunx (or set MMDC to an mmdc binary), uv.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
MMDC=${MMDC:-"bunx -p @mermaid-js/mermaid-cli@11.4.2 mmdc"}

python3 - "$tmp" <<'EOF'
import re, sys
s = open("docs/ARCHITECTURE.md").read()
blocks = re.findall(r"```mermaid\n(.*?)```", s, re.S)
for i, b in enumerate(blocks, 1):
    open(f"{sys.argv[1]}/d{i:02}.mmd", "w").write(b)
print(f"{len(blocks)} diagrams")
EOF

for f in "$tmp"/d*.mmd; do
    $MMDC -q -f -i "$f" -o "${f%.mmd}.pdf"
done

uv run -q --with pypdf==5.1.0 python3 - "$tmp" <<'EOF'
import glob, sys
from pypdf import PdfWriter
w = PdfWriter()
for f in sorted(glob.glob(f"{sys.argv[1]}/d*.pdf")):
    w.append(f)
w.write("docs/ARCHITECTURE_DIAGRAMS.pdf")
print("docs/ARCHITECTURE_DIAGRAMS.pdf")
EOF
