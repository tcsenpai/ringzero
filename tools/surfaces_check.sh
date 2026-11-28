#!/usr/bin/env bash
# Consistency check of the public surfaces listed in docs/SURFACES.md.
# Fails (exit 1) on: private data, em dashes, leftover old name, Italian outside the allowed places,
# broken relative links in README/INFO/docs/site, editions missing from a surface.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
say() { echo "surfaces: $*"; fail=1; }

USER_FACING=(README.md docs/BETA.md docs/DEMOS.md site/index.html release/hf/README.md)
ALL=("${USER_FACING[@]}" INFO.md docs/ARCHITECTURE.md docs/MANUAL.md docs/SURFACES.md release/hf/LICENSE.md)
[ -f release/NOTES.md ] && ALL+=(release/NOTES.md) && USER_FACING+=(release/NOTES.md)
ALL+=(openspec/specs/*/spec.md)  # behaviour specs: same private-data, em dash and link rules

for f in "${ALL[@]}"; do [ -f "$f" ] || { say "missing $f"; continue; }; done

# private data
grep -nEH '\b100\.[0-9]+\.[0-9]+\.[0-9]+\b|\b192\.168\.|fedorascale|fedoralan|tailscale|/Users/|/home/[a-z]' \
  "${ALL[@]}" 2>/dev/null && say "private-looking data above"

# the repo is public: no host names or addresses in any doc, agent instructions or tool data either
grep -nEiH '\b100\.[0-9]+\.[0-9]+\.[0-9]+\b|\b192\.168\.|fedoralan|fedorascale|tailscale|/Users/' \
  AGENTS.md CLAUDE.md docs/*.md docs/research/*.md tools/*.py 2>/dev/null && say "private-looking data above (docs, tools)"

# em dashes (en dashes in number ranges are tolerated)
grep -nH $'\xe2\x80\x94' "${ALL[@]}" 2>/dev/null && say "em dash above"

# old project name and old VNC password in docs
grep -niEH '\baios\b' "${USER_FACING[@]}" INFO.md 2>/dev/null && say "old name 'aios' above"

# Italian in user-facing text (common words; quoted keyword lists live in INFO/ARCHITECTURE/MANUAL only)
IT='\b(quante|quanti|dimmi|scrivi|leggi|crea|cosa devo|come faccio|mi chiamo|righe|perch[eé]|della|delle|nella|spesa|latte|comprare|tabella|ciao)\b'
grep -niEH "$IT" "${USER_FACING[@]}" 2>/dev/null && say "Italian in a user-facing surface above"

# every edition named in the README model table appears on the download page and the site
for ed in 'MODEL=4b QUANT=q4' 'MODEL=2b'; do
  grep -qF "$ed" README.md || say "README lost edition '$ed'"
done
grep -q '4b-q4' release/hf/README.md || say "HF card misses the 4B q4 ISO"
grep -q '2b-q8' release/hf/README.md || say "HF card misses the 2B q8 ISO"

# relative links resolve (markdown and html)
python3 - "${ALL[@]}" <<'EOF' || fail=1
import os, re, sys
bad = 0
for f in sys.argv[1:]:
    if not os.path.exists(f): continue
    s = open(f, encoding="utf-8").read()
    links = re.findall(r'\]\(([^)\s]+)\)', s) + re.findall(r'(?:href|src)="([^"]+)"', s)
    for l in links:
        if re.match(r'(https?:|mailto:|#)', l): continue
        p = os.path.normpath(os.path.join(os.path.dirname(f), l.split('#')[0]))
        if l.split('#')[0] and not os.path.exists(p):
            print(f"surfaces: broken link in {f}: {l}"); bad = 1
sys.exit(bad)
EOF

# Built-in manual: agentd::selftest needs every `## ` section, as "[manual: TITLE] BODY", within fidx::CHUNK + 300 bytes
python3 - <<'EOF' || fail=1
s = open("docs/MANUAL.md", encoding="utf-8").read()
bad = 0
for sec in s.split("\n## ")[1:]:
    t, _, b = sec.partition("\n")
    n = len(f"[manual: {t.strip()}] {b.strip()}".encode())
    if n > 1300:
        print(f"surfaces: MANUAL.md section '{t.strip()}' is {n} bytes, max 1300 (boot self test)"); bad = 1
raise SystemExit(bad)
EOF

# OpenSpec (openspec/specs): commands and settings agree between the specs, INFO.md and /help
python3 tools/spec_check.py || fail=1
if command -v openspec >/dev/null; then
  OPENSPEC_TELEMETRY=0 openspec validate --specs --strict --no-interactive >/dev/null || { say "openspec validate --specs --strict failed"; }
else
  echo "surfaces: openspec CLI not installed, spec format not validated"
fi

[ $fail = 0 ] && echo "surfaces: ok (${#ALL[@]} files)"
exit $fail
