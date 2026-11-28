#!/usr/bin/env bash
# Regenerate the GitHub wiki (github.com/tcsenpai/ringzero/wiki) from openspec/specs and push it.
# The specs are the source of truth; the wiki repo only holds generated pages (tools/wiki_gen.py).
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git clone -q https://github.com/tcsenpai/ringzero.wiki.git "$tmp"
find "$tmp" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
python3 tools/wiki_gen.py "$tmp"
# no private data on a public page
if grep -nEi '100\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.|fedora|tailscale|/Users/' "$tmp"/*.md; then
  echo "publish_wiki: private-looking data above, not publishing" >&2; exit 1
fi
cd "$tmp"
git config user.name "$(git -C "$OLDPWD" config user.name)"; git config user.email "$(git -C "$OLDPWD" config user.email)"
git add -A
if git diff --cached --quiet; then echo "publish_wiki: no changes"; exit 0; fi
git commit -qm "docs: wiki from ringzero specs $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push -q origin HEAD:master
echo "publish_wiki: pushed"
