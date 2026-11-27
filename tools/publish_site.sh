#!/usr/bin/env bash
# Publish site/ to the public site repo (GitHub Pages: https://tcsenpai.github.io/ringzero-site/).
# site/ in this repo is the source of truth; the site repo only holds a copy.
set -euo pipefail
cd "$(dirname "$0")/.."
# no private data on a public page
if grep -nEi '100\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.|fedora|tailscale|/Users/|/home/' site/*.html site/*.css; then
  echo "publish_site: private-looking data above, not publishing" >&2; exit 1
fi
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git clone -q https://github.com/tcsenpai/ringzero-site.git "$tmp"
find "$tmp" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R site/. "$tmp"/ && touch "$tmp/.nojekyll"
cd "$tmp"
git config user.name "$(git -C "$OLDPWD" config user.name)"; git config user.email "$(git -C "$OLDPWD" config user.email)"
git add -A
if git diff --cached --quiet; then echo "publish_site: no changes"; exit 0; fi
git commit -qm "docs: site update from ringzero $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push -q origin main
echo "publish_site: pushed, live in about a minute"
