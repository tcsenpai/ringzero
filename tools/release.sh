#!/usr/bin/env bash
# tools/release.sh VERSION : build release VERSION (e.g. 0.1.1) on the Linux build box and print the publish
# commands. Publishes nothing. Steps (INFO.md "Releases" has the checklist around it):
#   1. checks: clean tree, kernel/Cargo.toml and release/NOTES.md name VERSION, tools/surfaces_check.sh passes
#   2. make beta into build/release-VERSION (both ISOs, the data disk; the manual-vector pass boots a VM, so it
#      runs under the shared VM lock ~/.ringzero-vm.lock)
#   3. renames to ringzero-VERSION-{4b-q4,2b-q8}.iso, xz -9 of the disk, SHA256SUMS, a Hugging Face upload folder
#   4. prints gh / hf / site commands to run after the checks (boot tests, upgrade test, tooleval)
# Env: WASM_DIR (reuse built WASI programs), LOCK (lock file).
set -euo pipefail
cd "$(dirname "$0")/.."
V=${1:?usage: tools/release.sh VERSION}
die() { echo "release: $*" >&2; exit 1; }

grep -q "^version = \"$V\"" kernel/Cargo.toml || die "kernel/Cargo.toml version is not $V"
head -1 release/NOTES.md | grep -qF "ringzero $V " || die "release/NOTES.md does not start with 'ringzero $V'"
git diff --quiet HEAD -- . ':!.mycelium' || die "uncommitted changes (commit first: the build epoch is the commit time)"
tools/surfaces_check.sh >/dev/null || die "tools/surfaces_check.sh failed"

OUT=build/release-$V
rm -rf "$OUT"
flock "${LOCK:-$HOME/.ringzero-vm.lock}" make beta BETA="$OUT"
cd "$OUT"
mv ringzero-beta-4b-q4.iso "ringzero-$V-4b-q4.iso"
mv ringzero-beta-2b-q8.iso "ringzero-$V-2b-q8.iso"
xz -9 -T1 -k -c disk.img > "ringzero-$V-disk.img.xz"
sha256sum "ringzero-$V-4b-q4.iso" "ringzero-$V-2b-q8.iso" "ringzero-$V-disk.img.xz" disk.img > SHA256SUMS
mkdir hf
for f in ../../release/hf/*; do ln -s "../$f" hf/; done
for f in "ringzero-$V-4b-q4.iso" "ringzero-$V-2b-q8.iso" SHA256SUMS; do ln -s "../$f" hf/; done
cd - >/dev/null
ls -l "$OUT"; cat "$OUT/SHA256SUMS"

cat <<EOF

release: $OUT ready for $V ($(git rev-parse --short HEAD)). Before publishing: INFO.md "Releases" checks.
Publish (from this checkout, after review):

  git tag v$V && git push origin HEAD:main v$V     # this commit becomes main (fast-forward only)
  gh release create v$V --prerelease --title "ringzero $V $(sed -n '1s/.*"\(.*\)".*/\1/p' release/NOTES.md) (beta)" \\
    --notes-file release/NOTES.md $OUT/ringzero-$V-disk.img.xz $OUT/SHA256SUMS
  hf upload Pleistocene/ringzero $OUT/hf . --repo-type model --commit-message "ringzero $V"
  tools/publish_site.sh
EOF
