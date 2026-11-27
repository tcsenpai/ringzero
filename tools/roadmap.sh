#!/usr/bin/env bash
# Regenerate docs/ROADMAP.md from the mycelium plan (myc list --all).
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo "# Roadmap"
  echo
  echo "Generated from the mycelium plan (\`myc list --all\`), $(date +%F)."
  echo
  echo '```text'
  myc list --all 2>/dev/null
  echo '```'
} > docs/ROADMAP.md
