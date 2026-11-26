#!/usr/bin/env bash
# fetch_model.sh [OUT]: download a model from Hugging Face (skipping files already there) and export it to the
# engine's .asmbin (Q8_0, format v3). Default: Qwen/Qwen3.5-2B. `make model` also runs it for the embedder
# (HF_REPO=Qwen/Qwen3-Embedding-0.6B EXPORT_ARGS=--v3). Skips everything when OUT already exists (FORCE=1 redoes
# the export). HF_DIR=dir reuses an existing download instead. Needs uv (https://docs.astral.sh/uv/); curl is the
# fallback downloader.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-models/qwen35-2b-q8_0-v3.asmbin}
REPO=${HF_REPO:-Qwen/Qwen3.5-2B}
FILES=${HF_FILES:-"config.json model.safetensors.index.json model.safetensors-00001-of-00001.safetensors tokenizer.json tokenizer_config.json"}
if [ -e "$OUT" ] && [ -z "${FORCE:-}" ]; then
    echo "have $OUT (FORCE=1 exports it again)"
    exit 0
fi
if [ -z "${HF_DIR:-}" ]; then
    HF_DIR=models/hf/$(basename "$REPO" | tr '[:upper:]' '[:lower:]')
    mkdir -p "$HF_DIR"
    for f in $FILES; do
        [ -s "$HF_DIR/$f" ] && { echo "have $HF_DIR/$f"; continue; }
        echo "download $REPO/$f"
        uvx --from huggingface_hub hf download "$REPO" "$f" --local-dir "$HF_DIR" \
            || curl -fL --retry 3 -o "$HF_DIR/$f" "https://huggingface.co/$REPO/resolve/main/$f"
    done
fi
[ -n "${SKIP_EXPORT:-}" ] && exit 0
mkdir -p "$(dirname "$OUT")"
# export to a temp name first: a half-written OUT would be skipped by the next run
uv run --no-project --with numpy python3 tools/export_weights.py --model-dir "$HF_DIR" --out "$OUT.part" ${EXPORT_ARGS:-}
mv "$OUT.part" "$OUT"
sha256sum "$OUT" 2>/dev/null || shasum -a 256 "$OUT"
