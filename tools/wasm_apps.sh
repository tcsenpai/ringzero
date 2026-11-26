#!/bin/sh
# M5b: build or fetch the wasm32-wasip1 programs for /bin into $1 (default build/wasm). Needs rustup's
# wasm32-wasip1 target (rustup target add wasm32-wasip1) and curl. Each step is skipped when its .wasm exists.
#   rztest.wasm     tools/wasm/rztest (our WASI checks)
#   coreutils.wasm  uutils coreutils, multicall (`coreutils ls /`), feature feat_wasm, from crates.io
#   rg.wasm         ripgrep, from crates.io
#   qjs.wasm        QuickJS-ng, the official prebuilt qjs-wasi.wasm from its GitHub release
set -eu
OUT=${1:-build/wasm}
COREUTILS=0.13.0
RIPGREP=15.2.0
QJS=v0.18.0
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
T=wasm32-wasip1
export CARGO_PROFILE_RELEASE_STRIP=true CARGO_PROFILE_RELEASE_DEBUG=false  # no debug info: smaller modules

[ -e "$OUT/rztest.wasm" ] || {
    cargo build -q --release --target $T --manifest-path tools/wasm/rztest/Cargo.toml --target-dir "$OUT/target"
    cp "$OUT/target/$T/release/rztest.wasm" "$OUT/"
}
[ -e "$OUT/coreutils.wasm" ] || {
    cargo install -q --locked --target $T --root "$OUT/root" --no-default-features --features feat_wasm coreutils@$COREUTILS
    cp "$OUT/root/bin/coreutils.wasm" "$OUT/"
}
[ -e "$OUT/rg.wasm" ] || {
    cargo install -q --locked --target $T --root "$OUT/root" ripgrep@$RIPGREP
    cp "$OUT/root/bin/rg.wasm" "$OUT/"
}
[ -e "$OUT/qjs.wasm" ] || curl -fsSL -o "$OUT/qjs.wasm" https://github.com/quickjs-ng/quickjs/releases/download/$QJS/qjs-wasi.wasm
ls -l "$OUT"/*.wasm
