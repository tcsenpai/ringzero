#!/bin/sh
# M5b + M8: build or fetch the wasm32-wasip1 programs for /bin into $1 (default build/wasm), and CPython's standard
# library into $1/lib. Needs rustup's wasm32-wasip1 target (rustup target add wasm32-wasip1), curl, unzip, tar and
# sha256sum. Each step is skipped when its output exists. Every download is pinned by version and SHA-256.
#   rztest.wasm     tools/wasm/rztest (our WASI checks)
#   coreutils.wasm  uutils coreutils, multicall (`coreutils ls /`), feature feat_wasm, from crates.io (--locked)
#   rg.wasm         ripgrep, from crates.io
#   qjs.wasm        QuickJS-ng, the official prebuilt qjs-wasi.wasm from its GitHub release
#   jq.wasm         jaq (a jq clone in Rust), from crates.io, without its mimalloc feature (C, not for WASI)
#   kibi.wasm       kibi (a kilo-like editor in Rust, opens /dev/tty), from crates.io
#   sqlite3.wasm    the SQLite shell from the amalgamation, built with wasi-sdk
#   lua.wasm        Lua, built with wasi-sdk; setjmp/longjmp via Binaryen's Asyncify (tools/wasm/c)
#   python3.wasm    CPython, the wasm32-wasip1 release build from brettcannon/cpython-wasi-build (a CPython
#                   core developer's builds, the ones the CPython docs point to), its stdlib in lib/python3.15
set -eu
OUT=${1:-build/wasm}
J=${J:-12} # build threads
COREUTILS=0.13.0
RIPGREP=15.2.0
QJS=v0.18.0
QJS_SHA=2c218412cb8b17e46ee91bea7556ec38f5b9b6bd6312fddd3ceeed14aa71eaaa
JAQ=3.2.0
KIBI=0.3.2 # 0.3.3 needs rustc 1.93
WASI_SDK=34
WASI_SDK_SHA=b761e3a0721dbae9c09a0059e5fdb2bf917d1b4a8a7b430fb3b5aafb0984b2c4
BINARYEN=133
BINARYEN_SHA=2dc9c7813f5375db93d96ead4b78222fcc3e2677bbb832297af4797782a37489
SQLITE=3540100 # 3.54.1
SQLITE_SHA=0a9c8e34352dad46006aa82836d5383bff9dc6bc9ce5a60553c05aa30e6ad1f9
LUA=5.4.8
LUA_SHA=4f18ddae154e793e46eeab727c59ef1c0c0c2b744e7b94219710d76f530629ae
PY=3.15.2
PY_SDK=33
PY_SHA=b8a63559c56760bbdc310dedb74fd69d10bcbafae26b705bea0ad353ad256888
mkdir -p "$OUT/dl"
OUT=$(cd "$OUT" && pwd)
DL=$OUT/dl
C=$(pwd)/tools/wasm/c
T=wasm32-wasip1
export CARGO_PROFILE_RELEASE_STRIP=true CARGO_PROFILE_RELEASE_DEBUG=false  # no debug info: smaller modules

# fetch URL SHA256: into $DL, checked (a mismatch deletes the file and stops)
fetch() {
    f=$DL/$(basename "$1")
    if [ ! -e "$f" ]; then curl -fsSL -o "$f.part" "$1" && mv "$f.part" "$f"; fi
    echo "$2  $f" | sha256sum -c --quiet - || { rm -f "$f"; echo "checksum mismatch: $1"; exit 1; }
}
crate() { # crate NAME VERSION [cargo flags...]: cargo install into $OUT/root
    n=$1 v=$2
    shift 2
    cargo install -q -j "$J" --locked --target $T --root "$OUT/root" "$@" "$n@$v"
}

[ -e "$OUT/rztest.wasm" ] || {
    cargo build -q --release --target $T --manifest-path tools/wasm/rztest/Cargo.toml --target-dir "$OUT/target"
    cp "$OUT/target/$T/release/rztest.wasm" "$OUT/"
}
[ -e "$OUT/coreutils.wasm" ] || { crate coreutils $COREUTILS --no-default-features --features feat_wasm; cp "$OUT/root/bin/coreutils.wasm" "$OUT/"; }
[ -e "$OUT/rg.wasm" ] || { crate ripgrep $RIPGREP; cp "$OUT/root/bin/rg.wasm" "$OUT/"; }
[ -e "$OUT/qjs.wasm" ] || { fetch https://github.com/quickjs-ng/quickjs/releases/download/$QJS/qjs-wasi.wasm $QJS_SHA; cp "$DL/qjs-wasi.wasm" "$OUT/qjs.wasm"; }
[ -e "$OUT/jq.wasm" ] || { crate jaq $JAQ --no-default-features; cp "$OUT/root/bin/jaq.wasm" "$OUT/jq.wasm"; }
[ -e "$OUT/kibi.wasm" ] || { crate kibi $KIBI; cp "$OUT/root/bin/kibi.wasm" "$OUT/"; }

# C programs: wasi-sdk (clang + wasi-libc)
SDK=$DL/wasi-sdk-$WASI_SDK.0-x86_64-linux
cc_init() {
    [ -d "$SDK" ] || { fetch https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-$WASI_SDK/wasi-sdk-$WASI_SDK.0-x86_64-linux.tar.gz $WASI_SDK_SHA
        tar xzf "$SDK.tar.gz" -C "$DL"; }
    CC="$SDK/bin/clang --target=$T --sysroot=$SDK/share/wasi-sysroot -O2 -Wno-deprecated-declarations"
}
[ -e "$OUT/sqlite3.wasm" ] || {
    cc_init
    fetch https://www.sqlite.org/2026/sqlite-amalgamation-$SQLITE.zip $SQLITE_SHA
    rm -rf "$DL/sqlite-amalgamation-$SQLITE" && unzip -q -o "$DL/sqlite-amalgamation-$SQLITE.zip" -d "$DL"
    s=$DL/sqlite-amalgamation-$SQLITE
    # no threads, no extensions (no dlopen); WAL needs shared memory: the shell falls back to the rollback journal
    $CC -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_GETPID \
        -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS "$s/shell.c" "$s/sqlite3.c" "$C/nosys.c" \
        -lwasi-emulated-mman -lwasi-emulated-getpid -lwasi-emulated-signal -lwasi-emulated-process-clocks \
        -Wl,--strip-all -o "$OUT/sqlite3.wasm"
}
[ -e "$OUT/lua.wasm" ] || {
    cc_init
    fetch https://www.lua.org/ftp/lua-$LUA.tar.gz $LUA_SHA
    B=$DL/binaryen-version_$BINARYEN
    [ -d "$B" ] || { fetch https://github.com/WebAssembly/binaryen/releases/download/version_$BINARYEN/binaryen-version_$BINARYEN-x86_64-linux.tar.gz $BINARYEN_SHA
        tar xzf "$B-x86_64-linux.tar.gz" -C "$DL"; }
    rm -rf "$DL/lua-$LUA" && tar xzf "$DL/lua-$LUA.tar.gz" -C "$DL"
    o=$DL/lua-obj && rm -rf "$o" && mkdir -p "$o"
    # tools/wasm/c/setjmp.h shadows wasi-libc's; lua.c's main becomes rz_main, called by rzsjlj.c's main
    F="-I$C -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -DL_tmpnam=32"
    for f in "$DL/lua-$LUA"/src/*.c; do
        n=$(basename "$f" .c)
        case $n in luac) ;; lua) $CC $F -Dmain=rz_main -c "$f" -o "$o/$n.o" ;; *) $CC $F -c "$f" -o "$o/$n.o" ;; esac
    done
    $CC -I"$C" -c "$C/rzsjlj.c" -o "$o/rzsjlj.o" && $CC -c "$C/nosys.c" -o "$o/nosys.o"
    $CC "$o"/*.o -lwasi-emulated-signal -lwasi-emulated-process-clocks -Wl,-z,stack-size=4194304 -o "$o/lua-plain.wasm"
    # every function that can reach setjmp/longjmp gets Asyncify's unwind/rewind code. --asyncify before -O2: an
    # -O2 first may inline rz_setjmp into its caller, and then a longjmp traps (unreachable)
    "$B/bin/wasm-opt" --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
        --asyncify -O2 --strip-debug "$o/lua-plain.wasm" -o "$OUT/lua.wasm"
}
[ -e "$OUT/python3.wasm" ] || {
    fetch https://github.com/brettcannon/cpython-wasi-build/releases/download/v$PY/python-$PY-wasi_sdk-$PY_SDK.zip $PY_SHA
    p=$DL/python-$PY && rm -rf "$p" && mkdir -p "$p" && unzip -q -o "$DL/python-$PY-wasi_sdk-$PY_SDK.zip" -d "$p"
    rm -rf "$OUT/lib" && mkdir -p "$OUT/lib" && cp -r "$p/lib/python3.15" "$OUT/lib/"
    cp "$p/python.wasm" "$OUT/python3.wasm"
}
ls -l "$OUT"/*.wasm
