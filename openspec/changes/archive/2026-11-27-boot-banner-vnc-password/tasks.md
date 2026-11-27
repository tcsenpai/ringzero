## 1. Banner
- [x] 1.1 banner() in main.rs: die + name art, version, edition, models, machine, plain log line
- [x] 1.2 version from Cargo.toml (0.1.0-beta)

## 2. Framebuffer SGR
- [x] 2.1 CSI parameter parser and SGR apply in console.rs (0, 1, 2, 22, 30-37, 39, 90-97, 38;5;N, 38;2;R;G;B)
- [x] 2.2 colored and bold glyphs, logo palette for the ANSI colors

## 3. Names
- [x] 3.1 VNC_PASS ringzero in the Makefile, INFO.md
- [x] 3.2 "AIOS" markers in engine/ comments, INFO.md, docs/ARCHITECTURE.md, LICENSE renamed

## 4. Check
- [x] 4.1 boot on the build box: banner on serial (raw bytes) and on the framebuffer (screendump)
