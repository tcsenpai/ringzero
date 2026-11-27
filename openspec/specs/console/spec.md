# console Specification

## Purpose

What the two consoles (COM1 serial and the Limine framebuffer) show: the boot banner, how the framebuffer draws ANSI SGR escapes, the palette, and the role colors of the shell with the color setting. kernel/src/console.rs writes every byte to both.

## Requirements

### Requirement: Boot banner
At boot the kernel SHALL print a banner of 14 lines and at most 70 columns on both consoles: the R0 die next to the name as ASCII art, a `version` line (kernel version from kernel/Cargo.toml, edition), a `models` line (agent and embedder names from the edition, module sizes in MiB), a `machine` line (cores, RAM, firmware, framebuffer size) and a line saying the boot log follows.
Source: kernel/src/main.rs banner, VERSION, EDITION, kmain; kernel/Cargo.toml

#### Scenario: 2B edition
- **WHEN** the default 2B ISO boots
- **THEN** the banner shows `version  0.1.0-beta   ringzero Qwen35@2b-Qwen3@0.6b (QweQwe)` and a models line `agent Qwen35@2b (... MiB), embedder Qwen3@0.6b (... MiB)`

### Requirement: Banner colors and log line
The banner SHALL color the pins and package gold, the die red, `R0` and the name bold red and the labels gray, only through the color setting, so it is plain text when color is off. The kernel log SHALL get one plain line with version, edition, cores, RAM and firmware.
Source: kernel/src/main.rs banner; kernel/src/console.rs sgr, COLOR

#### Scenario: Kernel log
- **WHEN** the user types `!log 300` after boot
- **THEN** a plain line `ringzero 0.1.0-beta ringzero Qwen35@2b-Qwen3@0.6b (QweQwe), 8 cores, ... MiB RAM, ...` is in it

### Requirement: Framebuffer reads ANSI SGR
The framebuffer console SHALL apply ANSI SGR sequences (`ESC [ ... m`): 0 reset, 1 bold (Noto Sans Mono Bold glyphs), 2 dim (5/8 of the color), 22 normal intensity, 30-37 and 90-97 foreground, 39 default foreground, 38;5;N (xterm 256 colors) and 38;2;R;G;B. Other SGR parameters and every other escape sequence SHALL be skipped without drawing. Serial output SHALL carry the same bytes unchanged.
Source: kernel/src/console.rs Console, sgr, fb_char, xterm

#### Scenario: A WASI program's colors
- **WHEN** the user runs `!run jq -C . /demo/data.json` (jq prints colored JSON)
- **THEN** the framebuffer shows the keys and strings in color with no escape characters, and the serial terminal gets jq's escape bytes as they are

#### Scenario: Unknown escape
- **WHEN** a program prints `ESC [ ? 25 l` (hide cursor)
- **THEN** nothing is drawn and the next text is drawn normally

### Requirement: Framebuffer palette
The framebuffer SHALL map the ANSI colors to the logo palette where it has one (31 = #C6140A, 33 = #C9A45C, 90 = #6B6B6B, 91 = #E8473C), draw 30 as dark gray #4A4A4A so it stays readable on black, and draw default text off-white #ECEAE6.
Source: kernel/src/console.rs PAL, FG

#### Scenario: Prompt color
- **WHEN** the prompt is printed with `ESC [ 1 ; 31 m`
- **THEN** the framebuffer draws it in #C6140A with bold glyphs

### Requirement: Role colors
With color on, the shell SHALL print the prompt bold red (`ESC[1;31m`), `[tool]` lines yellow (`ESC[33m`), `[memory]`, `[grammar]`, `[confidence]` and statistics gray (`ESC[90m`), y/N questions and errors bright red (`ESC[91m`), each closed by `ESC[0m`; user input and agent answers SHALL keep the default color.
Source: kernel/src/console.rs sgr; kernel/src/shell.rs show_prompt, confirm, err; kernel/src/agent.rs ask, conf_line; kernel/src/gram.rs finish; kernel/src/llm.rs print_speed

#### Scenario: A tool turn
- **WHEN** the agent calls a tool
- **THEN** the serial bytes of the call line start with `ESC[33m[tool] ` and the statistics line with `ESC[90m[prefill]`

#### Scenario: Refused setting
- **WHEN** the user types `!set think maybe`
- **THEN** the error line is printed in bright red

### Requirement: Color off is plain text
With color off, the shell's own lines (prompt, tool, memory, statistics, questions, errors, the banner) SHALL contain no escape sequences; the escapes that a WASI program prints itself SHALL still reach both consoles.
Source: kernel/src/console.rs sgr, COLOR; kernel/src/wasm.rs out

#### Scenario: jq with color off
- **WHEN** color is off and the user runs `!run jq -C . /demo/data.json`
- **THEN** the prompt has no escape codes and jq's own colors still show
