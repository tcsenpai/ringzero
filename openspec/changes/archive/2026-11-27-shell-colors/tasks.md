## 1. Colors
- [x] 1.1 console::sgr and the role constants (PROMPT, TOOL, DIM, ERR, RESET)
- [x] 1.2 prompt, y/N questions and errors in shell.rs; [tool], [memory], [confidence] in agent.rs; [grammar] in
  gram.rs; statistics and debug bench lines in llm.rs

## 2. Setting
- [x] 2.1 `!set color on|off` in set_setting, SETTINGS and the /help text
- [x] 2.2 hibernation image extra v2 with the color byte, v1 still read

## 3. Tools
- [x] 3.1 SGR stripped in the serial test drivers

## 4. Check
- [x] 4.1 serial raw capture shows the codes, the framebuffer screendump shows the colors, color off is plain,
  jq colors render, decode speed unchanged
