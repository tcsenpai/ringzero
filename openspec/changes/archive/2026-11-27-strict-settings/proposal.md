## Why
The OpenSpec baseline found settings that silently did the wrong thing on a typo (T52): any value other than
on/1/true turned think, constrain, confidence and gramdraft off, `!set spec 0` and `!set spec 16` turned
speculation off although the help says the longest draft is 1..15, the unknown-setting error listed 8 of the 13
keys, and kernel/build.rs did not declare two of the RINGZERO_* build variables.

## What Changes
- think, constrain, confidence, gramdraft: `on`, `1`, `true` turn the setting on, `off`, `0`, `false` turn it
  off, anything else is refused with `error: <key> must be on or off` and nothing changes.
- spec: on/true/1..15 on (1..15 also set the longest draft), off/false off, any other value (0, 16, a typo) is
  refused with `error: spec must be on, off or the longest draft from 1 to 15`.
- Unknown setting: the error lists every key of the settings table.
- kernel/build.rs declares rerun-if-env-changed for RINGZERO_HEAP_MIB and RINGZERO_WASM_MIB, so every RINGZERO_*
  variable the kernel reads at build time is declared.
- Docs: INFO.md documents every `!set` key and the shortcuts; docs/M48B_RESULT.md gets a note that the prompt
  lookup's minimum match is 3 tokens, not 4.

## Impact
- kernel/src/shell.rs set_setting, kernel/build.rs, INFO.md, docs/M48B_RESULT.md.
- Specs: settings (keys, think, constrain, spec, gramdraft, confidence), build-options (rebuild on the memory
  variables; the OPEN requirement is resolved).
