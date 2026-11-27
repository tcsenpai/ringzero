# Licenses of the files in this repository

The ISO images bundle several works under different licenses.

| Part | License | Source |
|---|---|---|
| ringzero kernel, tools and build scripts | GPL-2.0-or-later | https://github.com/tcsenpai/ringzero |
| llm.asm inference engine (vendored, modified) | MIT | https://github.com/alesha-pro/llm.asm |
| Limine bootloader | BSD-2-Clause | https://github.com/limine-bootloader/limine |
| MicroPython (built into the kernel) | MIT | https://github.com/micropython/micropython |
| wasmi (built into the kernel) | MIT OR Apache-2.0 | https://github.com/wasmi-labs/wasmi |
| Qwen3.5-4B, Qwen3.5-2B weights (converted) | Apache-2.0 | https://huggingface.co/Qwen |
| Qwen3-Embedding-0.6B weights (converted) | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B |

The Qwen weights are redistributed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
They were converted from the original safetensors to ringzero's .asmbin format (Q8_0 and Q4_0 quantization);
no other change was made. The full dependency list with versions is in INFO.md in the ringzero repository.
