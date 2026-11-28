# M0 result (2026-11-26)

All numbers below come from runs on the Fedora build box (Ryzen AI 9 HX 370,
QEMU/KVM `-cpu host -M q35 -m 4G -smp 8`, OVMF from `/usr/share/edk2/ovmf`).
Toolchain: rustc 1.92.0, NASM 3.02, Limine 11.2.1 (`v11.2.1-binary`, commit deba356b).

**Status: M0 PASS on BIOS and UEFI, with the 2 GB Qwen3.5-2B model as the Limine module.**

## Serial output

`make run-bios` (serial on stdio, QEMU killed by `timeout` after the run):

```
==============================================
 AIOS M0  (Rust no_std kernel + NASM LLM engine)
==============================================
boot: Limine 11.2.1, base revision Some(3)
boot: firmware BIOS
fb:   1280x800 32 bpp
cpu:  AMD Ryzen AI 9 HX 370 w/ Radeon 890M
cpu:  AVX=true AVX2=true FMA=true F16C=true XSAVE=true
cpu:  enabled: CR4.OSXSAVE=true XCR0=0x7 (want 0x7)
time: TSC rate unknown, entry TSC=71936937520
mem:  20 entries, usable 2171 MiB, bootloader-reclaimable 1 MiB, kernel+modules 1922 MiB, top 0x180000000
mem:  largest usable run 0x100000000 + 2048 MiB, HHDM offset 0xffff800000000000
smp:  8 CPUs reported by Limine MP, BSP LAPIC id 0
mod:  1 module(s)
mod:  /boot/model.asmbin size 2015746176 B (1922 MiB) at virt 0xffff800007cb9000 phys 0x7cb9000
mod:  magic "ASML" header version 3 dim 2048 layers 24 vocab 248320
engine: 9 entry points linked, e.g. model_init @ 0xffffffff80007b20, transformer_forward @ 0xffffffff8000826a
engine: silu_mul_avx2 n=64 max_rel_err=155e-9 -> PASS
engine: argmax_f32 = 62 (rust 62) -> PASS
M0 PASS
```

`make run-uefi` (OVMF's own escape sequences trimmed; `tools/boottest.py` run, host timestamps):

```
[   1.42s] BdsDxe: starting Boot0002 "UEFI QEMU DVD-ROM QM00005 " from PciRoot(0x0)/Pci(0x1F,0x2)/Sata(0x2,0xFFFF,0x0)
[  62.18s] ==============================================
[  62.18s]  AIOS M0  (Rust no_std kernel + NASM LLM engine)
[  62.18s] ==============================================
[  62.18s] boot: Limine 11.2.1, base revision Some(3)
[  62.18s] boot: firmware UEFI x64
[  62.18s] fb:   1280x800 32 bpp
[  62.18s] cpu:  AMD Ryzen AI 9 HX 370 w/ Radeon 890M
[  62.18s] cpu:  AVX=true AVX2=true FMA=true F16C=true XSAVE=true
[  62.18s] cpu:  enabled: CR4.OSXSAVE=true XCR0=0x7 (want 0x7)
[  62.18s] time: TSC rate unknown, entry TSC=123541886680
[  62.18s] mem:  42 entries, usable 2127 MiB, bootloader-reclaimable 40 MiB, kernel+modules 1922 MiB, top 0x180000000
[  62.18s] mem:  largest usable run 0x100000000 + 2048 MiB, HHDM offset 0xffff800000000000
[  62.18s] smp:  8 CPUs reported by Limine MP, BSP LAPIC id 0
[  62.18s] mod:  1 module(s)
[  62.18s] mod:  /boot/model.asmbin size 2015746176 B (1922 MiB) at virt 0xffff800001fa6000 phys 0x1fa6000
[  62.18s] mod:  magic "ASML" header version 3 dim 2048 layers 24 vocab 248320
[  62.18s] engine: 9 entry points linked, e.g. model_init @ 0xffffffff80007b20, transformer_forward @ 0xffffffff8000826a
[  62.18s] engine: silu_mul_avx2 n=64 max_rel_err=155e-9 -> PASS
[  62.18s] engine: argmax_f32 = 62 (rust 62) -> PASS
[  62.18s] M0 PASS
```

`make run-uefi` itself also printed the same lines up to `M0 PASS` (checked in `build/run-uefi.out`).

## Framebuffer

Screendumps taken through the QEMU monitor after `M0 PASS` show the full text above, white on
black, Noto Sans Mono 16 px, at 1280x800x32 in **both** BIOS and UEFI.
UEFI screenshot: [m0_uefi_screen.png](m0_uefi_screen.png). The old asm OS was blank on UEFI;
this is fixed by Limine's GOP framebuffer.

## Module load times

Host wall clock from QEMU start to the kernel banner on serial (`tools/boottest.py`). For UEFI the
time OVMF hands off to Limine is shown too; the rest is Limine loading kernel + module.

| Firmware | Model | Runs: kernel banner at (s) | OVMF hand-off (s) |
|---|---|---|---|
| BIOS | 2B, 1922 MiB | 25.6, 32.1, 20.2, 33.0, 38.6, 35.0 | n/a |
| BIOS | 0.6B (v2), 613 MiB | 12.5, 12.3, 18.2, 11.8 | n/a |
| UEFI | 2B, 1922 MiB | 44.2, 49.0, 98.4, 62.2 | 1.4 - 2.2 (one run 4.3) |
| UEFI | 0.6B (v2), 613 MiB | 44.5, 49.5, 28.9 | 30.6, 8.5, 3.1 |

Caveat: during every run the box was heavily loaded by other jobs (load average 30-60, two
torch/HF processes at ~13 and ~4 cores, RAM 85-90% used, so the 2 GB ISO was not reliably in page
cache). Treat these as upper bounds with large variance. Still, the pattern held in every run:

- 2 GB as a Limine module **works on both firmwares**. No fallback to 0.6B was needed.
- BIOS loads the 2B module in roughly 20-38 s (~50-90 MiB/s).
- **UEFI is slower than BIOS**, about 45-60 s after hand-off for the 2B (~30-45 MiB/s). Limine on
  UEFI reads the ISO through OVMF's block I/O on the emulated AHCI CD-ROM.

## Sizes

- Kernel ELF (release): 106,472 B. `.text` 36,222, `.rodata` 16,612, `.data` 688, `.bss` 19,200
  (mostly engine `g_ctx` and buffers), `.got` 264.
- ISO with the 2B model: 2,019,825,664 B.

## Deviations from the report (aios_reuse.md)

1. **`limine` crate 0.6.x needs nightly** (`#![feature(ptr_metadata)]`), and 0.6.x lives at
   `robotman2412/limine-boot-rust`, not the repo named in the report. Used **0.5.0**
   (`limine-bootloader/limine-rs`, stable, base revision 3) on stable 1.92 instead. Limine 11.2.1
   accepts revision 3. 0.5.0 has no TSC-frequency request.
2. Limine lives on **Codeberg** now (`codeberg.org/Limine/Limine`); the GitHub mirror stops at 9.6.7.
3. Rust's `x86_64-unknown-none` is soft-float, so **float arguments in xmm cannot be passed from
   Rust** (`rmsnorm_avx2` eps, `sample_top_p` temp/topp). M0 tests `silu_mul_avx2` and
   `argmax_f32` (pointer/int args only). `model_init` and `transformer_forward` take only ints.
4. Linking the engine objects is not enough to prove linkage: with `--gc-sections` lld drops
   unreferenced engine code and ignores its undefined symbols. The kernel keeps a `#[used]` table of
   9 engine entry points so the whole engine stays live and all its imports must resolve
   (they do: `g_ctx`, `gqa_attn_range`, `deltanet_heads_range` are in the final ELF).
5. CPUID gives no TSC rate on this AMD host under KVM (leaves 0x15/0x16 absent or zero), so the
   in-guest time is printed as a raw TSC; load times come from the host side.
6. The GPL change: the project is GPL-2.0-or-later; the vendored engine stays MIT
   (`engine/LICENSE`).

## What is stubbed

`tp_matmul_q8_0`, `tp_gqa_attn`, `tp_deltanet` and `sys_mmap_anon` panic with a message naming M1.
`sys_munmap` is a no-op, `sys_write_all` writes to the console, `str_len` is real.
