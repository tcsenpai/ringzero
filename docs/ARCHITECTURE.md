# AIOS architecture (as of M0)

## Boot flow

1. Firmware (SeaBIOS or UEFI/OVMF) starts Limine from the ISO (El Torito: `limine-bios-cd.bin`
   for BIOS, `limine-uefi-cd.bin` / `BOOTX64.EFI` for UEFI).
2. Limine reads `limine.conf`, loads `/boot/kernel` (ELF, linked at `0xffffffff80000000`) and
   `/boot/model.asmbin` as a module, sets a GOP/VBE framebuffer, builds 4-level paging with the
   HHDM (base revision 3), starts every AP and parks it, then jumps to `kmain` on the BSP in
   long mode with a Limine stack. The IDT is undefined at this point.
3. `kmain` (kernel/src/main.rs): console on COM1 + framebuffer, CPUID check, `enable_avx_fpu`,
   memory map / MP / module report, engine self-test, then `hlt`.

## Memory

- Kernel image: higher half, 3 PT_LOAD segments (RX text, R rodata, RW data+bss). The engine's
  `.text/.rodata/.bss` (incl. `g_ctx`) are part of these segments. Engine objects are
  PC32-relative only, so they link at any address within +-2 GiB of their callers.
- Physical memory is reached through the HHDM (`phys + hhdm_offset`). In base revision 3 the
  HHDM covers usable, bootloader, kernel/module and framebuffer regions only; MMIO
  (LAPIC, IOAPIC, PCI BARs) must be mapped by us.
- The model is a Limine module: physically contiguous, already in RAM, mapped in the HHDM.
  `model_init` (M1) gets its HHDM virtual address. With `-m 4G` on q35 the model lands below
  4 GiB and 2 GiB of usable RAM remain above 4 GiB (`0x100000000`), which is where the
  engine's arena (`sys_mmap_anon`) will come from.
- No heap, no frame allocator and no page-table changes yet.

## Where the engine sits

```
Rust kernel (ring 0, soft-float)          NASM engine (ring 0, AVX2/FMA/F16C)
  kmain ---- calls ---------------------->  silu_mul_avx2, argmax_f32 (M0 self-test)
                                             model_init, transformer_forward, tok_* (M1)
  sys_mmap_anon / sys_munmap   <-- imports -- transformer.asm
  sys_write_all / str_len      <-- imports -- tokenizer.asm
  tp_matmul_q8_0 / tp_gqa_attn /
  tp_deltanet (stubs, panic)   <-- imports -- transformer.asm  (SMP dispatch, M1)
```

The Rust target is soft-float, so Rust never touches xmm/ymm and the engine owns the vector
state. Consequence: engine functions that take float args in xmm registers need a small asm shim
before Rust can call them. `enable_avx_fpu` must run on every core that executes engine code.

## Still asm vs Rust

| Part | Language |
|---|---|
| Inference engine (math, matmul, tokenizer, transformer), vendored unmodified | NASM |
| `enable_avx_fpu` (engine/cpu.asm, from the old boot.asm) | NASM |
| Boot, paging, AP startup | Limine |
| Console, CPU checks, memory/MP/module reporting, engine imports | Rust |
| SMP dispatch (`tp_*` + worker loop from the old smp.asm) | not yet ported (M1: keep in asm) |
