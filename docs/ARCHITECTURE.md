# AIOS architecture (as of M2.5)

## Diagram

```mermaid
flowchart TB
    subgraph FW["Firmware + boot"]
        BIOS["SeaBIOS / OVMF"] --> LIM["Limine v11<br/>paging, HHDM, MP, framebuffer,<br/>kernel ELF + model.asmbin module"]
    end
    LIM --> KMAIN

    subgraph K["Rust no_std kernel (ring 0, soft-float)"]
        KMAIN["main.rs kmain<br/>boot sequence"]
        IDT["idt.rs<br/>exceptions, wake IPI, idle timer"]
        MEM["mem.rs<br/>bump arena, lower-half identity map"]
        SMP["smp.rs<br/>TSC, x2APIC, AP release, parking"]
        CON["console.rs<br/>COM1 + framebuffer glyphs"]
        KBD["kbd.rs<br/>PS/2 set 1, layouts us / it, AltGr, Ctrl"]
        LOG["log.rs<br/>ring of 256 lines, level + timestamp"]
        SYS["sys.rs<br/>meminfo, cpuinfo, cores, uptime,<br/>poweroff, reboot"]
        SH["shell.rs<br/>text -> agent, !machine, /session commands,<br/>Tab completion, suggestions, history, Ctrl-C"]
        AG["agent.rs<br/>system prompt + tool registry, parser, validator,<br/>tool loop, 8 KiB result cap"]
        CM["agent.rs: context manager<br/>fill tracking, compaction (summary + rebuild),<br/>last 2 exchanges, raw transcript ring"]
        LLM["llm.rs<br/>golden self-test, bench, /fill, /gen, tokenizer split,<br/>feed / generate, prefix snapshot"]
        ENG["engine.rs<br/>FFI, output capture"]
    end

    subgraph A["NASM engine (ring 0, AVX2/FMA/F16C)"]
        TR["transformer.asm<br/>Qwen3.5: 6 full-attn + 18 DeltaNet layers"]
        KV["KV cache (6 full-attn layers)<br/>FP16 via F16C, 32768 tokens, 384 MiB<br/>(AIOS_KV=f32: FP32, 768 MiB)"]
        MM["matmul.asm Q8_0"]
        MA["math.asm"]
        TOK["tokenizer.asm BPE"]
        WK["smp.asm<br/>tp_* dispatch + worker loop on APs 1..7"]
        CPU["cpu.asm<br/>enable_avx_fpu, float shims"]
    end

    subgraph D["Devices (QEMU q35)"]
        COM1["COM1 16550<br/>TCP 4555 + log file"]
        PS2["8042 PS/2<br/>(VNC keys)"]
        GOP["framebuffer<br/>(VNC :1)"]
        LAPIC["x2APIC / TSC-deadline"]
        ACPI["ICH9 PM (S5), 8042 reset"]
    end

    KMAIN --> IDT & MEM & SMP & CON & LLM & AG & SH
    SH -- "!cmd" --> SYS
    SH -- "plain text" --> AG
    SH -- "/ctx /compact /history" --> CM
    AG -- "fill >= 75%" --> CM
    CM -- "summary turn, snapshot_restore,<br/>re-prefill" --> LLM
    AG -- "tool calls" --> SYS
    AG -- "read_log" --> LOG
    AG -- "set_setting" --> SH
    AG -- "y/N confirm" --> SH
    AG --> LLM --> ENG --> TR
    TR --> MM & MA & WK
    TR -- "vcvtps2ph / vcvtph2ps" --> KV
    LLM --> TOK
    TOK -- "sys_write_all" --> ENG --> CON
    TR -- "sys_mmap_anon" --> MEM
    SH --> KBD --> PS2
    SH --> CON --> COM1 & GOP
    SMP --> LAPIC
    SYS --> ACPI
    LOG -.->|klog! prints| CON
```

Data flow of `how much free memory?` (plain text goes to the agent):

```mermaid
sequenceDiagram
    participant U as User (serial / VNC)
    participant S as shell.rs
    participant A as agent.rs
    participant L as llm.rs + engine
    participant T as sys.rs (tool)
    U->>S: "how much free memory?" (Ctrl-C polled between tokens)
    S->>A: ask(q)
    A->>A: fill >= 75% of 32768? compact first (see below)
    A->>L: fresh conversation? snapshot_restore (KV rows 0..743 + DeltaNet state, ~1.5 ms)
    A->>L: encode user turn, feed (prefill), generate (stream to console)
    L-->>A: stop at </tool_call>, captured text (markup hidden on screen)
    A->>A: lenient parse + validate against the schema
    A->>T: meminfo()
    T-->>A: "RAM total 4095 MiB, free 2057 MiB, ..."
    A->>L: <tool_response> turn, feed, generate
    L-->>U: "La memoria libera è di 2057 MiB."
    Note over A: up to 4 tool rounds per user turn, poweroff/reboot ask y/N at the shell first
```

Compaction (auto at 75% fill, or `/compact`):

```mermaid
sequenceDiagram
    participant A as agent.rs
    participant L as llm.rs + engine
    A->>L: internal user turn "summarize facts, preferences, open tasks, tool results (<= 120 words)"
    L-->>A: summary (at most 300 tokens, streamed, captured)
    A->>L: snapshot_restore (boot prefix: tools + system prompt, without the closing <|im_end|>)
    A->>L: feed "Summary of the earlier part of this session ... trust them: SUMMARY<|im_end|>" (inside the system turn) + last 2 exchanges verbatim (<= 3000 tok)
    Note over A: DeltaNet state cannot be cropped, so the context is rebuilt from the snapshot.<br/>The raw transcript (64 KiB ring) is kept for /history. Every compaction is logged (fill before/after, summary tokens, ms).
```

Legend:

- **Boot**: Limine loads the kernel and the 1.9 GiB model as a module. `kmain` brings up the IDT, memory,
  SMP (APs enter the asm worker loop), checks the engine ABI, runs the golden self-test and the bench, then
  `agent::init` prefills the system prompt + tool list once (743 tokens, about 15 s) and snapshots the
  engine state. Then `shell::run`.
- **Shell**: plain text goes to the agent; `!cmd` runs a machine command (mem, cpu, cores, log, uptime,
  kbd, set, park, stats, poweroff, reboot), `/cmd` a session command (help, clear, ctx, compact, history,
  think, temp, bench, fill, gen; help and stats answer to both). Tab completes command names and first
  arguments, a bare `/` or `!` (pause or Enter) lists the commands, up/down recall the last 16 lines. Input from COM1 and PS/2 is polled into
  a type-ahead ring; 0x03 / Ctrl+C sets a flag that `llm::feed` and `llm::generate` check between tokens.
- **Agent**: Qwen3.5 native tool format (JSON tool lines in the system turn, `<tool_call><function=..>`
  calls, `<tool_response>` results in a user turn). The tokenizer splits special tokens and applies the
  Qwen pre-tokenizer regex before the engine BPE, so the prefix ids equal the HF tokenizer's
  (`tools/hf_prefix.py`, FNV hash printed at boot).
- **Prefix snapshot**: KV cache rows `0..P` of the 6 full-attention layers (head-major layout, FP16 or
  FP32 elements as the engine reports in `kv_elt_bytes`) plus the DeltaNet recurrent and conv state
  (27 MiB with FP16 KV). `/clear` marks the conversation fresh; the next turn restores it instead of
  prefilling again. Multi-turn keeps the live state.
- **KV cache**: 32768 tokens. FP16 (default build) stores K/V rows with `vcvtps2ph` and reads them with
  `vcvtph2ps` inside `gqa_attn_range` (12 KiB/token, 384 MiB); `AIOS_KV=f32 make iso` builds the FP32
  cache (24 KiB/token, 768 MiB). `model_init` takes the 32768 override although the .asmbin header says 4096.
- **Context manager**: fill is tracked in tokens (`/ctx`, stats line). Past `compact` percent (75 by
  default, `!set compact N`) the next turn first compacts (diagram above), if more than 2 exchanges came
  after the last compaction (no compaction loop at low thresholds). The boot snapshot stops 2 tokens short
  of the system turn's `<|im_end|>\n`, so the summary lands in the system prompt (the model distrusted it
  as a user turn); a fresh conversation feeds those 2 tokens. Tool results are capped at
  8 KiB with a visible truncation marker.
- **Log**: `klog!` prints and records `[sec.ms] L text` (I info, W warn, E error, A agent) in a 256-line
  ring, read by `log` and by the `read_log` tool.
- **Keyboard**: scancode set 1, layouts as data (`us`, `it` default), Shift, Ctrl, AltGr (E0 38, and left
  Alt for macOS VNC clients). `kbd` builtin or `set_setting keyboard`.

## Boot flow

1. Firmware (SeaBIOS or UEFI/OVMF) starts Limine from the ISO (El Torito: `limine-bios-cd.bin`
   for BIOS, `limine-uefi-cd.bin` / `BOOTX64.EFI` for UEFI).
2. Limine reads `limine.conf`, loads `/boot/kernel` (ELF, linked at `0xffffffff80000000`) and
   `/boot/model.asmbin` as a module, sets a GOP/VBE framebuffer, builds 4-level paging with the
   HHDM (base revision 3), starts every AP and parks it, then jumps to `kmain` on the BSP in
   long mode with a Limine stack. The IDT is undefined at this point.
3. `kmain` (kernel/src/main.rs): console on COM1 + framebuffer, IDT, CPUID check, `enable_avx_fpu`,
   TSC calibration (PIT ch2), arena + model identity maps, AP release, `model_init`, golden
   self-test, bench, agent prefix prefill + snapshot (kernel/src/agent.rs), then the shell
   (kernel/src/shell.rs).

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
- The engine treats a negative `sys_mmap_anon` result as an error (transformer.asm:529), and HHDM
  addresses are negative. So the arena (largest usable region, bump allocator, kernel/src/mem.rs) and
  the model are **identity-mapped in the lower half with 2 MiB pages**, added to Limine's page tables
  before the APs start (they share CR3). `model_init` gets the model's physical (= identity) address.
- No heap and no general frame allocator yet.

## Where the engine sits

```
Rust kernel (ring 0, soft-float)          NASM engine (ring 0, AVX2/FMA/F16C)
  kmain ---- calls ---------------------->  silu_mul_avx2, argmax_f32 (M0 self-test)
                                             model_init, transformer_forward, tok_* (M1)
  sys_mmap_anon / sys_munmap   <-- imports -- transformer.asm   (mem.rs)
  sys_write_all / str_len      <-- imports -- tokenizer.asm     (engine.rs)
  tp_matmul_q8_0 / tp_gqa_attn /
  tp_deltanet                  <-- imports -- transformer.asm   (engine/smp.asm)
```

## SMP

Limine MP (x2APIC requested) starts every AP. `smp::start` writes `tid` into each AP's
`extra_argument` and its `goto_address` = `ap_limine_entry` (engine/smp.asm): own 64 KiB stack,
`ap_init` (lidt, LAPIC enable), `enable_avx_fpu`, then the asm OS worker loop. The BSP is tid 0 and
runs its own slice inside `tp_*`, so `transformer_forward` is called from the BSP only.

Idle workers spin for `tp_park_tsc` cycles, then, if `tp_park_enable`, count themselves in
`tp_n_parked` and `hlt`. A dispatcher that sees `tp_n_parked != 0` sends one broadcast IPI (vector
0xF0, EOI-only handler). Default policy: parking disabled during a generation, enabled between
requests (parking inside a generation costs 27-46% decode, see M1_RESULT.md). The BSP waits for input
in `hlt` with a 1 kHz TSC-deadline timer (vector 0xF1).

Interrupts stay off (IF=0) everywhere except inside those `sti; hlt` windows, so the engine is never
interrupted and the soft-float Rust handlers never need to save vector state.

The Rust target is soft-float, so Rust never touches xmm/ymm and the engine owns the vector
state. Consequence: engine functions that take float args in xmm registers need a small asm shim
before Rust can call them. `enable_avx_fpu` must run on every core that executes engine code.

## Still asm vs Rust

| Part | Language |
|---|---|
| Inference engine (math, matmul, tokenizer, transformer), vendored; AIOS changes marked in transformer.asm: FP16 KV store/read, max_seq override | NASM |
| `enable_avx_fpu` (engine/cpu.asm, from the old boot.asm) | NASM |
| Boot, paging, AP startup | Limine |
| `sample_top_p_bits` float shim (engine/cpu.asm) | NASM |
| SMP dispatch (`tp_*`), worker loop, AP entry, parking (engine/smp.asm, adapted from the asm OS) | NASM |
| Console, IDT, CPU checks, TSC calibration, memory maps, AP release, generation loop | Rust |
| Shell (! and / commands, completion, history), keyboard layouts, kernel log, agent (prompt, tools, parser, loop), context compaction, transcript, tokenizer pre-split, prefix snapshot | Rust |
