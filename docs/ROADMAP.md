# Roadmap

Generated from the mycelium plan (`myc list --all`), 2026-11-27.

```text

📋 Tasks (Tree View)
✓ 🔴 Release fixes from the user's manual test [E1]
│   └── ✓ 🔴 Beta release (today, 2026-11-27) [E1]
│       ├── ○ 🟢 After the beta: tell llm.asm and ProbeLogits authors about ringzero [E1]
│       ├── ○ 🟢 POSIX shell or a compiler on ringzero (post-beta study) [E1]
│       ├── ○ 🟠 Unix compatibility layer (design principle) [E1]
│       └── ○ 🟢 Release cadence: post-beta updates, then weekly [E1]
✓ 🔴 M0 scaffold: Limine boot + Rust hello [E1]
│   └── ✓ 🔴 M1 engine in ring 0 [E1]
│       └── ✓ 🟠 M2 console and shell with @ agent prefix [E1]
│           └── ✓ 🟠 M3 agent loop and system tools [E1]
│               ├── ✓ 🟠 M4 storage and filesystem tools [E1]
│               │   ├── ○ 🟢 M6 real PC [E1]
│               │   └── ✓ 🟠 M4.5 agentd: deterministic context and memory subsystem [E1]
│               │       ├── ✓ 🟠 M4.8 base: batched prefill in the engine [E1]
│               │       │   ├── ○ 🟠 M4.8a Multi-mind batched decode [E1]
│               │       │   │   └── ○ 🔵 Quorum check on risky tool calls via batched forks [E1] [blocked by T25]
│               │       │   ├── ✓ 🟠 M4.8b Speculative decoding by prompt lookup [E1]
│               │       │   │   └── ✓ 🟠 Beta release gate [E1]
│               │       │   │       ├── T37: After the beta: tell llm.asm and ProbeLogits authors about ringzero (circular)
│               │       │   │       ├── ○ 🟢 Decode bandwidth A/B: vCPU count, lm_head bytes [E1]
│               │       │   │       ├── ○ 🟠 MTP draft head: Qwen3.5-4B's own multi-token prediction layer [E1]
│               │       │   │       ├── ○ 🟢 Decision model (user note, post-beta) [E1]
│               │       │   │       ├── ○ 🟢 Engine kernels for other model architectures (e.g. Gemma) [E1]
│               │       │   │       └── T51: Beta release (today, 2026-11-27) (circular)
│               │       │   └── ○ 🟠 M4.8c Precomputed document states [E1]
│               │       ├── ✓ 🟠 M4.7b Hibernate the agent: snapshots to disk [E1]
│               │       ├── ○ 🟠 Kernel event bus to the agent (deterministic perimeter) [E1]
│               │       ├── ✓ 🟠 Built-in OS manual in memory [E1]
│               │       │   ├── T34: Beta release gate (circular)
│               │       │   └── ✓ 🟢 Manual vectors baked into the kernel binary (build flag, default on) [E1]
│               │       │       ├── T34: Beta release gate (circular)
│               │       │       └── ✓ 🟠 Grammar as a speculative draft source [E1]
│               │       │           ├── ✓ 🟢 Uncertainty-aware agent [E1]
│               │       │           │   └── T34: Beta release gate (circular)
│               │       │           └── T34: Beta release gate (circular)
│               │       ├── ○ 🟢 Deterministic replay as audit log [E1]
│               │       └── ✓ 🟢 M4.7a Constrained decoding for tool calls [E1]
│               └── ✓ 🟢 M5 Code execution: MicroPython freestanding [E1]
│                   └── ✓ 🟢 M5b WASI runtime [E1]
│                       └── ✓ 🟢 M8 Demo apps on ringzero (Python + WASM) [E1]
│                           └── T34: Beta release gate (circular)
✓ 🟠 Fixes found by the OpenSpec baseline (before release) [E1]
│   └── T51: Beta release (today, 2026-11-27) (circular)
✓ 🟠 GitHub Pages site from the branding kit [E1]
✓ 🟠 Branding: boot banner overhaul, logo palette, VNC password ringzero [E1]
│   ├── T34: Beta release gate (circular)
│   ├── ✓ 🟠 Shell colors by role (serial + framebuffer), !set color [E1]
│   │   ├── T34: Beta release gate (circular)
│   │   └── T51: Beta release (today, 2026-11-27) (circular)
│   └── T51: Beta release (today, 2026-11-27) (circular)
✓ 🟠 agentd memory slot relevance fix (before beta) [E1]
│   └── T34: Beta release gate (circular)
✓ 🟠 Compaction recall [E1]
✓ 🟠 State snapshots: design (killer feature) [E1]
✓ 🟠 M2.5 shell UX + 32k FP16 context + native context management [E1]
│   ├── ✓ 🟠 Snapshots v1: undo/redo, retry, auto-checkpoints, branches [E1]
│   │   └── ✓ 🟠 Embedder in kernel: Qwen3-Embedding-0.6B next to Qwen3.5-2B [E1]
│   │       └── T12: M4.5 agentd: deterministic context and memory subsystem (circular)
│   └── ✓ 🟢 Rename aios to ringzero [E1]
○ 🟢 Type-ahead prefill: prefill while the user types [E1]
○ 🟢 Interruptible between layers [E1]
○ 🟢 Tools as capabilities in ring 3 [E1]
✓ 🟢 Spike: embedding model in kernel (Qwen3-Embedding-0.6B) [E1]
│   └── T16: Embedder in kernel: Qwen3-Embedding-0.6B next to Qwen3.5-2B (circular)
✓ 🟢 Spike: 2B tool-calling reliability on Linux [E1]
│   └── T4: M3 agent loop and system tools (circular)
○ 🔵 Live logit view on the framebuffer [E1]
○ 🔵 Shareable mind-state files [E1]
○ 🔵 Spike: zero-shot Safe/Dangerous probe on Qwen3.5-4B [E1]
○ 🔵 Native ELF in ring 3 (only if needed) [E1]
○ 🔵 Console: VNC-friendly scrolling [E1]
✓ 🔵 M7 Q4 weights for Qwen3.5-4B [E1]
    └── T34: Beta release gate (circular)

ℹ️ 56 task(s) shown
```
