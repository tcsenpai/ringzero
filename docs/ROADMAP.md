# Roadmap

Generated from the mycelium plan (`myc list --all`), 2026-11-26.

```text

📋 Tasks (Tree View)
✓ 🔴 M0 scaffold: Limine boot + Rust hello [E1]
│   └── ✓ 🔴 M1 engine in ring 0 [E1]
│       └── ✓ 🟠 M2 console and shell with @ agent prefix [E1]
│           └── ✓ 🟠 M3 agent loop and system tools [E1]
│               ├── ✓ 🟠 M4 storage and filesystem tools [E1]
│               │   ├── ○ 🟢 M6 real PC [E1]
│               │   └── ✓ 🟠 M4.5 agentd: deterministic context and memory subsystem [E1]
│               └── ○ 🟢 M5 code execution tool [E1]
○ 🟠 Compaction recall [E1]
✓ 🟠 State snapshots: design (killer feature) [E1]
○ 🟠 Batched prefill in the engine [E1]
✓ 🟠 M2.5 shell UX + 32k FP16 context + native context management [E1]
│   ├── ✓ 🟠 Snapshots v1: undo/redo, retry, auto-checkpoints, branches [E1]
│   │   └── ✓ 🟠 Embedder in kernel: Qwen3-Embedding-0.6B next to Qwen3.5-2B [E1]
│   │       └── T12: M4.5 agentd: deterministic context and memory subsystem (circular)
│   └── ✓ 🟢 Rename aios to ringzero [E1]
✓ 🟢 Spike: embedding model in kernel (Qwen3-Embedding-0.6B) [E1]
│   └── T16: Embedder in kernel: Qwen3-Embedding-0.6B next to Qwen3.5-2B (circular)
✓ 🟢 Spike: 2B tool-calling reliability on Linux [E1]
│   └── T4: M3 agent loop and system tools (circular)
○ 🔵 Console: VNC-friendly scrolling [E1]
○ 🔵 Q4 weights [E1]

ℹ️ 19 task(s) shown
```
