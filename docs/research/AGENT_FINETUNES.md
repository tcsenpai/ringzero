# Agent and tool-calling fine-tunes up to 4B (T64 survey)

Date: 2026-11-28. Question: which open fine-tunes made for agents, tool calling or function calling, at most 4B
parameters, could replace the stock Qwen3.5-4B q4 working model? This list is wider than
[MODEL_SURVEY.md](MODEL_SURVEY.md) (which covers all small tool models) and skips the two models already tested in
T59 (NeoHorse-1-4B, QwenPaw-Flash-4B, see [MODEL_EVAL.md](../MODEL_EVAL.md)). The test results of the picks are in
the T64 section of MODEL_EVAL.md.

How it was searched: the Hugging Face API (`filter=base_model:finetune:` for Qwen/Qwen3.5-4B, Qwen/Qwen3.5-2B,
Qwen/Qwen3.5-0.8B, unsloth/Qwen3.5-4B and -2B, about 400 repos matched agent/tool keywords; keyword search for agent,
tool, function calling, xLAM, ToolACE, Hammer, BFCL, tau; Qwen3-4B and Qwen3-1.7B fine-tunes), the BFCL v4
leaderboard CSV, arXiv and HN. Reddit refused automated access (HTTP 403), so it is not covered.

Sources: for every row the claims come from three files fetched on 2026-11-28: the model card
(`huggingface.co/REPO`, README.md), `config.json`, and the API record (`huggingface.co/api/models/REPO`: date,
license, file list). Extra sources are named in the row. All benchmark numbers are the authors' own runs (self
reported) unless stated, with different harnesses and BFCL versions, so they are not comparable across rows: the
base Qwen3.5-4B scores 30.35, 50.3 or 70.95 on "BFCL" depending on the card.

"Runs as is" means: our engine runs Qwen3.5 (Gated DeltaNet plus gated attention, tied embeddings, vocab 248320)
with the tool grammar of the Qwen3.5 XML format (`<tool_call><function=..><parameter=..>`), and Qwen3 dense attention
kernels (the embedder path). A Qwen3 dense model would need a Hermes JSON grammar and agent prompt; other families
need new ops (see MODEL_SURVEY.md section 4). Every Qwen3.5 row below was checked in its config.json: 32 layers
(24 for 2B), hidden 2560 (2048), layer pattern 3 linear + 1 full attention, vocab 248320, tied embeddings.

## Qwen3.5 base (runs as is)

| Model | Base | Params | License | Date | Training claim | Tool / agent benchmarks (self reported) | Fit for ringzero |
|---|---|---|---|---|---|---|---|
| [InternScience/Agents-A1-4B](https://huggingface.co/InternScience/Agents-A1-4B) | Qwen3.5-4B, with vision | 4B | Apache-2.0 | 2026-07-13 | full-domain agentic SFT, domain teachers, multi-teacher on-policy distillation (search, engineering, research, instruction following, tool calling); report arXiv 2606.30616 | vs Qwen3.5-4B in one table: tau2 78.2 vs 79.9, VitaBench 40.3 vs 22.0, IFEval 94.8 vs 89.8, GAIA 95.1 vs 58.3, BrowseComp 66.8 vs 47.2 | general agent from a lab, same tool format; chat template only adds a default system prompt when none is given |
| [birgermoell/Qwen3.5-4B-EU-Tool](https://huggingface.co/birgermoell/Qwen3.5-4B-EU-Tool) | Qwen3.5-4B, text only | 4B | Apache-2.0 | 2026-07-04 | SFT on ~590k examples: EU multilingual instructions (Italian listed), reasoning traces, ~55k function-calling examples (Glaive-v2, ToolACE, Hermes) converted to the Qwen3.5 format; no-think by default; OpenEuroLLM | own BFCL-style set, 300 prompts, base vs this: args_exact 0.873 vs 0.96, relevance 0.807 vs 0.84, name_acc 0.993 vs 0.98 | the only card aimed at function calling plus European languages in non-thinking mode; its usage example shows a JSON call inside `<tool_call>`, a sign of the Hermes data |
| [Minbyul/AgentMercury-Qwen3.5-4B](https://huggingface.co/Minbyul/AgentMercury-Qwen3.5-4B) | Qwen3.5-4B, with vision | 4B | Apache-2.0 | 2026-08-14 | on-policy GRPO, 200 steps, ~2.3k MCP tool environments, reward from final environment state | base vs this: BFCL 30.35 vs 31.93, tau2 0.448 vs 0.457, tau3 0.706 vs 0.747 (card lists only rows that improved) | general tool-use RL; small, honest gains; short RL means weights close to the base |
| [reasonwang/SkillGym-Qwen3.5-4B](https://huggingface.co/reasonwang/SkillGym-Qwen3.5-4B) | Qwen3.5-4B, with vision | 4B | Apache-2.0 | 2026-09-22 | SFT on SkillGym trajectories: read a skill folder, act through tool calls or shell commands in a sandbox; arXiv 2609.37539 | base vs this: SkillGym 33.8 vs 47.0, SkillEval 62.8 vs 70.8, SkillsBench 10.1 vs 14.3, Skill-Use-Bench 8.8 vs 48.7 | file and shell work through tools, close to the OS job |
| [LakoMoor/QClaw-4B](https://huggingface.co/LakoMoor/QClaw-4B) | Qwen3.5-4B, with vision | 4B | Apache-2.0 | 2026-04-24 | "agentic task trajectories, instruction following, code" for OpenClaw | ClawBench 84.8, Ru-Arena 80.7; no base column | chat template has no tools section and no system prompt; no base comparison |
| [smirki/qwen3.5-4b-terminal-v1](https://huggingface.co/smirki/qwen3.5-4b-terminal-v1) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-09-03 | terminal and coding agent tune, native XML tool format | Terminal-Bench 2.1 (thinking on, 76 tasks): mean reward 0.145 vs 0.211, solved 15 vs 21 | coding terminal agent, thinking mode |
| [FACET-Terminal/FACET-Terminal-Qwen3.5-4B](https://huggingface.co/FACET-Terminal/FACET-Terminal-Qwen3.5-4B) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-08-19 | SFT on 1.2K terminal trajectories from 6,078 skill-derived tasks; arXiv 2608.18580 | Terminal-Bench 2.1: 17.60 vs 24.72 | terminal coding agent |
| [TMaxxx/tmax-4b-v1.1](https://huggingface.co/TMaxxx/tmax-4b-v1.1) | Qwen3.5-4B, text only | 4B | Apache-2.0 | 2026-09-20 | DPPO RL on TMax-13K terminal tasks; arXiv 2606.23321 | Terminal-Bench 2.1 24.62 +- 1.82 (no base in the same row) | terminal coding agent; template differs from stock in one reasoning line |
| [apodex/Apodex-1.0-4B-SFT](https://huggingface.co/apodex/Apodex-1.0-4B-SFT) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-06-07 | SFT, agentic DPO, RL on long rollouts; ReAct deep-research agent with native function calling | deep-research benchmarks, no tool-calling set | web research, needs search tools we do not have |
| [BAAI/AREX-Turbo](https://huggingface.co/BAAI/AREX-Turbo) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-07-23 | recursive self-improving deep-research agent | xbench-2510 and research sets | web research |
| [antibyte/AuraGo-Qwen3.5-4B](https://huggingface.co/antibyte/AuraGo-Qwen3.5-4B) | Qwen3.5-4B, text only | 4B | Apache-2.0 | 2026-07-28 | LoRA for native tool calls in the AuraGo framework | own: schema valid 95.2%, exact case 97.3%; missed its own 97% gate | framework-specific |
| [agentscope-ai/QwenPaw-Flash-2B](https://huggingface.co/agentscope-ai/QwenPaw-Flash-2B) | Qwen3.5-2B, with vision | 2B | Apache-2.0 | 2026-03-30 | agent trajectories from QwenPaw: tool calls, CLI, file I/O, memory | numbers only in images | the 4B sibling failed the M7 gate (T59); a 2B is below the 4B baseline on every Qwen table |
| [FineEnvs/Qwen3.5-2B-multiharness-RL](https://huggingface.co/FineEnvs/Qwen3.5-2B-multiharness-RL) | Qwen3.5-2B | 2B | Apache-2.0 | 2026-10-01 | async GRPO through OpenCode, Claude Code, Codex, mini-SWE-agent harnesses | 37.0% pass@1 over four harnesses (data analysis) | data-analysis coding agent |
| [Sarthak213/doosra-qwen3.5-4b-toolcalls-v2](https://huggingface.co/Sarthak213/doosra-qwen3.5-4b-toolcalls-v2) | Qwen3.5-4B | 4B (LoRA + GGUF) | Apache-2.0 | 2026-10-02 | 6,695 conversations with a cricket app's 25 tools | app eval 33/40 vs 29 (base) | one domain; no merged bf16 weights |
| [CodeXomics/CodeXomics-ToolAgent-4B-v1](https://huggingface.co/CodeXomics/CodeXomics-ToolAgent-4B-v1) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-08-10 | tool calls for a genome browser | own registry eval | one domain |
| [McGill-NLP/A3-Qwen3.5-4B](https://huggingface.co/McGill-NLP/A3-Qwen3.5-4B) | Qwen3.5-4B | 4B | not set | 2026-04-06 | web agent SFT on A3-Synth; arXiv 2604.07776 | web agent sets | browser agent |
| [KORMo-VL/KORMo-Web-4B](https://huggingface.co/KORMo-VL/KORMo-Web-4B) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-11-14 | web agent | web sets | browser agent |
| [Hcompany/Holo-3.1-4B](https://huggingface.co/Hcompany/Holo-3.1-4B), [microsoft/Fara1.5-4B](https://huggingface.co/microsoft/Fara1.5-4B) | Qwen3.5-4B | 4B | Apache-2.0 / MIT | 2026-05-29 / 2026-07-17 | computer-use agents that act from screenshots | GUI and web sets | need vision; we run text only |
| [Mindcraft-CE/Andy-4.2-Air](https://huggingface.co/Mindcraft-CE/Andy-4.2-Air) | Qwen3.5-4B | 4B | not set | 2026-03-26 | Minecraft agent | game only | one domain |
| [RecursiveMAS/Deliberation-Toolcaller-Qwen3.5-4B](https://huggingface.co/RecursiveMAS/Deliberation-Toolcaller-Qwen3.5-4B) | Qwen3.5-4B | 4B | MIT | 2026-04-27 | tool-caller role inside a latent multi-agent system; arXiv 2604.25917 | numbers only in images | card: not a standalone model |
| [empero-ai/Qwen3.8-4B-Distill](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-08-15 | 45k reasoning traces from a larger teacher; "native function calling" inherited | gsm8k_cot 0.850 vs 0.785 (lower); no tool set | reasoning distill, not a tool tune |
| [ferry-el-kom/qwen3.8-4b-empero-v24-Hermes-agentic-subagent-toolcalls](https://huggingface.co/ferry-el-kom/qwen3.8-4b-empero-v24-Hermes-agentic-subagent-toolcalls) | empero Qwen3.8-4B-Distill (Qwen3.5) | 4B | Apache-2.0 | 2026-11-15 | LoRA on Hermes agent harness traces from cloud teachers | smoke tests only | GGUF only: no bf16 reference for the M7 gate; card warns it runs sudo and reflashes firmware |
| [Lalakai/Hunter-1-4B](https://huggingface.co/Lalakai/Hunter-1-4B), [barryzhang/ornith-harness-4b-v2](https://huggingface.co/barryzhang/ornith-harness-4b-v2), [visnia-ai/Qwen3.5-4B-Browser-Agent-SFT-FP8](https://huggingface.co/visnia-ai/Qwen3.5-4B-Browser-Agent-SFT-FP8) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-07 to 2026-09 | reasoning / tool use / harness / browser tunes | own evals | GGUF or FP8 only, no bf16 reference |
| [khazarai/Qwen3.5-4B-Agentic-Coding](https://huggingface.co/khazarai/Qwen3.5-4B-Agentic-Coding), [mahiatlinux/qwen3.5-4b-agentic-coder-v4](https://huggingface.co/mahiatlinux/qwen3.5-4b-agentic-coder-v4) | Qwen3.5-4B | 4B | Apache-2.0 | 2026-09-15 / 2026-06-30 | agentic coding chain-of-thought SFT (English only on the first card) | none | no evidence |
| [JJarvinen/Qwen3.5-2B-EU-Tool](https://huggingface.co/JJarvinen/Qwen3.5-2B-EU-Tool), [ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM](https://huggingface.co/ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM) | Qwen3.5-2B | 2B | Apache-2.0 | 2026-09-08 / 2026-04-20 | small tool SFTs (English/Finnish; xLAM 60k) | none | 2B, no evidence |
| [Quazim0t0/Qwenx86-4B-compact](https://huggingface.co/Quazim0t0/Qwenx86-4B-compact) | Qwen3.5-4B plus 13 compiled layers | larger | Apache-2.0 | 2026-11-05 | an x86 emulator compiled into extra layers, called through tool calls | demo | **does not run as is**: 61 layers, hidden 3968, vocab 478656 |

Not agents, listed so nobody re-checks them: org2ai/Wald-4B, OmniJev/OneJev-4B, agentmessier/messier-one,
frontier-infra/jebadiah-4b-v2 and sky7350/Mica (decision models that return option probabilities),
AI45Research/AgentDoG1.5 (safety judges of agent traces), s-nlp/ToolHACE 2B (tool hallucination checker),
NikitaLazarev/Qwen3.5-2B-Tools-SoftMoE (steering patch, no training), roskosmos19/Orca-4B-flash (card: the weights
are the unchanged Qwen3.5-4B weights), Kndy666/CoACT (observation compressor), about 40 identical
`*/Qwen3.5-2B-ToolCalling-LoRA` course repos. yantri-tool/Qwen3.5-4B-FC-R2 is gated (card not readable).

## Qwen3 dense base (kernels exist, need a Hermes JSON grammar)

All have the standard Qwen3 config (`Qwen3ForCausalLM`, 36 layers, or 28 for 1.7B, vocab 151936, tied, no sliding
window). None uses the `<function=` XML format, so each needs a JSON tool grammar and a Qwen3 system prompt in
`agent.rs` and `gram.rs` [E: about 1-2 days once for all of them]. The base Qwen3-4B scores below Qwen3.5-4B on
Qwen's own tables (MODEL_SURVEY.md section 2).

| Model | Base | License | Date | Training claim | Benchmarks (self reported) |
|---|---|---|---|---|---|
| [openbmb/AgentCPM-Explore](https://huggingface.co/openbmb/AgentCPM-Explore) | Qwen3-4B-Thinking-2507 | Apache-2.0 | 2026-01-11 | full-parameter agent RL, open training stack | GAIA text 63.9, BrowseComp 25.0, Xbench-DS 70.0 |
| [Team-ACE/EnvACE-Qwen3-1.7B](https://huggingface.co/Team-ACE/EnvACE-Qwen3-1.7B) | Qwen3-1.7B | Apache-2.0 | 2026-06-12 | role-wise GRPO, the model plays agent and environment | card names BFCL-v4, tau2, VitaBench without numbers (arXiv 2608.06197 not read) |
| [teng-pan/Agent-R2-4B](https://huggingface.co/teng-pan/Agent-R2-4B) | Qwen3-4B-Thinking-2507 | Apache-2.0 | 2026-11-04 | SFT then rubric RL on office and file workflows | GDPval 51.0 vs 32.4, OfficeQA 14.8 vs 2.2 |
| [LocoreMind/LocoOperator-4B](https://huggingface.co/LocoreMind/LocoOperator-4B) | Qwen3-4B-Instruct-2507 | MIT | 2026-02-23 | 170k samples distilled from Qwen3-Coder-Next, codebase exploration | own: first tool match 65.6% |
| [bue0912/ToolOmni-Qwen3-4B](https://huggingface.co/bue0912/ToolOmni-Qwen3-4B) | Qwen3-4B | Apache-2.0 | 2026-04-15 | proactive tool retrieval and grounded execution (ACL 2026) | none on the card |
| [katanemo/Plano-Orchestrator-4B](https://huggingface.co/katanemo/Plano-Orchestrator-4B) | Qwen3-4B-Instruct-2507 | katanemo license | 2025-12-04 | routing between agents | numbers only in images |
| [jupyter-agent/jupyter-agent-qwen3-4b-instruct](https://huggingface.co/jupyter-agent/jupyter-agent-qwen3-4b-instruct) | Qwen3-4B-Instruct-2507 | Apache-2.0 | 2025-09-08 | notebook agent SFT | DABStep easy 70.8 vs 44.0 |
| [Menlo/Jan-nano](https://huggingface.co/Menlo/Jan-nano), [janhq/Jan-v1-4B](https://huggingface.co/janhq/Jan-v1-4B) | Qwen3-4B | Apache-2.0 | 2025-06 / 2025-08 | MCP search agents | SimpleQA 91.1 (Jan-v1); rest in images |

## Other architectures (new ops needed)

| Model | Arch | Missing for our engine | Benchmarks (self reported) |
|---|---|---|---|
| [Salesforce/xLAM-2-3b-fc-r](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r), [MadeAgents/Hammer2.1-3b](https://huggingface.co/MadeAgents/Hammer2.1-3b), [katanemo/Arch-Agent-3B](https://huggingface.co/katanemo/Arch-Agent-3B) | Qwen2.5 | QKV bias, JSON grammar; xLAM is CC-BY-NC-4.0 | BFCL v4 leaderboard: 41.22, 29.71, 35.36 |
| [Nanbeige/ToolMind-Web-3B](https://huggingface.co/Nanbeige/ToolMind-Web-3B) | Llama | `LlamaTokenizer` (SentencePiece class) [U on byte fallback] | GAIA 0.670, Xbench-DS 0.751 |
| [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) | Mamba-2 hybrid | SSM scan | BFCL v3 61.1, tau2 Retail 34.8 |
| [LiquidAI/LFM2-1.2B-Tool](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool) | LFM2 conv + attention | short conv layers | numbers only in images |
| [XHToken/Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) | own `Spark2_5ForCausalLM` | GELU, sliding window | BFCL v4 65.1, tau2 75.1 |
| [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) | Gemma 4 E4B | per-layer embeddings, sliding window, byte fallback | BFCL v3 68.51 vs Qwen3.5-4B 70.95 in the same table |

BFCL v4 leaderboard (gorilla.cs.berkeley.edu `data_overall.csv`, page "Last Updated 2026-04-12"): still no
Qwen3.5 model; the best entries at or below 4B are Nanbeige4-3B-Thinking-2511 51.40, xLAM-2-3b-fc-r 41.22,
Qwen3-4B-Instruct-2507 (FC) 35.68.

## Ranking and the picks for T64

Rank by: runs as is (Qwen3.5, bf16 weights available, so the M7 gate can be run), trained for general tool use
rather than one app or web search, evidence against the base model in the same table, and closeness to our job
(file and shell tools, memory, non-thinking, Italian).

1. **Agents-A1-4B**: the strongest lab release; broad agent training including tool calling; large gains on
   agent and instruction sets; tau2 is 1.7 points below the base, so it is not a sure win for us.
2. **Qwen3.5-4B-EU-Tool**: the only one trained for function calling in non-thinking mode with Italian in the
   mix; small self-run eval. Risk: the tool data looks like Hermes JSON, while our grammar forces the XML format.
3. **SkillGym-Qwen3.5-4B**: tool calls and shell commands on files in a sandbox, clear gains on four sets.
4. **AgentMercury-Qwen3.5-4B**: general MCP tool RL, small gains on BFCL and tau; 200 RL steps keep the weights
   close to the base, which should help at Q4 (T59 failed on fine-tunes that moved further) [E].

Not picked: the terminal tunes (smirki, FACET, TMax) are coding agents measured in thinking mode on Terminal-Bench;
QClaw has no base comparison and a template without tools; QwenPaw-Flash-2B is a 2B and its 4B sibling failed the
gate; the Qwen3 dense rows need a new grammar and start from a weaker base; the rest are one domain, web or GUI
agents, or have no bf16 weights.
