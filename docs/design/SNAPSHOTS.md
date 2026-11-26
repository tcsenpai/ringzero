# Design: engine state snapshots (undo, branches, warm boot)

Status: proposal, 2026-11-26. Nothing here is implemented beyond the existing boot prefix snapshot.
Targets the tree after M2.5 (32k context, FP16 KV, compaction). Start implementing only after M2.5 lands:
`kernel/src/agent.rs` and `kernel/src/llm.rs` are being edited for it.

Tags: **[R file:line]** = read in the code, **[D file:line]** = read in the docs, **[INF]** = inferred or
estimated, not measured.

## 0. Why this matters more here than in llama.cpp

In a pure-attention model, undo is free: set `pos` back, the KV rows past it are ignored. Qwen3.5 is
hybrid: 18 of 24 layers are Gated DeltaNet, whose recurrent state is a running integral of the whole
history and cannot be rewound (agent.rs already says "DeltaNet state cannot be cropped" [R agent.rs:6-7]).
So on this model **a snapshot is the only cheap way to go back**. Without one, going back means
re-prefilling the conversation at ~47 tok/s [D M2M3_RESULT.md:56]: 25 s for 1k tokens, over 10 min at 32k.

With one, going back costs about one millisecond. That is less than decoding one token (~28-34 ms at
29-36 tok/s [D M2M3_RESULT.md:59-60]). That ratio is the whole case for this feature.

## 1. What a snapshot is in this engine

### 1.1 The state, field by field

| Part | Where | Layout | Size |
|---|---|---|---|
| K cache, 6 full-attn layers | `CTX_BUF_K_CACHE` [R transformer.asm:499-509] | head-major `[layer][kv_head][pos][head_dim]`, plane stride `max_seq*head_dim*elt` [R transformer.asm:1000-1004, llm.rs:499-512] | 6 KiB/token (FP16) |
| V cache, same | `CTX_BUF_V_CACHE` | same | 6 KiB/token (FP16) |
| DeltaNet S, 18 layers | `CTX_BUF_DN_STATE` | `[n_lin][Hv][128][128]` F32 [R transformer.asm:512-517] | 18 MiB, fixed |
| DeltaNet conv history | `CTX_BUF_DN_CONV`, adjacent to S | `[n_lin][3][conv_dim]` F32, conv_dim = 2·Hk·Dk + Hv·Dv [R transformer.asm:291-294, 518-521] | 1.27 MiB, fixed |
| Position | `agent::POS` (Rust; the engine gets `pos` as an argument) [R agent.rs:62, llm.rs:418-427] | u64 | 8 B |
| Sampler | `llm::RNG`, `TEMP_BITS`, `THINK` (Rust) [R llm.rs:34-36] | u64, u32, u64 | 24 B |
| Agent session | `FRESH`, `EX` ring (last 2 Q/A for compaction re-prefill), `N_COMPACT` [R agent.rs:62-68, 138-143] | | 16 KiB |
| Logits | `CTX_BUF_LOGITS` | vocab F32 | ~0.95 MiB, **not needed** (see below) |

Model dimensions, derived and cross-checked against two measurements:
- n_full = 6, n_kv_heads = 2, head_dim = 256 (kv_dim 512), n_lin = 18, Hv = Hk = 16, Dk = Dv = 128.
- Check 1: the M3 snapshot (FP32 KV, 743 tokens) is 743 × 24 KiB + 19.27 MiB = 36.7 MiB, logged as "36 MiB" [D M2M3_RESULT.md:47, 81].
- Check 2: the M1 arena at 2048 ctx FP32 is 48 MiB KV + 19.27 MiB DeltaNet + about 2 MiB buffers, about 69 MiB, logged as "69 MiB" [D M1_RESULT.md:52].
- Check 3: M3 meminfo reports "engine arena 106 MiB" = 69 + 36 snapshot [D M2M3_RESULT.md:88].
- I did not read the header values directly. [INF from the three checks above, which agree]

**KV per token, FP16 (M2.5):** 6 layers × 2 heads × 256 dims × 2 (K,V) × 2 B = **12 288 B = 12 KiB**
(confirmed: `shl r15, 1` / `vcvtps2ph` store path [R transformer.asm:999-1028]).

**Recurrent block R:** S = 18 × 16 × 128 × 128 × 4 = 18 874 368 B; conv = 18 × 3 × 6144 × 4 =
1 327 104 B; **R = 20 201 472 B = 19.27 MiB**. It is one contiguous span (`CTX_DN_ZERO_BYTES`),
already copied as one region by `snap_regions` [R transformer.asm:523-527, llm.rs:511].

The logits are not needed: every restore point is a turn boundary, and the next step is always
`feed()`, which recomputes logits for the last fed token [R llm.rs:418-427, agent.rs:430].

### 1.2 Size as a function of context fill P (FP16 KV)

| P (tokens) | Full snapshot R + 12 KiB·P | Of which KV |
|---|---|---|
| 743 (boot prefix) | 28.0 MiB | 8.7 MiB |
| 2 048 | 43.3 MiB | 24 MiB |
| 4 096 | 67.3 MiB | 48 MiB |
| 8 192 | 115.3 MiB | 96 MiB |
| 16 384 | 211.3 MiB | 192 MiB |
| 24 576 (75%, compaction point [R agent.rs:67]) | 307.3 MiB | 288 MiB |
| 32 768 | 403.3 MiB | 384 MiB |

### 1.3 What can be shared: append-only KV, fixed R

The KV cache is append-only: a turn writes rows `pos..pos+d` and never touches rows below. Rows past
`pos` are always written before they are read [D M2M3_RESULT.md:193]. So a child state shares all KV rows
`0..fork_pos` with its parent byte for byte. Only two things differ:

- its own KV rows (`fork_pos..pos`, 12 KiB/token),
- its own R (19.27 MiB, dense, rewritten on every token, cannot be shared or diffed).

**Proposal: delta nodes, no MMU tricks.** A node stores `parent`, `pos`, the KV rows
`[parent.pos, pos)`, its token ids (4 B/token), and R. A typical turn (d = 100-600 tokens) costs
**20.4-26.3 MiB**, dominated by R.

**Copy-on-write through page tables (considered, rejected for v1/v2).** Each of the 24 KV planes is
16 MiB at 32k (`32768 × 256 × 2`). A 4 KiB page holds 8 positions and a 2 MiB page holds 4096. The arena is
mapped with 2 MiB pages [R mem.rs:26-48], so CoW at 4k-position granularity is useless. CoW would need:
- remapping the 384 MiB KV range with 4 KiB pages (98 304 PTEs, 768 KiB of tables) [INF],
- PTE rewrites on every switch,
- a TLB shootdown on the 7 APs, which spin in the asm worker loop [D ARCHITECTURE.md:150-158] and
  would need a new "flush" command in `smp.asm`.

The gain is small. Delta nodes already give the memory sharing, and copying a target's divergent rows
costs at most ~20 ms (next section). CoW only pays off if two branches must be live at once, and the
engine has one global `g_ctx` and one KV buffer, so they cannot be. Keep the MMU for something cheaper: see
§4.4, write-protecting the snapshot pool.

### 1.4 Copy times

Measured: the 36 MiB restore takes 1.15-1.9 ms [D M2M3_RESULT.md:58], so effective bandwidth is
**~19-31 GB/s** (24 KV runs + 1 R run, BSP only). Estimates at that rate [INF]:

| Operation | Bytes | Time |
|---|---|---|
| Save a turn node (R + d rows) | 20-26 MiB | 0.7-1.4 ms (after the pool is touched, §4.1) |
| **Undo along the current chain** (R only; KV rows `0..target.pos` are still valid in the live buffer) | 19.27 MiB | **0.65-1.05 ms** |
| Switch to a sibling branch (R + its rows above the common ancestor) | R + 12 KiB/row | 1-10 ms |
| Restore a full chain from root at 32k | 403 MiB | 13.6-22 ms |
| Re-prefill instead, for comparison | | 25 s per 1k tokens |

The "undo = R only" case works because of the invariant in §5 step 1: the live rows `0..current.pos` always
belong to the current node's chain.

## 2. Use cases, ranked by value for this OS

Costs assume the v1 design (delta nodes, 1 GiB pool). Complexity: S = under 150 lines, M = 150-400,
L = needs another milestone.

| # | Use case | What the user gets | Cost | Complexity |
|---|---|---|---|---|
| 1 | **Undo last turn** (`/undo`, `/redo`) | Take back a bad answer, a wrong question, a derailed tool loop. Today the only exit is `/clear` (lose everything) | ~1 ms, one node per turn | S |
| 2 | **Ctrl-C and failure recovery** | Today Ctrl-C during prefill resets the whole conversation [R agent.rs:439-442], and an interrupted compaction also resets it [R agent.rs:198-202, 229-232]. With a pre-turn node: restore it, nothing is lost | free (reuses #1) | S |
| 3 | **`/bench`, `/fill`, `/gen` keep the conversation** | They overwrite from pos 0 and reset today [R shell.rs:109-111, llm.rs:226-242]. Checkpoint first, then restore afterwards | one full chain copy, ≤ 22 ms | S |
| 4 | **Checkpoint before compaction** | The 2B model writes the summary and it is lossy. `/undo` right after a compaction returns to the full context | R + rows already stored per turn, so nearly free | S |
| 5 | **Clean retry** (`/retry`; automatic on an invalid tool call) | Resample the answer from the same point. An invalid call is resampled instead of feeding an error turn back into the context. Targets the 4% of spike failures [D spikes/TOOLCALL.md:7-23] without spending context | 1 ms + re-feed of the turn (0.5-4 s), or a scratch slot with logits | S-M |
| 6 | **Branches and tree** (`/branch`, `/switch`, `/tree`, `/drop`) | Explore two directions from the same point, ChatGPT-style but under your control | 20 MiB per branch tip | M |
| 7 | **Named save and load in RAM** (`/save`, `/load`) | Bookmark a good state ("tuned the settings, explained the system") and come back to it | pinned node | S (falls out of #6) |
| 8 | **Try-in-a-branch for risky actions** (agent tool `try_in_branch`, user `/try`) | The agent runs its plan in a trial branch. Irreversible tools are simulated there. You see what it would do, then commit or discard. Matters once M4 brings `rm` and `write_file` | 1 node + trial decode time | M (needs tool classes) |
| 9 | **Warm boot** (persist the root and memory prefixes to the FS) | Skip the 15 s boot prefill [D M2M3_RESULT.md:56], and the much larger one once agentd adds a memory slot. The OS boots straight into a thinking state | 28 MiB file, ~30-150 ms read [INF] | L (needs M4 virtio-blk + FS) |
| 10 | **Resume the session across reboots** | Reboot, and the conversation continues exactly where it was | R + P × 12 KiB on disk | L (with #9) |
| 11 | **agentd prefix layers** | Snapshot at each compiled-context slot boundary (system+tools, then memory facts, then session summary). Only the slots after a change are re-prefilled. Agentd can also pre-warm the next session's prefix while the machine is idle | 1 node per layer | M (with M4.5 T12) |
| 12 | **Replay and debugging** | Store token ids, RNG and settings per node. A greedy (or seeded) replay must reproduce the stored state hash, so the journal becomes testable. Lets you diff the logit margins of two branches (`/gen` already prints them [R llm.rs:231-242]) | 4 B/token | S |
| 13 | **Personas as roots** | Different system prompts and tool sets as separate roots ("sysadmin", "tutor"); switching is ~1 ms | 28 MiB+ per root | M (tool registry per root [R agent.rs:19-45]) |
| 14 | **Parallel what-if, best-of-N** | Restore one node N times, sample N answers, let the kernel pick by a deterministic rule (validator pass, majority tool call). Time-sliced, not truly parallel: one `g_ctx` | N × decode time | M |
| 15 | **`/diff`** | Show the two transcripts after their common ancestor | text only | S |

Notes:
- Snapshots roll back the **model**, not the **world** (#8). This is the main trap for users and for the
  agent. See §3.4.
- #2-#4 are bug-class fixes that come almost for free once #1 exists. Do them in the same session.

## 3. UX

### 3.1 Model in one paragraph (for `/help`)

Every finished turn is saved automatically as a numbered point (`#7`). A **branch** is a name pointing
at the newest point of a line of work. The prompt shows it: `aios[main]>`. `/undo` steps back, `/redo`
steps forward again. Typing after an undo starts a side line, and the old line stays reachable for a
while (shown in `/tree`). `/save NAME` keeps a point forever, `/branch NAME` starts a named line. Only the
model's memory goes back in time; changes to the machine (settings, files) stay.

### 3.2 Commands

v1 (session commands, `/` prefix, added to `CMDS` like the others [R shell.rs:91-111], with Tab completion
of branch, save and `#id` arguments):

| Command | Does |
|---|---|
| `/undo [N]` | back N turns (default 1) on the current branch; R-only restore |
| `/redo [N]` | forward again, if no new turn was typed since |
| `/retry` | undo the last turn and ask the same question again (with temp 0, the same answer: that is the determinism test) |
| `/tree [all]` | compact tree (below); `all` also shows the inner points of linear runs |
| `/branch NAME` | name the current point as a new branch and switch to it |
| `/switch NAME\|#ID` | restore that point; the prompt changes |
| `/save NAME` | pin the current point under NAME (never evicted); v2: also write it to disk |
| `/load NAME` | `/switch` to a saved point |
| `/drop NAME\|#ID` | delete a branch, a save or a point, and every point that only it uses |
| `/snap` | status: current point, branch, pool use, thinned points |

v2: `/diff A B`, `/try TEXT` (one turn with the dry-run tool policy), `/save NAME` persisted to the FS
and listed by `/load` after a reboot, `/persona NAME` (switch root).

`!` stays reserved for machine commands; none of these touch the machine.

### 3.3 What the user sees

Prompt: `aios[main]> `. After an undo the prompt shows the step back: `aios[main~1]> `. A new turn
from there forks and names the old line `main@7` (the id of its tip). Do not show a dirty `*` marker:
every finished turn is saved, so there is never unsaved work. The only partial state is a turn
interrupted by Ctrl-C, and that is restored to the pre-turn point right away.

```
aios[main]> /tree
#0 boot prefix                         743 tok
`- #1 how much free memory?             812
   |- #2 and the model?                 905
   |  `- #4 set temperature 0.7       1 020  <- main
   `- #3 tell me a joke                 870  joke
#5 [compact 1] summary of #4          1 210  .
pool 214/1024 MiB, 6 points (1 thinned '.'), 2 branches, 0 saves
```

- The tree is ASCII only (`|- \- |`). The framebuffer font has only Basic Latin, Latin-1 and Specials,
  no box-drawing glyphs [R kernel/Cargo.toml:16].
- One line per point: the first 32 characters of the user's message, the context fill, and the branch
  or save names.
- A linear run of more than 3 points collapses to `... 5 more` unless `all` is given.
- A compacted line hangs off the root with a `[compact N]` label, because its KV rows restart at 743.

Undo output, one line plus a warning when the world changed:
```
[undo] back to #2 "and the model?" (ctx 905/32768), 0.9 ms. /redo brings #4 back.
[undo] note: #4 ran set_setting temperature=0.7; the machine keeps it (/temp 0 to change it back).
```

### 3.4 Automatic checkpoints and kernel policy

- **When:** after every finished turn, before every compaction, before `/bench`, `/fill` and `/gen`,
  and before every `try_in_branch`. Never mid-turn: a turn is atomic, and Ctrl-C restores its parent.
- **Ring:** keep the last **16** points on the current path fully restorable. Older unnamed points are
  *thinned* (R freed, KV rows kept; see §4.3), never silently lost.
- **Naming:** points are `#N` (monotonic, never reused in a boot). Branches and saves use `[a-z0-9_-]{1,16}`.
  Auto names: `main@N` for a line left by undo, `try-N` for trials, `pre-compact-N`.
- **The world is not rolled back.** The kernel classifies every tool:
  - `pure`: meminfo, cpuinfo, list_cores, read_log, uptime.
  - `reversible`: set_setting; the kernel records the old value, and v2 can offer to revert it.
  - `irreversible`: poweroff, reboot; later rm, write_file, run_code.
  `/undo` prints the irreversible and reversible calls it steps over.

### 3.5 The agent using snapshots (v2), under deterministic policy

The model proposes and the kernel decides. Only two tools, both kernel-enforced:

1. **`try_in_branch(goal)`**
   - The kernel checkpoints the current point C and opens `try-N`.
   - It feeds `[kernel] Trial branch: irreversible tools are simulated. Do: GOAL`.
   - It runs the normal tool loop with the dry-run policy: irreversible calls return
     `simulated: would run X(args)` and are recorded with their exact arguments.
   - It restores C and feeds a `<tool_response>` with a bounded (1 KiB) summary: the trial's answer
     plus the recorded calls.
   - The user sees:
     ```
     [try-3] would run: rm path=/logs/old.txt (irreversible)
     [try-3] answer: removed 1 file, 4 KiB freed
     run these 1 call(s) for real? [y/N]
     ```
   - On yes, the kernel executes **the recorded calls**, not a regenerated plan, so what you saw is
     what runs, and feeds their results in main.
   - Limits: 1 trial per user turn, 512 tokens, 4 rounds (same as `MAX_ROUNDS` [R agent.rs:50]). The
     trial branch is dropped after commit or discard unless the user `/save`s it.
2. **Clean retry** is not a tool. The kernel applies it: when parse or validate fails
   [R agent.rs:~440-470], it restores the pre-generation scratch state and resamples (temp ≥ 0.3,
   seeded), at most 2 times, then falls back to today's error turn.

The model never switches, drops or restores the user's branches. Every snapshot action is logged with
`klog!(b'A', ...)`, including id, reason, bytes and µs, so `read_log` and agentd's journal see it.

### 3.6 v1 and v2

- **v1** (RAM only, no agent tools): delta nodes, pool, auto-checkpoint per turn, `/undo /redo /retry
  /tree /branch /switch /save /load /drop /snap`, Ctrl-C/compaction/bench recovery, thinning with replay.
- **v2**: `/diff`, `/try`, `try_in_branch`, clean retry, tool classes with reversible-setting revert,
  FS persistence (`/save` to disk, warm boot root, resume after reboot), personas, agentd prefix layers,
  best-of-N.

## 4. Memory budget and policies

### 4.1 Budget (4 GiB guest)

- The arena region is 2048 MiB, above 4 GiB [D M1_RESULT.md:38, 50]. The model (1922 MiB) is below 4 GiB.
- M2.5 engine arena at 32k FP16 [INF]:
  - KV: 384 MiB.
  - R: 19.27 MiB.
  - Attention scores: 1 MiB (`n_heads × max_seq × 4` [R transformer.asm:459-462], 8 heads [INF]).
  - Logits, probs and misc: about 2 MiB.
  - Total about **406 MiB**, plus the boot prefix snapshot (28 MiB).
- Left in the arena region: **about 1.6 GiB** [INF; check `!mem` on the M2.5 boot].
- Proposed **snapshot pool: 1 GiB**, allocated once at boot right after `agent::init`, setting
  `/set snappool MiB` (boot-time only). This leaves about 600 MiB for M4 (virtio queues, FS cache, a real
  allocator), since the bump arena never frees [R mem.rs:76-93].
- `sys_mmap_anon` zeroes what it hands out [R mem.rs:84], so the pool is touched at boot (~1 GiB memset,
  roughly 0.05-0.3 s [INF]). This also removes the 31-188 ms first-touch cost of today's first save
  [D M2M3_RESULT.md:57].
- With `-m 6G`, the pool could grow by 2 GiB. Check the rule first: keep Fedora under 90% RAM, one
  guest only (AGENTS.md).

How many fit in 1 GiB (1 MiB chunks, see §4.2):

| Scenario | Count |
|---|---|
| Restorable points (R = 20 chunks) when turns are short | ~45-48 |
| Same, after a full 32k line of unique rows (384 chunks) is stored | ~30 |
| Full self-contained snapshots at 75% fill (308 MiB each) | 3 (why deltas are required) |
| Default ring (16 points × 20 MiB) + a 24k line of rows (288 MiB) | 608 MiB, fits with room for saves |

### 4.2 Pool allocator

No heap exists [D M2M3_RESULT.md:216]. Use the simplest allocator that frees:
- a **bitmap of 1 MiB chunks** (1024 bits for 1 GiB).
- R takes 20 chunks (20 MiB; 0.73 MiB wasted).
- A KV segment is a list of chunks. Each chunk holds 85 rows of all 24 planes, plane-major:
  24 runs of 85 × 512 B = 1 044 480 B ≤ 1 MiB. One restore is 24 memcpy calls per chunk.
- Average tail waste is 0.5 MiB per point. No fragmentation, no merging.
- Node metadata lives in a static table of 256 entries: parent, pos, seg chunk list, R chunks or none,
  token-id chunk, RNG, temp, think, EX copy, the turn's user and answer text capped at 2 KiB each, flags,
  and an FNV of R.
- Token ids (4 B/token) go into the same chunks as the KV segment: 85 rows × 4 B fit in the 4 KiB slack.

### 4.3 Eviction

Run when free chunks < 64 (64 MiB) before a save, oldest last-visit first:

1. **Thin**: free R (20 MiB) of an unnamed point that is not current, not a branch tip, not a save,
   not the root, and not one of the last 16 points on the current path. The point stays in the tree
   (`.` in `/tree`), and its KV rows stay because descendants need them. To restore a thinned point,
   restore its nearest restorable ancestor and re-prefill the stored token ids up to it (45 tok/s,
   printed with an ETA, Ctrl-C-able). This is deterministic, so the state hash must match (§5 step 3).
2. **Drop**: delete leaf points of anonymous lines (`main@N`, `try-N`) whose tip is older than 32 turns,
   including their KV rows.
3. If that is still not enough: refuse the save with `[snap] pool full: /drop something or /save less`.
   The turn itself still works. Never evict named things automatically.

### 4.4 Safety

The pool sits in the same ring-0 address space as everything else. Optional, cheap:
- Map the pool's 2 MiB pages read-only.
- Lift the write bit only inside save, so a stray write faults instead of silently corrupting a saved
  state. This needs CR0.WP=1, which Limine is expected to set [INF, verify].
- Store an FNV of R per point and check it on `/switch` in debug builds.

### 4.5 Compaction (M2.5)

`compact()` restores the root and re-prefills summary plus last exchanges [R agent.rs:169-244].
With snapshots:
- Before compaction: checkpoint `pre-compact-N` (pinned until the next compaction). This is cheap because
  the line's rows are already stored per turn.
- After: the new live state becomes point `[compact N]`, a child of root #0 (its rows restart at 743).
- An interrupted summary or re-prefill restores `pre-compact-N` instead of `reset()`
  [R agent.rs:198-202, 229-232].
- `/undo` after a compaction returns to the uncompacted line, if it still fits the context. Next
  compaction it unpins and becomes thinnable.
- `llm::snapshot_restore()` stays as a thin wrapper over "restore #0", so `compact()` needs no change in
  step 1.

## 5. Implementation plan

Each step is one agent session unless noted, and each leaves the boot self-test passing.

1. **Generalize the snapshot API, add the pool (1 session).**
   - Split `snap_regions` [R llm.rs:502-512] into `kv_rows(a, b, f)` and `rec(f)`.
   - Add the chunk pool and the node table. Node #0 = boot prefix (replaces `SNAP_BUF`; keep
     `snapshot_save/restore/pos` as wrappers).
   - Track `LIVE: Option<node>`, with the invariant "live KV rows `0..node.pos` = that node's chain".
     Set it to `None` by anything that writes from pos 0: bench, fill, gen, selftest [R llm.rs:99-115].
   - Done when: the boot restore through the new path gives byte-identical KV rows `0..743` and R
     (FNV compared with the old path); SELFTEST PASS; `/ctx` and compaction unchanged; save and restore
     µs logged.
2. **Per-turn checkpoints, undo, redo, retry, recovery (1 session).**
   - Save a node at the end of `ask()` after the final answer.
   - Restore the parent on Ctrl-C and on an interrupted prefill.
   - `/bench`, `/fill` and `/gen` checkpoint and restore.
   - Prompt `aios[main~N]>`.
   - Done when (through `tools/accept.py`, greedy):
     - "my name is Ada" → "my name is Bob" → `/undo` → "what is my name?" answers Ada.
     - `/retry` at temp 0 gives identical `GEN_TOKENS` (FNV) twice.
     - Ctrl-C during a prefill, then a follow-up question, still sees the earlier turns.
     - `/bench` then a follow-up still sees the conversation.
     - Undo ≤ 2 ms logged.
3. **Branches, tree, eviction (1-2 sessions).**
   - `/branch /switch /tree /save /load /drop /snap`, common-ancestor restore (copy only the target's
     rows above the common ancestor), anonymous `main@N` lines, thinning plus replay.
   - Done when:
     - 3 branches, each told a different name, each answer with their own after `/switch`.
     - `/set snappool 128` forces thinning, and restoring a thinned point by replay gives the same
       R FNV as the stored one.
     - Pool bitmap use matches the sum of node chunks.
4. **Compaction integration (1 session, can merge with 3).**
   - `pre-compact-N`, restore instead of reset on interrupt, `/undo` after a compaction.
   - Done when: `/compact` then `/undo` returns to the old `pos`, and `/ctx` shows the old fill; Ctrl-C
     during the summary keeps the conversation.
5. **Agent tools and tool classes (1-2 sessions, after M4 brings file tools).**
   - Tool class table, dry-run policy, `try_in_branch`, commit of recorded calls with y/N, clean retry
     on invalid calls.
   - Done when:
     - A trial containing `poweroff` or `rm` never executes it (simulated in the log); commit executes
       exactly the recorded arguments.
     - Retry rate and final validity are measured on the 30 spike prompts (`toolcall/`).
6. **Persistence and warm boot (1-2 sessions, after M4 virtio-blk + FS).**
   - Self-contained file per saved point:
     - header: magic, version, model id (header + size + sampled pages hash), engine commit,
       `kv_elt_bytes`, dims, prefix-token FNV (already computed at boot [D M2M3_RESULT.md:65]);
     - then the tokens, KV rows plane-major, R, and a CRC.
   - Warm boot loads the root when the prefix FNV matches, else it prefills and rewrites the file.
   - `/load` lists disk saves.
   - Done when: after a reboot the first `@` answers with no prefix prefill (log shows load ms, not 15 s),
     and a changed system prompt triggers a re-prefill and a new file.

Risks:
- **M2.5 churn**: agent.rs and llm.rs are moving. Start after the merge and rebase on the final
  `compact()` and `ask()`.
- **Position counters**: `pos` lives in Rust (`agent::POS`, local `pos` in `feed`/`generate`), and
  `generate()` feeds the trailing token and `<|im_end|>\n` after it stops [R llm.rs:471-482]. Save
  only after that, so `node.pos` is exactly where the next turn starts. The engine zeroes R whenever it
  sees pos == 0 [R transformer.asm:706-716]: never restore and then call with pos 0.
- **KV layout**:
  - Plane stride depends on `max_seq` [R llm.rs:504]. Store rows compactly so snapshots do not depend
    on `max_seq`.
  - FP16 and FP32 builds are incompatible: record `kv_elt_bytes` and refuse a mismatch.
- **SMP**:
  - Copies run on the BSP between turns, when the workers are idle in their loop. Never during
    `transformer_forward`.
  - Determinism across thread counts is unproven: `tp_*` splits rows and heads, so it is probably
    bit-exact [INF]. The step 2 and 3 hash tests are the proof. If it fails, record the thread count
    per node and replay with it.
- **Prefix snapshot**: node #0 must stay byte-identical to today's snapshot (step 1 gate), and the
  agentd prefix layers (#11) must key on token-id FNV, not on text.
- **Sampler**: with temp > 0, restore `RNG` with the node, or `/retry` becomes non-reproducible. Decide
  whether `/retry` reseeds on purpose (it should: a retry that repeats itself is useless at temp > 0).
- **User confusion about the world**: covered by the undo warning and the tool classes. Test it on a real
  user once.
- **Memory competition with M4**: the pool size is a boot setting; M4 sizing decides it.
- **Redundancy**: the `EX` ring [R agent.rs:138-165] could be derived from the node chain (last 2
  points). Leave it alone in v1 (surgical), note it for cleanup.

## 6. Prior art and what is new here

- **llama.cpp**:
  - `llama_state_get_data/set_data` and `llama_state_save_file/load_file` (whole context), per-sequence
    `llama_state_seq_*`.
  - Server slot save and restore (`--slot-save-path`, `POST /slots/{id}?action=save|restore`).
  - `--prompt-cache`.
  - For recurrent and hybrid models it cannot rewind by truncating the KV cache. The server keeps
    context checkpoints of the recurrent state for that, the same problem as here (from memory, check the
    exact flag names before citing).
- **SGLang RadixAttention, vLLM prefix caching**: a tree of shared KV prefixes with LRU eviction,
  close to §4.3. Their hybrid Mamba support also had to add per-block state checkpoints, because recurrent
  state is not prefix-shareable.
- **ChatGPT and Claude.ai branching**: edit a message to get `< 2/3 >` siblings. Branching exists only in
  the UI; the server recomputes or caches without the user seeing it, and there is no undo of tool side
  effects.
- **git**: the naming model (points = commits, branches = movable names, saves = tags, `main@N` ≈ reflog,
  thinning ≈ gc that keeps history recomputable).
- **VM and process snapshots** (QEMU savevm, Firecracker, CRIU, fork CoW): whole-machine or
  whole-process images, seconds and gigabytes, opaque to the workload.

What only an OS that owns the memory can do:
- **A snapshot is a kernel object cheaper than one token.** It is a direct memcpy of the engine's own
  buffers, with no process boundary and no serialization, in a pool the kernel sizes and protects.
- **It is coupled to the world.** The same kernel classifies and executes every tool, so it can make
  agent actions transactional: dry-run in a trial, commit the exact recorded calls. Later, an FS overlay
  can join the snapshot, and then the files go back too.
- **Policy is deterministic and outside the model.** The kernel decides when to checkpoint, what to
  evict, and what a trial may do, and journals it all. The model cannot skip a checkpoint or roll itself
  back.
- **Idle time and boot belong to the OS.** It can pre-warm the next session's prefix while idle, and
  boot straight into a saved mind (warm boot), the way hibernate works for RAM.
- **Snapshots are a cache, never the source of truth.** Token ids and the agentd journal can rebuild any
  point deterministically. Losing the pool costs time, never data.
