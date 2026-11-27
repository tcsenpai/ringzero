## MODIFIED Requirements

### Requirement: Consolidation leaves the conversation untouched
The system SHALL run consolidation greedily, generating at most 220 tokens into a capture buffer that is not shown. `/consolidate`, and an idle consolidation without a background stream that fits, SHALL run on top of the current snapshot point (temperature forced to 0 and restored after) and SHALL restore the current point after it finishes, when input arrives at the prompt, or on Ctrl-C during `/consolidate`.
Source: kernel/src/agentd.rs CONS_MAX, cons_begin, cons_abort, consolidate_now

#### Scenario: Interrupted by the user
- **WHEN** the user presses Ctrl-C during `/consolidate`
- **THEN** "^C [agentd] consolidation stopped, conversation restored (ctx N)" is printed and the context position is the one before the consolidation

## ADDED Requirements

### Requirement: Consolidation on the background stream
An idle consolidation SHALL run on a copy of the conversation in an extra stream when the conversation, the instruction and 220 tokens fit in it, decoded alone in idle slices and one token per foreground decode step. Input at the prompt SHALL NOT stop it; turns added meanwhile stay for the next one; `/multi` stops it.
Source: kernel/src/agentd.rs cons_begin, consolidate_bg, preempt, stop_bg; kernel/src/multi.rs bg_start, bg_run; kernel/src/llm.rs pass_bg

#### Scenario: Started by hand
- **WHEN** the user types `/consolidate &`
- **THEN** the shell prints `[agentd] consolidating on the background stream` and returns to the prompt at once; with no stream to lend it prints `[agentd] no background stream now (see /agentd log), use /consolidate`

#### Scenario: Typing during a background consolidation
- **WHEN** the user types a question while an idle consolidation runs on the background stream
- **THEN** the consolidation goes on during the answer's decode steps, and `/agentd log` later shows `consolidate #N (idle): T tok in M ms on the background stream (R steps next to the foreground's, A alone), ids 0x...`

### Requirement: Consolidation output ids
The finish line of every consolidation in the decision log SHALL end with `ids 0x...`, the FNV-1a of the output token ids; the same session gives the same ids on both paths.
Source: kernel/src/agentd.rs cons_run, cons_finish

#### Scenario: Same session, both paths
- **WHEN** the same session is consolidated once with `/consolidate` and once in idle time on the background stream
- **THEN** both finish lines show the same ids hash
