## MODIFIED Requirements

### Requirement: Baked manual vectors
With MANUAL_VECTORS=1 the kernel SHALL carry the sections' and the 54 hub probes' Q8_0 vectors and use them without embedder work when their key matches the manual, the probes, the embedder file, the engine build, the chunker version and the index dim; on a mismatch, or with MANUAL_VECTORS=0, the sections SHALL be embedded in idle time and the probes at the first search.
Source: kernel/src/agentd.rs queue_manual, baked, man_key, ensure_probes; Makefile target model, target kernel

#### Scenario: Key mismatch
- **WHEN** the index dim was changed with `/index dim 256`
- **THEN** the baked vectors are not used and the sections are embedded at the prompt

### Requirement: Recall returns a manual section
The recall tool SHALL put the best manual section before its top 3 hits when the section goes first (a how-to question in the user's message or the query, or a score 0.06 over the best other hit) and its raw cosine is 0.55 or more.
Source: kernel/src/agentd.rs howto, recall, ranked

#### Scenario: How-to query
- **WHEN** the model calls recall with query `how to hibernate`
- **THEN** the result includes a line starting with `ringzero manual, Hibernation:`
