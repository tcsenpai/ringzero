## ADDED Requirements

### Requirement: Multi-stream batched decode
The system SHALL be able to run rows of several streams in one engine pass, each row at its own position on its own KV cache and DeltaNet state, so the weights are read once for all of them, and every row's logits SHALL be bit-identical to decoding that stream alone.
Source: engine/prefill.asm transformer_prefill, pf_ms, MS_ROW; kernel/src/multi.rs step

#### Scenario: two streams in one pass
- **WHEN** a foreground decode step runs while agentd's consolidation decodes on an extra stream
- **THEN** one engine pass feeds the foreground token and the background token, and the foreground's token ids are the same as without the background row

### Requirement: Extra stream slots
Up to 3 extra streams of 4096 positions SHALL exist only while used, each in a contiguous run of free snapshot pool chunks lent while at least 256 chunks stay free and given back on close. Lent chunks SHALL count as used and SHALL NOT be saved in snapshots or the hibernation image.
Source: kernel/src/multi.rs SLOT_CAP, open, close; kernel/src/snap.rs lend, take_back

#### Scenario: pool too full to lend
- **WHEN** a slot would leave fewer than 256 free chunks in the pool
- **THEN** the slot is not opened: `/multi` runs fewer streams and says so, and a consolidation runs on top of the current point
