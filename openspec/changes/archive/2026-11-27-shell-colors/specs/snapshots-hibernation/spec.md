## MODIFIED Requirements

### Requirement: What the image holds
The hibernation image SHALL hold a 64-byte `RZHB` version 1 header (key, state hash, sizes, checksum, pool chunk count), the metadata of every non-root point (with its question, answer and side-effect texts) and every name, the KV row chunks of every non-root point, R of the current point only, and the agent settings (temperature, think, constrain, compaction percent and count, color) and transcript.
Source: kernel/src/snap.rs load_prefix, hib_save; kernel/src/hib.rs key, extra

#### Scenario: Mid-turn poweroff
- **WHEN** `!poweroff` runs while the live state is ahead of the current point
- **THEN** the image holds the current point, not the unsaved live state
