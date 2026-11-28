## MODIFIED Requirements

### Requirement: ringzero data under /var/lib/ringzero
The system SHALL keep its own files under /var/lib/ringzero (mem/journal.log, mem/facts.log, mem/episodes.log, mem/index.bin, mem/index.tsv, hib/image.bin, bootsnap.bin, layout), and the file indexer SHALL not walk that tree.
Source: kernel/src/layout.rs JOURNAL, FACTS, EPISODES, INDEX, HIB, BOOTSNAP; kernel/src/agentd.rs INDEX_LIST; kernel/src/fs.rs indexed, walk

#### Scenario: Paths in the status
- **WHEN** the user types `/agentd` with a disk mounted
- **THEN** the stores are reported in /var/lib/ringzero/mem
