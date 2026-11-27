## MODIFIED Requirements

### Requirement: Boot snapshot cache
The system SHALL write the prefilled boot prefix (token ids, every K/V plane's rows, R) to `/var/lib/ringzero/bootsnap.bin` with a 48-byte `RZBS` header keyed by the prefix token ids, the model hash, the engine build hash and the KV geometry, and SHALL load it at the next boot instead of prefilling only when magic, version, key, geometry, size, checksum and token count all match.
Source: kernel/src/snap.rs init_root, load_prefix; kernel/src/agent.rs build_prefix, init

#### Scenario: Engine rebuilt
- **WHEN** the kernel boots with a different engine build than the one that wrote `/var/lib/ringzero/bootsnap.bin`
- **THEN** "boot cache: miss (key differs (model, prefix or engine changed)), prefilling N tokens" is logged and the cache is rewritten

### Requirement: Hibernation commands
The system SHALL write the hibernation image `/var/lib/ringzero/hib/image.bin` on `/hibernate` and on every `!poweroff` (not on `!reboot`), SHALL on `/hibernate discard` delete the image and stop the next poweroffs of this session from writing one until a plain `/hibernate`, and SHALL on `/hibernate status` print the image size or "[hib] no image: the next boot starts fresh".
Source: kernel/src/hib.rs SKIP, boot, cmd; kernel/src/sys.rs poweroff, reboot; kernel/src/shell.rs run

#### Scenario: Discard then poweroff
- **WHEN** the user types `/hibernate discard` then `!poweroff`
- **THEN** the image is deleted, the log says "hib: poweroff without an image (/hibernate discard)", and the next boot starts a fresh conversation

### Requirement: Resume at boot
The system SHALL at boot resume the image only if magic, version, key (boot snapshot key mixed with "ringzero hibernate v1"), pool chunk count, boot prefix length, sizes and checksum match, restoring tree, names, settings and transcript with every point but the current one thinned, and logging the state hash as "same as saved" or "DIFFERS from the saved one"; else it SHALL start fresh.
Source: kernel/src/hib.rs key, apply, save, boot; kernel/src/snap.rs hdr_u, txt; kernel/src/main.rs kmain

#### Scenario: Model changed
- **WHEN** the kernel boots with a different model than the one that wrote `/var/lib/ringzero/hib/image.bin`
- **THEN** "hib: image /var/lib/ringzero/hib/image.bin not resumed (key differs ...), fresh conversation; /hibernate discard deletes it" is logged

#### Scenario: Same model
- **WHEN** the kernel boots with the same model, prefix, engine and pool size
- **THEN** "hib: resumed #N on BRANCH (ctx P), K points ... same as saved" is logged and `/tree` shows the hibernated tree

### Requirement: No partial image
The system SHALL delete `/var/lib/ringzero/hib/image.bin` when writing it fails, and SHALL write nothing when no disk is mounted or the boot has not finished.
Source: kernel/src/hib.rs apply, save

#### Scenario: Write error
- **WHEN** the disk write of the image fails during `/hibernate`
- **THEN** the file is removed and "hib: hibernate not saved (...), the next boot starts fresh" is logged
