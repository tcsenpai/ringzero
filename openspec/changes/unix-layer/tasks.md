## 1. Tooleval gate (before the code that removes tools)
- [ ] 1.1 Add 8 info prompts to tools/tooleval.py INOS (memory, cores, uptime, cpu model, pci devices, disk size, hostname, version) with `cat <file>` accepted as a right first tool; count hidden-alias hits and rounds per prompt
- [ ] 1.2 Baseline run on the current ISO (spike,inos), save JSON under docs/
- [ ] 1.3 Run on the unix-layer ISO with the four tools hidden; record prefix tokens, cold prefill time and boot snapshot size
- [ ] 1.4 Decide per D5 (at most 1 prompt worse, no more round-limit hits, uptime and cores answers right by hand); if it fails, keep the tools in the prompt and revise specs/agent-tools before the apply

## 2. Mount table and virtual trees
- [ ] 2.1 fs.rs `mount_of`, routing in read, entry, list, ls, open, check_rw; `ls /` shows proc, sys, dev
- [ ] 2.2 procfs.rs: kmsg, mounts, uptime readable form, cpuinfo flags once
- [ ] 2.3 sysfs.rs: class/block/vda, bus/pci/devices/<addr>, devices/system/cpu, power/state (writable: disk hibernates, the agent asks y/N)
- [ ] 2.4 devfs.rs and wasm.rs `Fd::Dev`: null, zero, random, urandom, tty, console, vda (read-only); shell and agent cat refuse devices
- [ ] 2.5 /etc: hostname default at boot, banner; virtual os-release and passwd
- [ ] 2.6 /tmp: made at boot, emptied at a fresh boot, kept on hibernation resume; fs::walk skips /tmp and /var/lib/ringzero
- [ ] 2.7 Protection: agent fs_tool and wasm.rs (`sys_ro` renamed) refuse writes under /var/lib/ringzero

## 3. Data move and migration
- [ ] 3.1 layout.rs path constants; mstore.rs, agentd.rs, hib.rs, agent.rs BOOT_SNAP use them
- [ ] 3.2 layout.rs migration per D6 (copy, verify, commit marker, remove; RAM-only agentd on failure)
- [ ] 3.3 Fixture: an old-layout disk image (facts, journal, index, image, bootsnap, a stray /sys file); boot it, check facts in `/facts`, `/search` hits, legacy/ contents, `layout:` lines
- [ ] 3.4 Crash test: kill QEMU after each migration step (debug env var that stops after step N), reboot, check the end state is the same as an uninterrupted run
- [ ] 3.5 Rollback test: boot the previous release ISO on the migrated disk, then the new one; check legacy/sys-<time>/ and the warning; `/agentd merge-legacy` merges it once, refuses an incompatible store
- [ ] 3.6 Disk-full test: migration pending, agentd RAM-only, retry succeeds after freeing space
- [ ] 3.7 agentd.rs merge_legacy: `/agentd merge-legacy [DIR]` (compatibility check, facts through remember, new journal and episode records, index queue, `merged` marker)

## 4. Agent text
- [ ] 4.1 SYSTEM sentence, ls and cat descriptions (D8); TOOLS and SPECS with hidden aliases (if 1.4 passes)
- [ ] 4.2 tools/wasm/rztest: /sys write test moved to /var/lib/ringzero, new EROFS tests for /sys and /dev/vda

## 5. Docs and specs
- [ ] 5.1 INFO.md file tree table, migration and rollback (mtools recipe), README.md, docs/MANUAL.md, docs/BETA.md, site/index.html, release notes upgrade note, /help
- [ ] 5.2 docs/ARCHITECTURE.md diagram (mount table, layout.rs), tools/arch_pdf.sh
- [ ] 5.3 docs/T55_RESULT.md with the tooleval numbers and migration timings; tools/roadmap.sh
- [ ] 5.4 tools/surfaces_check.sh passes; `openspec archive unix-layer`
