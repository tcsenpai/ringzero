#!/usr/bin/env python3
"""manvec.py ISO OUT : T38 pass 1. Boot ISO headless (tools/boottest.py, no disk, MEM env = guest RAM), type
`/man dump`, and write the manual vectors the kernel printed (`MV <hex>` lines, then `MV end <fnv> ...`) to OUT.
The blob is checked against the FNV-1a and the size the kernel printed; agentd.rs checks its header at boot."""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
iso, out = sys.argv[1], sys.argv[2]
logdir = out + ".boot"
env = {k: v for k, v in os.environ.items() if k != "DISK"}  # no data disk: the dump needs only the embedder
subprocess.run([sys.executable, f"{HERE}/boottest.py", iso, "bios", logdir, "8", "/man dump"], env=env,
               stdout=subprocess.DEVNULL, check=True)
log = open(f"{logdir}/timed.log", errors="replace").read()
end = re.search(r"MV end ([0-9a-f]{16}) (\d+) bytes", log)
if not end:
    sys.exit(f"manvec: no 'MV end' line in {logdir}/timed.log")
blob = bytes.fromhex("".join(re.findall(r"\] MV ([0-9a-f]+)\s*$", log, re.M)))
h = 0xcbf29ce484222325
for b in blob:
    h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
if len(blob) != int(end.group(2)) or h != int(end.group(1), 16) or blob[:4] != b"RZMV":
    sys.exit(f"manvec: blob check failed ({len(blob)} bytes, fnv {h:016x}, kernel said {end.group(0)})")
open(out, "wb").write(blob)
print(f"manvec: {out}, {len(blob)} bytes, fnv {h:016x}")
