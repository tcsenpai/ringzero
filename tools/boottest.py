#!/usr/bin/env python3
"""boottest.py ISO bios|uefi OUTDIR [SMP] : boot ISO in QEMU/KVM headless, log serial with host
timestamps, stop at 'M0 PASS'/'M0 FAIL'/'PANIC' (or 300 s), save a screenshot as PNG."""
import os, shutil, socket, struct, subprocess, sys, time, zlib

iso, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
smp = sys.argv[4] if len(sys.argv) > 4 else "8"
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) // 1024
if avail < 6144: sys.exit(f"refusing to start a 4 GiB guest: only {avail} MiB available (want > 6144)")
ser, mon_path = f"{out}/serial.log", f"{out}/mon"
cmd = ["qemu-system-x86_64", "-enable-kvm", "-cpu", "host", "-M", "q35", "-m", "4G", "-smp", smp,
       "-cdrom", iso, "-boot", "d", "-vga", "std", "-display", "none", "-no-reboot",
       "-serial", f"file:{ser}", "-monitor", f"unix:{mon_path},server,nowait"]
if mode == "uefi":
    shutil.copy("/usr/share/edk2/ovmf/OVMF_VARS.fd", f"{out}/vars.fd")
    cmd += ["-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/edk2/ovmf/OVMF_CODE.fd",
            "-drive", f"if=pflash,format=raw,file={out}/vars.fd"]

def mon(c):
    s = socket.socket(socket.AF_UNIX); s.connect(mon_path); s.settimeout(2)
    try: s.recv(4096)
    except Exception: pass
    s.sendall((c + "\n").encode()); time.sleep(0.5)
    s.close()

t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
seen, stamps, log = 0, [], ""
while time.time() - t0 < 300 and p.poll() is None:
    time.sleep(0.05)
    try: log = open(ser, errors="replace").read()
    except FileNotFoundError: continue
    lines = log.split("\n")
    for l in lines[seen:len(lines) - 1]:
        stamps.append((time.time() - t0, l.rstrip("\r")))
    seen = len(lines) - 1
    if any(k in log for k in ("M0 PASS", "M0 FAIL", "PANIC")): break
time.sleep(1); mon(f"screendump {out}/screen.ppm"); time.sleep(1)
p.terminate(); p.wait(timeout=10)

with open(f"{out}/timed.log", "w") as f:
    for t, l in stamps: f.write(f"[{t:7.2f}s] {l}\n")

# PPM (P6) -> PNG, stdlib only
d = open(f"{out}/screen.ppm", "rb").read()
parts = d.split(b"\n", 3); w, h = map(int, parts[1].split()); px = parts[3]
raw = b"".join(b"\0" + px[y * w * 3:(y + 1) * w * 3] for y in range(h))
chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
open(f"{out}/screen.png", "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                                      + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
at = lambda k: next((round(t, 2) for t, l in stamps if k in l), None)
print(f"[{mode}] OVMF hands off to Limine at {at('BdsDxe: starting')}s, kernel banner at {at('AIOS M0')}s, result: "
      f"{'PASS' if 'M0 PASS' in log else 'FAIL' if 'M0 FAIL' in log else 'PANIC' if 'PANIC' in log else 'TIMEOUT'}")
print(open(f"{out}/timed.log").read())
