#!/usr/bin/env python3
"""boottest.py ISO bios|uefi OUTDIR [SMP] [CMD ...] : boot ISO in QEMU/KVM headless, log serial with
host timestamps, type each CMD at the 'aios> ' prompt (one per prompt), stop at the prompt after the
last CMD (or at PANIC/EXCEPTION/M0 FAIL, or 900 s), save a screenshot as PNG."""
import os, shutil, socket, struct, subprocess, sys, time, zlib

iso, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
smp = sys.argv[4] if len(sys.argv) > 4 else "8"
cmds = sys.argv[5:]
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) // 1024
if avail < 6144: sys.exit(f"refusing to start a 4 GiB guest: only {avail} MiB available (want > 6144)")
ser_path, mon_path = f"{out}/ser", f"{out}/mon"
cmd = ["qemu-system-x86_64", "-enable-kvm", "-cpu", "host", "-M", "q35", "-m", "4G", "-smp", smp,
       "-cdrom", iso, "-boot", "d", "-vga", "std", "-display", "none", "-no-reboot",
       "-serial", f"unix:{ser_path},server,nowait", "-monitor", f"unix:{mon_path},server,nowait"]
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
ser = None
while ser is None and time.time() - t0 < 10:
    try: ser = socket.socket(socket.AF_UNIX); ser.connect(ser_path)
    except OSError: ser = None; time.sleep(0.1)
ser.settimeout(0.05)
buf, stamps, prompts, sent = b"", [], 0, 0
tlog, last_rx = open(f"{out}/timed.log", "w"), time.time()
while time.time() - t0 < 900 and p.poll() is None:
    try:
        d = ser.recv(65536)
        if not d: break
        buf += d; last_rx = time.time()
    except socket.timeout: pass
    while b"\n" in buf:
        l, buf = buf.split(b"\n", 1)
        stamps.append((time.time() - t0, l.decode(errors="replace").rstrip("\r")))
        tlog.write(f"[{stamps[-1][0]:7.2f}s] {stamps[-1][1]}\n"); tlog.flush()
    text = "\n".join(l for _, l in stamps) + buf.decode(errors="replace")
    if any(k in text for k in ("PANIC:", "-> halt", "M0 FAIL", "model_init failed")): break
    if stamps and time.time() - last_rx > 180: break  # silent for 3 min after output started: hung
    if text.count("aios> ") > prompts:
        prompts = text.count("aios> ")
        if sent < len(cmds):
            time.sleep(0.2); ser.sendall((cmds[sent] + "\r").encode()); sent += 1
        else:
            break
if buf:
    stamps.append((time.time() - t0, buf.decode(errors="replace"))); tlog.write(f"[{stamps[-1][0]:7.2f}s] {stamps[-1][1]}\n")
tlog.close()
try: time.sleep(1); mon(f"screendump {out}/screen.ppm"); time.sleep(1)
except OSError: pass
p.terminate(); p.wait(timeout=10)
if not os.path.exists(f"{out}/screen.ppm"): print(open(f"{out}/timed.log").read()); sys.exit("no screendump")

# PPM (P6) -> PNG, stdlib only
d = open(f"{out}/screen.ppm", "rb").read()
parts = d.split(b"\n", 3); w, h = map(int, parts[1].split()); px = parts[3]
raw = b"".join(b"\0" + px[y * w * 3:(y + 1) * w * 3] for y in range(h))
chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
open(f"{out}/screen.png", "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                                      + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
log = "\n".join(l for _, l in stamps)
at = lambda k: next((round(t, 2) for t, l in stamps if k in l), None)
res = "PASS" if "SELFTEST PASS" in log else "FAIL" if "SELFTEST FAIL" in log else \
      "CRASH" if ("PANIC" in log or "EXCEPTION" in log) else "TIMEOUT"
print(f"[{mode} smp={smp}] banner at {at('AIOS M1')}s, selftest {res}, {sent}/{len(cmds)} commands sent")
print(open(f"{out}/timed.log").read())
