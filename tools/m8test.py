#!/usr/bin/env python3
"""m8test.py ISO OUTDIR STEPS : boot ISO headless (BIOS, QEMU/KVM, MEM=6G, DISK=img attaches the data disk) and
play STEPS, a file of lines 'REGEX<TAB>TEXT': wait until REGEX matches the serial output that came after the
previous step, then type TEXT (Python escapes: \\r = Enter, \\x04 = Ctrl-D, \\x03 = Ctrl-C). 'PROMPT' stands for the
ringzero shell prompt. A line with REGEX only waits; '#' lines are comments. Each step waits at most 600 s (a hang
or a panic ends the run; STEP_TIMEOUT=s changes it). Serial with host timestamps in OUTDIR/timed.log, the transcript in OUTDIR/serial.txt."""
import os, re, shutil, socket, subprocess, sys, time

iso, out, steps_file = sys.argv[1:4]
PROMPT = r"ringzero(?:\[[^\]\n]*\])?> "
steps = []
for line in open(steps_file, encoding="utf-8"):
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        continue
    rx, _, text = line.partition("\t")
    steps.append((re.compile(rx.replace("PROMPT", PROMPT)), text.encode().decode("unicode_escape").encode("latin-1") if text else None))

shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
mem = os.environ.get("MEM", "6G")
avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) // 1024
if avail < int(mem.rstrip("G")) * 1024 + 2048:
    sys.exit(f"refusing to start a {mem} guest: only {avail} MiB available")
ser_path = f"{out}/ser"
cmd = ["qemu-system-x86_64", "-enable-kvm", "-cpu", "host", "-M", "q35", "-m", mem, "-smp", os.environ.get("SMP", "8"),
       "-cdrom", iso, "-boot", "d", "-vga", "std", "-display", "none", "-no-reboot", "-serial", f"unix:{ser_path},server,nowait"]
if os.environ.get("DISK"):
    cmd += ["-drive", f"file={os.environ['DISK']},if=none,id=disk0,format=raw", "-device", "virtio-blk-pci,drive=disk0"]
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ser = None
while ser is None and time.time() - t0 < 10:
    try:
        ser = socket.socket(socket.AF_UNIX)
        ser.connect(ser_path)
    except OSError:
        ser = None
        time.sleep(0.1)
ser.settimeout(0.05)
tlog, raw = open(f"{out}/timed.log", "w"), open(f"{out}/serial.txt", "wb")
seen, line, i, t_step = b"", b"", 0, time.time()
while p.poll() is None and i < len(steps):
    try:
        d = ser.recv(65536)
        if not d:
            break
    except socket.timeout:
        d = b""
    raw.write(d)
    raw.flush()
    seen += d
    line += d
    while b"\n" in line:
        l, line = line.split(b"\n", 1)
        tlog.write(f"[{time.time() - t0:7.2f}s] {l.decode(errors='replace').rstrip()}\n")
        tlog.flush()
    text = seen.decode(errors="replace")
    if any(k in text for k in ("PANIC:", "-> halt", "M0 FAIL")) or time.time() - t_step > float(os.environ.get("STEP_TIMEOUT", "600")):
        print(f"stopped at step {i + 1}: {'panic' if 'PANIC' in text else 'timeout'}")
        break
    rx, send = steps[i]
    if rx.search(text):
        if send is not None:
            time.sleep(0.3)
            ser.sendall(send)
        tlog.write(f"[{time.time() - t0:7.2f}s] >>> step {i + 1}: {send!r}\n")
        seen, i, t_step = b"", i + 1, time.time()
time.sleep(1)
p.terminate()
p.wait(timeout=10)
tlog.close()
raw.close()
print(f"{i}/{len(steps)} steps in {time.time() - t0:.0f} s")
