#!/usr/bin/env python3
"""uxtest.py HOST PORT OUT STEP... : type each STEP on the AIOS serial shell byte by byte (Tab = \\t),
then wait until the output ends with the prompt and stays quiet for QUIET seconds (suggestion lines
redraw the prompt mid-line, so a plain read-until-prompt would desync). 'y:'/'n:' prefix answers a
[y/N] prompt. Saves the transcript (\\r removed, \\x08 shown as <BS>) to OUT."""
import os, socket, sys, time
host, port, out, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
QUIET = float(os.environ.get("QUIET", "1.5"))
s = socket.create_connection((host, port)); s.settimeout(0.1)
def drain(end=b"aios> ", timeout=3600, ans=None):
    buf, last, t = b"", time.time(), time.time()
    while time.time() - t < timeout:
        try:
            d = s.recv(65536)
            if d: buf += d; last = time.time()
        except socket.timeout: pass
        if ans and buf.endswith(b"[y/N] "): s.sendall(ans.encode() + b"\r"); ans = None
        if buf.endswith(end) and time.time() - last > QUIET: return buf
    return buf + b"\n<<timeout>>"
s.sendall(b"\r"); drain(timeout=20)
log = ["aios> "]
for st in steps:
    ans = None
    if st[:2] in ("y:", "n:"): ans, st = st[0], st[2:]
    t0 = time.time()
    if len(st) > 200: s.sendall(st.encode())  # long pasted text: QEMU's chardev paces it to the UART
    else:
        for ch in st.encode():
            s.sendall(bytes([ch])); time.sleep(0.03)
    s.sendall(b"\r")
    txt = drain(ans=ans).decode(errors="replace").replace("\r", "").replace("\x08 \x08", "<BS>")
    log.append(txt + f"    (step took {time.time()-t0:.1f} s)\n")
    print(log[-1], flush=True)
open(out, "w").write("".join(log))
