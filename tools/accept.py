#!/usr/bin/env python3
"""accept.py HOST PORT OUT STEP... : drive the AIOS shell over the serial TCP port (from the Mac).
STEP is a line to type at the prompt, 'y:'/'n:' + line (answer y or n at the [y/N] prompt it causes),
or '^C<sec>:' + line (send Ctrl-C <sec> seconds after the line). Saves the transcript to OUT."""
import socket, sys, time
host, port, out, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
s = socket.create_connection((host, port)); s.settimeout(0.2)
buf = b""
def read_until(keys, timeout=300):
    global buf
    t = time.time()
    while time.time() - t < timeout:
        for k in keys:
            if k in buf:
                i = buf.index(k) + len(k); seen, buf = buf[:i], buf[i:]; return seen.decode(errors="replace")
        try: buf += s.recv(65536)
        except socket.timeout: pass
    return buf.decode(errors="replace") + "\n<<timeout>>"
log = []
s.sendall(b"\r"); read_until([b"aios> "], 20)
for st in steps:
    ans, intr = None, None
    if st[:2] in ("y:", "n:"): ans, st = st[0], st[2:]
    if st.startswith("^C"): d, st = st[2:].split(":", 1); intr = float(d)
    t0 = time.time(); s.sendall(st.encode() + b"\r")
    txt = "aios> "
    if intr is not None:
        time.sleep(intr); s.sendall(b"\x03")
    if ans:
        txt += read_until([b"[y/N] "]); s.sendall(ans.encode() + b"\r")
    txt += read_until([b"aios> "])
    txt = txt[:-len("aios> ")] if txt.endswith("aios> ") else txt
    log.append(f"{txt.rstrip()}\n    (step took {time.time()-t0:.1f} s)\n")
    print(log[-1], flush=True)
open(out, "w").write("\n".join(log).replace("\r", ""))
