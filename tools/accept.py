#!/usr/bin/env python3
"""accept.py HOST PORT OUT STEP... : drive the ringzero shell over the serial TCP port (from the Mac).
STEP is a line to type at the prompt, 'y:'/'n:' + line (answer y or n at the [y/N] prompt it causes),
or '^C<sec>:' + line (send Ctrl-C <sec> seconds after the line). Saves the transcript to OUT."""
import re, socket, sys, time
PROMPT = re.compile(rb"ringzero(?:\[[^\]\n]*\])?> ")  # ringzero> or ringzero[branch~N]>
host, port, out, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
s = socket.create_connection((host, port)); s.settimeout(0.2)
buf = b""
def read_until(pat, timeout=3600):
    """Text up to the first match of pat (bytes or compiled regex), and the matched part."""
    global buf
    t = time.time()
    while time.time() - t < timeout:
        m = pat.search(buf) if hasattr(pat, "search") else re.search(re.escape(pat), buf)
        if m:
            seen, buf = buf[:m.start()], buf[m.end():]
            return seen.decode(errors="replace"), m.group(0).decode()
        try: buf += s.recv(65536)
        except socket.timeout: pass
    return buf.decode(errors="replace") + "\n<<timeout>>", ""
log = []
s.sendall(b"\r"); prompt = read_until(PROMPT, 20)[1] or "ringzero> "
for st in steps:
    ans, intr = None, None
    if st[:2] in ("y:", "n:"): ans, st = st[0], st[2:]
    if st.startswith("^C"): d, st = st[2:].split(":", 1); intr = float(d)
    t0 = time.time(); s.sendall(st.encode() + b"\r")
    txt = prompt
    if intr is not None:
        time.sleep(intr); s.sendall(b"\x03")
    if ans:
        seen, m = read_until(b"[y/N] "); txt += seen + m; s.sendall(ans.encode() + b"\r")
    seen, prompt = read_until(PROMPT)
    txt += seen
    log.append(f"{txt.rstrip()}\n    (step took {time.time()-t0:.1f} s)\n")
    print(log[-1], flush=True)
open(out, "w").write("\n".join(log).replace("\r", ""))
