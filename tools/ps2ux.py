#!/usr/bin/env python3
"""ps2ux.py MONITOR_SOCK SERIAL_LOG: drive the new shell UX on the PS/2 keyboard with QEMU `sendkey`
(guest layout it): !mem, /he<Tab> -> /help, a bare / (pause -> suggestions), Up (history), a plain message
to the agent. Prints what the shell echoed on serial."""
import socket, sys, time
m = socket.socket(socket.AF_UNIX); m.connect(sys.argv[1]); m.settimeout(1)
try: m.recv(4096)
except socket.timeout: pass
n0 = len(open(sys.argv[2], "rb").read())
def keys(ks, gap=0.12):
    for k in ks:
        m.sendall(f"sendkey {k}\n".encode()); time.sleep(gap)
word = lambda w: [{" ": "spc"}.get(c, c) for c in w]
SL, BANG = "shift-7", "shift-1"  # '/' and '!' on the it layout
keys([BANG] + word("mem") + ["ret"]); time.sleep(1)
keys([SL] + word("he") + ["tab"]); time.sleep(0.5); keys(["ret"]); time.sleep(1)
keys([SL]); time.sleep(1.5); keys(["backspace"]); time.sleep(0.3)   # bare '/' then pause: suggestion line
keys([BANG] + word("kb") + ["tab", "tab"]); time.sleep(0.5); keys(["ret"]); time.sleep(1)
keys(["up"]); time.sleep(0.3); keys(["up"]); time.sleep(0.3); keys(["ret"]); time.sleep(2)  # history: /help again
keys(word("quanti core hai") + ["shift-minus", "ret"]); time.sleep(12)  # ? = shift + the key right of 0 (it)
print(open(sys.argv[2], "rb").read()[n0:].decode(errors="replace").replace("\r", ""))
