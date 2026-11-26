#!/usr/bin/env python3
"""kbdtest.py MONITOR_SOCK SERIAL_LOG: type keys on the guest's PS/2 keyboard with QEMU `sendkey`
(physical key codes, independent of -k) and print what the shell echoed on serial."""
import socket, sys, time
keys = ["x", "apostrophe", "semicolon", "bracket_left", "shift-bracket_left", "backslash", "equal",
        "alt_r-semicolon", "alt_r-apostrophe", "alt_r-bracket_left", "alt_r-bracket_right",
        "alt_r-shift-bracket_left", "alt_r-shift-bracket_right", "alt_r-e", "shift-3", "less", "shift-less",
        "shift-2", "minus", "shift-minus", "shift-7", "ret"]
m = socket.socket(socket.AF_UNIX); m.connect(sys.argv[1]); m.settimeout(1)
try: m.recv(4096)
except socket.timeout: pass
n0 = len(open(sys.argv[2], "rb").read())
for k in keys:
    m.sendall(f"sendkey {k}\n".encode()); time.sleep(0.15)
time.sleep(1)
print(open(sys.argv[2], "rb").read()[n0:].decode(errors="replace"))
