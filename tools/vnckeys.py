#!/usr/bin/env python3
"""vnckeys.py HOST PORT PASSWORD TEXT: type TEXT + Enter into the guest as a plain VNC client does
(one keysym per character, no QEMU extended key events), so QEMU's -k keymap turns keysyms into
scancodes. Needs pycryptodome (VNC auth is DES with bit-reversed key bytes)."""
import os, socket, struct, sys, time
from Crypto.Cipher import DES
host, port, pw, text = sys.argv[1], int(sys.argv[2]), sys.argv[3].encode(), sys.argv[4]
s = socket.create_connection((host, port))
rd = lambda n: b"".join(iter(lambda: s.recv(1), b""))[:0] or s.recv(n)
s.recv(12); s.sendall(b"RFB 003.008\n")
n = s.recv(1)[0]; types = s.recv(n)
assert 2 in types, types
s.sendall(b"\x02"); chal = s.recv(16)
key = bytes(int(f"{b:08b}"[::-1], 2) for b in (pw + b"\0" * 8)[:8])
s.sendall(DES.new(key, DES.MODE_ECB).encrypt(chal))
assert struct.unpack(">I", s.recv(4))[0] == 0, "VNC auth failed"
s.sendall(b"\x01"); s.recv(4096)  # ClientInit (shared), ServerInit
def key_ev(sym):
    for down in (1, 0):
        s.sendall(struct.pack(">BBHI", 4, down, 0, sym)); time.sleep(0.05)
SHIFT, ALT = 0xFFE1, int(os.environ.get("ALTSYM", "0xFFE9"), 16)  # Shift_L, Alt_L (macOS Option)
# what a client on an Italian keyboard holds down for each character (Option = AltGr on the Mac)
MOD = {**{c: SHIFT for c in 'éç°§£"!$%&/()=?^*;:_>|'}, **{c: ALT for c in "@#[]€{}~`"},
       **{c: (SHIFT, ALT) for c in "{}"}}
def mods(ch):
    m = MOD.get(ch, SHIFT if ch.isupper() else ())
    return m if isinstance(m, tuple) else (m,)
for ch in text:
    for m in mods(ch): s.sendall(struct.pack(">BBHI", 4, 1, 0, m))
    key_ev({"€": 0x20AC}.get(ch, ord(ch)))
    for m in mods(ch): s.sendall(struct.pack(">BBHI", 4, 0, 0, m))
key_ev(0xFF0D)  # Return
time.sleep(1); s.close()
