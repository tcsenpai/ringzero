#!/usr/bin/env python3
"""screenshot.py MONITOR_SOCK OUT.png : QEMU monitor `screendump` of the framebuffer, saved as PNG
(stdlib only; read-only for the guest)."""
import os, socket, struct, sys, time, zlib

def ppm_to_png(ppm, png):
    d = open(ppm, "rb").read()
    parts = d.split(b"\n", 3); w, h = map(int, parts[1].split()); px = parts[3]
    raw = b"".join(b"\0" + px[y * w * 3:(y + 1) * w * 3] for y in range(h))
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
    open(png, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                          + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

def mon(sock, cmd):
    s = socket.socket(socket.AF_UNIX); s.connect(sock); s.settimeout(2)
    try: s.recv(4096)
    except Exception: pass
    s.sendall((cmd + "\n").encode()); time.sleep(0.5); s.close()

if __name__ == "__main__":
    ppm = os.path.abspath(sys.argv[2]) + ".ppm"
    mon(sys.argv[1], f"screendump {ppm}"); time.sleep(1)
    ppm_to_png(ppm, sys.argv[2]); os.remove(ppm); print(sys.argv[2])
