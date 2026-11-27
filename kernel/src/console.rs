// Text output to COM1 and to the Limine framebuffer (Noto Sans Mono 16px glyphs).
use core::fmt::{self, Write};
use noto_sans_mono_bitmap::{get_raster, get_raster_width, FontWeight, RasterHeight};
use spin::Mutex;
use uart_16550::{backend::PioBackend, Config, Uart16550Tty};

const H: RasterHeight = RasterHeight::Size16;
const W: FontWeight = FontWeight::Regular;

pub struct Fb {
    pub addr: *mut u8,
    pub width: usize,
    pub height: usize,
    pub pitch: usize,
    pub bpp: usize,
}

struct Console {
    serial: Option<Uart16550Tty<PioBackend>>,
    fb: Option<Fb>,
    x: usize,
    y: usize,
    // partial UTF-8 sequence for the framebuffer (a token can end mid-character)
    ubuf: [u8; 4],
    ulen: usize,
    // last char had no glyph: a run of unknown glyphs prints one placeholder
    missing: bool,
    // M8: inside an ANSI escape sequence (colors from jq, sqlite3, ...): 1 after ESC, 2 inside ESC [ ...
    esc: u8,
}

unsafe impl Send for Console {}

static CONSOLE: Mutex<Console> =
    Mutex::new(Console { serial: None, fb: None, x: 0, y: 0, ubuf: [0; 4], ulen: 0, missing: false, esc: 0 });

pub fn init(fb: Option<Fb>) {
    let mut c = CONSOLE.lock();
    // SAFETY: 0x3F8 is COM1 on PC/q35 and nothing else drives it. Bytes typed on the serial line while the kernel
    // loads can land in the init's loopback self-test and fail it (then there is no serial console at all); the
    // test drains the FIFO first, so another try passes.
    c.serial = (0..8).find_map(|_| unsafe { Uart16550Tty::new_port(0x3F8, Config::default()) }.ok());
    if let Some(f) = &fb {
        // shortcut: 32 bpp only (what QEMU std-vga and OVMF GOP give), add other depths for real HW.
        unsafe { core::ptr::write_bytes(f.addr, 0, f.pitch * f.height) };
    }
    c.fb = fb;
}

impl Console {
    fn newline(&mut self) {
        let Some(fb) = &self.fb else { return };
        let lh = H.val();
        self.x = 0;
        self.y += lh;
        if self.y + lh > fb.height {
            // scroll one text line up
            unsafe {
                core::ptr::copy(fb.addr.add(lh * fb.pitch), fb.addr, (fb.height - lh) * fb.pitch);
                core::ptr::write_bytes(fb.addr.add((fb.height - lh) * fb.pitch), 0, lh * fb.pitch);
            }
            self.y -= lh;
        }
    }

    fn fb_char(&mut self, ch: char) {
        let Some(fb) = &self.fb else { return };
        if fb.bpp != 32 {
            return;
        }
        // the serial terminal gets escape sequences as they are; the framebuffer skips them (no ANSI support)
        match (self.esc, ch) {
            (0, '\x1b') => return self.esc = 1,
            (1, '[') => return self.esc = 2,
            (1, _) | (2, '\x40'..='\x7e') => return self.esc = 0,
            (2, _) => return,
            _ => {}
        }
        if ch == '\n' {
            return self.newline();
        }
        let cw = get_raster_width(W, H);
        if ch == '\u{8}' || ch == '\u{7f}' {
            // backspace: step back one cell and blank it
            if self.x >= cw {
                self.x -= cw;
                for dy in 0..H.val() {
                    let off = (self.y + dy) * fb.pitch + self.x * 4;
                    unsafe { core::ptr::write_bytes(fb.addr.add(off), 0, cw * 4) };
                }
            }
            return;
        }
        if (ch as u32) < 0x20 {
            return;
        }
        // emoji, pictographs, variation selectors, ZWJ: no glyphs in the font, skip silently
        if matches!(ch as u32, 0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0xFE00..=0xFE0F | 0x200D | 0x2B00..=0x2BFF) {
            return;
        }
        // '€' is outside the font's ranges: overlay 'C' and '='
        let glyphs = if ch == '€' { ['C', '='] } else { [ch, ch] };
        let Some(r) = get_raster(glyphs[0], W, H) else {
            if !core::mem::replace(&mut self.missing, true) {
                self.fb_char('\u{FFFD}');
                self.missing = true;
            }
            return;
        };
        self.missing = false;
        let r2 = get_raster(glyphs[1], W, H).unwrap_or(get_raster(' ', W, H).unwrap());
        if self.x + cw > fb.width {
            self.newline();
        }
        let Some(fb) = &self.fb else { return };
        for (dy, (row, row2)) in r.rows().zip(r2.rows()).enumerate() {
            for (dx, (&a, &b)) in row.iter().zip(row2.iter()).enumerate() {
                let v = a.max(b) as u32;
                let px = v << 16 | v << 8 | v;
                let off = (self.y + dy) * fb.pitch + (self.x + dx) * 4;
                unsafe { (fb.addr.add(off) as *mut u32).write_volatile(px) };
            }
        }
        self.x += cw;
    }
}

impl Write for Console {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if let Some(sp) = &mut self.serial {
            let _ = sp.write_str(s); // the tty turns \n into \r\n
        }
        for ch in s.chars() {
            self.fb_char(ch);
        }
        Ok(())
    }
}

pub fn print(args: fmt::Arguments) {
    // try_lock: a panic while printing must still reach the screen instead of deadlocking.
    if let Some(mut c) = CONSOLE.try_lock() {
        let _ = c.write_fmt(args);
    } else {
        unsafe { CONSOLE.force_unlock() };
        let _ = CONSOLE.lock().write_fmt(args);
    }
}

/// Raw bytes (model output): serial gets them as is (\n -> \r\n), the framebuffer decodes UTF-8
/// across calls.
pub fn write_bytes(b: &[u8]) {
    let mut c = match CONSOLE.try_lock() {
        Some(c) => c,
        None => {
            unsafe { CONSOLE.force_unlock() };
            CONSOLE.lock()
        }
    };
    if let Some(sp) = &mut c.serial {
        for &x in b {
            if x == b'\n' {
                sp.inner_mut().send_bytes_exact(b"\r");
            }
            sp.inner_mut().send_bytes_exact(&[x]);
        }
    }
    for &x in b {
        if c.ulen == 0 && x < 0x80 {
            c.fb_char(x as char);
            continue;
        }
        if c.ulen > 0 && x & 0xC0 != 0x80 {
            c.ulen = 0; // broken sequence: drop it
            if x < 0x80 {
                c.fb_char(x as char);
                continue;
            }
        }
        let n = c.ulen;
        c.ubuf[n] = x;
        c.ulen += 1;
        let want = match c.ubuf[0] {
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => 1, // stray continuation byte
        };
        if c.ulen >= want {
            let ch = core::str::from_utf8(&c.ubuf[..c.ulen]).ok().and_then(|s| s.chars().next()).unwrap_or('?');
            c.ulen = 0;
            c.fb_char(ch);
        }
    }
}

/// Clear the framebuffer and the serial terminal (ANSI), cursor home.
pub fn clear_screen() {
    let mut c = CONSOLE.lock();
    if let Some(sp) = &mut c.serial {
        sp.inner_mut().send_bytes_exact(b"\x1b[2J\x1b[H");
    }
    if let Some(f) = &c.fb {
        unsafe { core::ptr::write_bytes(f.addr, 0, f.pitch * f.height) };
    }
    c.x = 0;
    c.y = 0;
}

/// Non-blocking read of one byte from COM1.
pub fn serial_getc() -> Option<u8> {
    CONSOLE.lock().serial.as_mut().and_then(|s| s.inner_mut().try_receive_byte().ok())
}

#[macro_export]
macro_rules! println {
    ($($t:tt)*) => { $crate::console::print(format_args!("{}\n", format_args!($($t)*))) };
}

#[macro_export]
macro_rules! print {
    ($($t:tt)*) => { $crate::console::print(format_args!($($t)*)) };
}
