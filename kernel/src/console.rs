// Text output to COM1 and to the Limine framebuffer (Noto Sans Mono 16px glyphs). T46: the framebuffer reads ANSI
// SGR (colors, bold, dim) like the serial terminal does, so both consoles show the shell's role colors and the
// escapes of WASI programs (jq, ...) the same way.
use core::fmt::{self, Write};
use core::sync::atomic::{AtomicBool, Ordering::Relaxed};
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
    // T46: CSI parameters seen so far (np of them, the last one still growing) and the SGR state they set
    par: [u16; 8],
    np: usize,
    fg: u32,
    bold: bool,
    dim: bool,
}

unsafe impl Send for Console {}

static CONSOLE: Mutex<Console> = Mutex::new(Console {
    serial: None, fb: None, x: 0, y: 0, ubuf: [0; 4], ulen: 0, missing: false, esc: 0,
    par: [0; 8], np: 0, fg: FG, bold: false, dim: false,
});

/// `!set color on|off` (T46): the shell's own lines get role colors. Programs' escapes pass through either way.
pub static COLOR: AtomicBool = AtomicBool::new(true);
/// Role colors (assets/branding/README.md): prompt R0 red bold, tool lines pin gold, statistics gray,
/// confirmations and errors bright red.
pub const PROMPT: &str = "\x1b[1;31m";
pub const TOOL: &str = "\x1b[33m";
pub const DIM: &str = "\x1b[90m";
pub const ERR: &str = "\x1b[91m";
pub const RESET: &str = "\x1b[0m";

/// The escape for a role, or "" when color is off.
pub fn sgr(code: &'static str) -> &'static str {
    if COLOR.load(Relaxed) { code } else { "" }
}

/// Framebuffer default text color: off-white.
const FG: u32 = 0xECEAE6;
/// ANSI 30-37 and 90-97 on the framebuffer, mapped to the logo palette where it has a match: red #C6140A,
/// yellow = pin gold #C9A45C, bright black = package gray #6B6B6B, bright red #E8473C. Black is a dark gray
/// so "bold black" (jq's null) stays readable on the black background.
const PAL: [u32; 16] = [
    0x4A4A4A, 0xC6140A, 0x4E9A06, 0xC9A45C, 0x3465A4, 0x75507B, 0x06989A, 0xC0C0C0,
    0x6B6B6B, 0xE8473C, 0x8AE234, 0xFCE94F, 0x729FCF, 0xAD7FA8, 0x34E2E2, 0xFFFFFF,
];

/// xterm 256-color index to RGB (16 base, 6x6x6 cube, 24 grays).
fn xterm(n: u16) -> u32 {
    match n {
        0..=15 => PAL[n as usize],
        16..=231 => {
            let lv = |v: u16| if v == 0 { 0 } else { 55 + 40 * v as u32 };
            let i = n - 16;
            lv(i / 36) << 16 | lv(i / 6 % 6) << 8 | lv(i % 6)
        }
        _ => {
            let g = 8 + 10 * (n.min(255) as u32 - 232);
            g << 16 | g << 8 | g
        }
    }
}

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
    /// ESC [ ... m on the framebuffer: 0 reset, 1 bold, 2 dim, 22 normal, 30-37 / 90-97 / 39 foreground,
    /// 38;5;N and 38;2;R;G;B; anything else (backgrounds, underline) is ignored.
    fn sgr(&mut self) {
        let n = self.np.max(1);
        let p = self.par;
        let mut i = 0;
        while i < n {
            match p[i] {
                0 => (self.fg, self.bold, self.dim) = (FG, false, false),
                1 => self.bold = true,
                2 => self.dim = true,
                22 => (self.bold, self.dim) = (false, false),
                30..=37 => self.fg = PAL[p[i] as usize - 30],
                90..=97 => self.fg = PAL[p[i] as usize - 82],
                39 => self.fg = FG,
                38 if i + 2 < n && p[i + 1] == 5 => {
                    self.fg = xterm(p[i + 2]);
                    i += 2;
                }
                38 if i + 4 < n && p[i + 1] == 2 => {
                    let c = |v: u16| v.min(255) as u32;
                    self.fg = c(p[i + 2]) << 16 | c(p[i + 3]) << 8 | c(p[i + 4]);
                    i += 4;
                }
                _ => {}
            }
            i += 1;
        }
    }

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
        // the serial terminal gets escape sequences as they are; the framebuffer applies SGR and skips the rest
        match (self.esc, ch) {
            (0, '\x1b') => return self.esc = 1,
            (1, '[') => {
                (self.esc, self.np, self.par) = (2, 0, [0; 8]);
                return;
            }
            (1, _) => return self.esc = 0,
            (2, '0'..='9') => {
                let k = self.np.max(1) - 1;
                self.np = self.np.max(1);
                if k < self.par.len() {
                    self.par[k] = self.par[k].saturating_mul(10).saturating_add(ch as u16 - '0' as u16);
                }
                return;
            }
            (2, ';') => return self.np = self.np.max(1) + 1,
            (2, '\x40'..='\x7e') => {
                if ch == 'm' {
                    self.np = self.np.min(self.par.len());
                    self.sgr();
                }
                return self.esc = 0;
            }
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
        let w = if self.bold { FontWeight::Bold } else { W };
        let Some(r) = get_raster(glyphs[0], w, H) else {
            if !core::mem::replace(&mut self.missing, true) {
                self.fb_char('\u{FFFD}');
                self.missing = true;
            }
            return;
        };
        self.missing = false;
        let r2 = get_raster(glyphs[1], w, H).unwrap_or(get_raster(' ', w, H).unwrap());
        if self.x + cw > fb.width {
            self.newline();
        }
        let Some(fb) = &self.fb else { return };
        let k = if self.dim { 5 } else { 8 }; // dim: 5/8 of the color
        let (cr, cg, cb) = ((self.fg >> 16) * k / 8, (self.fg >> 8 & 0xFF) * k / 8, (self.fg & 0xFF) * k / 8);
        for (dy, (row, row2)) in r.rows().zip(r2.rows()).enumerate() {
            for (dx, (&a, &b)) in row.iter().zip(row2.iter()).enumerate() {
                let v = a.max(b) as u32;
                let px = (v * cr / 255) << 16 | (v * cg / 255) << 8 | v * cb / 255;
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

/// /proc/devices: (serial console present, framebuffer width, height, bpp).
pub fn devices() -> (bool, Option<(usize, usize, usize)>) {
    let c = CONSOLE.lock();
    (c.serial.is_some(), c.fb.as_ref().map(|f| (f.width, f.height, f.bpp)))
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
