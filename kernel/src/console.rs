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
}

unsafe impl Send for Console {}

static CONSOLE: Mutex<Console> = Mutex::new(Console { serial: None, fb: None, x: 0, y: 0 });

pub fn init(fb: Option<Fb>) {
    let mut c = CONSOLE.lock();
    // SAFETY: 0x3F8 is COM1 on PC/q35 and nothing else drives it.
    c.serial = unsafe { Uart16550Tty::new_port(0x3F8, Config::default()) }.ok();
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
        if ch == '\n' {
            return self.newline();
        }
        let cw = get_raster_width(W, H);
        if self.x + cw > fb.width {
            self.newline();
        }
        let Some(fb) = &self.fb else { return };
        let r = get_raster(ch, W, H).unwrap_or_else(|| get_raster('?', W, H).unwrap());
        for (dy, row) in r.rows().enumerate() {
            for (dx, &v) in row.iter().enumerate() {
                let px = (v as u32) << 16 | (v as u32) << 8 | v as u32;
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

pub fn write_bytes(b: &[u8]) {
    print(format_args!("{}", core::str::from_utf8(b).unwrap_or("<non-utf8>")));
}

#[macro_export]
macro_rules! println {
    ($($t:tt)*) => { $crate::console::print(format_args!("{}\n", format_args!($($t)*))) };
}
