// Kernel log: a ring of the last LINES lines, each "[  sec.ms] L text". klog! prints the text on the
// console (as before) and records it here; `log` (shell) and read_log (agent) read it back.
use core::fmt::{self, Write};
use spin::Mutex;

/// Fixed-capacity string buffer (no heap). Writes past N bytes are dropped at a char boundary.
pub struct SBuf<const N: usize> {
    pub b: [u8; N],
    pub n: usize,
}

impl<const N: usize> SBuf<N> {
    pub const fn new() -> Self {
        SBuf { b: [0; N], n: 0 }
    }
    pub fn as_str(&self) -> &str {
        unsafe { core::str::from_utf8_unchecked(&self.b[..self.n]) }
    }
    pub fn clear(&mut self) {
        self.n = 0;
    }
    pub fn push(&mut self, s: &[u8]) {
        let k = s.len().min(N - self.n);
        self.b[self.n..self.n + k].copy_from_slice(&s[..k]);
        self.n += k;
    }
    /// push, never cutting inside a UTF-8 sequence (the buffer is read back as &str); returns the bytes dropped.
    pub fn push_cut(&mut self, s: &[u8]) -> usize {
        let mut k = s.len().min(N - self.n);
        while k > 0 && k < s.len() && s[k] & 0xC0 == 0x80 {
            k -= 1;
        }
        self.push(&s[..k]);
        s.len() - k
    }
}

impl<const N: usize> Write for SBuf<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let mut k = s.len().min(N - self.n);
        while !s.is_char_boundary(k) {
            k -= 1;
        }
        self.push(&s.as_bytes()[..k]);
        Ok(())
    }
}

const LINES: usize = 256;
const W: usize = 128;

struct Ring {
    l: [[u8; W]; LINES],
    len: [u8; LINES],
    next: usize, // total lines ever written
}

static RING: Mutex<Ring> = Mutex::new(Ring { l: [[0; W]; LINES], len: [0; LINES], next: 0 });

pub fn ms_since_boot() -> u64 {
    let hz = crate::smp::TSC_HZ.load(core::sync::atomic::Ordering::Relaxed);
    if hz == 0 { 0 } else { crate::smp::tsc_to_us(crate::smp::rdtsc()) / 1000 }
}

/// Record one message (may hold several lines) at `level` (b'I', b'W', b'E', b'A' = agent).
pub fn record(level: u8, msg: &str) {
    let t = ms_since_boot();
    let mut r = RING.lock();
    for line in msg.split('\n').filter(|l| !l.is_empty()) {
        let mut s = SBuf::<W>::new();
        let _ = write!(s, "[{:5}.{:03}] {} {}", t / 1000, t % 1000, level as char, line);
        let i = r.next % LINES;
        r.l[i] = s.b;
        r.len[i] = s.n as u8;
        r.next += 1;
    }
}

/// Call f on each of the last n lines, oldest first.
pub fn last(n: usize, mut f: impl FnMut(&str)) {
    let r = RING.lock();
    let n = n.min(r.next).min(LINES);
    for k in r.next - n..r.next {
        let i = k % LINES;
        f(unsafe { core::str::from_utf8_unchecked(&r.l[i][..r.len[i] as usize]) });
    }
}

pub fn emit(level: u8, args: fmt::Arguments) {
    let mut s = SBuf::<512>::new();
    let _ = s.write_fmt(args);
    crate::console::print(format_args!("{}\n", s.as_str()));
    record(level, s.as_str());
}

/// println! plus a log record: klog!(b'I', "fmt", ...)
#[macro_export]
macro_rules! klog {
    ($lvl:expr, $($t:tt)*) => { $crate::log::emit($lvl, format_args!($($t)*)) };
}
