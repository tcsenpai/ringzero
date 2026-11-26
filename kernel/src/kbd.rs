// PS/2 keyboard, scancode set 1, polled. Layouts are data: a US base table plus per-layout
// overrides [normal, shift, altgr, altgr+shift] ('\0' = nothing). Shift, Ctrl, AltGr (E0 38).
// Ctrl+letter gives the control code (Ctrl+C = 0x03, used to interrupt a generation).
use core::arch::asm;

const US_LO: &[u8; 0x3A] = b"\0\x1b1234567890-=\x08\tqwertyuiop[]\n\0asdfghjkl;'`\0\\zxcvbnm,./\0*\0 ";
const US_HI: &[u8; 0x3A] = b"\0\x1b!@#$%^&*()_+\x08\tQWERTYUIOP{}\n\0ASDFGHJKL:\"~\0|ZXCVBNM<>?\0*\0 ";

pub struct Layout {
    pub name: &'static str,
    keys: &'static [(u8, [char; 4])],
}

const N: char = '\0';
pub static LAYOUTS: [Layout; 2] = [
    Layout { name: "us", keys: &[(0x56, ['\\', '|', N, N])] },
    // Italian (IT, ISO 105), as the Windows/Linux "it" layout
    Layout {
        name: "it",
        keys: &[
            (0x29, ['\\', '|', N, N]),
            (0x03, ['2', '"', N, N]),
            (0x04, ['3', '£', N, N]),
            (0x07, ['6', '&', N, N]),
            (0x08, ['7', '/', '{', N]),
            (0x09, ['8', '(', '[', N]),
            (0x0A, ['9', ')', ']', N]),
            (0x0B, ['0', '=', '}', N]),
            (0x0C, ['\'', '?', '`', N]),
            (0x0D, ['ì', '^', '~', N]),
            (0x12, ['e', 'E', '€', N]),
            (0x1A, ['è', 'é', '[', '{']),
            (0x1B, ['+', '*', ']', '}']),
            (0x27, ['ò', 'ç', '@', N]),
            (0x28, ['à', '°', '#', N]),
            (0x2B, ['ù', '§', '`', N]),
            (0x33, [',', ';', N, N]),
            (0x34, ['.', ':', N, N]),
            (0x35, ['-', '_', N, N]),
            (0x56, ['<', '>', N, N]),
        ],
    },
];

static mut LAYOUT: usize = 1; // default: it
static mut SHIFT: bool = false;
static mut CTRL: bool = false;
static mut ALTGR: bool = false;
static mut E0: bool = false;

pub fn layout() -> &'static str {
    unsafe { LAYOUTS[LAYOUT].name }
}

pub fn set_layout(name: &str) -> bool {
    match LAYOUTS.iter().position(|l| l.name == name) {
        Some(i) => {
            unsafe { LAYOUT = i };
            true
        }
        None => false,
    }
}

fn lookup(sc: u8) -> char {
    let (shift, altgr) = unsafe { (SHIFT, ALTGR) };
    let col = (shift as usize) | (altgr as usize) << 1;
    if let Some((_, k)) = unsafe { LAYOUTS[LAYOUT].keys }.iter().find(|(s, _)| *s == sc) {
        return k[col];
    }
    if altgr || (sc as usize) >= US_LO.len() {
        return N;
    }
    (if shift { US_HI } else { US_LO })[sc as usize] as char
}

/// Decode one scancode byte, if the controller has one. Returns a character for key presses.
pub fn getc() -> Option<char> {
    let (st, sc): (u8, u8);
    unsafe {
        asm!("in al, 0x64", out("al") st, options(nomem, nostack));
        if st & 1 == 0 {
            return None;
        }
        asm!("in al, 0x60", out("al") sc, options(nomem, nostack));
        if st & 0x20 != 0 {
            return None; // mouse byte
        }
        if sc == 0xE0 {
            E0 = true;
            return None;
        }
        let ext = E0;
        E0 = false;
        match (ext, sc) {
            (_, 0x1D) => CTRL = true,
            (_, 0x9D) => CTRL = false,
            // left Alt acts as AltGr too: VNC clients on macOS send Option as Alt_L, and nothing here
            // uses Alt shortcuts
            (_, 0x38) => ALTGR = true,
            (_, 0xB8) => ALTGR = false,
            (false, 0x2A | 0x36) => SHIFT = true,
            (false, 0xAA | 0xB6) => SHIFT = false,
            (true, 0x1C) => return Some('\n'), // keypad enter
            (true, _) => {}                    // shortcut: arrows and other extended keys ignored
            (false, s) if s < 0x80 => {
                let c = lookup(s);
                if c == N {
                    return None;
                }
                if CTRL && c.is_ascii_alphabetic() {
                    return Some((c.to_ascii_lowercase() as u8 - b'a' + 1) as char);
                }
                return Some(c);
            }
            _ => {}
        }
    }
    None
}
