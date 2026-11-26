// Shell on COM1 + PS/2: a line is a builtin command, `@ <text>` goes to the agent. Input is polled
// into a type-ahead ring; Ctrl-C (0x03) sets a flag the generation loop checks between tokens.
use crate::log::SBuf;
use crate::{agent, console, kbd, llm, smp, sys};
use crate::{print, println};
use core::fmt::Write;
use core::sync::atomic::Ordering::Relaxed;

static mut TA: [u8; 512] = [0; 512];
static mut TA_HEAD: usize = 0; // total bytes pushed
static mut TA_TAIL: usize = 0; // total bytes popped
static mut INTR: bool = false;

fn push(b: u8) {
    unsafe {
        if TA_HEAD - TA_TAIL < 512 {
            TA[TA_HEAD % 512] = b;
            TA_HEAD += 1;
        }
    }
}

/// Drain COM1 and PS/2 into the type-ahead ring. Ctrl-C sets the interrupt flag instead.
fn poll_input() {
    while let Some(b) = console::serial_getc() {
        if b == 3 {
            unsafe { INTR = true };
        } else {
            push(b);
        }
    }
    while let Some(c) = kbd::getc() {
        if c == '\u{3}' {
            unsafe { INTR = true };
        } else {
            let mut u = [0u8; 4];
            c.encode_utf8(&mut u).bytes().for_each(push);
        }
    }
}

/// True once per Ctrl-C. Called between tokens.
pub fn poll_intr() -> bool {
    poll_input();
    take_intr()
}

fn take_intr() -> bool {
    unsafe {
        let i = INTR;
        INTR = false;
        i
    }
}

fn getb() -> Option<u8> {
    poll_input();
    unsafe {
        if TA_TAIL == TA_HEAD {
            return None;
        }
        TA_TAIL += 1;
        Some(TA[(TA_TAIL - 1) % 512])
    }
}

/// Read a line (echoed, UTF-8) into buf. Ctrl-C cancels the line (returns 0).
pub fn readline(buf: &mut [u8]) -> usize {
    let mut n = 0;
    let mut last_cr = false;
    loop {
        if take_intr() {
            println!("^C");
            return 0;
        }
        let Some(c) = getb() else {
            smp::idle_wait();
            continue;
        };
        if c == b'\n' && core::mem::replace(&mut last_cr, false) {
            continue; // \r\n
        }
        last_cr = c == b'\r';
        match c {
            b'\r' | b'\n' => {
                console::write_bytes(b"\n");
                return n;
            }
            8 | 127 => {
                if n > 0 {
                    while n > 0 {
                        n -= 1;
                        if buf[n] & 0xC0 != 0x80 {
                            break; // removed a whole UTF-8 character
                        }
                    }
                    console::write_bytes(b"\x08 \x08");
                }
            }
            c if c >= 0x20 && n < buf.len() => {
                buf[n] = c;
                n += 1;
                console::write_bytes(&[c]);
            }
            _ => {}
        }
    }
}

/// Ask the human at the shell. Only an explicit y/yes confirms.
pub fn confirm(question: &str) -> bool {
    print!("{} [y/N] ", question);
    let mut b = [0u8; 16];
    let n = readline(&mut b);
    let a = core::str::from_utf8(&b[..n]).unwrap_or("").trim();
    let yes = a.eq_ignore_ascii_case("y") || a.eq_ignore_ascii_case("yes");
    crate::log::record(b'A', if yes { "confirmation: yes" } else { "confirmation: no" });
    yes
}

/// Settings shared by the `set`/`kbd`/`cores`/`temp` builtins and the agent's set_setting tool.
pub fn set_setting(key: &str, val: &str, o: &mut SBuf<1024>) -> bool {
    let ok = match key {
        "temperature" | "temp" => match val.parse::<f32>() {
            Ok(t) if (0.0..=2.0).contains(&t) => {
                unsafe { llm::TEMP_BITS = if t > 0.0 { t.to_bits() } else { 0 } };
                let _ = write!(o, "temperature = {} ({})", val, if t > 0.0 { "top-p 0.95 sampling" } else { "greedy" });
                true
            }
            _ => {
                let _ = write!(o, "error: temperature must be a number from 0 to 2");
                false
            }
        },
        "cores" => match val.parse::<u64>() {
            Ok(k) if k >= 1 && k <= smp::MAX_THREADS.load(Relaxed) => {
                smp::set_threads(k);
                let _ = write!(o, "cores = {} of {} used for inference", smp::threads(), smp::MAX_THREADS.load(Relaxed));
                true
            }
            _ => {
                let _ = write!(o, "error: cores must be 1..{}", smp::MAX_THREADS.load(Relaxed));
                false
            }
        },
        "keyboard" | "kbd" => {
            let ok = kbd::set_layout(val);
            if ok {
                let _ = write!(o, "keyboard layout = {}", val);
            } else {
                let _ = write!(o, "error: keyboard layout must be one of:");
                kbd::LAYOUTS.iter().for_each(|l| { let _ = write!(o, " {}", l.name); });
            }
            ok
        }
        "think" => {
            let on = matches!(val, "on" | "1" | "true");
            unsafe { llm::THINK = on as u64 };
            let _ = write!(o, "think = {}", if on { "on" } else { "off" });
            true
        }
        _ => {
            let _ = write!(o, "error: unknown setting '{}' (temperature, cores, keyboard, think)", key);
            false
        }
    };
    if ok {
        crate::log::record(b'I', o.as_str());
    }
    ok
}

const HELP: &str = "builtins: help, clear, mem, cpu, cores [N], log [N], uptime, bench, stats, kbd [us|it], \
set <key> <value>, temp X, think on|off, park off|idle|<usec>, poweroff, reboot\n\
@ <text>   ask the system agent (Qwen3.5-2B in ring 0, with tools). Ctrl-C interrupts a generation.";

fn show(f: fn(&mut SBuf<1024>)) {
    let mut o = SBuf::<1024>::new();
    f(&mut o);
    println!("{}", o.as_str());
}

pub fn run() -> ! {
    println!("AIOS shell. {}", HELP);
    static mut LINE: [u8; 1024] = [0; 1024];
    loop {
        print!("\naios> ");
        let line = unsafe { &mut *core::ptr::addr_of_mut!(LINE) };
        let n = readline(line);
        let s = core::str::from_utf8(&line[..n]).unwrap_or("").trim();
        if s.is_empty() {
            continue;
        }
        if let Some(q) = s.strip_prefix('@') {
            match q.trim() {
                "" => println!("usage: @ <question or request>"),
                q => agent::ask(q),
            }
            continue;
        }
        let s = s.strip_prefix('/').unwrap_or(s); // M1 slash commands still work
        let (cmd, arg) = s.split_once(' ').map(|(a, b)| (a, b.trim())).unwrap_or((s, ""));
        let mut o = SBuf::<1024>::new();
        match cmd {
            "help" => println!("{}", HELP),
            "clear" | "new" | "reset" => {
                console::clear_screen();
                agent::reset();
                println!("[agent] new conversation (the system prompt is restored from the boot snapshot on the next @)");
            }
            "mem" => show(sys::meminfo),
            "cpu" => show(sys::cpuinfo),
            "uptime" => show(sys::uptime),
            "cores" => {
                if !arg.is_empty() {
                    set_setting("cores", arg, &mut o);
                    println!("{}", o.as_str());
                }
                show(sys::cores);
            }
            "log" => {
                let k = arg.parse::<usize>().unwrap_or(20);
                crate::log::last(k, |l| println!("{}", l));
            }
            "bench" => {
                llm::bench();
                agent::reset(); // the bench ran from position 0
            }
            "stats" => println!(
                "[stats] threads {}, parked now {}, wake IPIs {}, arena used {} MiB, agent ctx {}, kbd {}",
                smp::threads(), smp::parked(), smp::wakeups(), crate::mem::used() >> 20, agent::pos(), kbd::layout()
            ),
            "kbd" | "set" | "temp" | "think" => {
                let (k, v) = match cmd {
                    "set" => arg.split_once(' ').map(|(a, b)| (a, b.trim())).unwrap_or((arg, "")),
                    c => (c, arg),
                };
                if cmd == "kbd" && v.is_empty() {
                    println!("keyboard layout {} (available: us, it)", kbd::layout());
                } else {
                    set_setting(k, v, &mut o);
                    println!("{}", o.as_str());
                }
            }
            "park" => {
                let p = match arg {
                    "off" => Some(smp::Park::Off),
                    "idle" => Some(smp::Park::Idle),
                    a => a.parse::<u64>().ok().map(smp::Park::Always),
                };
                if let Some(p) = p {
                    unsafe { smp::PARK = p };
                    smp::park_phase(false);
                }
                let desc = match unsafe { smp::PARK } {
                    smp::Park::Off => "off (always spin)",
                    smp::Park::Idle => "idle (spin during a generation, park between requests)",
                    smp::Park::Always(_) => "always (park after the spin window, also during a generation)",
                };
                println!("[park] {}", desc);
            }
            "poweroff" | "halt" => sys::poweroff(),
            "reboot" => sys::reboot(),
            "toks" => agent::dump_prefix_tokens(),
            // test hook for the exception handler: read an unmapped lower-half address -> #PF, halt
            "fault" => unsafe {
                core::ptr::read_volatile(0xdead_b000usize as *const u64);
            },
            _ => println!("unknown command '{}'. Type 'help', or '@ {}' to ask the agent.", cmd, s),
        }
    }
}
