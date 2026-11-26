// M5: MicroPython v1.29.0 (ports/embed, MIT) in ring 0, built freestanding by build.rs (kernel/upy/).
// A fixed GC heap and a private 1 MiB stack come from the engine arena at boot (before the snapshot pool
// takes the rest). Every run is a fresh interpreter: `!py CODE`, `!py -f PATH` and the agent's run_code tool.
// stdout goes to the console, or into a buffer for the agent. The VM calls upy_check every 256 backward
// jumps / returns: Ctrl-C or the time budget raise KeyboardInterrupt; a program that keeps running GRACE_MS
// after that (it swallowed the exception) is aborted outright. Pure C loops (a huge bigint power, a giant
// str()) never reach the hook: the budget cannot stop those (see docs/M5_MICROPYTHON_RESULT.md).
use crate::log::SBuf;
use crate::{klog, println, shell, smp};
use core::fmt::Write;

unsafe extern "C" {
    fn upy_setup(heap: *mut u8, heap_size: usize, stack: *mut u8, stack_size: usize);
    fn upy_exec(src: *const u8, len: usize, name: *const u8, name_len: usize, single: i32, used: *mut usize) -> i32;
}

const STACK: usize = 1 << 20;
const SHELL_MS: u64 = 10_000;
const TOOL_MS: u64 = 5_000;
const GRACE_MS: u64 = 1_000;
/// Captured output for the agent (the tool result cap is 8 KiB, the status line needs some room).
const OUT_CAP: usize = 6144;

static mut HEAP_BYTES: usize = 0;
static mut OUT: SBuf<OUT_CAP> = SBuf::new();
static mut CAPTURE: bool = false;
static mut DROPPED: usize = 0;
static mut DEADLINE: u64 = 0;
static mut HARD: u64 = 0;
static mut NEXT_POLL: u64 = 0;
static mut WHY: Why = Why::None;

#[derive(Clone, Copy, PartialEq)]
enum Why {
    None,
    Intr,
    Budget,
}

pub fn heap_mib() -> usize {
    unsafe { HEAP_BYTES >> 20 }
}

/// Boot: GC heap (RINGZERO_UPY_HEAP_MIB at build time, default 8) and stack from the arena.
pub fn init() {
    let mib = option_env!("RINGZERO_UPY_HEAP_MIB").and_then(|s| s.parse::<usize>().ok()).unwrap_or(8).clamp(1, 256);
    let heap = crate::mem::sys_mmap_anon((mib << 20) as u64);
    let stack = crate::mem::sys_mmap_anon(STACK as u64);
    if (heap as i64) <= 0 || (stack as i64) <= 0 {
        klog!(b'W', "upy:  arena full, MicroPython is off");
        return;
    }
    unsafe {
        upy_setup(heap, mib << 20, stack, STACK);
        HEAP_BYTES = mib << 20;
    }
    klog!(b'I', "upy:  MicroPython 1.29.0 (embed port, freestanding), GC heap {} MiB + stack {} KiB from the arena", mib, STACK >> 10);
}

#[unsafe(no_mangle)]
extern "C" fn upy_out(s: *const u8, n: usize) {
    let b = unsafe { core::slice::from_raw_parts(s, n) };
    unsafe {
        if !CAPTURE {
            crate::console::write_bytes(b);
            return;
        }
        let o = &mut *core::ptr::addr_of_mut!(OUT);
        let mut k = b.len().min(OUT_CAP - o.n);
        while k > 0 && k < b.len() && b[k] & 0xC0 == 0x80 {
            k -= 1; // never cut a UTF-8 sequence: OUT is read back as &str
        }
        o.push(&b[..k]);
        DROPPED += b.len() - k;
    }
}

/// 0 go on, 1 raise KeyboardInterrupt, 2 abort the run (glue.c upy_poll).
#[unsafe(no_mangle)]
extern "C" fn upy_check() -> i32 {
    let now = smp::rdtsc();
    unsafe {
        if now >= NEXT_POLL {
            NEXT_POLL = now + smp::us_to_tsc(10_000); // input every 10 ms: a port read is a VM exit
            if shell::poll_intr() {
                if WHY != Why::None {
                    return 2; // second Ctrl-C
                }
                WHY = Why::Intr;
                HARD = now + smp::us_to_tsc(GRACE_MS * 1000);
            }
        }
        if WHY == Why::None && now >= DEADLINE {
            WHY = Why::Budget;
            HARD = now + smp::us_to_tsc(GRACE_MS * 1000);
        }
        match WHY {
            Why::None => 0,
            _ if now >= HARD => 2,
            _ => 1,
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn upy_fatal(msg: *const u8) -> ! {
    let n = (0..).take_while(|&i| unsafe { *msg.add(i) } != 0).count();
    panic!("{}", core::str::from_utf8(unsafe { core::slice::from_raw_parts(msg, n) }).unwrap_or("?"));
}

pub struct Run {
    pub rc: i32,
    pub us: u64,
    pub used: usize,
    why: Why,
}

impl Run {
    /// "ok in 3 ms, heap 12 KiB of 8 MiB" or why it stopped.
    pub fn status<const N: usize>(&self, o: &mut SBuf<N>, budget_ms: u64) {
        let what = match (self.rc, self.why) {
            (0, _) => "ok",
            (2, Why::None) => "aborted",
            (_, Why::Intr) => "interrupted (Ctrl-C)",
            (1, Why::Budget) => "stopped: time budget exceeded",
            (2, Why::Budget) => "aborted: time budget exceeded and KeyboardInterrupt was ignored",
            _ => "error",
        };
        let _ = write!(o, "{} in {} ms (budget {} ms), GC heap {} KiB used of {} MiB", what, self.us / 1000, budget_ms,
            self.used >> 10, heap_mib());
    }
}

fn run(src: &str, name: &str, budget_ms: u64, capture: bool) -> Result<Run, &'static str> {
    if unsafe { HEAP_BYTES } == 0 {
        return Err("MicroPython is not available (no arena memory at boot)");
    }
    let single = !src.trim().contains('\n');
    let t0 = smp::rdtsc();
    let mut used = 0;
    let rc = unsafe {
        CAPTURE = capture;
        (*core::ptr::addr_of_mut!(OUT)).clear();
        DROPPED = 0;
        WHY = Why::None;
        DEADLINE = t0 + smp::us_to_tsc(budget_ms * 1000);
        NEXT_POLL = 0;
        let s = if single { src.trim() } else { src };
        upy_exec(s.as_ptr(), s.len(), name.as_ptr(), name.len(), single as i32, &mut used)
    };
    unsafe { CAPTURE = false };
    let r = Run { rc, us: smp::tsc_to_us(smp::rdtsc() - t0), used, why: unsafe { WHY } };
    let mut l = SBuf::<160>::new();
    let _ = write!(l, "py {}: ", name);
    r.status(&mut l, budget_ms);
    crate::log::record(b'I', l.as_str());
    Ok(r)
}

/// `!py CODE` or `!py -f PATH`: output straight to the console.
pub fn shell_cmd(arg: &str) {
    let (src, name) = match arg.split_once(' ') {
        Some(("-f", p)) => match crate::fs::read(p.trim(), 0, 256 << 10) {
            Ok((b, size)) if (b.len() as u64) == size && crate::fs::is_text(&b) => match alloc::string::String::from_utf8(b) {
                Ok(s) => (s, p.trim()),
                Err(_) => return println!("[py] {}: not UTF-8 text", p.trim()),
            },
            Ok((_, size)) if size > 256 << 10 => return println!("[py] {}: larger than 256 KiB", p.trim()),
            Ok(_) => return println!("[py] {}: not a text file", p.trim()),
            Err(e) => return println!("[py] {}: {}", p.trim(), e),
        },
        _ if arg.is_empty() || arg == "-f" => return println!("usage: !py CODE | !py -f PATH"),
        _ => (alloc::string::String::from(arg), "<shell>"),
    };
    match run(&src, name, SHELL_MS, false) {
        Ok(r) => {
            let mut o = SBuf::<256>::new();
            r.status(&mut o, SHELL_MS);
            println!("[py] {}", o.as_str());
        }
        Err(e) => println!("[py] {}", e),
    }
}

/// The agent's run_code tool: captured stdout + traceback + one status line.
pub fn tool<const N: usize>(code: &str, o: &mut SBuf<N>) {
    match run(code, "<run_code>", TOOL_MS, true) {
        Ok(r) => {
            let out = unsafe { &*core::ptr::addr_of!(OUT) };
            o.push(out.as_str().trim_end().as_bytes());
            let dropped = unsafe { DROPPED };
            if dropped > 0 {
                let _ = write!(o, "\n[... {} more bytes of output dropped]", dropped);
            }
            let _ = write!(o, "{}[python: ", if out.n > 0 { "\n" } else { "" });
            r.status(o, TOOL_MS);
            o.push(b"]");
        }
        Err(e) => {
            let _ = write!(o, "error: {}", e);
        }
    }
}
