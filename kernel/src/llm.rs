// Generation (mirror of kernel.asm run_inference / pick_next_token), the golden boot self-test and
// the REPL on COM1 + PS/2 keyboard.
use crate::engine::*;
use crate::smp::{self, Park};
use crate::{console, print, println};

// engine/constants.inc (QWEN35)
const CTX_ARENA_SIZE: usize = 24;
const CTX_MAX_SEQ: usize = 160;
const CTX_EOS_1: usize = 184;
const CTX_EOS_2: usize = 192;
const CTX_BUF_LOGITS: usize = 280;
const CTX_BUF_PROMPT: usize = 344;
const CTX_BUF_GEN: usize = 352;
const CTX_BUF_PROBS: usize = 360;
const CTX_SAMPLE_N: usize = 488;
const CTX_TOK_IM_END: usize = 520;
const CTX_TOK_NL: usize = 560;
const MAX_PROMPT_TOKENS: u64 = 8192;
const MAX_GEN_TOKENS: u64 = 8192;
pub const CTX_LEN: u64 = 2048; // kernel.asm opt_ctx_len default

// docs/qwen35-2b/golden_2b.json
const GOLDEN_PROMPT: &[u8] = b"Say hi in 5 words.\0";
const GOLDEN_PROMPT_IDS: [u32; 19] =
    [248045, 846, 198, 44240, 15131, 303, 220, 20, 4105, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271];
const GOLDEN_GEN_IDS: [u32; 3] = [9419, 1017, 0];
const BENCH_PROMPT: &[u8] = b"Explain how a CPU cache works in 3 concise sentences.\0";

static mut CHAT_POS: u64 = 0;
static mut THINK: u64 = 0;
static mut TEMP_BITS: u32 = 0; // 0.0f = greedy argmax
static mut RNG: u64 = 42;
const TOPP_BITS: u32 = 0x3F73_3333; // 0.95f

struct Opts {
    single_turn: bool,
    max_tok: u64,
    print_toks: bool,
}

pub struct Stats {
    pub n_prompt: u64,
    pub n_gen: u64,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub pos: u64,
}

fn rate(n: u64, us: u64) -> (u64, u64) {
    let h = n * 100_000_000 / us.max(1); // tok/s * 100
    (h / 100, h % 100)
}

fn pick_next() -> u64 {
    let logits = ctx(CTX_BUF_LOGITS) as *const f32;
    let n = ctx(CTX_SAMPLE_N);
    unsafe {
        if TEMP_BITS == 0 {
            argmax_f32(logits, n)
        } else {
            sample_top_p_bits(logits, ctx(CTX_BUF_PROBS) as *mut f32, n, core::ptr::addr_of_mut!(RNG), TEMP_BITS, TOPP_BITS)
        }
    }
}

fn print_csv(label: &str, t: &[u32]) {
    print!("{}", label);
    for (i, x) in t.iter().enumerate() {
        print!("{}{}", if i > 0 { "," } else { "" }, x);
    }
    println!("");
}

/// prompt must be NUL-terminated.
fn run(prompt: &[u8], o: &Opts) -> Stats {
    let max_seq = ctx(CTX_MAX_SEQ);
    let ptoks = ctx(CTX_BUF_PROMPT) as *mut u32;
    let gtoks = ctx(CTX_BUF_GEN) as *mut u32;
    let mut n = unsafe { tok_build_chat_prompt(prompt.as_ptr(), ptoks, MAX_PROMPT_TOKENS, 0, THINK) };
    let mut pos = unsafe { CHAT_POS };
    if o.single_turn {
        pos = 0;
    } else if pos > 0 && pos + n + 64 >= max_seq {
        pos = 0;
        println!("[chat] context full, resetting KV-cache");
    }
    n = n.min(max_seq);
    if o.print_toks {
        print_csv("PROMPT_TOKENS ", unsafe { core::slice::from_raw_parts(ptoks, n as usize) });
    }
    smp::park_phase(true);

    let t0 = smp::rdtsc();
    for i in 0..n {
        unsafe { transformer_forward(*ptoks.add(i as usize), pos, (i + 1 == n) as i32) };
        pos += 1;
    }
    let t1 = smp::rdtsc();
    let mut next = pick_next();

    let t2 = smp::rdtsc();
    OUT_TSC.store(0, core::sync::atomic::Ordering::Relaxed);
    let (eos1, eos2) = (ctx(CTX_EOS_1), ctx(CTX_EOS_2));
    let max_tok = o.max_tok.min(MAX_GEN_TOKENS);
    let mut n_gen = 0u64;
    let mut unfed = false;
    while n_gen < max_tok && next != eos1 && next != eos2 {
        unsafe { *gtoks.add(n_gen as usize) = next as u32 };
        n_gen += 1;
        unsafe { tok_write_stdout(next) };
        if n_gen >= max_tok {
            unfed = true;
            break;
        }
        if pos >= max_seq {
            break;
        }
        unsafe { transformer_forward(next as u32, pos, 1) };
        pos += 1;
        next = pick_next();
    }
    let t3 = smp::rdtsc();
    println!("");

    // close the assistant turn in the KV cache so the next <|im_start|>user lines up
    if o.single_turn {
        unsafe { CHAT_POS = 0 };
    } else {
        unsafe {
            if unfed && pos < max_seq {
                transformer_forward(next as u32, pos, 0);
                pos += 1;
            }
            if pos + 2 < max_seq {
                transformer_forward(ctx(CTX_TOK_IM_END) as u32, pos, 0);
                transformer_forward(ctx(CTX_TOK_NL) as u32, pos + 1, 0);
                pos += 2;
            }
            CHAT_POS = pos;
        }
    }
    smp::park_phase(false);

    let s = Stats {
        n_prompt: n,
        n_gen,
        prefill_us: smp::tsc_to_us(t1 - t0),
        decode_us: smp::tsc_to_us(t3 - t2),
        pos,
    };
    if o.print_toks {
        print_csv("GEN_TOKENS ", unsafe { core::slice::from_raw_parts(gtoks, n_gen as usize) });
    }
    let (pa, pb) = rate(s.n_prompt, s.prefill_us);
    let (da, db) = rate(s.n_gen.saturating_sub(1).max(1), s.decode_us);
    println!(
        "[prefill] {} tok in {}.{:02} ms ({}.{:02} tok/s)  [decode] {} tok in {}.{:02} ms ({}.{:02} tok/s, output {} ms)  cores {}  ctx {}/{}",
        s.n_prompt, s.prefill_us / 1000, s.prefill_us % 1000 / 10, pa, pb,
        s.n_gen, s.decode_us / 1000, s.decode_us % 1000 / 10, da, db,
        smp::tsc_to_us(OUT_TSC.load(core::sync::atomic::Ordering::Relaxed)) / 1000,
        smp::threads(), s.pos, max_seq
    );
    s
}

pub fn init(base: *const u8, size: u64) -> bool {
    let t0 = smp::rdtsc();
    let r = unsafe { model_init(base, size, CTX_LEN) };
    if r < 0 {
        println!("model: model_init failed ({})", r);
        return false;
    }
    println!(
        "model: model_init ok in {} ms, max_seq {}, arena {} MiB, vocab sample_n {}",
        smp::tsc_to_us(smp::rdtsc() - t0) / 1000,
        ctx(CTX_MAX_SEQ),
        ctx(CTX_ARENA_SIZE) >> 20,
        ctx(CTX_SAMPLE_N)
    );
    true
}

/// Golden boot self-test (greedy, single turn). Prints SELFTEST PASS/FAIL.
pub fn selftest() -> bool {
    println!("selftest: prompt \"Say hi in 5 words.\" (golden_2b.json), greedy, {} threads", smp::threads());
    let s = run(GOLDEN_PROMPT, &Opts { single_turn: true, max_tok: 16, print_toks: true });
    let p = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_PROMPT) as *const u32, s.n_prompt as usize) };
    let g = unsafe { core::slice::from_raw_parts(ctx(CTX_BUF_GEN) as *const u32, s.n_gen as usize) };
    let ok = p == GOLDEN_PROMPT_IDS && g == GOLDEN_GEN_IDS;
    println!(
        "SELFTEST {} (prompt ids {}, generated ids {} vs golden 9419,1017,0 + <|im_end|>)",
        if ok { "PASS" } else { "FAIL" },
        if p == GOLDEN_PROMPT_IDS { "match" } else { "DIFFER" },
        if g == GOLDEN_GEN_IDS { "match" } else { "DIFFER" }
    );
    ok
}

pub fn bench() -> Stats {
    unsafe { CHAT_POS = 0 };
    let s = run(BENCH_PROMPT, &Opts { single_turn: true, max_tok: 64, print_toks: false });
    unsafe { CHAT_POS = 0 };
    s
}

// ---- input: COM1 + PS/2 (scancode set 1, US layout) ----

const SC_LO: &[u8; 0x3A] = b"\0\x1b1234567890-=\x08\tqwertyuiop[]\n\0asdfghjkl;'`\0\\zxcvbnm,./\0*\0 ";
const SC_HI: &[u8; 0x3A] = b"\0\x1b!@#$%^&*()_+\x08\tQWERTYUIOP{}\n\0ASDFGHJKL:\"~\0|ZXCVBNM<>?\0*\0 ";
static mut SHIFT: bool = false;
static mut E0: bool = false;

fn ps2_getc() -> Option<u8> {
    let (st, sc): (u8, u8);
    unsafe {
        core::arch::asm!("in al, 0x64", out("al") st, options(nomem, nostack));
        if st & 1 == 0 {
            return None;
        }
        core::arch::asm!("in al, 0x60", out("al") sc, options(nomem, nostack));
        if st & 0x20 != 0 {
            return None; // mouse byte
        }
        if sc == 0xE0 {
            E0 = true;
            return None;
        }
        let ext = E0;
        E0 = false;
        if ext {
            return None; // shortcut: extended keys (arrows, keypad enter) ignored
        }
        match sc {
            0x2A | 0x36 => SHIFT = true,
            0xAA | 0xB6 => SHIFT = false,
            _ => {}
        }
        if (sc as usize) < SC_LO.len() {
            let c = if SHIFT { SC_HI[sc as usize] } else { SC_LO[sc as usize] };
            if c != 0 {
                return Some(c);
            }
        }
    }
    None
}

/// Read a line (echoed) into buf, NUL-terminated. Returns its length.
fn readline(buf: &mut [u8]) -> usize {
    let mut n = 0;
    let mut last_cr = false;
    loop {
        let Some(c) = console::serial_getc().or_else(ps2_getc) else {
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
                buf[n] = 0;
                return n;
            }
            8 | 127 => {
                while n > 0 {
                    n -= 1;
                    if buf[n] & 0xC0 != 0x80 {
                        break; // removed a whole UTF-8 character
                    }
                }
                console::write_bytes(b"\x08 \x08");
            }
            c if c >= 0x20 && n + 1 < buf.len() => {
                buf[n] = c;
                n += 1;
                console::write_bytes(&[c]);
            }
            _ => {}
        }
    }
}

const HELP: &str = "commands: /clear (reset chat), /think, /nothink, /bench, /cores N, /park off|idle|<usec>, /temp X (0 = greedy), /stats, /help";

pub fn repl() -> ! {
    println!("AIOS chat on COM1 (Qwen3.5-2B in ring 0). {}", HELP);
    let mut line = [0u8; 1024];
    loop {
        print!("\naios> ");
        let n = readline(&mut line);
        let s = core::str::from_utf8(&line[..n]).unwrap_or("").trim();
        if s.is_empty() {
            continue;
        }
        let (cmd, arg) = s.split_once(' ').map(|(a, b)| (a, b.trim())).unwrap_or((s, ""));
        match cmd {
            "/clear" | "/new" | "/reset" => {
                unsafe { CHAT_POS = 0 };
                println!("[chat] history cleared (ctx=0/{})", ctx(CTX_MAX_SEQ));
            }
            "/think" => unsafe {
                THINK = 1;
                println!("[mode] think=1");
            },
            "/nothink" => unsafe {
                THINK = 0;
                println!("[mode] think=0");
            },
            "/bench" => {
                bench();
            }
            "/cores" => {
                if let Ok(k) = arg.parse::<u64>() {
                    smp::set_threads(k);
                }
                println!("[smp] {} of {} threads", smp::threads(), smp::MAX_THREADS.load(core::sync::atomic::Ordering::Relaxed));
            }
            "/park" => {
                let p = match arg {
                    "off" => Some(Park::Off),
                    "idle" => Some(Park::Idle),
                    a => a.parse::<u64>().ok().map(Park::Always),
                };
                if let Some(p) = p {
                    unsafe { smp::PARK = p };
                    smp::park_phase(false);
                }
                let desc = match unsafe { smp::PARK } {
                    Park::Off => "off (always spin)",
                    Park::Idle => "idle (spin during a generation, park between requests)",
                    Park::Always(_) => "always (park after the spin window, also during a generation)",
                };
                println!("[park] {}", desc);
            }
            "/temp" => {
                if let Ok(t) = arg.parse::<f32>() {
                    unsafe { TEMP_BITS = if t > 0.0 { t.to_bits() } else { 0 } };
                }
                println!("[sample] temp bits {:#x} ({}), top-p 0.95, seed 42", unsafe { TEMP_BITS },
                    if unsafe { TEMP_BITS } == 0 { "greedy" } else { "sampling" });
            }
            "/stats" => println!(
                "[stats] threads {}, parked now {}, wake IPIs {}, arena used {} MiB, chat pos {}",
                smp::threads(), smp::parked(), smp::wakeups(), crate::mem::used() >> 20, unsafe { CHAT_POS }
            ),
            "/help" => println!("{}", HELP),
            // test hook for the exception handler: read an unmapped lower-half address -> #PF, halt
            "/fault" => unsafe {
                core::ptr::read_volatile(0xdead_b000usize as *const u64);
            },
            _ => {
                // the line buffer is NUL-terminated at n; trimming only matters for commands
                run(&line[..=n], &Opts { single_turn: false, max_tok: 512, print_toks: false });
            }
        }
    }
}
