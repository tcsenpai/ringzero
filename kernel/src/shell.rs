// Shell on COM1 + PS/2. Plain text goes to the agent; `!cmd` runs a machine command (memory, CPU,
// cores, log, settings, power), `/cmd` a session command (help, clear, context, history, benches).
// Line editor: Tab completes command names and simple arguments, a bare `/` or `!` lists the
// commands, up/down walk the history. Input is polled into a type-ahead ring; Ctrl-C (0x03) sets a
// flag the generation loop checks between tokens.
use crate::log::SBuf;
use crate::{agent, console, embed, kbd, llm, memidx, smp, snap, sys};
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

fn room() -> bool {
    unsafe { TA_HEAD - TA_TAIL < 512 - 8 }
}

/// Drain COM1 and PS/2 into the type-ahead ring while it has room (bytes left in the UART are held back
/// by QEMU, so a long pasted line is never cut). Ctrl-C sets the interrupt flag instead.
fn poll_input() {
    while room() {
        let Some(b) = console::serial_getc() else { break };
        if b == 3 {
            unsafe { INTR = true };
        } else {
            push(b);
        }
    }
    while room() {
        let Some(c) = kbd::getc() else { break };
        match c {
            '\u{3}' => unsafe { INTR = true },
            kbd::UP => b"\x1b[A".iter().copied().for_each(push),
            kbd::DOWN => b"\x1b[B".iter().copied().for_each(push),
            c => {
                let mut u = [0u8; 4];
                c.encode_utf8(&mut u).bytes().for_each(push);
            }
        }
    }
}

/// Input waiting (a byte or a Ctrl-C), without taking it: ends an idle-indexer slice.
fn input_waiting() -> bool {
    poll_input();
    unsafe { TA_HEAD != TA_TAIL || INTR }
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

/// One typed byte, if any (M8: also the console stdin of a WASI program, wasm.rs).
pub fn getb() -> Option<u8> {
    poll_input();
    unsafe {
        if TA_TAIL == TA_HEAD {
            return None;
        }
        TA_TAIL += 1;
        Some(TA[(TA_TAIL - 1) % 512])
    }
}

// ---- commands ----

const BANG: u8 = 1; // `!name`: acts on or reads the machine
const SLASH: u8 = 2; // `/name`: session, agent and LLM commands

struct Cmd {
    name: &'static str,
    pre: u8,
    args: &'static [&'static str], // Tab completion of the first argument
    help: &'static str,
}

const fn c(name: &'static str, pre: u8, args: &'static [&'static str], help: &'static str) -> Cmd {
    Cmd { name, pre, args, help }
}

const SETTINGS: &[&str] = &["temperature", "cores", "keyboard", "think", "compact", "snappool", "consolidate", "slot", "constrain", "spec", "gramdraft", "confidence", "seed"];
const CMDS: &[Cmd] = &[
    c("help", BANG | SLASH, &[], "this help"),
    c("mem", BANG, &[], "memory: total, free, model, engine arena"),
    c("cpu", BANG, &[], "CPU model, clock, features"),
    c("cores", BANG, &[], "[N]  cores online; N = cores used for inference"),
    c("log", BANG, &[], "[N]  last N kernel log lines (20)"),
    c("uptime", BANG, &[], "time since boot"),
    c("kbd", BANG, &["us", "it"], "[us|it]  PS/2 keyboard layout"),
    c("set", BANG, SETTINGS, "<key> <value>  temperature, cores, keyboard, think, compact (percent), snappool (MiB), consolidate (idle s, 0 = off), slot (tokens, 0 = off), constrain (on|off: tool-call grammar), spec (on|off|N: speculative decoding, N = longest draft), gramdraft (on|off: tool-call text the grammar implies as a draft), confidence (on|off: margin of a tool call's decision tokens, low ones ask first), seed (sampler RNG)"),
    c("park", BANG, &["off", "idle"], "off|idle|<usec>  worker core parking"),
    c("stats", BANG | SLASH, &[], "threads, parking, arena, context"),
    c("ls", BANG, &[], "[PATH]  list a directory on the disk (default /)"),
    c("cat", BANG, &[], "PATH  print a file"),
    c("write", BANG, &[], "PATH TEXT  create or replace a file (\\n in TEXT = new line)"),
    c("mkdir", BANG, &[], "PATH  create a directory (and missing parents)"),
    c("rm", BANG, &["-r"], "[-r] PATH  delete a file, or a directory with -r (asks y/N)"),
    c("df", BANG, &[], "disk size and free space"),
    c("sync", BANG, &[], "flush the disk (poweroff and reboot do it too)"),
    c("py", BANG, &["-f"], "CODE | -f PATH  run MicroPython (one line echoes values; 10 s budget, Ctrl-C stops it)"),
    c("wasm", BANG, &[], "PATH|NAME [ARGS] [< FILE]  run a WASI program (NAME = /bin/NAME.wasm, else a coreutils tool; 60 s budget, Ctrl-C stops it)"),
    c("run", BANG, &[], "NAME [ARGS] [< FILE]  same as !wasm"),
    c("poweroff", BANG, &[], "power off now"),
    c("reboot", BANG, &[], "reboot now"),
    c("clear", SLASH, &[], "clear the screen, new conversation"),
    c("ctx", SLASH, &[], "context fill (tokens in use / 32768)"),
    c("compact", SLASH, &[], "summarize the conversation and rebuild the context now"),
    c("history", SLASH, &[], "[N]  raw transcript, kept across compactions (40)"),
    c("think", SLASH, &["on", "off"], "on|off  model thinking"),
    c("temp", SLASH, &[], "<0..2>  sampling temperature, 0 = greedy"),
    c("undo", SLASH, &[], "[N]  back N turns (the model's state only, not the machine)"),
    c("redo", SLASH, &[], "[N]  forward again after an undo"),
    c("retry", SLASH, &[], "ask the last question again (temp 0: the same answer)"),
    c("tree", SLASH, &["all"], "[all]  points, branches and saves"),
    c("branch", SLASH, &[], "NAME  start a branch here and switch to it"),
    c("switch", SLASH, &[], "NAME|#ID  go to a branch, a save or a point"),
    c("save", SLASH, &[], "NAME  keep this point (never evicted)"),
    c("load", SLASH, &[], "NAME  go to a saved point"),
    c("drop", SLASH, &[], "NAME|#ID  delete a branch, a save or a point"),
    c("snap", SLASH, &[], "snapshot status: point, pool, timings"),
    c("hibernate", SLASH, &["discard", "status"], "[discard|status]  save the conversation and its points to the disk (!poweroff does it too); the next boot resumes it"),
    c("agentd", SLASH, &["log"], "[log [N]]  memory subsystem: stores, index, queue, consolidation, decisions"),
    c("facts", SLASH, &[], "saved facts (key = value, source, time)"),
    c("remember", SLASH, &[], "KEY VALUE  save a fact"),
    c("forget", SLASH, &[], "KEY  delete a fact (asks y/N)"),
    c("consolidate", SLASH, &[], "summarize this session into an episode and facts now"),
    c("embed", SLASH, &["-q", "-v"], "[-q] [-v] TEXT  embed TEXT (-q: as a query, -v: print all values)"),
    c("index", SLASH, &["add", "stats", "clear", "dim"], "add TEXT | stats | clear (rebuild) | dim N  the memory index"),
    c("search", SLASH, &[], "QUERY  top 5 in the memory index (embedding + gated BM25)"),
    c("man", SLASH, &[], "[TOPIC|check]  search the built-in manual (alone: list its sections; check: baked vectors vs a fresh embedding)"),
    c("bench", SLASH, &[], "64-token decode bench (the conversation is restored after)"),
    c("fill", SLASH, &[], "<tokens>  prefill + decode speed at that context fill"),
    c("gen", SLASH, &[], "<n> <text>  debug: greedy ids + logit margins"),
    c("prefill", SLASH, &["avx2", "vnni"], "[B] [avx2|vnni]  prompt block (1 = per token, 64 max) and matmul kernel: time the boot prefix + state hash"),
    c("spec", SLASH, &[], "[N]  debug: speculative verifier logits for N positions (16) vs one transformer_forward each, and their times"),
];

fn pre_of(b: u8) -> u8 {
    match b {
        b'!' => BANG,
        b'/' => SLASH,
        _ => 0,
    }
}

fn find(pre: u8, name: &str) -> Option<&'static Cmd> {
    CMDS.iter().find(|c| c.pre & pre != 0 && c.name == name)
}

fn help() {
    println!("Type a message to talk to the system agent (Qwen3.5 in ring 0, with tools; Qwen3-Embedding-0.6B for /embed,");
    println!("/index, /search and its recall tool). Ctrl-C stops the agent. agentd keeps facts, episodes and a journal on");
    println!("the disk (/sys/mem) and puts saved facts in front of the first message of every conversation.");
    for (pre, label) in [(BANG, "!  machine commands"), (SLASH, "/  session commands")] {
        println!("{}:", label);
        for c in CMDS.iter().filter(|c| c.pre & pre != 0) {
            let p = if pre == BANG { '!' } else { '/' };
            println!("  {}{:<9} {}", p, c.name, c.help);
        }
    }
    println!("Tab completes commands and arguments, a bare / or ! (pause or Enter) lists them, up/down recall history.");
    println!("Every finished turn is saved as a point (#7). The prompt shows the branch, [main], and [main~1] after");
    println!("/undo; /redo steps forward again. Typing after an undo starts a side line, the old one stays in /tree.");
    println!("/save NAME keeps a point, /branch NAME starts a line. Only the model's memory goes back in time:");
    println!("changes to the machine (settings) stay.");
}

// ---- line editor ----

static mut HIST: [[u8; 1024]; 16] = [[0; 1024]; 16];
static mut HIST_LEN: [u16; 16] = [0; 16];
static mut HIST_N: usize = 0; // lines ever stored

fn erase(buf: &[u8]) {
    for &b in buf {
        if b & 0xC0 != 0x80 {
            console::write_bytes(b"\x08 \x08");
        }
    }
}

fn redraw(prompt: &str, buf: &[u8]) {
    print!("\n{}", prompt);
    console::write_bytes(buf);
}

/// Candidates for the word being typed: command names (no space yet) or the first argument.
/// Calls f(candidate) for each; returns (count, start of the word in buf).
fn candidates(buf: &[u8], mut f: impl FnMut(&'static str)) -> (usize, usize) {
    let Ok(s) = core::str::from_utf8(buf) else { return (0, 0) };
    let pre = pre_of(buf[0]);
    let mut n = 0;
    match s[1..].split_once(' ') {
        None => {
            for c in CMDS.iter().filter(|c| c.pre & pre != 0 && c.name.starts_with(&s[1..])) {
                f(c.name);
                n += 1;
            }
            (n, 1)
        }
        Some((cmd, a)) if !a.contains(' ') => {
            for x in find(pre, cmd).map(|c| c.args).unwrap_or(&[]).iter().filter(|x| x.starts_with(a)) {
                f(x);
                n += 1;
            }
            (n, s.len() - a.len())
        }
        _ => (0, 0),
    }
}

fn suggest(prompt: &str, buf: &[u8]) {
    print!("\n ");
    candidates(buf, |x| print!(" {}{}", if buf.contains(&b' ') { "" } else { core::str::from_utf8(&buf[..1]).unwrap_or("") }, x));
    redraw(prompt, buf);
}

/// Tab: complete the longest common prefix; one match also gets a trailing space; several print a list.
fn complete(prompt: &str, buf: &mut [u8], n: &mut usize) {
    if *n == 0 || pre_of(buf[0]) == 0 {
        return;
    }
    let mut first: Option<&str> = None;
    let mut common = 0;
    let (count, start) = candidates(&buf[..*n], |x| match first {
        None => {
            first = Some(x);
            common = x.len();
        }
        Some(f) => common = common.min(f.bytes().zip(x.bytes()).take_while(|(a, b)| a == b).count()),
    });
    let Some(f) = first else { return };
    let typed = *n - start;
    fn add(buf: &mut [u8], n: &mut usize, b: &[u8]) {
        let k = b.len().min(buf.len() - *n);
        buf[*n..*n + k].copy_from_slice(&b[..k]);
        *n += k;
        console::write_bytes(&b[..k]);
    }
    if common > typed {
        add(buf, n, &f.as_bytes()[typed..common]);
    }
    if count == 1 {
        add(buf, n, b" ");
    } else if common <= typed {
        suggest(prompt, &buf[..*n]);
    }
}

/// Read a line (echoed, UTF-8) into buf. Ctrl-C cancels the line (returns 0). With a prompt: Tab
/// completion, suggestions on a bare / or !, history on up/down.
pub fn readline(buf: &mut [u8], prompt: Option<&str>) -> usize {
    let mut n = 0;
    let mut last_cr = false;
    let mut esc = 0u8; // 1 = got ESC, 2 = got ESC [ or ESC O
    let mut h = unsafe { HIST_N }; // history cursor
    let mut sug_at = 0u64; // TSC deadline: list the commands if a bare / or ! stays alone this long
    loop {
        if take_intr() {
            crate::agentd::preempt();
            println!("^C");
            return 0;
        }
        let Some(c) = getb() else {
            if sug_at != 0 && smp::rdtsc() >= sug_at {
                sug_at = 0;
                if let (Some(p), 1) = (prompt, n) {
                    suggest(p, &buf[..1]);
                }
            }
            // at the shell prompt only (not in a y/N question): one slice of agentd's idle work, ended by any input
            if prompt.is_some() && crate::agentd::idle(&mut input_waiting) {
                continue;
            }
            smp::idle_wait();
            continue;
        };
        crate::agentd::preempt(); // input: a consolidation in flight gives the engine back first
        sug_at = 0;
        if c == b'\n' && core::mem::replace(&mut last_cr, false) {
            continue; // \r\n
        }
        last_cr = c == b'\r';
        match (esc, c) {
            (0, 0x1b) => {
                esc = 1;
                continue;
            }
            (1, b'[' | b'O') => {
                esc = 2;
                continue;
            }
            (2, b'A' | b'B') => {
                esc = 0;
                if prompt.is_none() {
                    continue;
                }
                let hn = unsafe { HIST_N };
                let lo = hn.saturating_sub(16);
                let nh = if c == b'A' { h.saturating_sub(1).max(lo) } else { (h + 1).min(hn) };
                if nh == h {
                    continue;
                }
                h = nh;
                erase(&buf[..n]);
                n = 0;
                if h < hn {
                    let (l, len) = unsafe { (&HIST[h % 16], HIST_LEN[h % 16] as usize) };
                    n = len.min(buf.len());
                    buf[..n].copy_from_slice(&l[..n]);
                    console::write_bytes(&buf[..n]);
                }
                continue;
            }
            (1 | 2, _) => {
                esc = 0;
                continue; // other escape sequences: ignored
            }
            _ => {}
        }
        match c {
            b'\r' | b'\n' => {
                console::write_bytes(b"\n");
                if prompt.is_some() && n > 0 && n <= 1024 {
                    unsafe {
                        let prev = HIST_N.checked_sub(1).map(|i| &HIST[i % 16][..HIST_LEN[i % 16] as usize]);
                        if prev != Some(&buf[..n]) {
                            HIST[HIST_N % 16][..n].copy_from_slice(&buf[..n]);
                            HIST_LEN[HIST_N % 16] = n as u16;
                            HIST_N += 1;
                        }
                    }
                }
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
            b'\t' => {
                if let Some(p) = prompt {
                    complete(p, buf, &mut n);
                }
            }
            c if c >= 0x20 && n < buf.len() => {
                buf[n] = c;
                n += 1;
                console::write_bytes(&[c]);
                if prompt.is_some() && n == 1 && pre_of(c) != 0 {
                    sug_at = smp::rdtsc() + smp::us_to_tsc(400_000);
                }
            }
            _ => {}
        }
    }
}

/// Ask the human at the shell. Only an explicit y/yes confirms.
pub fn confirm(question: &str) -> bool {
    print!("{} [y/N] ", question);
    let mut b = [0u8; 16];
    let n = readline(&mut b, None);
    let a = core::str::from_utf8(&b[..n]).unwrap_or("").trim();
    let yes = a.eq_ignore_ascii_case("y") || a.eq_ignore_ascii_case("yes");
    crate::log::record(b'A', if yes { "confirmation: yes" } else { "confirmation: no" });
    yes
}

/// on/1/true or off/0/false (T52: a typo is an error, it no longer turns the setting off).
fn onoff(v: &str) -> Option<bool> {
    match v {
        "on" | "1" | "true" => Some(true),
        "off" | "0" | "false" => Some(false),
        _ => None,
    }
}

/// Settings shared by the `!set`/`!kbd`/`!cores`/`/temp`/`/think` commands and the agent's set_setting tool.
pub fn set_setting<const N: usize>(key: &str, val: &str, o: &mut SBuf<N>) -> bool {
    let ok = match key {
        "think" | "constrain" | "confidence" | "gramdraft" if onoff(val).is_none() => {
            let _ = write!(o, "error: {} must be on or off", key);
            false
        }
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
        "constrain" => {
            let on = onoff(val) == Some(true);
            unsafe { crate::gram::ON = on };
            let _ = write!(o, "constrain = {}", if on { "on (tool calls follow the tool schemas)" } else { "off (the grammar only checks tool calls)" });
            true
        }
        "confidence" => {
            let on = onoff(val) == Some(true);
            unsafe { crate::gram::CONF_ON = on };
            if on {
                let _ = write!(o, "confidence = on (a [confidence] line per tool call; a margin below {:.2} asks first on tools that change the machine)", crate::agent::CONF_LO);
            } else {
                let _ = write!(o, "confidence = off");
            }
            true
        }
        "seed" => match val.parse::<u64>() {
            Ok(r) if r != 0 => {
                llm::set_rng(r);
                let _ = write!(o, "seed = {} (sampler RNG state)", r);
                true
            }
            _ => {
                let _ = write!(o, "error: seed must be a positive integer");
                false
            }
        },
        "spec" => {
            let on = match val.parse::<usize>() {
                Ok(k) if (1..llm::SPEC_ROWS).contains(&k) => {
                    unsafe { llm::SPEC_K = k };
                    Some(true)
                }
                Ok(_) => None,
                Err(_) => onoff(val),
            };
            let Some(on) = on else {
                let _ = write!(o, "error: spec must be on, off or the longest draft from 1 to {}", llm::SPEC_ROWS - 1);
                return false;
            };
            unsafe { llm::SPEC_ON = on };
            let _ = write!(o, "spec = {}", if !unsafe { llm::SPEC_ON } { "off (one token per pass)" } else if llm::spec_ok() { "on" } else { "on, but not available on this build" });
            if unsafe { llm::SPEC_ON } {
                let _ = write!(o, " (prompt lookup, drafts up to {} tokens)", unsafe { llm::SPEC_K });
            }
            true
        }
        "gramdraft" => {
            unsafe { llm::GRAM_DRAFT = onoff(val) == Some(true) };
            let _ = write!(o, "gramdraft = {} (inside a tool call the grammar's continuation is the speculative draft)",
                if unsafe { llm::GRAM_DRAFT } { "on" } else { "off" });
            true
        }
        "think" => {
            let on = onoff(val) == Some(true);
            unsafe { llm::THINK = on as u64 };
            let _ = write!(o, "think = {}", if on { "on" } else { "off" });
            true
        }
        "snappool" => match val.parse::<usize>() {
            Ok(m) if snap::set_cap(m) => {
                let _ = write!(o, "snappool = {} MiB of {} usable for snapshots", snap::cap_mib(), snap::max_mib());
                true
            }
            _ => {
                let _ = write!(o, "error: snappool must be 64..{} MiB", snap::max_mib());
                false
            }
        },
        "slot" => match val.parse::<usize>() {
            Ok(n) if n <= 1000 => {
                unsafe { crate::agentd::SLOT_BUDGET = n };
                let _ = write!(o, "slot = {} tokens{}", n, if n == 0 { " (off: no memory slot)" } else { " for the memory slot of a new conversation" });
                true
            }
            _ => {
                let _ = write!(o, "error: slot must be a token budget from 0 (off) to 1000");
                false
            }
        },
        "consolidate" => match val.parse::<u64>() {
            Ok(n) => {
                unsafe { crate::agentd::CONS_IDLE_S = n };
                let _ = write!(o, "consolidate = {} s{}", n, if n == 0 { " (off: /consolidate only)" } else { " of idle time at the prompt" });
                true
            }
            _ => {
                let _ = write!(o, "error: consolidate must be idle seconds (0 = off)");
                false
            }
        },
        "compact" => match val.parse::<u64>() {
            Ok(p) if (1..=95).contains(&p) => {
                unsafe { agent::COMPACT_PCT = p };
                let _ = write!(o, "compact = {}% (the context is compacted when the fill reaches it)", p);
                true
            }
            _ => {
                let _ = write!(o, "error: compact must be a percent from 1 to 95");
                false
            }
        },
        _ => {
            let _ = write!(o, "error: unknown setting '{}' (", key);
            for (i, k) in SETTINGS.iter().enumerate() {
                let _ = write!(o, "{}{}", if i > 0 { ", " } else { "" }, k);
            }
            let _ = write!(o, ")");
            false
        }
    };
    if ok {
        crate::log::record(b'I', o.as_str());
    }
    ok
}

/// `/embed [-q] [-v] TEXT`
fn embed_cmd(mut a: &str) {
    let (mut q, mut v) = (false, false);
    loop {
        match a.split_once(' ') {
            Some(("-q", r)) => (q, a) = (true, r.trim_start()),
            Some(("-v", r)) => (v, a) = (true, r.trim_start()),
            _ => break,
        }
    }
    if a.is_empty() {
        println!("usage: /embed [-q] [-v] TEXT");
        return;
    }
    let mut e = [0.0f32; embed::DIM];
    smp::park_phase(true);
    let r = embed::embed(a, q, embed::DIM, &mut e);
    smp::park_phase(false);
    match r {
        Ok(i) => {
            let n = embed::sqrt(e.iter().map(|&x| x as f64 * x as f64).sum());
            print!("[embed] {} dim {}, norm {:.6} (pooled |x*w| {:.3}), {} tok in {}.{:03} ms, first:", if q { "query" } else { "doc" },
                embed::DIM, n, i.raw_norm, i.n_tok, i.us / 1000, i.us % 1000);
            e[..8].iter().for_each(|x| print!(" {:.5}", x));
            println!("");
            if v {
                print!("EMB");
                e.iter().for_each(|x| print!(" {:.7}", x));
                println!("");
            }
        }
        Err(m) => println!("[embed] {}", m),
    }
}

fn index_cmd(a: &str) {
    let (sub, rest) = a.split_once(' ').map(|(x, y)| (x, y.trim())).unwrap_or((a, ""));
    match sub {
        "add" if !rest.is_empty() => {
            smp::park_phase(true);
            let r = memidx::add(rest);
            smp::park_phase(false);
            match r {
                Ok(x) => println!("[index] #{} added ({} tok embedded in {}.{:03} ms, {} entries)", x.id, x.info.n_tok,
                    x.info.us / 1000, x.info.us % 1000, memidx::len()),
                Err(m) => println!("[index] {}", m),
            }
        }
        "stats" | "" => {
            let mut o = SBuf::<512>::new();
            memidx::stats(&mut o);
            println!("[index] {}", o.as_str());
        }
        "clear" => {
            crate::agentd::rebuild("/index clear");
            println!("[index] cleared: every file and memory record is queued again (idle time at the prompt)");
        }
        "dim" => match rest.parse::<usize>().map_err(|_| "usage: /index dim N").and_then(memidx::set_dim) {
            Ok(()) => println!("[index] dim {}", memidx::dim()),
            Err(m) => println!("[index] {}", m),
        },
        _ => println!("usage: /index add TEXT | stats | clear | dim N"),
    }
}

fn search_cmd(q: &str) {
    if q.is_empty() {
        println!("usage: /search QUERY");
        return;
    }
    let mut h = [memidx::Hit { id: 0, cos: 0.0, bm25: 0.0, via: memidx::Via::Emb }; 5];
    smp::park_phase(true);
    let r = memidx::search(q, &mut h);
    smp::park_phase(false);
    match r {
        Ok(f) => {
            for (i, x) in h[..f.n].iter().enumerate() {
                let t = memidx::text(x.id);
                let t = t[..agent::floor_char(t, 100)].replace('\n', " ");
                println!("{}. #{:<4} cos {:.3}  bm25 {:>5.2}  {:<8} {}", i + 1, x.id, x.cos, x.bm25, x.via.name(), t);
            }
            println!("[search] {} entries, dim {}, query {} tok embedded in {}.{:03} ms, search {} us, BM25 {}",
                memidx::len(), memidx::dim(), f.embed.n_tok, f.embed.us / 1000, f.embed.us % 1000, f.search_us,
                if f.gate.is_empty() { "not used (embedding order)" } else { f.gate });
        }
        Err(m) => println!("[search] {}", m),
    }
}

/// `!ls !cat !write !mkdir !rm !df !sync`
fn fs_cmd(name: &str, arg: &str) {
    use crate::{fidx, fs};
    let r: Result<(), &str> = (|| {
        match name {
            "ls" => {
                let p = if arg.is_empty() { "/" } else { arg };
                let mut o = alloc::string::String::new();
                let n = fs::ls(p, &mut o)?;
                println!("{}", if n == 0 { "(empty)" } else { o.as_str() });
            }
            "cat" => {
                let (b, size) = fs::read(arg, 0, 64 << 10)?;
                println!("{}", alloc::string::String::from_utf8_lossy(&b));
                if (b.len() as u64) < size {
                    println!("[fs] first {} of {} bytes shown", b.len(), size);
                }
            }
            "write" => {
                let (p, text) = arg.split_once(' ').unwrap_or((arg, ""));
                let p = fs::canon(p)?;
                let data = text.trim().replace("\\n", "\n");
                fs::write(&p, data.as_bytes())?;
                fidx::enqueue(&p, data.as_bytes());
                println!("[fs] wrote {} bytes to {}{}", data.len(), p, if fidx::pending() { " (queued for the memory index)" } else { "" });
            }
            "mkdir" => {
                let p = fs::canon(arg)?;
                println!("[fs] {} {}", p, if fs::mkdir(&p)? { "created" } else { "already exists" });
            }
            "rm" => {
                let (rec, p) = match arg.split_once(' ') {
                    Some(("-r", p)) => (true, p.trim()),
                    _ => (false, arg),
                };
                let p = fs::canon(p)?;
                let (dir, k) = fs::stat(&p)?;
                if dir && k > 0 && !rec {
                    return Err("directory is not empty (use !rm -r)");
                }
                let mut q = SBuf::<320>::new();
                let _ = if dir { write!(q, "delete directory {}{}?", p, if k > 0 { alloc::format!(" and the {} entries inside", k) } else { alloc::string::String::new() }) }
                    else { write!(q, "delete file {} ({} B)?", p, k) };
                if confirm(q.as_str()) {
                    fs::rm(&p, rec)?;
                    let n = fidx::forget(&p);
                    println!("[fs] deleted {}{}", p, if n > 0 { alloc::format!(" ({} index chunks dropped)", n) } else { alloc::string::String::new() });
                } else {
                    println!("[fs] kept {}", p);
                }
            }
            "df" => {
                let mut o = SBuf::<256>::new();
                fs::df(&mut o);
                println!("[fs] {}", o.as_str());
            }
            _ => {
                fs::sync()?;
                println!("[fs] synced");
            }
        }
        Ok(())
    })();
    if let Err(e) = r {
        println!("[fs] {}: {}", name, e);
    }
}

fn show(f: fn(&mut SBuf<1024>)) {
    let mut o = SBuf::<1024>::new();
    f(&mut o);
    println!("{}", o.as_str());
}

pub fn run() -> ! {
    println!("ringzero shell. Type a message for the system agent, !cmd for the machine, /cmd for the session. /help lists them.");
    static mut LINE: [u8; 16384] = [0; 16384];
    loop {
        let mut prompt = SBuf::<64>::new();
        snap::prompt(&mut prompt);
        print!("\n{}", prompt.as_str());
        let line = unsafe { &mut *core::ptr::addr_of_mut!(LINE) };
        let n = readline(line, Some(prompt.as_str()));
        let s = core::str::from_utf8(&line[..n]).unwrap_or("").trim();
        if s.is_empty() {
            continue;
        }
        let pre = pre_of(s.as_bytes()[0]);
        if pre == 0 {
            agent::ask(s.strip_prefix('@').map(str::trim).unwrap_or(s)); // `@ text` (M3) still works
            continue;
        }
        let body = s[1..].trim_start();
        let (cmd, arg) = body.split_once(' ').map(|(a, b)| (a, b.trim())).unwrap_or((body, ""));
        if cmd.is_empty() {
            print!("commands:");
            candidates(&s.as_bytes()[..1], |x| print!(" {}{}", &s[..1], x));
            println!("");
            continue;
        }
        if cmd == "toks" {
            agent::dump_prefix_tokens(); // hidden: prefix ids for tools/hf_prefix.py
            continue;
        }
        if cmd == "fault" {
            // hidden test hook for the exception handler: read an unmapped lower-half address -> #PF, halt
            unsafe { core::ptr::read_volatile(0xdead_b000usize as *const u64) };
        }
        let Some(c) = find(pre, cmd) else {
            let other = if pre == BANG { SLASH } else { BANG };
            match find(other, cmd) {
                Some(_) => println!("'{}' is a {} command: try {}{}", cmd, if other == BANG { "machine" } else { "session" },
                    if other == BANG { '!' } else { '/' }, cmd),
                None => println!("unknown command '{}'. Type /help, or type without {} to ask the agent.", s, &s[..1]),
            }
            continue;
        };
        let mut o = SBuf::<1024>::new();
        match c.name {
            "help" => help(),
            "clear" => {
                console::clear_screen();
                agent::reset();
                println!("[agent] new conversation from the boot prefix (the old one stays in /tree)");
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
            "ctx" => agent::ctx_info(),
            "compact" => {
                agent::compact("manual");
                agent::ctx_info();
            }
            "history" => agent::history(arg.parse::<usize>().unwrap_or(40)),
            "bench" => {
                snap::invalidate(); // the bench runs from position 0
                llm::bench();
                agent::resume();
            }
            "fill" => {
                snap::invalidate();
                llm::bench_fill(arg.parse::<u64>().unwrap_or(1024));
                agent::resume();
            }
            "prefill" => {
                snap::invalidate(); // the bench prefills from position 0
                llm::prefill_bench(arg, agent::prefix_tokens());
                agent::resume();
            }
            "spec" => {
                snap::invalidate(); // the check runs from position 0
                llm::spec_check(&agent::prefix_tokens()[..200], arg.parse::<usize>().unwrap_or(llm::SPEC_ROWS));
                agent::resume();
            }
            "gen" => {
                let (k, text) = arg.split_once(' ').unwrap_or(("", ""));
                match k.parse::<u64>() {
                    Ok(k) if !text.trim().is_empty() => {
                        snap::invalidate();
                        llm::gen_debug(k, text.trim());
                        agent::resume();
                    }
                    _ => println!("usage: /gen <max tokens> <prompt text>"),
                }
            }
            "stats" => println!(
                "[stats] threads {}, parked now {}, wake IPIs {}, arena used {} MiB, heap used {} KiB, agent ctx {}/{} ({}%), KV {}, kbd {}",
                smp::threads(), smp::parked(), smp::wakeups(), crate::mem::used() >> 20, crate::mem::heap_used() >> 10, agent::pos(), llm::ctx_max_seq(),
                agent::pos() * 100 / llm::ctx_max_seq().max(1), llm::kv_type(), kbd::layout()
            ),
            "kbd" | "set" | "temp" | "think" => {
                let (k, v) = match c.name {
                    "set" => arg.split_once(' ').map(|(a, b)| (a, b.trim())).unwrap_or((arg, "")),
                    n => (n, arg),
                };
                if c.name == "kbd" && v.is_empty() {
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
            "agentd" => {
                if let Some(r) = arg.strip_prefix("log") {
                    crate::agentd::log_last(r.trim().parse::<usize>().unwrap_or(30));
                } else {
                    crate::agentd::status();
                    crate::agentd::log_last(10);
                }
            }
            "facts" => crate::agentd::list_facts(),
            "remember" => match arg.split_once(' ') {
                Some((k, v)) => match crate::agentd::remember(k, v, "user", false) {
                    Ok(m) | Err(m) => println!("[facts] {}", m),
                },
                None => println!("usage: /remember KEY VALUE"),
            },
            "forget" if arg.is_empty() => println!("usage: /forget KEY"),
            "forget" => {
                let mut q = SBuf::<96>::new();
                let _ = write!(q, "forget the fact '{}'?", arg);
                if confirm(q.as_str()) {
                    match crate::agentd::forget(arg) {
                        Ok(m) | Err(m) => println!("[facts] {}", m),
                    }
                } else {
                    println!("[facts] kept");
                }
            }
            "consolidate" => crate::agentd::consolidate_now(),
            "embed" => embed_cmd(arg),
            "index" => index_cmd(arg),
            "search" => search_cmd(arg),
            "man" => {
                smp::park_phase(true);
                crate::agentd::man(arg);
                smp::park_phase(false);
            }
            "undo" => snap::undo(arg.parse::<usize>().unwrap_or(1)),
            "redo" => snap::redo(arg.parse::<usize>().unwrap_or(1)),
            "retry" => agent::retry(),
            "tree" => snap::tree(arg == "all"),
            "snap" => snap::status(),
            "hibernate" => crate::hib::cmd(arg),
            "branch" | "save" | "switch" | "load" | "drop" if arg.is_empty() => println!("usage: /{} NAME", c.name),
            "branch" => snap::branch(arg),
            "save" => snap::save_cmd(arg),
            "switch" | "load" => snap::switch(arg, c.name),
            "drop" => snap::drop_cmd(arg),
            "ls" | "cat" | "write" | "mkdir" | "rm" | "df" | "sync" => fs_cmd(c.name, arg),
            "py" => crate::upy::shell_cmd(arg), // M5
            "wasm" | "run" => crate::wasm::shell_cmd(arg), // M5b
            "poweroff" => sys::poweroff(),
            "reboot" => sys::reboot(),
            _ => {}
        }
    }
}
