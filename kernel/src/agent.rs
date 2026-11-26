// The system agent: Qwen3.5 native tool calling in ring 0. The system prompt + tool list is prefilled
// once at boot and snapshotted (llm::snapshot_*); each `@` turn streams the answer, parses a tool
// call if the model makes one, validates it against the tool schema, runs it, feeds the result back
// as a <tool_response> turn and loops (at most MAX_ROUNDS tool rounds per user turn).
// Context management (M2.5): when the fill crosses COMPACT_PCT (or on /compact) the model writes a short
// summary, the boot snapshot is restored (it ends just before the system turn's <|im_end|>), the summary
// is appended to the system prompt and the last exchanges are prefilled again (DeltaNet state cannot be
// cropped, so the context is rebuilt). The raw transcript stays in HIST.
use crate::engine::{CAP, CAPTURE};
use crate::log::SBuf;
use crate::{klog, llm, print, println, shell, smp, sys};
use core::fmt::Write;

const SYSTEM: &str = "You are the system agent of ringzero, an experimental operating system with its own Rust \
kernel (not Linux). You run inside the kernel (ring 0) and answer the user at the shell. Use the tools to read real data from this machine \
or to act on it, and never invent numbers. Reply in English or in Italian, matching the language of the \
user's latest message, briefly, in plain text without markdown headers.";

// JSON lines exactly as the HF template's `tool | tojson` renders them (", " and ": " separators).
const TOOLS: [&str; 8] = [
    r#"{"type": "function", "function": {"name": "meminfo", "description": "Memory of this machine: free, used and total RAM in MiB.", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "cpuinfo", "description": "CPU model, number of cores, clock and features.", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "list_cores", "description": "List the CPU cores and what each one does.", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "read_log", "description": "Last n lines of the kernel log.", "parameters": {"type": "object", "properties": {"n": {"type": "integer", "description": "number of lines, default 10"}}}}}"#,
    r#"{"type": "function", "function": {"name": "uptime", "description": "Time since boot.", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "poweroff", "description": "Shut down the machine (asks the user to confirm).", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "reboot", "description": "Reboot the machine (asks the user to confirm).", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "set_setting", "description": "Change a system setting. temperature: sampling temperature, 0 = greedy. cores: CPU cores used for inference. keyboard: PS/2 keyboard layout, us or it.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "enum": ["temperature", "cores", "keyboard"]}, "value": {"type": "string"}}, "required": ["key", "value"]}}}"#,
];

enum Ty {
    Int,
    Str(&'static [&'static str]), // allowed values, empty = any
}

// (name, params (name, type, required)), same order and content as TOOLS
const SPECS: [(&str, &[(&str, Ty, bool)]); 8] = [
    ("meminfo", &[]),
    ("cpuinfo", &[]),
    ("list_cores", &[]),
    ("read_log", &[("n", Ty::Int, false)]),
    ("uptime", &[]),
    ("poweroff", &[]),
    ("reboot", &[]),
    ("set_setting", &[("key", Ty::Str(&["temperature", "cores", "keyboard"]), true), ("value", Ty::Str(&[]), true)]),
];

const TEMPLATE_TOOLS_HEAD: &str = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TEMPLATE_TOOLS_TAIL: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

const MAX_ROUNDS: usize = 4;
const MAX_ANSWER: u64 = 512;
/// Tool results longer than this are cut, with a marker the model can see.
const RES_CAP: usize = 8192;
const SUMMARY_MAX: u64 = 300;
/// Token budget for the exchanges re-prefilled verbatim after a compaction.
const KEEP_BUDGET: usize = 3000;
const KEEP: usize = 2;
const COMPACT_ASK: &str = "[kernel] The context is almost full and is about to be compacted. Write a short note for \
yourself. First a line starting with 'User:' with every fact the user told you (name, numbers, preferences), \
including those in the earlier summary of the system prompt if there is one. Then 'Tasks:' with open tasks, then \
'Tool results:' with results that still matter. At most 120 words, plain text.";
/// Appended to the system prompt after a compaction (the snapshot ends just before its <|im_end|>).
const SUMMARY_HEAD: &str = "\n\nSummary of the earlier part of this session (older turns were dropped to free \
context; these are facts you noted yourself, trust them):\n";

static mut POS: u64 = 0;
static mut FRESH: bool = true;
static mut TEXT: SBuf<32768> = SBuf::new();
static mut TOKS: [u32; 16384] = [0; 16384];
static mut RES: SBuf<{ RES_CAP + 512 }> = SBuf::new();
pub static mut COMPACT_PCT: u64 = 75;
static mut N_COMPACT: u64 = 0;
static mut EX_AT_COMPACT: usize = 0; // EX_N at the last compaction (auto needs new exchanges past the kept ones)
/// "<|im_end|>\n" closing the system turn: not in the snapshot, fed when a conversation starts fresh.
static mut SYS_END: [u32; 2] = [0; 2];

fn text() -> &'static mut SBuf<32768> {
    unsafe { &mut *core::ptr::addr_of_mut!(TEXT) }
}
fn toks() -> &'static mut [u32; 16384] {
    unsafe { &mut *core::ptr::addr_of_mut!(TOKS) }
}

pub fn pos() -> u64 {
    unsafe { POS }
}

/// Next turn starts a fresh conversation from the boot snapshot.
pub fn reset() {
    unsafe {
        if !FRESH {
            hist("kernel", "--- new conversation ---");
        }
        FRESH = true;
        EX_N = 0;
        EX_AT_COMPACT = 0;
    }
}


// ---- raw transcript (memory only): "role: text" lines, oldest dropped when full ----

static mut HIST: SBuf<65536> = SBuf::new();

fn hist(role: &str, s: &str) {
    let h = unsafe { &mut *core::ptr::addr_of_mut!(HIST) };
    let s = &s[..floor_char(s, 16384)];
    let need = role.len() + s.len() + 3;
    if h.n + need > h.b.len() {
        let mut cut = (need + h.b.len() / 4).min(h.n);
        while cut < h.n && h.b[cut - 1] != b'\n' {
            cut += 1;
        }
        h.b.copy_within(cut..h.n, 0);
        h.n -= cut;
    }
    let _ = write!(h, "{}: {}\n", role, s);
}

fn floor_char(s: &str, max: usize) -> usize {
    let mut k = s.len().min(max);
    while !s.is_char_boundary(k) {
        k -= 1;
    }
    k
}

/// `/history [N]`: the last N transcript lines (default 40), kept across compactions.
pub fn history(n: usize) {
    let h = unsafe { (*core::ptr::addr_of!(HIST)).as_str() };
    let total = h.lines().count();
    h.lines().skip(total.saturating_sub(n)).for_each(|l| println!("{}", l));
    println!("[history] {} of {} lines, {} bytes", n.min(total), total, h.len());
}

/// `/ctx`: context fill.
pub fn ctx_info() {
    let (p, m) = (pos(), llm::ctx_max_seq());
    let fresh = unsafe { FRESH };
    println!("[ctx] {}/{} tokens ({}%){}, boot prefix {} tok, KV {}, compaction at {}%, {} compaction(s) so far, {} exchange(s) kept for re-prefill",
        p, m, p * 100 / m, if fresh { " (fresh: next turn restores the prefix)" } else { "" },
        llm::snapshot_pos(), llm::kv_type(), unsafe { COMPACT_PCT }, unsafe { N_COMPACT }, unsafe { EX_N.min(KEEP) });
}

// ---- last exchanges (question + final answer), re-prefilled after a compaction ----

struct Ex {
    q: SBuf<4096>,
    a: SBuf<4096>,
}
static mut EX: [Ex; KEEP] = [Ex { q: SBuf::new(), a: SBuf::new() }, Ex { q: SBuf::new(), a: SBuf::new() }];
static mut EX_N: usize = 0; // exchanges recorded since the last reset

fn push_ex(q: &str, a: &str) {
    unsafe {
        let e = &mut (*core::ptr::addr_of_mut!(EX))[EX_N % KEEP];
        e.q.clear();
        e.a.clear();
        let _ = write!(e.q, "{}", q);
        let _ = write!(e.a, "{}", a);
        EX_N += 1;
    }
}

/// Render the last k exchanges as chat turns (oldest first).
fn render_ex(t: &mut SBuf<32768>, k: usize) {
    unsafe {
        for i in EX_N - k..EX_N {
            let e = &(*core::ptr::addr_of!(EX))[i % KEEP];
            let _ = write!(t, "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n",
                e.q.as_str(), e.a.as_str());
        }
    }
}

/// Summarize, restore the boot prefix, prefill the summary and the last exchanges. false = interrupted
/// or no room (the conversation is reset).
pub fn compact(why: &str) -> bool {
    if unsafe { FRESH } {
        println!("[compact] nothing to compact (fresh conversation)");
        return false;
    }
    let max_seq = llm::ctx_max_seq();
    let before = pos();
    let t = text();
    t.clear();
    let _ = write!(t, "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", COMPACT_ASK);
    let n = llm::encode(&t.b[..t.n], toks());
    if before + n as u64 + SUMMARY_MAX + 8 >= max_seq {
        println!("[compact] no room left to summarize, starting a fresh conversation");
        reset();
        return false;
    }
    println!("[compact] context {}/{} ({}%, {}): summarizing...", before, max_seq, before * 100 / max_seq, why);
    smp::park_phase(true);
    let t0 = smp::rdtsc();
    let mut pos = before;
    unsafe {
        (*core::ptr::addr_of_mut!(CAP)).clear();
        CAPTURE = true;
    }
    let fed = llm::feed(&toks()[..n], &mut pos);
    let (stop, n_sum) = if fed { llm::generate(&mut pos, SUMMARY_MAX) } else { (llm::Stop::Intr, 0) };
    unsafe { CAPTURE = false };
    println!("");
    let t1 = smp::rdtsc();
    if stop == llm::Stop::Intr {
        println!("^C [compact] interrupted, conversation reset");
        smp::park_phase(false);
        reset();
        return false;
    }
    let mut sum = SBuf::<4096>::new();
    let cap = unsafe { (*core::ptr::addr_of!(CAP)).as_str() };
    let cap = cap.rsplit("</think>").next().unwrap_or(cap);
    let cap = cap.split("<tool_call>").next().unwrap_or(cap).trim();
    let _ = write!(sum, "{}", cap);

    unsafe { POS = llm::snapshot_restore() }; // system turn still open: the summary is appended to it
    let ex_avail = unsafe { EX_N.min(KEEP) };
    let mut n = 0;
    for k in (0..=ex_avail).rev() {
        t.clear();
        let _ = write!(t, "{}{}<|im_end|>\n", SUMMARY_HEAD, sum.as_str());
        render_ex(t, k);
        n = llm::encode(&t.b[..t.n], toks());
        if k == 0 || n <= KEEP_BUDGET {
            if k < ex_avail {
                println!("[compact] kept {} of {} recent exchange(s) (token budget {})", k, ex_avail, KEEP_BUDGET);
            }
            break;
        }
    }
    pos = unsafe { POS };
    let ok = llm::feed(&toks()[..n], &mut pos);
    let t2 = smp::rdtsc();
    smp::park_phase(false);
    if !ok {
        println!("^C [compact] interrupted during re-prefill, conversation reset");
        reset();
        return false;
    }
    unsafe {
        POS = pos;
        N_COMPACT += 1;
        EX_AT_COMPACT = EX_N;
    }
    let (us1, us2) = (smp::tsc_to_us(t1 - t0), smp::tsc_to_us(t2 - t1));
    // one log line (the ring keeps 128 bytes per line)
    klog!(b'A', "compact #{} {}: {}->{} tok ({}%->{}%), summary {} tok {} ms, re-prefill {} tok {} ms, total {} ms",
        unsafe { N_COMPACT }, why, before, pos, before * 100 / max_seq, pos * 100 / max_seq, n_sum, us1 / 1000, n, us2 / 1000, (us1 + us2) / 1000);
    hist("kernel", "[compaction] summary written by the model:");
    hist("summary", sum.as_str());
    true
}

fn build_prefix() -> usize {
    let t = text();
    t.clear();
    t.push(TEMPLATE_TOOLS_HEAD.as_bytes());
    for j in TOOLS {
        t.push(b"\n");
        t.push(j.as_bytes());
    }
    t.push(TEMPLATE_TOOLS_TAIL.as_bytes());
    let _ = write!(t, "\n\n{}<|im_end|>\n", SYSTEM);
    llm::encode(&t.b[..t.n], toks())
}

/// Boot: prefill the system prompt + tools from position 0 and snapshot the engine state.
pub fn init() {
    let n = build_prefix();
    let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV-1a of the ids, to compare with the HF tokenizer
    toks()[..n].iter().for_each(|&x| h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
    klog!(b'I', "agent: system prompt + {} tools = {} tokens (fnv {:#x}), prefilling once...", TOOLS.len(), n, h);
    smp::park_phase(true);
    let t0 = smp::rdtsc();
    unsafe { SYS_END = [toks()[n - 2], toks()[n - 1]] }; // <|im_end|> \n of the system turn
    let mut pos = 0;
    while !llm::feed(&toks()[..n - 2], &mut pos) {
        pos = 0; // Ctrl-C at boot: no shell yet, prefill again
    }
    let t1 = smp::rdtsc();
    let bytes = llm::snapshot_save(pos);
    let t2 = smp::rdtsc();
    smp::park_phase(false);
    let us = smp::tsc_to_us(t1 - t0);
    klog!(b'I', "agent: prefix prefill {} tok in {}.{:03} s ({} tok/s), snapshot {} MiB saved in {} ms",
        pos, us / 1_000_000, us / 1000 % 1000, n as u64 * 1_000_000 / us.max(1), bytes >> 20, smp::tsc_to_us(t2 - t1) / 1000);
    unsafe { FRESH = true };
}

/// Hidden `toks` builtin: print the prefix token ids (checked against the HF tokenizer offline).
pub fn dump_prefix_tokens() {
    let n = build_prefix();
    for (i, t) in toks()[..n].iter().enumerate() {
        print!("{}{}", if i > 0 { "," } else { "" }, t);
    }
    println!("");
    unsafe { FRESH = true }; // nothing in the engine changed, but keep the contract simple
}

struct Call<'a> {
    name: &'a str,
    args: [(&'a str, &'a str); 4],
    n: usize,
}

/// Lenient parse of the first <tool_call>: closing </parameter>, </function> and </tool_call> may be
/// missing. None = no tool call in the text.
fn parse(s: &str) -> Option<Result<Call<'_>, &'static str>> {
    let s = s.rsplit("</think>").next().unwrap_or(s);
    let body = &s[s.find("<tool_call>")? + 11..];
    let body = body.split("</tool_call>").next().unwrap_or(body);
    let Some(f) = body.find("<function=") else { return Some(Err("missing <function=NAME>")) };
    let rest = &body[f + 10..];
    let Some(e) = rest.find('>') else { return Some(Err("unterminated <function=NAME>")) };
    let mut c = Call { name: rest[..e].trim(), args: [("", ""); 4], n: 0 };
    let mut rest = &rest[e + 1..];
    while let Some(p) = rest.find("<parameter=") {
        let r = &rest[p + 11..];
        let Some(e) = r.find('>') else { return Some(Err("unterminated <parameter=KEY>")) };
        let key = r[..e].trim();
        let v = &r[e + 1..];
        let end = ["</parameter>", "<parameter=", "</function>"].iter().filter_map(|t| v.find(t)).min().unwrap_or(v.len());
        if c.n == c.args.len() {
            return Some(Err("too many parameters"));
        }
        c.args[c.n] = (key, v[..end].trim());
        c.n += 1;
        rest = &v[end..];
    }
    Some(Ok(c))
}

/// Check a call against SPECS. Ok(spec index) or an error for the model.
fn validate<const N: usize>(c: &Call, err: &mut SBuf<N>) -> Option<usize> {
    let Some(i) = SPECS.iter().position(|(n, _)| *n == c.name) else {
        let _ = write!(err, "error: unknown tool '{}'", c.name);
        return None;
    };
    let params = SPECS[i].1;
    for (k, v) in &c.args[..c.n] {
        let Some((_, ty, _)) = params.iter().find(|(p, _, _)| p == k) else {
            let _ = write!(err, "error: {} has no parameter '{}'", c.name, k);
            return None;
        };
        let bad = match ty {
            Ty::Int => v.parse::<i64>().is_err(),
            Ty::Str(allowed) => !allowed.is_empty() && !allowed.contains(v),
        };
        if bad {
            let _ = write!(err, "error: bad value '{}' for {}.{}", v, c.name, k);
            return None;
        }
    }
    if let Some((p, _, _)) = params.iter().find(|(p, _, req)| *req && !c.args[..c.n].iter().any(|(k, _)| k == p)) {
        let _ = write!(err, "error: {} needs parameter '{}'", c.name, p);
        return None;
    }
    Some(i)
}

fn arg<'a>(c: &Call<'a>, k: &str) -> Option<&'a str> {
    c.args[..c.n].iter().find(|(a, _)| *a == k).map(|(_, v)| *v)
}

fn exec<const N: usize>(c: &Call, o: &mut SBuf<N>) {
    match c.name {
        "meminfo" => sys::meminfo(o),
        "cpuinfo" => sys::cpuinfo(o),
        "list_cores" => sys::cores(o),
        "uptime" => sys::uptime(o),
        "read_log" => {
            let k = arg(c, "n").and_then(|v| v.parse::<usize>().ok()).unwrap_or(10).clamp(1, 256);
            let mut first = true;
            crate::log::last(k, |l| {
                if !core::mem::replace(&mut first, false) {
                    o.push(b"\n");
                }
                o.push(l.as_bytes());
            });
        }
        "poweroff" | "reboot" => {
            let what = if c.name == "poweroff" { "POWER OFF" } else { "REBOOT" };
            let mut q = SBuf::<128>::new();
            let _ = write!(q, "\n[agent] wants to {} the machine. Proceed?", what);
            if shell::confirm(q.as_str()) {
                if c.name == "poweroff" { sys::poweroff() } else { sys::reboot() }
            }
            let _ = write!(o, "cancelled: the user answered no, the machine stays on");
        }
        "set_setting" => {
            shell::set_setting(arg(c, "key").unwrap_or(""), arg(c, "value").unwrap_or(""), o);
        }
        _ => {}
    }
}

/// One user turn at the shell.
pub fn ask(q: &str) {
    let max_seq = llm::ctx_max_seq();
    let mut l = SBuf::<160>::new();
    let _ = write!(l, "user: {}", q);
    crate::log::record(b'A', l.as_str());
    unsafe {
        if !FRESH && POS * 100 >= max_seq * COMPACT_PCT && EX_N - EX_AT_COMPACT > KEEP {
            compact("auto");
        }
        if !FRESH && POS + q.len() as u64 + 300 >= max_seq {
            println!("[agent] context full ({}/{}), starting a fresh conversation", pos(), max_seq);
            reset();
        }
        if FRESH {
            let t0 = smp::rdtsc();
            POS = llm::snapshot_restore();
            for t in SYS_END {
                crate::engine::transformer_forward(t, POS, 0); // close the system turn
                POS += 1;
            }
            FRESH = false;
            klog!(b'A', "[agent] fresh conversation: prefix ({} tok) restored in {} us", pos(), smp::tsc_to_us(smp::rdtsc() - t0));
        }
    }
    hist("user", q);
    let t = text();
    t.clear();
    let think = unsafe { llm::THINK } != 0;
    let gen_prompt = if think { "<|im_start|>assistant\n<think>\n" } else { "<|im_start|>assistant\n<think>\n\n</think>\n\n" };
    let _ = write!(t, "<|im_start|>user\n{}<|im_end|>\n{}", q, gen_prompt);
    smp::park_phase(true);
    for round in 0..=MAX_ROUNDS {
        let n = llm::encode(&t.b[..t.n], toks());
        let mut pos = unsafe { POS };
        if pos + n as u64 + 64 >= max_seq {
            println!("[agent] context full, type 'clear' to start over");
            break;
        }
        unsafe {
            (*core::ptr::addr_of_mut!(CAP)).clear();
            CAPTURE = true;
        }
        crate::engine::OUT_TSC.store(0, core::sync::atomic::Ordering::Relaxed);
        let t0 = smp::rdtsc();
        let fed = llm::feed(&toks()[..n], &mut pos);
        let t1 = smp::rdtsc();
        let (stop, n_gen) = if fed { llm::generate(&mut pos, MAX_ANSWER) } else { (llm::Stop::Intr, 0) };
        let t2 = smp::rdtsc();
        unsafe {
            CAPTURE = false;
            POS = pos;
        }
        println!("");
        llm::print_speed(&llm::Stats { n_prompt: n as u64, n_gen, prefill_us: smp::tsc_to_us(t1 - t0), decode_us: smp::tsc_to_us(t2 - t1), pos });
        if !fed {
            println!("^C [agent] interrupted during prefill, conversation reset");
            reset();
            break;
        }
        if stop == llm::Stop::Intr {
            println!("^C [agent] interrupted");
            crate::log::record(b'A', "turn interrupted by Ctrl-C");
            break;
        }
        let cap = unsafe { (*core::ptr::addr_of!(CAP)).as_str() };
        let res = unsafe { &mut *core::ptr::addr_of_mut!(RES) };
        res.clear();
        match parse(cap) {
            None => {
                let mut a = SBuf::<160>::new();
                let _ = write!(a, "answer ({} tok): {}", n_gen, cap.trim().lines().next().unwrap_or(""));
                crate::log::record(b'A', a.as_str());
                let ans = cap.rsplit("</think>").next().unwrap_or(cap).trim();
                hist("assistant", ans);
                push_ex(q, ans);
                break;
            }
            Some(_) if round == MAX_ROUNDS => {
                println!("[agent] tool round limit ({}) reached", MAX_ROUNDS);
                break;
            }
            Some(Err(e)) => {
                let _ = write!(res, "error: malformed tool call ({})", e);
                println!("[tool] malformed call: {}", e);
            }
            Some(Ok(c)) => {
                print!("[tool] {}(", c.name);
                for (i, (k, v)) in c.args[..c.n].iter().enumerate() {
                    print!("{}{}={}", if i > 0 { ", " } else { "" }, k, v);
                }
                println!(")");
                if validate(&c, res).is_some() {
                    exec(&c, res);
                }
                if res.n > RES_CAP {
                    res.n = floor_char(res.as_str(), RES_CAP - 96);
                    let _ = write!(res, "\n[... truncated by the kernel: the result was longer than {} bytes]", RES_CAP);
                }
                let mut h = SBuf::<512>::new();
                let _ = write!(h, "{}(...) -> {}", c.name, &res.as_str()[..floor_char(res.as_str(), 400)]);
                hist("tool", h.as_str());
                let mut l = SBuf::<200>::new();
                let _ = write!(l, "tool {} -> {}", c.name, res.as_str());
                crate::log::record(b'A', l.as_str());
            }
        }
        println!("[tool] -> {}", res.as_str());
        t.clear();
        let _ = write!(t, "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n{}", res.as_str(), gen_prompt);
    }
    smp::park_phase(false);
}

/// Boot self-check of the lenient parser and the validator (no model involved).
pub fn selftest() -> bool {
    let ok1 = matches!(parse("ok <tool_call>\n<function=read_log>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>"),
        Some(Ok(ref c)) if c.name == "read_log" && c.n == 1 && c.args[0] == ("n", "5"));
    let ok2 = matches!(parse("<tool_call>\n<function=meminfo>\n</function>"), Some(Ok(ref c)) if c.name == "meminfo" && c.n == 0);
    let ok3 = parse("no call here").is_none();
    let ok4 = matches!(parse("<tool_call>\n<function=set_setting>\n<parameter=key>\ncores\n<parameter=value>\n4\n"),
        Some(Ok(ref c)) if c.n == 2 && c.args[1] == ("value", "4"));
    let mut e = SBuf::<1024>::new();
    let bad = Call { name: "set_setting", args: [("key", "colour"), ("value", "x"), ("", ""), ("", "")], n: 2 };
    let miss = Call { name: "set_setting", args: [("key", "cores"), ("", ""), ("", ""), ("", "")], n: 1 };
    let ok5 = validate(&bad, &mut e).is_none() && validate(&miss, &mut e).is_none();
    ok1 && ok2 && ok3 && ok4 && ok5
}
