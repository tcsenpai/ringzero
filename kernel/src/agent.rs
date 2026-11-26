// The system agent: Qwen3.5 native tool calling in ring 0. The system prompt + tool list is prefilled
// once at boot and snapshotted (llm::snapshot_*); each `@` turn streams the answer, parses a tool
// call if the model makes one, validates it against the tool schema, runs it, feeds the result back
// as a <tool_response> turn and loops (at most MAX_ROUNDS tool rounds per user turn).
use crate::engine::{CAP, CAPTURE};
use crate::log::SBuf;
use crate::{klog, llm, print, println, shell, smp, sys};
use core::fmt::Write;

const SYSTEM: &str = "You are the system agent of AIOS, an experimental operating system with its own Rust \
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

static mut POS: u64 = 0;
static mut FRESH: bool = true;
static mut TEXT: SBuf<8192> = SBuf::new();
static mut TOKS: [u32; 4096] = [0; 4096];

fn text() -> &'static mut SBuf<8192> {
    unsafe { &mut *core::ptr::addr_of_mut!(TEXT) }
}
fn toks() -> &'static mut [u32; 4096] {
    unsafe { &mut *core::ptr::addr_of_mut!(TOKS) }
}

pub fn pos() -> u64 {
    unsafe { POS }
}

/// Next `@` starts a fresh conversation from the boot snapshot.
pub fn reset() {
    unsafe { FRESH = true };
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
    let mut pos = 0;
    while !llm::feed(&toks()[..n], &mut pos) {
        pos = 0; // Ctrl-C at boot: no shell yet, prefill again
    }
    let t1 = smp::rdtsc();
    let bytes = llm::snapshot_save(pos);
    let t2 = smp::rdtsc();
    smp::park_phase(false);
    let us = smp::tsc_to_us(t1 - t0);
    klog!(b'I', "agent: prefix prefill {} tok in {}.{:03} s ({} tok/s), snapshot {} MiB saved in {} ms",
        n, us / 1_000_000, us / 1000 % 1000, n as u64 * 1_000_000 / us.max(1), bytes >> 20, smp::tsc_to_us(t2 - t1) / 1000);
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
fn validate(c: &Call, err: &mut SBuf<1024>) -> Option<usize> {
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

fn exec(c: &Call, o: &mut SBuf<1024>) {
    match c.name {
        "meminfo" => sys::meminfo(o),
        "cpuinfo" => sys::cpuinfo(o),
        "list_cores" => sys::cores(o),
        "uptime" => sys::uptime(o),
        "read_log" => {
            let k = arg(c, "n").and_then(|v| v.parse::<usize>().ok()).unwrap_or(10).clamp(1, 30);
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
    let _ = write!(l, "user: @ {}", q);
    crate::log::record(b'A', l.as_str());
    unsafe {
        if !FRESH && POS + q.len() as u64 + 300 >= max_seq {
            println!("[agent] context full ({}/{}), starting a fresh conversation", pos(), max_seq);
            FRESH = true;
        }
        if FRESH {
            let t0 = smp::rdtsc();
            POS = llm::snapshot_restore();
            FRESH = false;
            klog!(b'A', "[agent] fresh conversation: prefix ({} tok) restored in {} us", pos(), smp::tsc_to_us(smp::rdtsc() - t0));
        }
    }
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
        let mut res = SBuf::<1024>::new();
        match parse(cap) {
            None => {
                let mut a = SBuf::<160>::new();
                let _ = write!(a, "answer ({} tok): {}", n_gen, cap.trim().lines().next().unwrap_or(""));
                crate::log::record(b'A', a.as_str());
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
                if validate(&c, &mut res).is_some() {
                    exec(&c, &mut res);
                }
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
