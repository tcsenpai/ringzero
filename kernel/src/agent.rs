// The system agent: Qwen3.5 native tool calling in ring 0. The system prompt + tool list is prefilled
// once at boot and saved as snapshot point #0 (snap.rs); each turn streams the answer, parses a tool
// call if the model makes one, validates it against the tool schema, runs it, feeds the result back
// as a <tool_response> turn and loops (at most MAX_ROUNDS tool rounds per user turn).
// Context management (M2.5): when the fill crosses COMPACT_PCT (or on /compact) the model writes a short
// summary, the boot snapshot is restored (it ends just before the system turn's <|im_end|>), the summary
// is appended to the system prompt and the last exchanges are prefilled again (DeltaNet state cannot be
// cropped, so the context is rebuilt). The raw transcript stays in HIST.
// Snapshots (snap.rs): every finished turn is saved as a point; Ctrl-C or an interrupted compaction
// restores the point the turn started from, and the exchanges for the re-prefill come from the points.
use crate::engine::{CAP, CAPTURE};
use crate::log::SBuf;
use crate::{klog, llm, print, println, shell, smp, snap, sys};
use core::fmt::Write;

const SYSTEM: &str = "You are the system agent of ringzero, an experimental operating system with its own Rust \
kernel (not Linux). You run inside the kernel (ring 0) and answer the user at the shell. Use the tools to read real data from this machine \
or to act on it, and never invent numbers. Reply in English or in Italian, matching the language of the \
user's latest message, briefly, in plain text without markdown headers.";

// JSON lines exactly as the HF template's `tool | tojson` renders them (", " and ": " separators).
const TOOLS: [&str; 18] = [
    r#"{"type": "function", "function": {"name": "meminfo", "description": "Memory of this machine: free, used and total RAM in MiB.", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "cpuinfo", "description": "CPU model, number of cores, clock and features.", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "list_cores", "description": "List the CPU cores and what each one does.", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "read_log", "description": "Last n lines of the kernel log.", "parameters": {"type": "object", "properties": {"n": {"type": "integer", "description": "number of lines, default 10"}}}}}"#,
    r#"{"type": "function", "function": {"name": "uptime", "description": "Time since boot.", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "poweroff", "description": "Shut down the machine (asks the user to confirm).", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "reboot", "description": "Reboot the machine (asks the user to confirm).", "parameters": {"type": "object", "properties": {}}}}"#,
    r#"{"type": "function", "function": {"name": "set_setting", "description": "Change a system setting. temperature: sampling temperature, 0 = greedy. cores: CPU cores used for inference. keyboard: PS/2 keyboard layout, us or it.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "enum": ["temperature", "cores", "keyboard"]}, "value": {"type": "string"}}, "required": ["key", "value"]}}}"#,
    r#"{"type": "function", "function": {"name": "remember", "description": "Save a fact the user told you, kept across reboots.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "snake_case, e.g. user_name"}, "value": {"type": "string"}}, "required": ["key", "value"]}}}"#,
    r#"{"type": "function", "function": {"name": "forget", "description": "Delete a saved fact (asks the user).", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}"#,
    r#"{"type": "function", "function": {"name": "recall", "description": "Search saved facts, past conversations and files on the disk.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}"#,
    r#"{"type": "function", "function": {"name": "ls", "description": "List a directory on the disk.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}"#,
    r#"{"type": "function", "function": {"name": "cat", "description": "Read a text file on the disk.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}"#,
    r#"{"type": "function", "function": {"name": "write_file", "description": "Create or replace a text file on the disk.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}"#,
    r#"{"type": "function", "function": {"name": "mkdir", "description": "Create a directory on the disk.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}"#,
    r#"{"type": "function", "function": {"name": "rm", "description": "Delete a file or an empty directory (asks the user).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}"#,
    r#"{"type": "function", "function": {"name": "run_code", "description": "Run Python code (MicroPython) and return what it prints.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}"#,
    r#"{"type": "function", "function": {"name": "run_wasm", "description": "Run a program from /bin and return what it prints: coreutils (ls, wc, sort, head, ...), rg, jq, sqlite3, lua, python3 (CPython), qjs. Commands look like: wc -l /a.txt, jq .name /a.json, sqlite3 /a.db \"select 1;\", lua /a.lua, python3 /a.py", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "the program"}, "args": {"type": "string", "description": "the rest of the command line"}}, "required": ["path"]}}}"#,
];

pub(crate) enum Ty {
    Int,
    Str(&'static [&'static str]), // allowed values, empty = any
}

// (name, params (name, type, required)), same order and content as TOOLS
pub(crate) const SPECS: [(&str, &[(&str, Ty, bool)]); 18] = [
    ("meminfo", &[]),
    ("cpuinfo", &[]),
    ("list_cores", &[]),
    ("read_log", &[("n", Ty::Int, false)]),
    ("uptime", &[]),
    ("poweroff", &[]),
    ("reboot", &[]),
    ("set_setting", &[("key", Ty::Str(&["temperature", "cores", "keyboard"]), true), ("value", Ty::Str(&[]), true)]),
    ("remember", &[("key", Ty::Str(&[]), true), ("value", Ty::Str(&[]), true)]),
    ("forget", &[("key", Ty::Str(&[]), true)]),
    ("recall", &[("query", Ty::Str(&[]), true)]),
    ("ls", &[("path", Ty::Str(&[]), true)]),
    ("cat", &[("path", Ty::Str(&[]), true)]),
    ("write_file", &[("path", Ty::Str(&[]), true), ("content", Ty::Str(&[]), true)]),
    ("mkdir", &[("path", Ty::Str(&[]), true)]),
    ("rm", &[("path", Ty::Str(&[]), true)]),
    ("run_code", &[("code", Ty::Str(&[]), true)]), // M5
    ("run_wasm", &[("path", Ty::Str(&[]), true), ("args", Ty::Str(&[]), false)]), // M5b
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

static mut TEXT: SBuf<32768> = SBuf::new();
static mut TOKS: [u32; 16384] = [0; 16384];
static mut RES: SBuf<{ RES_CAP + 512 }> = SBuf::new();
pub static mut COMPACT_PCT: u64 = 75;
pub static mut N_COMPACT: u64 = 0;
/// Key of the boot snapshot cache (model, prefix ids, engine build, KV geometry); hib.rs derives its own.
static mut BOOT_KEY: u64 = 0;

pub fn boot_key() -> u64 {
    unsafe { BOOT_KEY }
}
/// "<|im_end|>\n" closing the system turn: not in the snapshot, fed when a conversation starts fresh.
static mut SYS_END: [u32; 2] = [0; 2];

fn text() -> &'static mut SBuf<32768> {
    unsafe { &mut *core::ptr::addr_of_mut!(TEXT) }
}
fn toks() -> &'static mut [u32; 16384] {
    unsafe { &mut *core::ptr::addr_of_mut!(TOKS) }
}

pub fn pos() -> u64 {
    snap::pos()
}

/// A fresh conversation from the boot prefix on the current branch (the old line stays in /tree).
pub fn reset() {
    if !snap::at_root() {
        hist("kernel", "--- new conversation ---");
    }
    snap::new_line();
}

/// A kernel line in the raw transcript (snapshot moves).
pub fn note(s: &str) {
    hist("kernel", s);
}

/// After /bench, /fill, /gen: put the conversation back (they ran from position 0).
pub fn resume() {
    let t0 = smp::rdtsc();
    snap::invalidate();
    let p = snap::ensure_live();
    println!("[snap] conversation restored (ctx {}) in {} us", p, smp::tsc_to_us(smp::rdtsc() - t0));
}

/// /retry: undo the last turn and ask its question again (temp 0: the same answer; temp > 0: reseeded).
pub fn retry() {
    let mut q = SBuf::<4096>::new();
    if !snap::last_question(&mut q) {
        println!("[retry] nothing to retry here (no question at this point)");
        return;
    }
    snap::undo(1);
    if unsafe { llm::TEMP_BITS } != 0 {
        llm::set_rng(unsafe { llm::RNG } ^ smp::rdtsc() | 1);
    }
    println!("[retry] {}", q.as_str());
    ask(q.as_str());
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

/// The raw transcript (hibernation image).
pub fn hist_text() -> &'static str {
    unsafe { (*core::ptr::addr_of!(HIST)).as_str() }
}
pub fn hist_restore(s: &str) {
    let h = unsafe { &mut *core::ptr::addr_of_mut!(HIST) };
    h.clear();
    h.push(s[..floor_char(s, h.b.len())].as_bytes());
}

pub fn floor_char(s: &str, max: usize) -> usize {
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
    let fresh = snap::at_root();
    println!("[ctx] {}/{} tokens ({}%){}, boot prefix {} tok, KV {}, compaction at {}%, {} compaction(s) so far, {} exchange(s) kept for re-prefill",
        p, m, p * 100 / m, if fresh { " (fresh: the next turn closes the system turn)" } else { "" },
        snap::root_pos(), llm::kv_type(), unsafe { COMPACT_PCT }, unsafe { N_COMPACT }, snap::ex_count().0.min(KEEP));
}

// ---- last exchanges (question + final answer, stored in the snapshot points), re-prefilled after a compaction ----

/// Render the last k exchanges of the current history as chat turns (oldest first).
fn render_ex(t: &mut SBuf<32768>, k: usize) {
    snap::exchanges(k, |q, a| {
        let _ = write!(t, "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n", q, a);
    });
}

/// Summarize, restore the boot prefix, prefill the summary and the last exchanges, save it as point
/// "[compact N]" (the uncompacted point stays as pre-compact-N, /undo goes back to it). false =
/// interrupted (the conversation is restored) or no room (a fresh conversation).
pub fn compact(why: &str) -> bool {
    if snap::at_root() {
        println!("[compact] nothing to compact (fresh conversation)");
        return false;
    }
    let max_seq = llm::ctx_max_seq();
    let before = snap::ensure_live();
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
        smp::park_phase(false);
        snap::rollback();
        println!("^C [compact] interrupted, conversation restored (ctx {})", snap::pos());
        return false;
    }
    let mut sum = SBuf::<4096>::new();
    let cap = unsafe { (*core::ptr::addr_of!(CAP)).as_str() };
    let cap = cap.rsplit("</think>").next().unwrap_or(cap);
    let cap = cap.split("<tool_call>").next().unwrap_or(cap).trim();
    let _ = write!(sum, "{}", cap);

    let root = snap::restore_root(); // system turn still open: the summary is appended to it
    let ex_avail = snap::ex_count().0.min(KEEP);
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
    pos = root;
    let ok = llm::feed(&toks()[..n], &mut pos);
    let t2 = smp::rdtsc();
    smp::park_phase(false);
    if !ok {
        snap::rollback();
        println!("^C [compact] interrupted during re-prefill, conversation restored (ctx {})", snap::pos());
        return false;
    }
    unsafe { N_COMPACT += 1 };
    snap::commit(pos, "", "", false, "", unsafe { N_COMPACT } as u32);
    let (us1, us2) = (smp::tsc_to_us(t1 - t0), smp::tsc_to_us(t2 - t1));
    // one log line (the ring keeps 128 bytes per line)
    klog!(b'A', "compact #{} {}: {}->{} tok ({}%->{}%), summary {} tok {} ms, re-prefill {} tok {} ms, total {} ms",
        unsafe { N_COMPACT }, why, before, pos, before * 100 / max_seq, pos * 100 / max_seq, n_sum, us1 / 1000, n, us2 / 1000, (us1 + us2) / 1000);
    hist("kernel", "[compaction] summary written by the model:");
    hist("summary", sum.as_str());
    crate::agentd::on_compact(sum.as_str());
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

/// Boot snapshot cache: the prefilled prefix state on the disk (snap.rs), keyed by model, prefix and engine.
const BOOT_SNAP: &str = "/sys/bootsnap.bin";

/// Boot: load the prefix state from the disk cache, or prefill the system prompt + tools from position 0
/// (and write the cache); then snapshot it as point #0.
pub fn init() {
    let n = build_prefix();
    let ids = unsafe { core::slice::from_raw_parts(toks().as_ptr() as *const u8, (n - 2) * 4) };
    let h = toks()[..n].iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
    klog!(b'I', "agent: system prompt + {} tools = {} tokens (fnv {:#x})", TOOLS.len(), n, h);
    use crate::mstore::fnv;
    let geo = [unsafe { crate::engine::kv_elt_bytes }, llm::ctx_max_seq(), llm::CTX_LEN];
    let key = geo.iter().fold(fnv(fnv(fnv(crate::mstore::FNV0, ids), &unsafe { llm::MODEL_HASH }.to_le_bytes()),
        env!("RINGZERO_ENGINE_HASH").as_bytes()), |h, g| fnv(h, &g.to_le_bytes()));
    unsafe { (SYS_END, BOOT_KEY) = ([toks()[n - 2], toks()[n - 1]], key) }; // <|im_end|> \n of the system turn
    smp::park_phase(true);
    let t0 = smp::rdtsc();
    let cached = if crate::fs::mounted() { snap::load_prefix(BOOT_SNAP, key) } else { Err("no disk") };
    let t1 = smp::rdtsc();
    let hit = matches!(cached, Ok(p) if p == (n - 2) as u64);
    let pos = match cached {
        Ok(p) if p == (n - 2) as u64 => {
            let us = smp::tsc_to_us(t1 - t0);
            klog!(b'I', "agent: boot snapshot cache hit ({}, key {:#x}): {} tok of state loaded in {}.{:03} s instead of a prefill",
                BOOT_SNAP, key, p, us / 1_000_000, us / 1000 % 1000);
            crate::dlog!("boot cache: hit, prefix state loaded from the disk in {} ms", us / 1000);
            p
        }
        r => {
            let why = match r {
                Err(e) => e,
                Ok(_) => "token count differs",
            };
            crate::dlog!("boot cache: miss ({}), prefilling {} tokens", why, n - 2);
            let mut pos = 0;
            while !llm::feed(&toks()[..n - 2], &mut pos) {
                pos = 0; // Ctrl-C at boot: no shell yet, prefill again
            }
            let us = smp::tsc_to_us(smp::rdtsc() - t1);
            klog!(b'I', "agent: prefix prefill {} tok in {}.{:03} s ({} tok/s)", pos, us / 1_000_000, us / 1000 % 1000,
                n as u64 * 1_000_000 / us.max(1));
            pos
        }
    };
    let t2 = smp::rdtsc();
    let bytes = snap::init_root(pos);
    let t3 = smp::rdtsc();
    if !hit && crate::fs::mounted() {
        match snap::save_prefix(BOOT_SNAP, key, pos) {
            Ok(b) => crate::dlog!("boot cache: written, {} MiB in {} ms", b >> 20, smp::tsc_to_us(smp::rdtsc() - t3) / 1000),
            Err(e) => crate::dlog!("boot cache: write failed ({})", e),
        }
    }
    smp::park_phase(false);
    klog!(b'I', "agent: snapshot #0 {} MiB in a {} MiB pool, saved in {} ms (pool touched)", bytes >> 20, snap::pool_bytes() >> 20,
        smp::tsc_to_us(t3 - t2) / 1000);
}

/// The boot prefix ids that init() prefills (without the closing <|im_end|>\n): the `/prefill` bench input.
pub fn prefix_tokens() -> &'static [u32] {
    let n = build_prefix();
    &toks()[..n - 2]
}

/// Hidden `toks` builtin: print the prefix token ids (checked against the HF tokenizer offline).
pub fn dump_prefix_tokens() {
    let n = build_prefix();
    for (i, t) in toks()[..n].iter().enumerate() {
        print!("{}{}", if i > 0 { "," } else { "" }, t);
    }
    println!("");
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

/// Tool calls of the current turn that changed the machine (printed by /undo, which cannot revert them).
static mut FXT: SBuf<160> = SBuf::new();

fn exec<const N: usize>(c: &Call, o: &mut SBuf<N>) {
    let fx = unsafe { &mut *core::ptr::addr_of_mut!(FXT) };
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
        // memory goes through agentd: the kernel validates, dedups and logs every change
        "recall" => crate::agentd::recall(arg(c, "query").unwrap_or(""), o),
        "remember" => match crate::agentd::remember(arg(c, "key").unwrap_or(""), arg(c, "value").unwrap_or(""), "agent", true) {
            Ok(m) => {
                let _ = write!(fx, "{}remember {}", if fx.n > 0 { ", " } else { "" }, arg(c, "key").unwrap_or(""));
                o.push(m.as_bytes());
            }
            Err(e) => o.push(e.as_bytes()),
        },
        "forget" => {
            let k = arg(c, "key").unwrap_or("");
            let mut q = SBuf::<160>::new();
            let _ = write!(q, "\n[agent] wants to forget the fact '{}'. Proceed?", k);
            if !crate::agentd::facts().iter().any(|f| Some(f.key.as_str()) == crate::agentd::norm_key(k).as_deref()) {
                if let Err(m) = crate::agentd::forget(k) {
                    o.push(m.as_bytes()); // unknown key: the error lists the saved keys, nothing is removed
                }
            } else if shell::confirm(q.as_str()) {
                match crate::agentd::forget(k) {
                    Ok(m) | Err(m) => o.push(m.as_bytes()),
                }
                let _ = write!(fx, "{}forget {}", if fx.n > 0 { ", " } else { "" }, k);
            } else {
                let _ = write!(o, "cancelled: the user answered no, the fact is kept");
            }
        }
        "ls" | "cat" | "write_file" | "mkdir" | "rm" => {
            if let Err(e) = fs_tool(c, o, fx) {
                let _ = write!(o, "error: {}", e);
            }
        }
        "run_code" => crate::upy::tool(arg(c, "code").unwrap_or(""), o), // M5
        "run_wasm" => crate::wasm::tool(arg(c, "path").unwrap_or(""), arg(c, "args").unwrap_or(""), o), // M5b
        "set_setting" => {
            let (k, v) = (arg(c, "key").unwrap_or(""), arg(c, "value").unwrap_or(""));
            if shell::set_setting(k, v, o) {
                let _ = write!(fx, "{}set_setting {}={}", if fx.n > 0 { ", " } else { "" }, k, v);
            }
        }
        _ => {}
    }
}

/// The disk tools. Paths are checked by fs::check (absolute, no '..'); rm always asks the human.
fn fs_tool<const N: usize>(c: &Call, o: &mut SBuf<N>, fx: &mut SBuf<160>) -> Result<(), &'static str> {
    use crate::{fidx, fs};
    let p = fs::canon(arg(c, "path").unwrap_or(""))?;
    let mut note = |what: &str| {
        let _ = write!(fx, "{}{} {}", if fx.n > 0 { ", " } else { "" }, what, p);
    };
    match c.name {
        "ls" => {
            if fs::ls(&p, o)? == 0 {
                let _ = write!(o, "(empty directory)");
            }
        }
        "cat" => {
            let (b, size) = fs::read(&p, 0, RES_CAP - 256)?;
            if !fs::is_text(&b) {
                return Err("not a text file");
            }
            o.push(&b);
            if (b.len() as u64) < size {
                let _ = write!(o, "\n[... truncated by the kernel: first {} of {} bytes]", b.len(), size);
            }
        }
        "write_file" => {
            let data = arg(c, "content").unwrap_or("");
            fs::write(&p, data.as_bytes())?;
            fidx::enqueue(&p, data.as_bytes());
            note("write_file");
            let _ = write!(o, "wrote {} bytes to {}", data.len(), p);
        }
        "mkdir" => {
            let made = fs::mkdir(&p)?;
            if made {
                note("mkdir");
            }
            let _ = write!(o, "{} {}", p, if made { "created" } else { "already exists" });
        }
        _ => {
            let (dir, k) = fs::stat(&p)?;
            if dir && k > 0 {
                return Err("directory is not empty; the agent removes only files and empty directories");
            }
            let mut q = SBuf::<320>::new();
            let _ = write!(q, "\n[agent] wants to delete {} {}{}. Proceed?", if dir { "directory" } else { "file" }, p,
                if dir { alloc::string::String::new() } else { alloc::format!(" ({} B)", k) });
            if shell::confirm(q.as_str()) {
                fs::rm(&p, false)?;
                fidx::forget(&p);
                note("rm");
                let _ = write!(o, "deleted {}", p);
            } else {
                let _ = write!(o, "cancelled: the user answered no, {} is kept", p);
            }
        }
    }
    Ok(())
}

/// One user turn at the shell.
pub fn ask(q: &str) {
    let max_seq = llm::ctx_max_seq();
    let mut l = SBuf::<160>::new();
    let _ = write!(l, "user: {}", q);
    crate::log::record(b'A', l.as_str());
    let p = snap::ensure_live();
    if !snap::at_root() && p * 100 >= max_seq * unsafe { COMPACT_PCT } && snap::ex_count().1 > KEEP {
        compact("auto");
    }
    if !snap::at_root() && pos() + q.len() as u64 + 300 >= max_seq {
        println!("[agent] context full ({}/{}), starting a fresh conversation", pos(), max_seq);
        reset();
    }
    // the live state is the current point here; a fresh one still has the system turn open
    let mut start = pos();
    let fresh = snap::at_root();
    if fresh {
        for t in unsafe { SYS_END } {
            llm::fwd(t, start, 0); // close the system turn
            start += 1;
        }
        crate::log::record(b'A', "[agent] fresh conversation from the boot prefix");
    }
    unsafe { (*core::ptr::addr_of_mut!(FXT)).clear() };
    let mut pos_now = start;
    let mut answer: Option<(usize, usize)> = None; // the final answer's span in CAP
    hist("user", q);
    let t = text();
    t.clear();
    let think = unsafe { llm::THINK } != 0;
    let gen_prompt = if think { "<|im_start|>assistant\n<think>\n" } else { "<|im_start|>assistant\n<think>\n\n</think>\n\n" };
    smp::park_phase(true);
    // agentd's context compiler: facts + recalled notes in front of the first message of a conversation
    let mut slot = alloc::string::String::new();
    let why = if fresh { Some("new conversation") } else if crate::agentd::take_after_compact() { Some("after compaction") } else { None };
    if let Some(w) = why {
        crate::agentd::slot(q, w, &mut slot);
    } else {
        crate::agentd::man_slot(q, &mut slot); // T22: a how-to message mid-conversation gets the manual line
    }
    slot.lines().skip(1).for_each(|l| println!("[memory] {}", l));
    crate::agentd::on_user(q);
    let _ = write!(t, "<|im_start|>user\n{}{}<|im_end|>\n{}", slot, q, gen_prompt);
    let mut intr = false;
    for round in 0..=MAX_ROUNDS {
        let n = llm::encode(&t.b[..t.n], toks());
        let mut pos = pos_now;
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
        unsafe { CAPTURE = false };
        let n_fed = if fed { n as u64 } else { pos - pos_now }; // Ctrl-C during the prefill: what was fed
        pos_now = pos;
        println!("");
        llm::print_speed(&llm::Stats { n_prompt: n_fed, n_gen, prefill_us: smp::tsc_to_us(t1 - t0), decode_us: smp::tsc_to_us(t2 - t1), pos });
        if !fed || stop == llm::Stop::Intr {
            intr = true;
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
                crate::agentd::on_answer(ans);
                let off = ans.as_ptr() as usize - cap.as_ptr() as usize;
                answer = Some((off, off + ans.len()));
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
                // M4.7a: a value the grammar had to choose is a guess; tools that change the machine do not run it
                const FX: [&str; 10] = ["set_setting", "remember", "forget", "write_file", "mkdir", "rm", "poweroff", "reboot", "run_code", "run_wasm"];
                if validate(&c, res).is_some() {
                    if crate::gram::last_steered() && FX.contains(&c.name) {
                        let _ = write!(res, "error: not run: a tool name, parameter or value was not in the schema and the kernel had to replace it. Call {} again with values from its schema, or answer without the tool", c.name);
                    } else {
                        exec(&c, res);
                    }
                }
                if res.n > RES_CAP {
                    res.n = floor_char(res.as_str(), RES_CAP - 96);
                    let _ = write!(res, "\n[... truncated by the kernel: the result was longer than {} bytes]", RES_CAP);
                }
                let mut h = SBuf::<512>::new();
                let _ = write!(h, "{}(...) -> {}", c.name, &res.as_str()[..floor_char(res.as_str(), 400)]);
                hist("tool", h.as_str());
                crate::agentd::on_tool(c.name, res.as_str());
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
    let fx = unsafe { (*core::ptr::addr_of!(FXT)).as_str() };
    if intr {
        // a turn is atomic: back to the point it started from, earlier turns stay
        let p = snap::rollback();
        println!("^C [agent] interrupted, turn rolled back (ctx {})", p);
        if !fx.is_empty() {
            println!("[agent] note: this turn ran {}; the machine keeps it.", fx);
        }
        crate::log::record(b'A', "turn interrupted by Ctrl-C, rolled back");
        hist("kernel", "[turn interrupted by Ctrl-C, rolled back]");
        return;
    }
    let cap = unsafe { (*core::ptr::addr_of!(CAP)).as_str() };
    let (a, ex) = match answer {
        Some((i, j)) => (&cap[i..j], true),
        None => ("", false),
    };
    snap::commit(pos_now, q, a, ex, fx, 0);
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
