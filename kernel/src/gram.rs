// Constrained decoding for tool calls (M4.7a). Once the model has emitted <tool_call>, every next token
// must keep the call valid against the tool schemas (agent::SPECS) in the native Qwen3.5 format
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>
// : a known tool name, its parameter names (each at most once), enum values, integers, every required
// parameter before </function>, then </tool_call>. Free string values stay free, except that they cannot
// contain </function> or <parameter= (the lenient parser would end the value there) and no special token.
// The grammar is a byte automaton (G); a token is allowed when the automaton accepts all its bytes.
//
// Fast path: the token the sampler picked is checked (a few bytes) and kept when it is allowed. Only
// when it is rejected, every rejected token's logit is set to -inf (a walk of the vocab trie below) and
// the sampler picks again; with one allowed token it is forced. Greedy: the same token as an argmax over
// the masked logits; top-p: the same distribution as sampling the masked logits (rejection + resample).
// Outside a tool call the cost is one bool test per token. `!set constrain off` keeps the automaton
// running as a passive checker (it reports where a call leaves the grammar) for A/B runs.
//
// Vocab trie: token ids sorted by their bytes plus each one's common-prefix length with the previous
// one (an implicit trie, 5 bytes per token, built once at boot). The walk keeps the automaton state per
// depth, starts every token at its shared prefix and skips every token under a rejected prefix.
use crate::agent::{Ty, SPECS};
use crate::engine::ctx;
use crate::log::SBuf;
use crate::{llm, println, smp};
use core::fmt::Write;

// engine/constants.inc (QWEN35)
const CTX_TOK_OFFS_PTR: usize = 40;
const CTX_TOK_BYTES_PTR: usize = 48;
const CTX_VOCAB_SIZE: usize = 152;
const CTX_BUF_LOGITS: usize = 280;
const CTX_SAMPLE_N: usize = 488;
/// <|endoftext|>: the special tokens start here, none of them is text inside a call
const SPECIAL_MIN: usize = 248044;
const MAXV: usize = 262144;
/// Longer tokens are never allowed inside a call (runs of spaces or '=' only).
const MAXLEN: usize = 128;
/// Whitespace bytes allowed between two structural parts (the format uses one '\n').
const WS_MAX: u8 = 4;
const NEG_INF: u32 = 0xff80_0000;

const FUNC: &[u8] = b"<function=";
const PARAM: &[u8] = b"<parameter=";
const CLOSE_P: &[u8] = b"</parameter>";
const CLOSE_F: &[u8] = b"</function>";

#[derive(Clone, Copy, PartialEq, Debug)]
enum St {
    Func,  // ws, then "<function="
    Name,  // a tool name, then '>'
    Tag,   // ws, then "<parameter=" (a parameter is left) or "</function>" (all required given)
    Key,   // an unused parameter name of the tool, then '>'
    Free,  // free text until "</parameter>"
    Enum,  // ws, one of the enum values, then Close
    Int,   // ws, -?digits, then Close
    Close, // ws, then "</parameter>"
    Done,  // ws, then the </tool_call> token
}

#[derive(Clone, Copy)]
pub struct G {
    st: St,
    buf: [u8; 24],
    n: u8,
    ws: u8,
    tool: u8,
    key: u8,
    used: u8, // bit j = parameter j given
}
const G0: G = G { st: St::Func, buf: [0; 24], n: 0, ws: 0, tool: 0, key: 0, used: 0 };

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

impl G {
    fn params(&self) -> &'static [(&'static str, Ty, bool)] {
        SPECS[self.tool as usize].1
    }
    fn space(&mut self) -> bool {
        self.ws += 1;
        self.ws <= WS_MAX
    }
    fn to(&mut self, st: St) {
        (self.st, self.n, self.ws) = (st, 0, 0);
    }
    fn push(&mut self, b: u8) -> bool {
        if self.n as usize == self.buf.len() {
            return false;
        }
        self.buf[self.n as usize] = b;
        self.n += 1;
        true
    }
    /// One byte. false = it leaves the grammar (the state is then unspecified).
    pub fn step(&mut self, b: u8) -> bool {
        match self.st {
            St::Func | St::Close => {
                let lit = if self.st == St::Func { FUNC } else { CLOSE_P };
                if self.n == 0 && is_ws(b) {
                    return self.space();
                }
                if lit[self.n as usize] != b {
                    return false;
                }
                self.n += 1;
                if self.n as usize == lit.len() {
                    self.to(if self.st == St::Func { St::Name } else { St::Tag });
                }
                true
            }
            St::Name | St::Key => {
                let key = self.st == St::Key;
                let (tool, used) = (self.tool, self.used);
                let cand = |j: usize| -> Option<&'static str> {
                    if key {
                        let p = SPECS[tool as usize].1;
                        (j < p.len() && used & 1 << j == 0).then(|| p[j].0)
                    } else {
                        SPECS.get(j).map(|s| s.0)
                    }
                };
                let w = &self.buf[..self.n as usize];
                if b == b'>' {
                    let Some(j) = (0..SPECS.len()).find(|&j| cand(j).is_some_and(|c| c.as_bytes() == w)) else { return false };
                    if !key {
                        self.tool = j as u8;
                        self.to(St::Tag);
                        return true;
                    }
                    self.key = j as u8;
                    self.used |= 1 << j;
                    self.to(match self.params()[j].1 {
                        Ty::Int => St::Int,
                        Ty::Str(a) if !a.is_empty() => St::Enum,
                        Ty::Str(_) => St::Free,
                    });
                    return true;
                }
                if !self.push(b) {
                    return false;
                }
                let w = &self.buf[..self.n as usize];
                (0..SPECS.len()).any(|j| cand(j).is_some_and(|c| c.as_bytes().starts_with(w)))
            }
            St::Tag => {
                if self.n == 0 && is_ws(b) {
                    return self.space();
                }
                let p = self.params();
                let left = (0..p.len()).any(|j| self.used & 1 << j == 0);
                let done = p.iter().enumerate().all(|(j, x)| !x.2 || self.used & 1 << j != 0);
                if !self.push(b) {
                    return false;
                }
                let w = &self.buf[..self.n as usize];
                let (p_ok, f_ok) = (left && PARAM.starts_with(w), done && CLOSE_F.starts_with(w));
                if p_ok && w.len() == PARAM.len() {
                    self.to(St::Key);
                } else if f_ok && w.len() == CLOSE_F.len() {
                    self.to(St::Done);
                }
                p_ok || f_ok
            }
            St::Free => {
                if b == b'<' {
                    (self.buf[0], self.n) = (b'<', 1);
                    return true;
                }
                if self.n == 0 {
                    return true;
                }
                self.push(b);
                let w = &self.buf[..self.n as usize];
                if w == CLOSE_P {
                    self.to(St::Tag);
                    return true;
                }
                if w == CLOSE_F || w == PARAM {
                    return false;
                }
                if !(CLOSE_P.starts_with(w) || CLOSE_F.starts_with(w) || PARAM.starts_with(w)) {
                    self.n = 0;
                }
                true
            }
            St::Enum | St::Int => {
                if is_ws(b) || b == b'<' {
                    if self.n == 0 {
                        return b != b'<' && self.space();
                    }
                    let w = &self.buf[..self.n as usize];
                    let ok = match self.params()[self.key as usize].1 {
                        Ty::Str(a) => a.iter().any(|v| v.as_bytes() == w),
                        Ty::Int => w != &b"-"[..],
                    };
                    if !ok {
                        return false;
                    }
                    self.to(St::Close);
                    return self.step(b);
                }
                if self.st == St::Int {
                    return (b.is_ascii_digit() || (b == b'-' && self.n == 0)) && self.n < 18 && self.push(b);
                }
                let Ty::Str(a) = self.params()[self.key as usize].1 else { return false };
                self.push(b) && a.iter().any(|v| v.as_bytes().starts_with(&self.buf[..self.n as usize]))
            }
            St::Done => is_ws(b) && self.space(),
        }
    }
}

// ---- vocab trie (sorted ids + LCP), built at boot ----

static mut ORDER: [u32; MAXV] = [0; MAXV];
static mut LCP: [u8; MAXV] = [0; MAXV];
static mut NV: usize = 0; // tokens in ORDER
static mut NSAMPLE: usize = 0;
static mut OFFS: *const u32 = core::ptr::null();
static mut BYTES: *const u8 = core::ptr::null();
static mut STACK: [G; MAXLEN + 1] = [G0; MAXLEN + 1];

fn tok(t: usize) -> &'static [u8] {
    unsafe {
        let (a, b) = (*OFFS.add(t) as usize, *OFFS.add(t + 1) as usize);
        core::slice::from_raw_parts(BYTES.add(a), b - a)
    }
}

/// Boot (2B g_ctx loaded): sort the text tokens by their bytes.
pub fn init() {
    let t0 = smp::rdtsc();
    unsafe {
        (OFFS, BYTES) = (ctx(CTX_TOK_OFFS_PTR) as *const u32, ctx(CTX_TOK_BYTES_PTR) as *const u8);
        NSAMPLE = ctx(CTX_SAMPLE_N) as usize;
    }
    let nv = (ctx(CTX_VOCAB_SIZE) as usize).min(SPECIAL_MIN).min(MAXV);
    let (o, l) = unsafe { (&mut *core::ptr::addr_of_mut!(ORDER), &mut *core::ptr::addr_of_mut!(LCP)) };
    let mut n = 0;
    for t in 0..nv {
        let b = tok(t);
        if !b.is_empty() && b.len() <= MAXLEN {
            o[n] = t as u32;
            n += 1;
        }
    }
    o[..n].sort_unstable_by(|&a, &b| tok(a as usize).cmp(tok(b as usize)));
    l[0] = 0;
    for i in 1..n {
        let (a, b) = (tok(o[i - 1] as usize), tok(o[i] as usize));
        l[i] = a.iter().zip(b).take_while(|(x, y)| x == y).count().min(255) as u8;
    }
    unsafe { NV = n };
    crate::klog!(b'I', "gram: vocab trie {} of {} tokens sorted in {} ms, {} KiB (ids + prefix lengths), {} tokens over {} B never allowed in a call",
        n, ctx(CTX_VOCAB_SIZE), smp::tsc_to_us(smp::rdtsc() - t0) / 1000, n * 5 / 1024, nv - n, MAXLEN);
}

// ---- the call in progress ----

pub static mut ON: bool = true;
static mut CUR: G = G0;
static mut ACTIVE: bool = false;

#[derive(Clone, Copy)]
struct Stats {
    toks: u32,
    masked: u32,
    forced: u32,
    check_cyc: u64,
    mask_cyc: u64,
    mask_max: u64,
    broken: bool,
    at_tok: u32,
    bad: u32,    // the rejected token id
    steer: bool, // a mask chose a tool name, key, enum value or integer the model did not pick
}
const S0: Stats = Stats { toks: 0, masked: 0, forced: 0, check_cyc: 0, mask_cyc: 0, mask_max: 0, broken: false, at_tok: 0, bad: 0, steer: false };
static mut LAST_STEER: bool = false;

/// The last call had a name, key or value chosen by the mask (agent.rs does not run it if it has side effects).
pub fn last_steered() -> bool {
    unsafe { LAST_STEER }
}
static mut ST: Stats = S0;

fn s() -> &'static mut Stats {
    unsafe { &mut *core::ptr::addr_of_mut!(ST) }
}

pub fn active() -> bool {
    unsafe { ACTIVE }
}

/// The model emitted <tool_call>.
pub fn begin() {
    unsafe {
        CUR = G0;
        LAST_STEER = false;
        ACTIVE = true;
        ST = S0;
    }
}

fn accept(g: &mut G, t: usize) -> bool {
    if t >= SPECIAL_MIN {
        return t as u64 == llm::TOK_TOOL_CALL_END && g.st == St::Done;
    }
    let b = tok(t);
    if b.is_empty() || b.len() > MAXLEN {
        return false;
    }
    let mut x = *g;
    if b.iter().all(|&c| x.step(c)) {
        *g = x;
        return true;
    }
    false
}

/// -inf on every token the automaton rejects from g. Returns (allowed, the last allowed id).
fn mask(g: &G) -> (usize, usize) {
    let (o, l, nv) = unsafe { (&*core::ptr::addr_of!(ORDER), &*core::ptr::addr_of!(LCP), NV) };
    let st = unsafe { &mut *core::ptr::addr_of_mut!(STACK) };
    let logits = ctx(CTX_BUF_LOGITS) as *mut u32;
    st[0] = *g;
    let (mut count, mut last, mut dead) = (0, 0, usize::MAX);
    for i in 0..nv {
        let id = o[i] as usize;
        let d0 = l[i] as usize;
        let ok = d0 <= dead && {
            dead = usize::MAX;
            let b = tok(id);
            let mut d = d0;
            loop {
                if d == b.len() {
                    break true;
                }
                let mut x = st[d];
                if !x.step(b[d]) {
                    dead = d;
                    break false;
                }
                st[d + 1] = x;
                d += 1;
            }
        };
        if ok {
            count += 1;
            last = id;
        } else {
            unsafe { *logits.add(id) = NEG_INF };
        }
    }
    // tokens outside the trie: specials (only </tool_call> at the end), empty or too long ones
    let n = unsafe { NSAMPLE };
    for t in 0..n {
        let b = if t < SPECIAL_MIN { tok(t) } else { &[][..] };
        if t >= SPECIAL_MIN || b.is_empty() || b.len() > MAXLEN {
            if t as u64 == llm::TOK_TOOL_CALL_END && g.st == St::Done {
                count += 1;
                last = t;
            } else {
                unsafe { *logits.add(t) = NEG_INF };
            }
        }
    }
    (count, last)
}

/// The sampler picked t inside a call: t if the grammar allows it, else the pick over the masked logits.
pub fn fix(t: u64) -> u64 {
    let g = unsafe { &mut *core::ptr::addr_of_mut!(CUR) };
    let x = s();
    x.toks += 1;
    let t0 = smp::rdtsc();
    let ok = accept(g, t as usize);
    let t1 = smp::rdtsc();
    x.check_cyc += t1 - t0;
    if ok {
        return t;
    }
    if !unsafe { ON } {
        // passive checker: note where the call breaks, stop checking
        (x.broken, x.at_tok, x.bad) = (true, x.toks, t as u32);
        unsafe { ACTIVE = false };
        return t;
    }
    x.steer |= matches!(g.st, St::Name | St::Key | St::Enum | St::Int);
    let (count, last) = mask(g);
    let pick = if count == 1 { last as u64 } else { llm::pick_next() };
    let c = smp::rdtsc() - t1;
    (x.masked, x.mask_cyc, x.mask_max) = (x.masked + 1, x.mask_cyc + c, x.mask_max.max(c));
    x.forced += (count == 1) as u32;
    if count == 0 || !accept(g, pick as usize) {
        (x.broken, x.at_tok, x.bad) = (true, x.toks, t as u32); // cannot happen: every byte is a token
        unsafe { ACTIVE = false };
        return t;
    }
    pick
}

/// generate() is done: one line about the call (if there was one), also in the kernel log.
pub fn finish(ended: bool) {
    if !unsafe { ACTIVE } && s().toks == 0 {
        return;
    }
    let valid = ended && !s().broken && unsafe { CUR.st } == St::Done;
    unsafe { (ACTIVE, LAST_STEER) = (false, s().steer) };
    let x = *s();
    let mut l = SBuf::<200>::new();
    let us = |c: u64| smp::tsc_to_us(c);
    let _ = write!(l, "[grammar] {}: {} tok, ", if unsafe { ON } { "on" } else { "off (check only)" }, x.toks);
    if x.broken {
        let mut b = SBuf::<48>::new();
        tok(x.bad as usize).iter().take(24).for_each(|&c| { let _ = b.write_char(if c.is_ascii_graphic() || c == b' ' { c as char } else { '.' }); });
        let _ = write!(l, "call leaves the grammar at token {} ({} '{}'), ", x.at_tok, x.bad, b.as_str());
    }
    let _ = write!(l, "{} masked ({} forced), mask {} us avg {} us max, check {} ns/tok -> {}",
        x.masked, x.forced, us(x.mask_cyc) / x.masked.max(1) as u64, us(x.mask_max),
        us(x.check_cyc * 1000) / x.toks.max(1) as u64, if !valid { if ended { "INVALID call" } else { "unfinished call" } } else if x.steer { "valid call, steered" } else { "valid call" });
    println!("{}", l.as_str());
    crate::log::record(b'A', l.as_str());
    *s() = S0;
}

/// Boot: the cost of one full mask from a few states (it overwrites the logits buffer, the next forward
/// call writes it again).
pub fn bench() {
    let mut l = SBuf::<256>::new();
    let _ = write!(l, "gram: full mask cost (allowed tokens):");
    let at: [(&str, &[u8]); 5] = [
        ("call start", b""),
        ("tool name", b"\n<function=re"),
        ("after name", b"\n<function=set_setting>"),
        ("free value", b"\n<function=cat>\n<parameter=path>\n/no"),
        ("enum value", b"\n<function=set_setting>\n<parameter=key>\n"),
    ];
    for (what, pre) in at {
        let mut g = G0;
        pre.iter().for_each(|&b| { g.step(b); });
        let t0 = smp::rdtsc();
        let (n, _) = mask(&g);
        let _ = write!(l, " {} {} us ({}),", what, smp::tsc_to_us(smp::rdtsc() - t0), n);
    }
    l.n -= 1;
    crate::klog!(b'I', "{}", l.as_str());
}

/// Boot self-check of the automaton on byte strings (no model involved).
pub fn selftest() -> bool {
    let run = |s: &[u8]| {
        let mut g = G0;
        s.iter().all(|&b| g.step(b)) && g.st == St::Done
    };
    let good: [&[u8]; 4] = [
        b"\n<function=read_log>\n<parameter=n>\n5\n</parameter>\n</function>\n",
        b"\n<function=meminfo>\n</function>\n",
        b"\n<function=set_setting>\n<parameter=value>\n4\n</parameter>\n<parameter=key>\ncores\n</parameter>\n</function>",
        b"<function=write_file>\n<parameter=path>\n/a.txt\n</parameter>\n<parameter=content>\nx < y </p\n</parameter>\n</function>\n",
    ];
    let bad: [&[u8]; 7] = [
        b"\n<function=mem_info>\n</function>\n",                                                     // unknown tool
        b"\n<function=set_setting>\n<parameter=key>\ncolour\n</parameter>\n",                        // not in the enum
        b"\n<function=set_setting>\n<parameter=key>\ncores\n</parameter>\n</function>",              // value missing
        b"\n<function=read_log>\n<parameter=n>\nten\n</parameter>\n</function>",                     // not an integer
        b"\n<function=cat>\n<parameter=path>\n/a\n</parameter>\n<parameter=path>\n/b\n</parameter>", // twice
        b"\n<function=cat>\n<parameter=path>\n/a</function>",                                        // value not closed
        b"\n<function=cat>\n\n\n\n\n\n<parameter=path>",                                             // whitespace run
    ];
    good.iter().all(|s| run(s)) && !bad.iter().any(|s| run(s))
}
