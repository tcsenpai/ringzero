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
        CF = CF0;
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

/// f(id, allowed) for every token of the trie, from automaton state g (skips every token under a rejected prefix).
fn walk(g: &G, mut f: impl FnMut(usize, bool)) {
    let (o, l, nv) = unsafe { (&*core::ptr::addr_of!(ORDER), &*core::ptr::addr_of!(LCP), NV) };
    let st = unsafe { &mut *core::ptr::addr_of_mut!(STACK) };
    st[0] = *g;
    let mut dead = usize::MAX;
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
        f(id, ok);
    }
}

/// -inf on every token the automaton rejects from g. Returns (allowed, the last allowed id).
fn mask(g: &G) -> (usize, usize) {
    let logits = ctx(CTX_BUF_LOGITS) as *mut u32;
    let (mut count, mut last) = (0, 0);
    walk(g, |id, ok| {
        if ok {
            count += 1;
            last = id;
        } else {
            unsafe { *logits.add(id) = NEG_INF };
        }
    });
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
    let g0 = *g; // T28: the state the pick was made from
    let x = s();
    x.toks += 1;
    let t0 = smp::rdtsc();
    let ok = accept(g, t as usize);
    let t1 = smp::rdtsc();
    x.check_cyc += t1 - t0;
    if ok {
        conf(&g0, t);
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
    conf(&g0, pick);
    pick
}

// ---- T28: confidence of the decision tokens (method from ProbeLogits, arXiv:2604.11943) ----
//
// At a decision token of a call the margin is p1 - p2 of the softmax restricted to the tokens the grammar allows
// from the state before that token (raw logits, temperature 1), over the outcomes of the decision: tokens that lead
// to the same set of names still open are summed ("cat" and "ca" both mean cat), tokens that narrow nothing ('='
// before a tool name) are left out. Decision tokens: the tool name, a parameter name when
// an optional parameter is among the candidates, another optional parameter or the end of the call, an enum value,
// an integer digit or its end; never a free string value, and not a byte the grammar leaves only one way to write.
// The logits are the ones the pick used (a single decode step, or the verify row of the M4.8b/T39 pass: every pick
// goes through fix() above), only read: no extra forward pass, the sampling does not change.

/// `!set confidence on|off` (default on).
pub static mut CONF_ON: bool = true;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Name,
    Param,
    Opt,
    Enum,
    Int,
}

impl Kind {
    pub fn what(self) -> &'static str {
        match self {
            Kind::Name => "tool name",
            Kind::Param => "parameter",
            Kind::Opt => "optional parameter",
            Kind::Enum => "enum value",
            Kind::Int => "integer",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Conf {
    pub n: u32,       // decision tokens of the call
    pub min: f32,     // the smallest margin
    pub kind: Kind,   // where it was
    pub top: [u32; 2], // the two most likely allowed tokens there
    pub m: [f32; 8],  // the first 8 margins, in call order
    pub cyc: u64,     // TSC spent computing them
}
const CF0: Conf = Conf { n: 0, min: 1.0, kind: Kind::Name, top: [0; 2], m: [0.0; 8], cyc: 0 };
static mut CF: Conf = CF0;
static mut LAST_CF: Conf = CF0;

/// The margins of the last call (None: confidence off, or no decision token).
pub fn last_conf() -> Option<Conf> {
    let c = unsafe { LAST_CF };
    (unsafe { CONF_ON } && c.n > 0).then_some(c)
}

/// The decision the next byte b makes from state x, if it is one (more than one way on that the model chooses).
fn choice(x: &G, b: u8) -> Option<Kind> {
    if x.n == 0 && is_ws(b) {
        return None; // layout whitespace
    }
    let w = &x.buf[..x.n as usize];
    match x.st {
        St::Name => (SPECS.iter().filter(|s| s.0.as_bytes().starts_with(w)).count() > 1).then_some(Kind::Name),
        St::Key => {
            let p = x.params();
            let (mut k, mut opt) = (0, false);
            for j in (0..p.len()).filter(|&j| x.used & 1 << j == 0 && p[j].0.as_bytes().starts_with(w)) {
                (k, opt) = (k + 1, opt || !p[j].2);
            }
            (k > 1 && opt).then_some(Kind::Param) // only required ones left: their order is not a decision
        }
        St::Tag => {
            let p = x.params();
            let left = (0..p.len()).any(|j| x.used & 1 << j == 0);
            let done = p.iter().enumerate().all(|(j, y)| !y.2 || x.used & 1 << j != 0);
            (left && done && PARAM.starts_with(w) && CLOSE_F.starts_with(w)).then_some(Kind::Opt)
        }
        St::Enum => {
            let Ty::Str(a) = x.params()[x.key as usize].1 else { return None };
            (a.iter().filter(|v| v.as_bytes().starts_with(w)).count() > 1).then_some(Kind::Enum)
        }
        St::Int => Some(Kind::Int),
        _ => None,
    }
}

/// Token t from state g makes a decision on one of its bytes.
fn decision(g: &G, t: usize) -> Option<Kind> {
    if t >= SPECIAL_MIN {
        return None;
    }
    let mut x = *g;
    for &b in tok(t) {
        if let Some(k) = choice(&x, b) {
            return Some(k);
        }
        if !x.step(b) {
            return None;
        }
    }
    None
}

/// e^x for x <= 0, 0 below -30 (engine.rs exp_ref in f32: Taylor on x/1024, then squared 10 times).
fn expn(x: f32) -> f32 {
    if !(x > -30.0) {
        return 0.0;
    }
    let r = x / 1024.0;
    let mut s = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    for _ in 0..10 {
        s *= s;
    }
    s
}

/// The candidates of a name decision still open in state x: bit j = tool j (Name), unused parameter j (Key), enum
/// value j (Enum) that starts with the bytes written so far.
fn open(x: &G) -> u32 {
    let w = &x.buf[..x.n as usize];
    let bits = |it: &mut dyn Iterator<Item = (usize, bool)>| it.fold(0u32, |m, (j, ok)| if ok { m | 1 << j } else { m });
    match x.st {
        St::Name => bits(&mut SPECS.iter().enumerate().map(|(j, s)| (j, s.0.as_bytes().starts_with(w)))),
        St::Key => bits(&mut x.params().iter().enumerate().map(|(j, p)| (j, x.used & 1 << j == 0 && p.0.as_bytes().starts_with(w)))),
        St::Enum => match x.params()[x.key as usize].1 {
            Ty::Str(a) => bits(&mut a.iter().enumerate().map(|(j, v)| (j, v.as_bytes().starts_with(w)))),
            Ty::Int => 0,
        },
        _ => 0,
    }
}

/// Which outcome of the decision token t takes from g (None: t does not narrow it, e.g. '=' before a tool name or
/// '<' before '<parameter=' / '</function>'). Name, Key, Enum: the set of names left open (or the one chosen);
/// integer and optional-parameter decisions: the token itself.
fn outcome(g: &G, t: usize) -> Option<u64> {
    let mut x = *g;
    let b = tok(t);
    let mut i = 0;
    let k = loop {
        let c = *b.get(i)?;
        if let Some(k) = choice(&x, c) {
            break k;
        }
        if !x.step(c) {
            return None;
        }
        i += 1;
    };
    let (st0, m0) = (x.st, open(&x));
    let mut m = m0;
    let mut num: Option<u64> = None; // Int: FNV of the digits when the token ends the number
    for &c in &b[i..] {
        if x.st == St::Int && st0 == St::Int && num.is_none() && (is_ws(c) || c == b'<') {
            num = Some(x.buf[..x.n as usize].iter().fold(FNV, |h, &d| (h ^ d as u64).wrapping_mul(0x100_0000_01b3)));
        }
        if x.st == st0 && matches!(st0, St::Name | St::Key | St::Enum) {
            if c == b'>' || (st0 == St::Enum && (is_ws(c) || c == b'<')) {
                // the name ends here: the one that equals the bytes written
                let w = &x.buf[..x.n as usize];
                let mut e = 0;
                for j in 0..32 {
                    if m & 1 << j != 0 {
                        let name = match st0 {
                            St::Name => SPECS[j].0,
                            St::Key => x.params()[j].0,
                            _ => match x.params()[x.key as usize].1 { Ty::Str(a) => a[j], Ty::Int => "" },
                        };
                        if name.as_bytes() == w {
                            e |= 1 << j;
                        }
                    }
                }
                m = e;
            }
        }
        if !x.step(c) {
            return None;
        }
        if x.st == st0 && matches!(st0, St::Name | St::Key | St::Enum) {
            m = open(&x);
        }
    }
    match k {
        Kind::Name | Kind::Param | Kind::Enum => (m != m0).then_some(m as u64),
        // another parameter (2) or the end of the call (1), however the token spells it
        Kind::Opt => match x.st {
            St::Tag => {
                let w = &x.buf[..x.n as usize];
                match (PARAM.starts_with(w), CLOSE_F.starts_with(w)) {
                    (true, true) => None,
                    (p, _) => Some(1 << 63 | if p { 2 } else { 1 }),
                }
            }
            St::Done | St::Func => Some(1 << 63 | 1),
            _ => Some(1 << 63 | 2),
        },
        // the number written so far, and whether the token ends it
        Kind::Int => Some(match num {
            Some(h) => 1 << 62 | h >> 2,
            None => x.buf[..x.n as usize].iter().fold(FNV, |h, &d| (h ^ d as u64).wrapping_mul(0x100_0000_01b3)) >> 2,
        }),
    }
}

const FNV: u64 = 0xcbf2_9ce4_8422_2325;
const NOUT: usize = 64; // distinct outcomes kept per decision (more are rare: names, digits)
const NTOK: usize = 4096; // allowed tokens kept per decision
static mut DT: [(u64, f32); NTOK] = [(0, 0.0); NTOK];

/// Margin of the decision made from g: P(best outcome) - P(second) under the softmax restricted to the tokens the
/// grammar allows from g that narrow the decision (current logits, temperature 1). Tokens with the same outcome
/// are summed. Also the best token of the two outcomes.
fn margin(g: &G) -> (f32, [u32; 2]) {
    let lg = ctx(CTX_BUF_LOGITS) as *const u32;
    let dt = unsafe { &mut *core::ptr::addr_of_mut!(DT) };
    let (mut n, mut mx) = (0, f32::NEG_INFINITY);
    walk(g, |id, ok| {
        if !ok || n == NTOK {
            return;
        }
        let Some(o) = outcome(g, id) else { return };
        let l = f32::from_bits(unsafe { *lg.add(id) });
        dt[n] = (o, l);
        n += 1;
        mx = if l > mx { l } else { mx };
    });
    // per outcome: (key, sum of e^(l - mx), best logit, its token): the token is found again below
    let mut out = [(0u64, 0.0f32, f32::NEG_INFINITY); NOUT];
    let (mut k, mut total) = (0, 0.0f32);
    for &(o, l) in &dt[..n] {
        let e = expn(l - mx);
        total += e;
        let j = match out[..k].iter().position(|x| x.0 == o) {
            Some(j) => j,
            None if k < NOUT => {
                k += 1;
                out[k - 1].0 = o;
                k - 1
            }
            None => continue,
        };
        out[j].1 += e;
        out[j].2 = if l > out[j].2 { l } else { out[j].2 };
    }
    let (mut a, mut b) = (usize::MAX, usize::MAX);
    for j in 0..k {
        if a == usize::MAX || out[j].1 > out[a].1 {
            (b, a) = (a, j);
        } else if b == usize::MAX || out[j].1 > out[b].1 {
            b = j;
        }
    }
    if a == usize::MAX || !(total > 0.0) {
        return (1.0, [0; 2]);
    }
    let p2 = if b == usize::MAX { 0.0 } else { out[b].1 };
    // the best token of each of the two outcomes (for the log line)
    let mut top = [0u32; 2];
    let want = [Some(out[a]), (b != usize::MAX).then(|| out[b])];
    walk(g, |id, ok| {
        if !ok {
            return;
        }
        for (i, w) in want.iter().enumerate() {
            if let Some((o, _, best)) = w {
                if f32::from_bits(unsafe { *lg.add(id) }) == *best && outcome(g, id) == Some(*o) {
                    top[i] = id as u32;
                }
            }
        }
    });
    ((out[a].1 - p2) / total, top)
}

/// fix() kept or picked t from state g0: the margin if t is a decision token.
fn conf(g0: &G, t: u64) {
    if !unsafe { CONF_ON } {
        return;
    }
    let t0 = smp::rdtsc();
    let Some(k) = decision(g0, t as usize) else { return };
    let (m, top) = margin(g0);
    let c = unsafe { &mut *core::ptr::addr_of_mut!(CF) };
    if (c.n as usize) < c.m.len() {
        c.m[c.n as usize] = m;
    }
    if c.n == 0 || m < c.min {
        (c.min, c.kind, c.top) = (m, k, top);
    }
    c.n += 1;
    c.cyc += smp::rdtsc() - t0;
}

/// A token's text for a log line (printable ASCII, '.' else, at most 16 bytes).
pub fn show(t: u32) -> SBuf<20> {
    let mut b = SBuf::<20>::new();
    if (t as usize) < SPECIAL_MIN {
        tok(t as usize).iter().take(16).for_each(|&c| b.push(&[if c.is_ascii_graphic() || c == b' ' { c } else { b'.' }]));
    }
    b
}

// ---- T39: the grammar as a draft source for the speculative verify (llm.rs generate) ----

enum Next {
    Bytes,  // the piece in `o` is the only way on (the format puts one '\n' between parts)
    Choice, // the model chooses (a name, a value, which parameter, ...): the piece so far ends the draft
}

/// The only name of `cands` that starts with w, if there is exactly one.
fn unique<'a>(w: &[u8], cands: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let mut it = cands.filter(|c| c.as_bytes().starts_with(w));
    let a = it.next()?;
    it.next().is_none().then_some(a)
}

/// The next piece the grammar implies from g, in the canonical layout `<function=NAME>\n<parameter=KEY>\nVALUE\n
/// </parameter>\n...</function>\n`.
fn canon(g: &G, o: &mut SBuf<48>) -> Next {
    let w = &g.buf[..g.n as usize];
    let nl = |o: &mut SBuf<48>| if g.n == 0 && g.ws == 0 { o.push(b"\n") };
    match g.st {
        St::Func | St::Close => {
            nl(o);
            o.push(&(if g.st == St::Func { FUNC } else { CLOSE_P })[g.n as usize..]);
            Next::Bytes
        }
        St::Name | St::Key => {
            let c = if g.st == St::Name {
                unique(w, SPECS.iter().map(|s| s.0))
            } else {
                unique(w, g.params().iter().enumerate().filter(|(j, _)| g.used & 1 << j == 0).map(|(_, x)| x.0))
            };
            let Some(c) = c else { return Next::Choice };
            o.push(&c.as_bytes()[w.len()..]);
            o.push(if g.st == St::Key { b">\n" } else { b">" });
            Next::Bytes
        }
        St::Tag => {
            let p = g.params();
            let left = (0..p.len()).any(|j| g.used & 1 << j == 0);
            let done = p.iter().enumerate().all(|(j, x)| !x.2 || g.used & 1 << j != 0);
            nl(o);
            let (p_ok, f_ok) = (left && PARAM.starts_with(w), done && CLOSE_F.starts_with(w));
            match (p_ok, f_ok) {
                (true, false) => o.push(&PARAM[w.len()..]),
                (false, true) => o.push(&CLOSE_F[w.len()..]),
                _ => return Next::Choice, // another parameter or the end: the model's choice
            }
            Next::Bytes
        }
        St::Free if g.n > 0 && CLOSE_P.starts_with(w) => {
            o.push(&CLOSE_P[w.len()..]);
            Next::Bytes
        }
        St::Enum => {
            let Ty::Str(a) = g.params()[g.key as usize].1 else { return Next::Choice };
            let Some(c) = unique(w, a.iter().copied()) else { return Next::Choice };
            o.push(&c.as_bytes()[w.len()..]);
            o.push(b"\n");
            Next::Bytes
        }
        St::Done => {
            nl(o); // then </tool_call>: not drafted, it comes as the pick after a fully accepted draft
            Next::Choice
        }
        St::Free | St::Int => Next::Choice,
    }
}

/// The tokens the grammar implies after the call so far (structural text up to the model's next choice), at
/// most out.len(); 0 outside a call or with the grammar in check-only mode. Only a draft: the verify pass picks.
pub fn draft(out: &mut [u32]) -> usize {
    if !unsafe { ACTIVE && ON } || out.is_empty() {
        return 0;
    }
    let mut g = unsafe { CUR };
    let mut b = SBuf::<160>::new();
    for _ in 0..8 {
        let mut o = SBuf::<48>::new();
        let r = canon(&g, &mut o);
        if o.n == 0 || b.n + o.n > b.b.len() || !o.b[..o.n].iter().all(|&c| g.step(c)) {
            break;
        }
        b.push(&o.b[..o.n]);
        if matches!(r, Next::Choice) {
            break;
        }
    }
    llm::encode_plain(&b.b[..b.n], out)
}

/// generate() is done: one line about the call (if there was one), also in the kernel log.
pub fn finish(ended: bool) {
    unsafe { (LAST_CF, CF) = (CF, CF0) }; // T28: CF is CF0 when this generate() made no call
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
    println!("{}{}{}", crate::console::sgr(crate::console::DIM), l.as_str(), crate::console::sgr(crate::console::RESET));
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
    // T39: the canonical pieces from a few states (no tokenizer involved)
    let chain = |pre: &[u8]| {
        let mut g = G0;
        pre.iter().for_each(|&c| { g.step(c); });
        let mut b = SBuf::<160>::new();
        for _ in 0..8 {
            let mut o = SBuf::<48>::new();
            let r = canon(&g, &mut o);
            if o.n == 0 || !o.b[..o.n].iter().all(|&c| g.step(c)) {
                break;
            }
            b.push(&o.b[..o.n]);
            if matches!(r, Next::Choice) {
                break;
            }
        }
        b
    };
    let d = chain(b"\n<function=ca").as_str() == "t>\n<parameter=path>\n"
        && chain(b"\n<function=cat>\n<parameter=path>\n/a.txt\n</").as_str() == "parameter>\n</function>\n"
        && chain(b"\n<function=meminfo").as_str() == ">\n</function>\n"
        && chain(b"\n<function=set_setting>\n<parameter=key>\ncor").as_str() == "es\n</parameter>\n<parameter=value>\n"
        && chain(b"").as_str() == "\n<function=";
    // T28: which next byte is a decision, and e^x
    let at = |pre: &[u8], b: u8| {
        let mut g = G0;
        pre.iter().for_each(|&c| { g.step(c); });
        choice(&g, b)
    };
    let c = at(b"\n<function=", b'c') == Some(Kind::Name)
        && at(b"\n<function=set_", b's').is_none()
        && at(b"\n<function=set_setting>\n<parameter=", b'k').is_none()
        && at(b"\n<function=run_wasm>\n<parameter=", b'p') == Some(Kind::Param)
        && at(b"\n<function=read_log>\n", b'<') == Some(Kind::Opt)
        && at(b"\n<function=meminfo>\n", b'<').is_none()
        && at(b"\n<function=set_setting>\n<parameter=key>\n", b'c') == Some(Kind::Enum)
        && at(b"\n<function=read_log>\n<parameter=n>\n", b'5') == Some(Kind::Int)
        && at(b"\n<function=read_log>\n<parameter=n>\n", b'\n').is_none()
        && at(b"\n<function=cat>\n<parameter=path>\n", b'/').is_none();
    let de = expn(-1.0) - 0.367_879;
    let e = expn(0.0) == 1.0 && de < 1e-4 && de > -1e-4 && expn(-31.0) == 0.0;
    good.iter().all(|s| run(s)) && !bad.iter().any(|s| run(s)) && d && c && e
}
