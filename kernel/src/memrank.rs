// T72: design A of docs/research/MEMORY_SYSTEMS.md, the ranking rules frozen on the host in T70
// (docs/research/MEMORY_AB_RESULT.md, tools/memeval.py `rank`, docs/spikes/memab/config.json "A"), no model call:
//  - hub penalty: an entry close to generic questions ("what's up?", "cosa ho?") wins too many short queries; its
//    hub score is the mean cosine to its HUB_K nearest PROBES and HUB_W of it is subtracted from its cosine
//  - owner: relation words (IT/EN, also in fact keys like sister_dog) and an alias table (name -> relation, from the
//    memories) say whose an entry is; a memory owned by someone the query does not name loses OWN_OTHER, a memory
//    that lacks the owner the query names loses OWN_MISSING
//  - supersession and the manual apart are applied by agentd on the ranked hits (SUP_*, MAN_MARGIN)
// The floors (agentd RECALL_MIN, RECALL_MIN_BM25) read the raw cosine, never the lowered score (T70 found the
// penalties push everything under them otherwise).
use crate::memidx;
use alloc::string::String;
use alloc::vec::Vec;

/// Generic questions for the hub score (tools/memeval.py PROBES, written for T70, none from the eval set).
pub const PROBES: [&str; NP] = [
    "what's up?", "tell me something", "can you help me?", "I have a question", "what do you know?",
    "what did we talk about?", "remind me", "what happened?", "is everything ok?", "what should I do?",
    "what is this?", "show me", "anything new?", "what did you do?", "can you check?", "did it work?",
    "what was that?", "tell me more", "explain", "what's the status?", "how are things?", "any news?",
    "what did I say?", "what do I have?", "where is it?", "when is it?", "who is it?", "how much?",
    "come va?", "dimmi qualcosa", "mi aiuti?", "ho una domanda", "cosa sai?", "di cosa abbiamo parlato?",
    "ricordami", "cosa e' successo?", "va tutto bene?", "cosa devo fare?", "cos'e' questo?", "fammi vedere",
    "novita'?", "cosa hai fatto?", "puoi controllare?", "ha funzionato?", "cos'era?", "dimmi di piu'",
    "spiegami", "a che punto siamo?", "cosa ho detto?", "cosa ho?", "dov'e'?", "quando?", "chi e'?",
    "quanto?",
];
pub const NP: usize = 54;
const HUB_K: usize = 10;
/// frozen config A: hub 0.5, owner 0.03 / 0.06, supersession (0.6, 0.2) over the top 8, manual margin 0.06
pub const HUB_W: f32 = 0.5;
const OWN_OTHER: f32 = 0.03;
const OWN_MISSING: f32 = 0.06;
/// supersession: of the first SUP_TOP non-manual hits, those within SUP_MARGIN of the first whose row has a cosine
/// of SUP_DD or more with the first's row are one group; the newest stands for it, the older ones leave the answer
pub const SUP_DD: f32 = 0.6;
pub const SUP_MARGIN: f32 = 0.2;
pub const SUP_TOP: usize = 8;
/// the best manual section goes first, for any query, when it scores this much above the best other hit
pub const MAN_MARGIN: f32 = 0.06;

/// Mean of the HUB_K largest of the probe cosines (c is reordered).
pub fn hub_of(c: &mut [f32]) -> f32 {
    c.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    c[..HUB_K].iter().sum::<f32>() / HUB_K as f32
}

// ---- owner ----

/// relation -> its words (the bit of a relation in an owner mask is its index)
const REL: [&[&str]; 7] = [
    &["sorella", "sister"],
    &["fratello", "brother"],
    &["madre", "mamma", "mother", "mum", "mom"],
    &["padre", "papa", "father", "dad"],
    &["collega", "colleague", "coworker"],
    &["amico", "amica", "friend"],
    &["ragazza", "fidanzata", "ragazzo", "fidanzato", "partner", "girlfriend", "boyfriend", "moglie", "marito", "wife", "husband"],
];
/// capitalized words that are not names
const CAPSTOP: [&str; 18] = ["User", "User's", "The", "Il", "La", "Lo", "Le", "Gli", "Un", "Una", "Task", "Tool", "Past", "Promemoria",
    "Errore", "Default", "Checking", "Acme"];

/// Relations named in s: runs of ASCII letters matched case-sensitively, so "Brother" (a printer) is a name.
fn rel_words(s: &str) -> u8 {
    let mut m = 0;
    for w in s.split(|c: char| !c.is_ascii_alphabetic()).filter(|w| !w.is_empty()) {
        for (i, ws) in REL.iter().enumerate() {
            if ws.contains(&w) {
                m |= 1 << i;
            }
        }
    }
    m
}

/// Words as memeval's `[^\W\d_]+'?[^\W\d_]*`: letters, then at most one apostrophe and more letters.
fn name_words(s: &str) -> Vec<&str> {
    let mut v = Vec::new();
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if !c.is_alphabetic() {
            continue;
        }
        let mut end = i + c.len_utf8();
        let mut apo = false;
        while let Some(&(j, d)) = it.peek() {
            if d.is_alphabetic() {
                end = j + d.len_utf8();
            } else if d == '\'' && !apo {
                apo = true;
                end = j + 1;
            } else {
                break;
            }
            it.next();
        }
        v.push(&s[i..end]);
    }
    v
}

/// The body of an index text (after its "[source] " tag).
fn body(t: &str) -> &str {
    t.split_once("] ").map_or(t, |x| x.1)
}

/// A memory entry (not a file chunk, not a manual section): only these get owner penalties.
pub fn is_memory(t: &str) -> bool {
    !(t.starts_with("[/") || t.starts_with("[manual: "))
}

/// Name -> relations: in a memory with a relation word, the capitalized word nearest to it (5 words either side;
/// the first in byte order on a tie) is that person's or pet's name ("mia sorella Chiara").
fn aliases<'a>(texts: impl Iterator<Item = &'a str>, al: &mut Vec<(String, u8)>) {
    for t in texts {
        let w = name_words(body(t));
        for (i, x) in w.iter().enumerate() {
            let r = rel_words(x);
            if r == 0 {
                continue;
            }
            let near = (i.saturating_sub(5)..(i + 6).min(w.len()))
                .filter(|&j| w[j].chars().next().is_some_and(|c| c.is_uppercase()) && !CAPSTOP.contains(&w[j]) && rel_words(w[j]) == 0)
                .map(|j| (j.abs_diff(i), w[j]))
                .min();
            if let Some((_, n)) = near {
                let n = n.to_lowercase();
                match al.iter_mut().find(|(k, _)| *k == n) {
                    Some(e) => e.1 |= r,
                    None => al.push((n, r)),
                }
            }
        }
    }
}

/// Whose s is: its relation words, and the relations of the names in it.
fn owner(s: &str, al: &[(String, u8)]) -> u8 {
    let l = s.to_lowercase();
    l.split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty())
        .fold(rel_words(s), |m, w| m | al.iter().find(|(k, _)| k == w).map_or(0, |e| e.1))
}

struct Cache {
    gen: u64,
    al: Vec<(String, u8)>,
    own: Vec<u8>, // per entry, 0xff = not a memory
}
static mut C: Cache = Cache { gen: u64::MAX, al: Vec::new(), own: Vec::new() };

/// Per entry, what design A subtracts from its cosine for query q: HUB_W x hub, plus the owner penalties.
pub fn penalties(q: &str) -> Vec<f32> {
    let c = unsafe { &mut *core::ptr::addr_of_mut!(C) };
    let n = memidx::len();
    if c.gen != memidx::gen() {
        c.al.clear();
        aliases((0..n).map(memidx::text).filter(|t| is_memory(t)), &mut c.al);
        c.own = (0..n).map(|d| {
            let t = memidx::text(d);
            if is_memory(t) { owner(t, &c.al) } else { 0xff }
        }).collect();
        c.gen = memidx::gen();
    }
    let qo = owner(q, &c.al);
    let hub = memidx::hubs();
    (0..n).map(|d| {
        let mut p = HUB_W * hub[d];
        let e = c.own[d];
        if e != 0xff {
            if e & !qo != 0 {
                p += OWN_OTHER;
            }
            if qo != 0 && e & qo == 0 {
                p += OWN_MISSING;
            }
        }
        p
    }).collect()
}

/// Boot self-check of the owner rule (no index, no model).
pub fn selftest() -> bool {
    let mut al = Vec::new();
    aliases(["[fact] sister_city: mia sorella Chiara vive a Parma", "La stampante e' una Brother HL-L2350DW",
        "[said] il cane di mia sorella si chiama Rocco"].into_iter(), &mut al);
    let sis = 1u8;
    let w = name_words("dell'utente e' Chiara's x2y") == ["dell'utente", "e'", "Chiara's", "x", "y"];
    w && rel_words("sister_dog") == sis && rel_words("Brother HL") == 0 && owner("dove vive Chiara?", &al) == sis
        && owner("che stampante ho?", &al) == 0 && al.iter().any(|(k, m)| k == "rocco" && *m == sis)
        && (hub_of(&mut [1.0f32; NP]) - 1.0).abs() < 1e-6
}
