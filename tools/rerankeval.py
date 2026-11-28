#!/usr/bin/env python3
"""T63 host spike: does Qwen3-Reranker-0.6B, reranking the top 20 of Qwen3-Embedding-0.6B, put the right entry first
more often than the embedder alone? One pooled index like the kernel's (memories, manual sections, file chunks).

  rerankeval.py run OUT.json      host only (torch, transformers; CPU, 8 threads): embed, rank, rerank, time
  rerankeval.py report OUT.json   top-1 / top-3 per pool and split, K picked on the tune split, latency
  rerankeval.py check             data self-check (every gold key exists), no model

Embedder as in kernel/src/embed.rs: plain BPE, <|endoftext|> appended, queries "Instruct: {TASK}\\nQuery:{q}",
documents bare, last-token pooling, 1024 dims, L2. Texts as agentd stores them: manual sections
"[manual: TITLE] body" (agentd::manual), file chunks "[PATH] text" cut like fidx::cut (1000 bytes). Reranker
prompt from its model card, with the kernel's TASK as the instruction (not tuned). Split: even query = tune,
odd = held-out."""
import ast, json, os, re, sys, time

RZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = re.search(r'pub const TASK: &str = "([^"]+)"', open(RZ + "/kernel/src/embed.rs").read()).group(1)
EOT = 151643

# the 40 memories and 20 queries of tools/retrieval.py (read as data, that script runs on import)
_rt = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(RZ + "/tools/retrieval.py").read()).body
       if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("MEM", "QS")}
MEM = _rt["MEM"] + [  # T63: in agentd's own formats, several of them near-misses of an older memory
    "[fact] sister_dog: il cane di mia sorella si chiama Rocco",  # 40
    "[fact] brother: mio fratello si chiama Luca e vive a Torino",  # 41
    "[episode 3] The user asked what is in /notes; the agent listed todo.txt, spesa.txt and idee.md.",  # 42
    "[fact] favourite_colour: verde",  # 43
    "[said] ricordati che lavoro come insegnante di matematica al liceo",  # 44
    "[fact] home_city: Bologna",  # 45
    "[episode 5] Disk check: !df showed 1.2 GB free of 4 GB on the ringzero disk.",  # 46
    "[fact] router: FritzBox 7590 in the hallway, admin page at fritz.box",  # 47
    "[fact] coffee_at_home: moka da 3 tazze, caffe' macinato fine",  # 48
    "[episode 7] The user ran /demo/stats.py with python3; it wrote /demo/report.json, mean age 38.8.",  # 49
    "[fact] doctor: la dottoressa Bianchi e' il medico di base",  # 50
    "[fact] cat_vet: il veterinario di Pixel e' in via Roma 12, Bologna",  # 51
    "[said] my favourite programming language is Rust",  # 52
    "[fact] laptop: ThinkPad X1 Carbon gen 11 with 32 GB",  # 53
    "[episode 9] The agent set the keyboard layout to it after the user asked for the Italian keyboard.",  # 54
]

FILES = {  # synthetic user files, plus the real demo files of the disk image (tools/demo -> /demo)
    "/notes/todo.txt": "- comprare il latte\n- chiamare il dentista lunedi' alle 9\n- finire il report M4\n"
                       "- rinnovare il certificato TLS del NAS\n- portare Pixel dal veterinario\n",
    "/notes/spesa.txt": "Lista della spesa\nlatte, uova, guanciale, pecorino romano, pane, mele, caffe' macinato\n",
    "/notes/idee.md": "# Idee\n- un gioco di carte in Lua per ringzero\n- dashboard del consumo elettrico con sqlite3\n"
                      "- tradurre il manuale in italiano\n",
    "/home/todo.md": "# TODO\n- [ ] backup photos to the NAS\n- [ ] update the router firmware\n- [x] book the dentist\n",
    "/home/ricetta_carbonara.txt": "Carbonara per 2: 200 g di spaghetti, 100 g di guanciale, 3 tuorli, 50 g di pecorino. "
                                   "Rosolare il guanciale senza olio, mescolare tuorli e pecorino, unire fuori dal fuoco "
                                   "con un po' di acqua di cottura. Niente panna.\n",
    "/home/meeting_acme.md": "# Acme, weekly call 2026-11-24\n- Acme wants the export in CSV, not JSON\n"
                             "- deadline moved to December 10\n- next call: Tuesday 10:00 CET\n",
    "/home/viaggio_giappone.md": "# Viaggio in Giappone\nPartenza 3 aprile da Bologna, ritorno 17 aprile.\n"
                                 "Tokyo 5 notti, poi Kyoto (ryokan vicino a Gion) e Osaka. JR Pass di 14 giorni.\n",
    "/etc/hostname": "ringzero\n",
    "/etc/os-release": 'NAME="ringzero"\nVERSION="0.8"\nID=ringzero\nPRETTY_NAME="ringzero 0.8"\n',
}
for f in sorted(os.listdir(RZ + "/tools/demo")):
    FILES["/demo/" + f] = open(RZ + "/tools/demo/" + f).read()


def cut(b):
    """fidx::cut: at most 1000 bytes, preferring a newline, then a space, past 500."""
    if len(b) <= 1000:
        return len(b)
    w = b[:1000]
    for c in (b"\n", b" "):
        i = w.rfind(c, 500)
        if i >= 0:
            return i + 1
    return 1000


def corpus():
    """[(key, pool, text)]: keys m<i>, man:<title>, f:<path>#<chunk>."""
    out = [(f"m{i}", "memory", t) for i, t in enumerate(MEM)]
    for sec in open(RZ + "/docs/MANUAL.md").read().split("\n## ")[1:]:
        t, b = (sec.split("\n", 1) + [""])[:2]
        out.append((f"man:{t.strip()}", "manual", f"[manual: {t.strip()}] {b.strip()}"))
    for p, txt in FILES.items():
        b, k = txt.encode(), 0
        while b:
            n = cut(b)
            out.append((f"f:{p}#{k}", "files", f"[{p}] {b[:n].decode(errors='replace')}".strip()))
            b, k = b[n:], k + 1
    return out


# (query, pool of the target, kind, gold keys: m<i>, man:<title>, f:<path> = any chunk of it)
# kind: para = paraphrase, exact = identifier or file token, hard = a near-miss entry exists (T49 error cases too)
NOTES = {"m42", "f:/notes/todo.txt", "f:/notes/spesa.txt", "f:/notes/idee.md"}
Q = [(q, "memory", "exact" if i >= 15 else "para", {f"m{g}" for g in gold})  # retrieval.py: 15 paraphrased, 5 exact
     for i, (q, gold) in enumerate(_rt["QS"])] + [
    ("come si chiama il mio gatto?", "memory", "hard", {"m11"}),
    ("come si chiama mio fratello?", "memory", "hard", {"m41"}),
    ("come si chiama la mia ragazza?", "memory", "hard", {"m25"}),
    ("qual e' il mio colore preferito?", "memory", "para", {"m43"}),
    ("che lavoro faccio?", "memory", "para", {"m44"}),
    ("in che citta' vivo?", "memory", "hard", {"m45"}),
    ("how much free space is left on the ringzero disk?", "memory", "hard", {"m46"}),
    ("what router do I have?", "memory", "para", {"m47"}),
    ("come faccio il caffe' a casa?", "memory", "hard", {"m48"}),
    ("what did the stats script print last time?", "memory", "hard", {"m49"}),
    ("come si chiama il mio medico?", "memory", "hard", {"m50"}),
    ("quando riceve il medico?", "memory", "hard", {"m37"}),
    ("dove porto Pixel dal veterinario?", "memory", "para", {"m51"}),
    ("which language do I like to code in?", "memory", "para", {"m52"}),
    ("che portatile ho?", "memory", "para", {"m53"}),
    ("did we change the keyboard layout?", "memory", "hard", {"m54"}),
    ("X1 Carbon", "memory", "exact", {"m53"}),
    ("FritzBox", "memory", "exact", {"m47"}),
    ("how is my partner called?", "memory", "para", {"m25"}),
    ("when is our anniversary?", "memory", "para", {"m25"}),
    ("quando vado in palestra?", "memory", "para", {"m29"}),
    ("how do I reach the Fedora server?", "memory", "para", {"m3"}),
    ("do I have a certificate to renew?", "memory", "para", {"m6"}),
    ("what do I listen to when I need to focus?", "memory", "para", {"m33"}),
    ("che editor uso?", "memory", "para", {"m26"}),
    ("come faccio a ibernare?", "manual", "para", {"man:Hibernation"}),
    ("how do I save the session to disk before turning off?", "manual", "para", {"man:Hibernation"}),
    ("come torno indietro di due turni?", "manual", "para", {"man:Undo and redo a turn"}),
    ("how do I switch the keyboard to Italian?", "manual", "hard", {"man:Keyboard layout"}),
    ("come eseguo uno script python?", "manual", "para", {"man:Running Python"}),
    ("how do I run a wasm program?", "manual", "para", {"man:Running WASI programs"}),
    ("quali programmi sono installati?", "manual", "para", {"man:Programs in /bin"}),
    ("how do I query a sqlite database?", "manual", "para", {"man:SQLite, Lua, JavaScript, jq and the editor"}),
    ("come cambio la temperatura?", "manual", "para", {"man:Temperature and thinking"}),
    ("how do I see the kernel log?", "manual", "para", {"man:Kernel log"}),
    ("perche' mi chiede Run it y/N?", "manual", "para", {"man:Confidence of tool calls"}),
    ("how do I turn off the memory slot?", "manual", "para", {"man:Memory slot and consolidation"}),
    ("come cerco nei miei file?", "manual", "para", {"man:Memory index and search"}),
    ("how full is the context?", "manual", "para", {"man:Context size and compaction"}),
    ("come ricomincio una nuova conversazione?", "manual", "para", {"man:New conversation"}),
    ("how do I create a folder?", "manual", "para", {"man:Writing and deleting files"}),
    ("how many tokens per second does it do?", "manual", "para", {"man:Speed and benchmarks"}),
    ("what model runs this OS?", "manual", "para", {"man:What ringzero is"}),
    ("come faccio a fargli dimenticare una cosa?", "manual", "para", {"man:Saved facts"}),
    ("how many tool calls can the agent make for one question?", "manual", "para", {"man:Tool round limit"}),
    ("dove tiene ringzero i suoi dati?", "manual", "para", {"man:Devices, /tmp and ringzero's own data"}),
    ("quanto e' grande il disco?", "manual", "hard", {"man:System information in /sys and /etc", "man:Listing and reading files"}),
    ("how do I give the agent a long document to read?", "manual", "para", {"man:Reading a long document"}),
    ("cosa ha deciso la memoria sul mio ultimo messaggio?", "manual", "para", {"man:What the memory system decided"}),
    ("which cores are used for inference?", "manual", "hard", {"man:System information in /sys and /etc", "man:CPU, cores, memory and uptime"}),
    ("how do I make a branch of the conversation?", "manual", "para", {"man:Branches and saved points"}),
    ("/hibernate discard", "manual", "exact", {"man:Hibernation"}),
    ("!set spec", "manual", "exact", {"man:Speed and benchmarks", "man:Settings"}),
    ("/agentd log", "manual", "exact", {"man:What the memory system decided"}),
    ("cosa devo comprare al supermercato?", "files", "para", {"f:/notes/spesa.txt"}),
    ("what's on my todo list?", "files", "para", {"f:/notes/todo.txt", "f:/home/todo.md"}),
    ("che idee avevo per i progetti?", "files", "para", {"f:/notes/idee.md"}),
    ("what's the machine's hostname?", "files", "hard", {"f:/etc/hostname"}),
    ("which version of ringzero is installed?", "files", "hard", {"f:/etc/os-release", "f:/demo/data.json"}),
    ("how do I make carbonara?", "files", "para", {"f:/home/ricetta_carbonara.txt"}),
    ("cosa abbiamo deciso alla riunione con Acme?", "files", "hard", {"f:/home/meeting_acme.md"}),
    ("quando parto per il Giappone?", "files", "hard", {"f:/home/viaggio_giappone.md"}),
    ("which script computes the mean age?", "files", "para", {"f:/demo/stats.py"}),
    ("who lives in Milano?", "files", "para", {"f:/demo/people.csv"}),
    ("people.csv", "files", "exact", {"f:/demo/people.csv"}),
    ("fib(22)", "files", "exact", {"f:/demo/hello.js"}),
    ("which program counts the words of the quick brown fox?", "files", "para", {"f:/demo/hello.lua"}),
    ("quali modelli sono elencati nel file json?", "files", "para", {"f:/demo/data.json"}),
    ("anna.rossi@example.org", "files", "exact", {"f:/demo/people.csv"}),
    ("dove ho annotato il dentista?", "files", "hard", {"f:/notes/todo.txt", "f:/home/todo.md"}),
    ("what did Acme ask for in the last call?", "files", "hard", {"f:/home/meeting_acme.md"}),
    ("what's in /notes?", "files", "hard", NOTES),
    ("ryokan Kyoto", "files", "exact", {"f:/home/viaggio_giappone.md"}),
]
KS = (3, 5, 10, 20)
CAP = 96  # the capped variant: each document cut to its first CAP tokens before reranking


def ok(key, gold):
    return key in gold or key.split("#")[0] in gold


def check():
    keys = {k for k, _, _ in corpus()}
    for q, _, _, gold in Q:
        assert all(any(ok(k, {g}) for k in keys) for g in gold), (q, gold)
    assert len(Q) == len({q for q, *_ in Q}), "duplicate query"
    print(f"ok: {len(keys)} entries, {len(Q)} queries")


def free_gb():
    return int(re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read()).group(1)) >> 20


EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"  # memeval.py --embedder DIR: a fine-tuned checkpoint (tools/embft/train.py)


def embedder():
    """emb(text) -> (L2-normalized last-token vector, tokens), as kernel/src/embed.rs (host: fp32, CPU, 8 threads).
    Dropping emb frees the model."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(8)
    et = AutoTokenizer.from_pretrained(EMBEDDER)
    em = AutoModel.from_pretrained(EMBEDDER, dtype=torch.float32).eval()

    def emb(text):
        ids = et(text, add_special_tokens=False)["input_ids"] + [EOT]
        with torch.no_grad():
            h = em(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
        return torch.nn.functional.normalize(h, dim=-1), len(ids)
    return emb


def run(out):
    """Embedder pass first (then freed), reranker one pair per forward (no padding); OUT.partial resumes a stop."""
    import gc, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    C = corpus()
    emb = embedder()
    D = torch.stack([emb(t)[0] for _, _, t in C])
    rows = []
    for i, (q, pool, kind, gold) in enumerate(Q):
        t0 = time.perf_counter()
        v, nq = emb(f"Instruct: {TASK}\nQuery:{q}")
        cos = (D @ v).tolist()
        e_ms = (time.perf_counter() - t0) * 1000
        order = sorted(range(len(C)), key=lambda d: -cos[d])
        rows.append({"i": i, "split": "tune" if i % 2 == 0 else "held-out", "q": q, "pool": pool, "kind": kind,
                     "gold": sorted(gold), "q_tokens": nq, "embed_ms": e_ms,
                     "top20": [{"key": C[d][0], "cos": cos[d]} for d in order[:20]],
                     "gold_rank_embed": next((n for n, d in enumerate(order) if ok(C[d][0], gold)), None)})
    del emb, D
    gc.collect()
    text = {k: t for k, _, t in C}
    rt = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
    rm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", dtype=torch.float32).eval()
    yes, no = rt.convert_tokens_to_ids("yes"), rt.convert_tokens_to_ids("no")
    pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
           "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    tk = lambda s: rt(s, add_special_tokens=False)["input_ids"]
    suf_ids = tk(suf)

    def rerank(q, docs, cap):
        head = tk(pre + f"<Instruct>: {TASK}\n<Query>: {q}\n<Document>: ")
        bodies = [tk(d)[:cap] if cap else tk(d) for d in docs]
        t0, py = time.perf_counter(), []
        with torch.no_grad():  # last position only: full-vocab logits for every row took 10 GB of RAM
            for b in bodies:
                lg = rm(input_ids=torch.tensor([head + b + suf_ids]), logits_to_keep=1).logits[0, -1]
                py.append(torch.softmax(torch.stack([lg[no], lg[yes]]), 0)[1].item())
        ms = (time.perf_counter() - t0) * 1000
        return py, {"shared": len(head), "unique": [len(b) + len(suf_ids) for b in bodies], "ms": ms}
    done = json.load(open(out + ".partial")) if os.path.exists(out + ".partial") else {}
    for r in rows:
        if str(r["i"]) in done:
            r.update(done[str(r["i"])])
            continue
        # the build box is shared: 8 GB free to start, then stop before the box passes 90% (32 GB: 3.2 GB free)
        assert free_gb() >= (8 if r is rows[0] or not done else 4), f"MemAvailable {free_gb()} GB, stopped (rerun to resume)"
        for name, cap in (("full", 0), ("cap", CAP)):
            py, info = rerank(r["q"], [text[h["key"]] for h in r["top20"]], cap)
            for h, p in zip(r["top20"], py):
                h["p_" + name] = p
            r["rr_" + name] = info
        done[str(r["i"])] = {k: r[k] for k in ("top20", "rr_full", "rr_cap")}
        json.dump(done, open(out + ".partial", "w"))
        g = set(r["gold"])
        e1, r1 = r["top20"][0]["key"], max(r["top20"], key=lambda h: h["p_full"])["key"]
        print(r["i"], "emb", "ok " if ok(e1, g) else "MISS", "rr", "ok " if ok(r1, g) else "MISS",
              f"{r['rr_full']['ms']:.0f}ms", f"free {free_gb()} GB", r["q"][:50], flush=True)
    json.dump({"task": TASK, "cap": CAP, "n_entries": len(C), "rows": rows}, open(out, "w"), indent=1)
    os.remove(out + ".partial")


def ranked(r, how, k):
    """Keys in final order: 'embed' = cosine; 'full'/'cap' = top k reranked by P(yes), the rest in cosine order."""
    h = r["top20"]
    if how == "embed":
        return [x["key"] for x in h]
    return [x["key"] for x in sorted(h[:k], key=lambda x: -x["p_" + how])] + [x["key"] for x in h[k:]]


def topn(rows, how, k, n):
    return sum(any(ok(x, set(r["gold"])) for x in ranked(r, how, k)[:n]) for r in rows)


def est_ms(r, how, k):
    """[E] ring-0 reranker time: the shared prefix once (snapshot after it), then each document + suffix.
    Per-token path 6 ms/token (the embedder, docs/EMBEDDER_RESULT.md); batched prefill 700-800 tok/s
    (DECISION_MODELS.md section 3): 1/0.75 ms per token as the midpoint."""
    t = r["rr_" + how]["shared"] + sum(r["rr_" + how]["unique"][:k])
    return t, t * 6.0, t / 0.75


def report(path):
    import statistics as st
    R = json.load(open(path))
    rows = R["rows"]
    pools = ("memory", "manual", "files")
    print(f"{R['n_entries']} entries, {len(rows)} queries; instruction: {R['task']}")
    # pick (variant, k) on the tune split: most top-1, then the cheaper one
    tune = [r for r in rows if r["split"] == "tune"]
    cands = [(topn(tune, how, k, 1), -st.median(est_ms(r, how, k)[0] for r in tune), how, k)
             for how in ("full", "cap") for k in KS]
    best = max(cands)
    print(f"picked on tune: rerank {best[2]} top-{best[3]} ({best[0]}/{len(tune)} top-1 on tune)")
    hdr = "| split | pool | n | embed top-1 | embed top-3 | " + " | ".join(
        f"rr {how} K={k} top-1" for how in ("full", "cap") for k in KS) + " |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for split in ("tune", "held-out", "all"):
        for pool in pools + ("all",):
            s = [r for r in rows if split in ("all", r["split"]) and pool in ("all", r["pool"])]
            cells = [f"{topn(s, 'embed', 0, 1)}", f"{topn(s, 'embed', 0, 3)}"] + [
                f"{topn(s, how, k, 1)}" for how in ("full", "cap") for k in KS]
            print(f"| {split} | {pool} | {len(s)} | " + " | ".join(cells) + " |")
    print("\ntop-3 after reranking (held-out / all):")
    for how in ("full", "cap"):
        for k in KS:
            ho = [r for r in rows if r["split"] == "held-out"]
            print(f"  {how} K={k}: {topn(ho, how, k, 3)}/{len(ho)}  {topn(rows, how, k, 3)}/{len(rows)}")
    print("\nby kind (all queries): embed top-1 vs rr full K=20 top-1")
    for kind in ("para", "exact", "hard"):
        s = [r for r in rows if r["kind"] == kind]
        print(f"  {kind}: n {len(s)}  embed {topn(s, 'embed', 0, 1)}  rr {topn(s, 'full', 20, 1)}  "
              f"rr K=5 {topn(s, 'full', 5, 1)}")
    miss20 = [r for r in rows if r["gold_rank_embed"] is None or r["gold_rank_embed"] >= 20]
    print(f"gold outside the embedder's top 20: {len(miss20)}: " + "; ".join(r["q"] for r in miss20))
    print("\nlatency (median over queries; host = PyTorch fp32, 8 threads, one pair per forward)")
    print(f"  embedder query: {st.median(r['embed_ms'] for r in rows):.0f} ms host, "
          f"{st.median(r['q_tokens'] for r in rows)} tokens")
    for how in ("full", "cap"):
        print(f"  host, all 20 pairs ({how}): {st.median(r['rr_' + how]['ms'] for r in rows):.0f} ms")
        for k in KS:
            e = [est_ms(r, how, k) for r in rows]
            print(f"  {how} K={k}: tokens {st.median(x[0] for x in e):.0f} (shared {st.median(r['rr_' + how]['shared'] for r in rows):.0f}), "
                  f"[E] ring 0 per-token {st.median(x[1] for x in e)/1000:.2f} s, batched {st.median(x[2] for x in e)/1000:.2f} s, "
                  f"p90 batched {sorted(x[2] for x in e)[int(.9 * len(e))]/1000:.2f} s")
    print("\nheld-out changes, rr full K=20 vs embed (top-1):")
    for r in rows:
        if r["split"] != "held-out":
            continue
        g = set(r["gold"])
        a, b = ok(ranked(r, "embed", 0)[0], g), ok(ranked(r, "full", 20)[0], g)
        if a != b or not b:
            print(f"  {'+' if b and not a else '-' if a and not b else '='} {r['q'][:48]:48} emb {ranked(r, 'embed', 0)[0][:34]:34} "
                  f"rr {ranked(r, 'full', 20)[0][:34]}")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "report": lambda: report(sys.argv[2]), "check": check}[sys.argv[1]]()
