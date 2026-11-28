#!/usr/bin/env python3
"""retrieval.py steps | score TRANSCRIPT : the embedder spike's retrieval set (40 synthetic OS-agent memories, IT/EN;
15 paraphrased + 5 exact-token queries, docs/spikes/EMBEDDER.md; two names replaced) through the kernel shell.
  steps            : one shell line per output line (40x /index add, then 20x /search), for tools/accept.py
  score TRANSCRIPT : top1 of the fused ranking (what /search prints first) and of the embedding alone (highest cos
                     among the printed hits), semantic and exact queries apart."""
import re, sys

MEM = [
    "User's name is Ada, he lives in Italy and prefers to be addressed informally.",  # 0
    "L'utente preferisce risposte brevi e in italiano, senza emoji.",  # 1
    "Default shell on the Fedora box is fish; bash heredocs do not work there.",  # 2
    "Il server Fedora si raggiunge con ssh buildbox e ha 32 GB di RAM.",  # 3
    "Tool result: df -h showed /home at 91% usage, 18 GB free.",  # 4
    "Tool result: free -m reported 2057 MiB free out of 4095 MiB total.",  # 5
    "Task: renew the TLS certificate for the home NAS before December 15.",  # 6
    "Promemoria: pagare la bolletta della luce entro il 30 del mese.",  # 7
    "User prefers bun over npm and uv for Python projects.",  # 8
    "L'utente usa Syncthing per sincronizzare il vault Obsidian tra i computer.",  # 9
    "Past tool call: ls /etc showed no nginx directory, nginx is not installed.",  # 10
    "Il gatto dell'utente si chiama Pixel ed e' allergico al pollo.",  # 11
    "Meeting with the client Acme every Tuesday at 10:00 CET.",  # 12
    "The kernel panics if the model module is missing at boot.",  # 13
    "Lo sviluppo di AIOS avviene su un branch chiamato agentd.",  # 14
    "User dislikes long explanations; wants the action first, details later.",  # 15
    "Tool result: uptime 18 days, load average 2.79.",  # 16
    "Password manager is Bitwarden, accessed with the bw CLI, read-only.",  # 17
    "La stampante dell'ufficio e' una Brother HL-L2350DW collegata via Wi-Fi.",  # 18
    "Task: write the M4 milestone report for the agent subsystem.",  # 19
    "User's birthday is March 3rd.",  # 20
    "Il caffe' preferito dell'utente e' un espresso ristretto senza zucchero.",  # 21
    "The backup job runs nightly at 02:00 with restic to a Backblaze bucket.",  # 22
    "Tool result: cat /proc/cpuinfo shows AMD Ryzen 9 5900X, 24 threads.",  # 23
    "Errore precedente: la build Rust e' fallita per una feature nightly mancante.",  # 24
    "User's partner is called Giulia; anniversary on June 12.",  # 25
    "Preferred editor is Helix with the catppuccin theme.",  # 26
    "Task completato: aggiornato il firmware del router a ottobre.",  # 27
    "The QEMU guest for AIOS runs with 4 GiB of RAM and 8 vCPUs.",  # 28
    "L'utente va in palestra il lunedi', mercoledi' e venerdi' alle 7.",  # 29
    "Tool result: ping 8.8.8.8 had 0% packet loss, 12 ms average.",  # 30
    "Do not run destructive commands without explicit confirmation from the user.",  # 31
    "L'auto dell'utente deve fare la revisione a febbraio.",  # 32
    "Favourite music for focus: lo-fi and Brian Eno ambient albums.",  # 33
    "Tool result: the file notes/todo.md contains 14 open items.",  # 34
    "La password del Wi-Fi di casa e' salvata in Bitwarden sotto 'casa-wifi'.",  # 35
    "User is learning Japanese, currently at JLPT N4 level.",  # 36
    "Il medico di base riceve il giovedi' pomeriggio, serve prenotare.",  # 37
    "Docker compose is preferred for multi-service apps.",  # 38
    "Tool result: systemctl status sshd reported active (running).",  # 39
]

# (query, set of relevant memory ids)
QS = [
    ("come si chiama il mio animale domestico?", {11}),
    ("how much disk space is left on my home partition?", {4}),
    ("quanta memoria libera aveva la macchina?", {5}),
    ("when do I have to pay the electricity bill?", {7}),
    ("which javascript package manager should you use for me?", {8}),
    ("what CPU does this machine have?", {23}),
    ("quando ho la riunione settimanale col cliente?", {12}),
    ("is the web server installed?", {10}),
    ("che tipo di risposte preferisco?", {1, 15}),
    ("where is the wifi password stored?", {35}),
    ("quando e' il compleanno dell'utente?", {20}),
    ("how are my files backed up?", {22}),
    ("is the network connection working?", {30}),
    ("che stampante ho in ufficio?", {18}),
    ("what should I remember to do about the car?", {32}),
    # exact-token queries (identifiers, file names): BM25 territory
    ("HL-L2350DW", {18}),
    ("restic Backblaze", {22}),
    ("notes/todo.md", {34}),
    ("casa-wifi", {35}),
    ("JLPT", {36}),
]
N_SEM = 15


if sys.argv[1] == "steps":
    for m in MEM:
        print(f"/index add {m}")
    for q, _ in QS:
        print(f"/search {q}")
    sys.exit(0)

txt = open(sys.argv[2], errors="replace").read()
blocks = re.split(r"ringzero(?:\[[^\]\n]*\])?> /search ", txt)[1:]
assert len(blocks) == len(QS), f"{len(blocks)} /search blocks, want {len(QS)}"
res = {"sem": [0, 0], "lex": [0, 0]}
for i, ((q, gold), b) in enumerate(zip(QS, blocks)):
    hits = [(int(h), float(c), via) for h, c, via in re.findall(r"^\d+\. #(\d+)\s+cos (-?[\d.]+)\s+bm25\s+[\d.]+\s+(\S+)", b, re.M)]
    gate = re.search(r"BM25 (.*)", b).group(1)
    fused, emb = hits[0][0], max(hits, key=lambda h: h[1])[0]
    k = "sem" if i < N_SEM else "lex"
    res[k][0] += emb in gold
    res[k][1] += fused in gold
    print(f"{'ok ' if fused in gold else 'MISS'} fused #{fused} ({hits[0][2]}) emb #{emb}  gold {sorted(gold)}  [{gate}]  {q}")
for k, n in (("sem", N_SEM), ("lex", len(QS) - N_SEM)):
    print(f"{k}: embedding top1 {res[k][0]}/{n}, fused top1 {res[k][1]}/{n}")
