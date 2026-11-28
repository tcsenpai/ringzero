#!/usr/bin/env python3
"""T71 host test of design C: wiki pages written by the stock 2B from session transcripts, validated by kernel rules.

  memwiki.py check                 the sessions carry every T69 memory once, on its date; no model
  memwiki.py oracle OUT            pages built by hand rules from the same sessions (upper bound of the structure)
  memwiki.py write MODEL OUT       host (torch, CPU): MODEL consolidates each session into pages, greedy, as agentd
  memwiki.py embed OUT             host: embed the queries and the wiki chunks (per line, per page) with the embedder
  memwiki.py report OUT [BASE]     page quality, then memeval per class (held-out, all) for each index layout

Sessions: the 138 T69 memories as the turns that produced them (user words, tool results), one session per date of
memeval.DATE (undated T63 memories go in sessions dated before any record that supersedes them). Every turn names the
memory it carries, so each wiki line can be traced back (provenance by word overlap) for kept / lost / invented.

Write path (MEMORY_SYSTEMS.md 6.C): after the session, one more kernel message in CONS_ASK's shape. It lists the
page names and the numbered live lines of the 3 pages closest to the session (embedder cosine); the model answers
"PAGE name: line" (at most 6) and "OLD name n". The kernel checks each line: page name as norm_key, line as
norm_value (160 bytes), grounded() against the session's user words and tool results, no duplicate of a live line;
an OLD needs an accepted PAGE line on the same page in the same answer and a shown live line n. It stamps the
session date and marks [superseded DATE]. The model never rewrites a page.

Index layouts: wiki chunks added to the T69 index ("add") or standing in for every memory entry ("replace"); a
chunk is a live line ("line": "[wiki: page] DATE text") or a page cut like a file ("page": title + live lines).
A wiki chunk counts as the memory keys its live lines were traced to."""
import json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memeval as me  # noqa: E402
import rerankeval as rr  # noqa: E402

CONS_MAX, MAX_LINES, SHOW_PAGES, LINE_MAX = 220, 6, 3, 160

# (date, [(role, text, memory, page)]): role u = user, t = tool ("cmd\nresult"), a = answer (default "Ok." after u)
# memory: a memeval name, m<i> or f:<path> (a tool result that shows a file); page: the page a careful writer picks
S = [
 ("2024-05-01", [("u", "la mia email e' ada.moretti@libero.it, segnatela", "mail_old", "user")]),
 ("2025-03-01", [("u", "Hi, I'm Ada, I live in Italy. Talk to me informally, please.", "m0", "user"),
                 ("u", "e rispondimi in italiano, risposte brevi, senza emoji", "m1", "preferences"),
                 ("u", "I dislike long explanations: the action first, the details later.", "m15", "preferences"),
                 ("u", "never run destructive commands without my explicit confirmation", "m31", "preferences"),
                 ("u", "sono vegetariano da due anni, niente carne e niente pesce", "diet_old", "diet")]),
 ("2025-04-10", [("u", "what CPU does this machine have?", "m23", "machine"),
                 ("t", "grep -m1 'model name' /proc/cpuinfo; nproc\nmodel name : AMD Ryzen 9 5900X 12-Core Processor\n24", "m23", "machine"),
                 ("a", "An AMD Ryzen 9 5900X with 24 threads.", "m23", ""),
                 ("u", "and how long has it been up?", "m16", "machine"),
                 ("t", "uptime\nup 18 days, load average: 2.79, 2.41, 2.20", "m16", "machine"),
                 ("a", "Up 18 days, load average 2.79.", "m16", ""),
                 ("u", "quanta memoria libera c'e'?", "m5", "machine"),
                 ("t", "free -m\nMem: total 4095 used 2038 free 2057", "m5", "machine"),
                 ("a", "2057 MiB liberi su 4095.", "m5", ""),
                 ("u", "e lo spazio su /home?", "m4", "machine"),
                 ("t", "df -h /home\n/dev/sda2 200G 182G 18G 91% /home", "m4", "machine"),
                 ("a", "/home e' al 91%, 18 GB liberi.", "m4", "")]),
 ("2025-04-12", [("u", "is nginx installed?", "m10", "machine"),
                 ("t", "ls /etc\nfstab group hostname hosts os-release passwd profile", "m10", "machine"),
                 ("a", "No: there is no nginx directory in /etc, so nginx is not installed.", "m10", ""),
                 ("u", "is sshd running?", "m39", "machine"),
                 ("t", "systemctl status sshd\nsshd.service - OpenSSH server\n   Active: active (running)", "m39", "machine"),
                 ("a", "Yes, sshd is active (running).", "m39", ""),
                 ("u", "does the network work?", "m30", "machine"),
                 ("t", "ping -c 4 8.8.8.8\n4 packets transmitted, 4 received, 0% packet loss, rtt avg 12 ms", "m30", "machine"),
                 ("a", "Yes: 0% packet loss, 12 ms on average.", "m30", "")]),
 ("2025-05-05", [("u", "il server Fedora lo raggiungo con ssh buildbox, ha 32 GB di RAM", "m3", "fedora_server"),
                 ("u", "on the Fedora box the default shell is fish, so bash heredocs don't work there", "m2", "fedora_server"),
                 ("u", "I prefer bun over npm, and uv for Python projects", "m8", "dev_tools"),
                 ("u", "for multi-service apps I prefer docker compose", "m38", "dev_tools")]),
 ("2025-05-20", [("u", "my editor is Helix, with the catppuccin theme", "m26", "dev_tools"),
                 ("u", "my favourite programming language is Rust", "m52", "dev_tools"),
                 ("u", "lo sviluppo di AIOS avviene su un branch chiamato agentd", "m14", "aios"),
                 ("u", "the QEMU guest for AIOS runs with 4 GiB of RAM and 8 vCPUs", "m28", "aios")]),
 ("2025-06-01", [("u", "il mio preside e' Paolo Ferri", "boss_old", "school"),
                 ("u", "ricordati che lavoro come insegnante di matematica al liceo", "m44", "user"),
                 ("u", "abito a Bologna", "m45", "user"),
                 ("u", "I have a meeting with the client Acme every Tuesday at 10:00 CET", "m12", "acme")]),
 ("2025-06-15", [("u", "compila il progetto Rust", "m24", "aios"),
                 ("t", "cargo build\nerror[E0554]: #![feature] may not be used on the stable release channel\nerror: could not compile `agentd`", "m24", "aios"),
                 ("a", "La build Rust e' fallita: manca una feature nightly.", "m24", ""),
                 ("u", "remember: the kernel panics if the model module is missing at boot", "m13", "aios")]),
 ("2025-07-01", [("u", "pago 750 euro al mese di affitto", "rent_old", "home"),
                 ("u", "my password manager is Bitwarden, I use it with the bw CLI, read-only", "m17", "passwords"),
                 ("u", "la password del Wi-Fi di casa e' salvata in Bitwarden sotto 'casa-wifi'", "m35", "passwords"),
                 ("u", "uso Syncthing per sincronizzare il vault Obsidian tra i computer", "m9", "dev_tools")]),
 ("2025-08-01", [("u", "metti la tastiera italiana", "m54", "machine"),
                 ("t", "loadkeys it\nkeyboard layout: it", "m54", "machine"),
                 ("a", "Fatto, la tastiera ora e' italiana.", "m54", ""),
                 ("u", "quanto spazio c'e' sul disco di ringzero?", "m46", "machine"),
                 ("t", "!df\nringzero disk: 1.2 GB free of 4 GB", "m46", "machine"),
                 ("a", "1.2 GB liberi su 4 GB.", "m46", "")]),
 ("2025-08-10", [("u", "run /demo/stats.py with python3", "m49", "demo"),
                 ("t", "python3 /demo/stats.py\n{\"people\": 5, \"mean_age\": 38.8, ...}\nwritten /demo/report.json", "m49", "demo"),
                 ("a", "Done: it wrote /demo/report.json, mean age 38.8.", "m49", "")]),
 ("2025-09-01", [("u", "ho una Fiat Panda grigia del 2015", "car_old", "car"),
                 ("u", "l'auto deve fare la revisione a febbraio", "m32", "car"),
                 ("u", "my birthday is March 3rd", "m20", "user"),
                 ("u", "my partner is called Giulia, our anniversary is on June 12", "m25", "giulia")]),
 ("2025-10-10", [("u", "mia madre vive a Ravenna", "mom_city", "mom"),
                 ("u", "mio fratello si chiama Luca e vive a Torino", "m41", "luca"),
                 ("u", "il cane di mia sorella si chiama Rocco", "m40", "rocco"),
                 ("u", "il mio gatto si chiama Pixel ed e' allergico al pollo", "m11", "pixel")]),
 ("2025-11-01", [("u", "il mio migliore amico e' Davide", "best_friend", "davide"),
                 ("u", "I'm learning Japanese, I'm at JLPT N4 level now", "m36", "japanese"),
                 ("u", "when I need to focus I listen to lo-fi and Brian Eno ambient albums", "m33", "music"),
                 ("u", "il mio colore preferito e' il verde", "m43", "user")]),
 ("2025-12-01", [("u", "il mio provider e' TIM, fibra 200 mega", "isp_old", "internet"),
                 ("u", "the router is a FritzBox 7590 in the hallway, admin page at fritz.box", "m47", "internet"),
                 ("u", "the backup job runs nightly at 02:00 with restic to a Backblaze bucket", "m22", "backup"),
                 ("u", "I have to renew the TLS certificate of the home NAS before December 15", "m6", "nas")]),
 ("2025-12-12", [("u", "Giulia lavora come architetta a Modena", "giulia_job", "giulia"),
                 ("u", "my laptop is a ThinkPad X1 Carbon gen 11 with 32 GB", "m53", "laptop"),
                 ("u", "il mio caffe' preferito e' un espresso ristretto senza zucchero", "m21", "coffee"),
                 ("u", "a casa uso una moka da 3 tazze, caffe' macinato fine", "m48", "coffee")]),
 ("2026-01-05", [("u", "il mio obiettivo di corsa: 10 km sotto i 50 minuti", "run_old", "running"),
                 ("u", "vado in palestra il lunedi', mercoledi' e venerdi' alle 7", "m29", "gym"),
                 ("u", "promemoria: devo pagare la bolletta della luce entro il 30 del mese", "m7", "bills"),
                 ("u", "il veterinario di Pixel e' in via Roma 12, Bologna", "m51", "pixel")]),
 ("2026-01-10", [("u", "il compleanno di Davide e' il 3 febbraio", "davide_bday", "davide"),
                 ("u", "I have to write the M4 milestone report for the agent subsystem", "m19", "work")]),
 ("2026-01-22", [("u", "il mio numero di scarpe e' 43", "r_shoes", "user"),
                 ("u", "how many open items are in notes/todo.md?", "m34", "todo"),
                 ("t", "grep -c '^- \\[ \\]' notes/todo.md\n14", "m34", "todo"),
                 ("a", "notes/todo.md has 14 open items.", "m34", "")]),
 ("2026-02-01", [("u", "il compleanno di Giulia e' l'8 settembre", "giulia_bday", "giulia")]),
 ("2026-02-10", [("u", "ho un Motorola g54, il numero finisce con 4411", "phone_old", "phone"),
                 ("u", "il medico di base riceve il giovedi' pomeriggio, bisogna prenotare", "m37", "doctor")]),
 ("2026-02-12", [("u", "my piano lessons are on Thursday at 18:00", "piano_old", "piano")]),
 ("2026-02-20", [("u", "il compleanno di mia sorella Chiara e' il 9 ottobre", "sister_bday", "chiara")]),
 ("2026-02-27", [("u", "I'm allergic to penicillin, write it down", "r_penicillin", "health")]),
 ("2026-03-03", [("u", "il medico di mio padre e' il dottor Neri", "dad_doctor", "dad")]),
 ("2026-03-15", [("u", "mia sorella Chiara vive a Parma", "sister_city", "chiara")]),
 ("2026-03-30", [("u", "mi piace correre la mattina presto prima del lavoro", "r_morning", "running")]),
 ("2026-04-02", [("u", "il compleanno di mia madre e' il 12 maggio", "mom_bday_a", "mom")]),
 ("2026-04-04", [("u", "il gatto di Giulia si chiama Mochi", "giulia_cat", "giulia")]),
 ("2026-04-11", [("u", "di solito vado a letto verso mezzanotte e mi sveglio alle 7", "r_sleep", "user")]),
 ("2026-05-01", [("u", "per il compleanno di mia madre voglio regalarle un corso di ceramica", "mom_gift", "mom")]),
 ("2026-05-03", [("u", "mi sbagliavo, il compleanno di mia madre e' il 21 maggio", "mom_bday_b", "mom")]),
 ("2026-05-05", [("u", "mio fratello lavora come ingegnere alla Ferrari", "brother_job", "luca")]),
 ("2026-05-19", [("u", "the spare house keys are with the neighbour on the second floor", "r_keys", "home")]),
 ("2026-06-01", [("u", "mia sorella usa un MacBook Air", "sister_laptop", "chiara"),
                 ("u", "la dottoressa Bianchi e' il mio medico di base", "m50", "doctor")]),
 ("2026-06-06", [("u", "Chiara ha due figli, Tommaso e Irene", "chiara_kids", "chiara")]),
 ("2026-06-10", [("u", "non sopporto il coriandolo, non mettermelo mai nelle ricette", "r_coriander", "food")]),
 ("2026-07-01", [("u", "I always forget to water the basil on the balcony", "r_basil", "home")]),
 ("2026-07-07", [("u", "mio fratello ha un iPhone 13", "brother_phone", "luca")]),
 ("2026-07-15", [("u", "ho messo le idee nel file idee.md", "p_idee", "ideas")]),
 ("2026-07-20", [("u", "Rocco, il cane di mia sorella, non puo' mangiare cereali", "rocco_food", "rocco")]),
 ("2026-08-08", [("u", "le ricette sono in /home", "p_recipes", "files")]),
 ("2026-08-21", [("u", "I only drink decaf after 4 pm", "r_decaf", "coffee")]),
 ("2026-08-30", [("u", "Davide si e' trasferito a Lisbona per lavoro", "davide_city", "davide")]),
 ("2026-09-01", [("u", "cosa c'e' in /notes?", "m42", "files"),
                 ("t", "ls /notes\nidee.md  spesa.txt  todo.txt", "m42", "files"),
                 ("a", "In /notes ci sono todo.txt, spesa.txt e idee.md.", "m42", ""),
                 ("u", "la lista della spesa e' in /notes/spesa.txt", "p_spesa", "files")]),
 ("2026-09-03", [("u", "la mia squadra del cuore e' il Bologna, ho l'abbonamento allo stadio", "r_team", "sport")]),
 ("2026-09-12", [("u", "sono passato a Fastweb, la fibra adesso e' da 1 giga", "isp_new", "internet")]),
 ("2026-09-15", [("u", "Marta e' la mia collega di matematica", "marta_who", "marta")]),
 ("2026-09-20", [("u", "Paolo Ferri e' andato in pensione, la nuova preside della scuola e' Elena Costa", "boss_new", "school")]),
 ("2026-09-25", [("u", "la stampante dell'ufficio e' una Brother HL-L2350DW collegata via Wi-Fi", "m18", "printer"),
                 ("u", "scrivi i dettagli in /home/stampante.txt: IP 10.0.0.50, toner TN-2420 comprato a settembre", "f:/home/stampante.txt", "printer"),
                 ("t", "write /home/stampante.txt\n3 lines written", "p_printer", "printer"),
                 ("a", "Fatto, ho scritto i dati della stampante in /home/stampante.txt.", "p_printer", "")]),
 ("2026-10-02", [("u", "fammi vedere il piano del viaggio in Giappone", "p_japan", "japan"),
                 ("t", "cat /home/viaggio_giappone.md\n" + rr.FILES["/home/viaggio_giappone.md"].strip(), "f:/home/viaggio_giappone.md", "japan"),
                 ("a", "Parti il 3 aprile da Bologna e torni il 17: 5 notti a Tokyo, poi Kyoto e Osaka.", "f:/home/viaggio_giappone.md", ""),
                 ("u", "il referente di Acme e' Marco Bianchi", "acme_contact", "acme")]),
 ("2026-10-05", [("u", "ho ricominciato a mangiare pesce, adesso sono pescetariano", "diet_new", "diet")]),
 ("2026-10-08", [("u", "I moved the piano lesson to Saturday morning at 10", "piano_new", "piano")]),
 ("2026-10-12", [("u", "my neighbour Franco has a spare charger for my e-bike", "r_charger", "ebike")]),
 ("2026-10-15", [("u", "ho venduto la Panda, adesso guido una Toyota Yaris hybrid", "car_new", "car")]),
 ("2026-10-20", [("u", "ho l'appuntamento dal dentista il 4 novembre alle 9", "dentist_old", "dentist"),
                 ("u", "fatto: ho aggiornato il firmware del router", "m27", "internet")]),
 ("2026-10-30", [("u", "in Giappone vado con Davide", "japan_with", "japan")]),
 ("2026-10-31", [("u", "ho paura di volare, in aereo prendo sempre il posto corridoio", "r_fly", "travel")]),
 ("2026-11-01", [("u", "il dentista ha spostato l'appuntamento al 18 novembre alle 15", "dentist_new", "dentist")]),
 ("2026-11-02", [("u", "ho cambiato telefono, adesso ho un iPhone 15", "phone_new", "phone")]),
 ("2026-11-03", [("u", "show me /demo/people.csv", "p_people", "people_csv"),
                 ("t", "cat /demo/people.csv\n" + rr.FILES["/demo/people.csv"].strip(), "f:/demo/people.csv", "people_csv"),
                 ("a", "people.csv has 5 people: Anna, Marco, Giulia, Luca and Sara.", "f:/demo/people.csv", "")]),
 ("2026-11-04", [("u", "devo restituire il libro in biblioteca il 2 dicembre", "book_a", "library")]),
 ("2026-11-05", [("u", "la dottoressa Bianchi mi ha prescritto la vitamina D", "prescription", "doctor")]),
 ("2026-11-06", [("u", "run hello.lua in /demo", "p_lua", "demo"),
                 ("t", "lua /demo/hello.lua\nthe 3\nbrown 1\ndog 1\nfib(20) = 6765 in 2 ms\npcall: false something went wrong\nwritten by Lua 5.4", "p_lua", "demo"),
                 ("a", "It printed the top words, fib(20) = 6765, and wrote /demo/lua-out.txt.", "p_lua", "")]),
 ("2026-11-08", [("u", "print /demo/data.json with jq", "p_json", "demo"),
                 ("t", "jq . /demo/data.json\n" + rr.FILES["/demo/data.json"].strip(), "f:/demo/data.json", "demo"),
                 ("a", "ringzero 0.8, models Qwen3.5-2B and Qwen3-Embedding-0.6B, 8 apps.", "f:/demo/data.json", "")]),
 ("2026-11-09", [("u", "la riunione con i genitori degli studenti e' in aula 12", "room_a", "school")]),
 ("2026-11-10", [("u", "Acme call today: the deadline for the CSV export is November 30", "acme_old", "acme"),
                 ("u", "sto leggendo Il nome della rosa, sono a meta'", "r_book", "books")]),
 ("2026-11-11", [("u", "Rocco deve fare il vaccino a gennaio", "rocco_vax", "rocco")]),
 ("2026-11-12", [("u", "la mia nuova email e' ada.m@proton.me, quella di libero non la uso piu'", "mail_new", "user")]),
 ("2026-11-14", [("u", "leggimi la ricetta della carbonara mentre cucino", "p_carbonara", "recipes"),
                 ("t", "cat /home/ricetta_carbonara.txt\n" + rr.FILES["/home/ricetta_carbonara.txt"].strip(), "f:/home/ricetta_carbonara.txt", "recipes"),
                 ("a", "Spaghetti, guanciale, 3 tuorli e pecorino; niente panna.", "f:/home/ricetta_carbonara.txt", "")]),
 ("2026-11-15", [("u", "il mio nuovo obiettivo e' la mezza maratona di Bologna a marzo", "run_new", "running")]),
 ("2026-11-16", [("u", "hanno cambiato aula: la riunione con i genitori e' in aula magna", "room_b", "school")]),
 ("2026-11-17", [("u", "aggiungi due cose a /home/todo.md: comprare le lampadine e chiamare l'idraulico", "p_todo", "todo"),
                 ("t", "append /home/todo.md\n2 lines added", "p_todo", "todo")]),
 ("2026-11-18", [("u", "sabato sera ceno da mia sorella", "sister_a", "chiara")]),
 ("2026-11-19", [("u", "Elena Costa ha fissato il collegio docenti per il 5 dicembre", "council", "school")]),
 ("2026-11-20", [("u", "il pacco di Amazon arriva giovedi'", "parcel_a", "parcel")]),
 ("2026-11-21", [("u", "did last night's backup run?", "backup_a", "backup"),
                 ("t", "tail -1 /var/log/backup.log\n2026-11-20 02:00 restic: backup finished without errors", "backup_a", "backup"),
                 ("a", "Yes, the November 20 backup finished without errors.", "backup_a", ""),
                 ("u", "la cena da mia sorella salta, ci vediamo domenica a pranzo", "sister_b", "chiara")]),
 ("2026-11-22", [("u", "check the backup log again, the bucket dashboard shows an error", "backup_b", "backup"),
                 ("t", "grep -i fatal /var/log/backup.log\n2026-11-20 02:14 restic: Fatal: unable to save snapshot: bucket full", "backup_b", "backup"),
                 ("a", "The November 20 backup failed: the bucket is full.", "backup_b", "")]),
 ("2026-11-23", [("u", "il corriere ha scritto che il pacco arriva solo lunedi' prossimo", "parcel_b", "parcel")]),
 ("2026-11-24", [("u", "my train to Milano leaves at 7:40 on Friday", "train_a", "travel")]),
 ("2026-11-25", [("u", "da gennaio l'affitto sale a 820 euro al mese", "rent_new", "home"),
                 ("u", "correction: the train to Milano leaves at 8:10, not 7:40", "train_b", "travel")]),
 ("2026-11-26", [("u", "Marta arriva in stazione alle 18", "marta_a", "marta")]),
 ("2026-11-26", [("u", "Marta ha perso il treno, arriva alle 20:30", "marta_b", "marta")]),
 ("2026-11-26", [("u", "quanto fa 17 per 23?", "hub_qa", ""), ("a", "391.", "hub_qa", ""),
                 ("u", "e qual e' la capitale del Portogallo?", "hub_qa", ""), ("a", "Lisbona.", "hub_qa", "")]),
 ("2026-11-27", [("u", "ciao!", "hub_hello", ""), ("a", "Ciao Ada!", "hub_hello", "")]),
 ("2026-11-28", [("u", "ho rinnovato il prestito, il libro va restituito il 16 dicembre", "book_b", "library")]),
]
NOISE = {"hub_qa", "hub_hello"}  # sessions with nothing worth a page line
# (older, newer) records of one thing: the newer one's session must mark the older line superseded
PAIRS = [(a, b) for a, b in (("phone_old", "phone_new"), ("car_old", "car_new"), ("dentist_old", "dentist_new"),
         ("isp_old", "isp_new"), ("boss_old", "boss_new"), ("diet_old", "diet_new"), ("run_old", "run_new"),
         ("rent_old", "rent_new"), ("piano_old", "piano_new"), ("mail_old", "mail_new"), ("backup_a", "backup_b"),
         ("parcel_a", "parcel_b"), ("sister_a", "sister_b"), ("train_a", "train_b"), ("mom_bday_a", "mom_bday_b"),
         ("room_a", "room_b"), ("marta_a", "marta_b"), ("book_a", "book_b"))]

# The ask, in CONS_ASK's shape ("FACT: key = value" is the wording the 2B follows, docs/M45_RESULT.md). Variant A
# ("PAGE name: one short line") was copied literally by the 2B in the smoke run: kept for the record.
ASKS = {
    "A": ("[kernel] Wiki update (internal: the user does not see this reply). The kernel keeps one page per person, "
          "thing or topic. Pages: {names}.\n{shown}Write what this conversation adds, in exactly this format:\n"
          "PAGE name: one short line\nOLD name n\nOne PAGE line per new fact the user stated or a tool result showed "
          "(page name in English snake_case, an existing page or a new one; the line short and in the user's own "
          "words), at most 6, none if there are none. One OLD line when a numbered line above is no longer true "
          "after this conversation. Nothing else."),
    "B": ("[kernel] Wiki update (internal: the user does not see this reply). Pages: {names}.\n{shown}Write what is "
          "worth keeping from this conversation, in exactly this format:\nWIKI: page = fact\nOne WIKI line per fact "
          "the user stated or a tool result showed (page: the person, thing or topic the fact is about, an existing "
          "page or a new one, in English snake_case; fact short and in the user's own words), at most 6, none if "
          "there are none.\nOLD: page n\nOne OLD line for each numbered line above that this conversation makes no "
          "longer true. Nothing else."),
}
# Variant K, after A, B and C (B with the answer started as "WIKI:") all failed on the 2B: agentd's own CONS_ASK,
# read from the kernel source, plus the saved keys. The kernel files each accepted FACT on the page named by its key
# and marks the live line of the same key superseded (agentd replaces a fact of the same key; here it keeps it,
# dated); EPISODE lines go on the page "episodes".
_CONS = re.search(r'const CONS_ASK: &str = "(.*?)";\n', open(rr.RZ + "/kernel/src/agentd.rs").read(), re.S).group(1)
CONS_ASK = re.sub(r"\\\n", "", _CONS).replace("\\n", "\n").replace("\\'", "'")
ASKS["K"] = CONS_ASK.replace(" Nothing else.", " Keys already saved (reuse one when the fact replaces it): {names}. "
                                               "Nothing else.")
ASK = os.environ.get("ASK", "B")
PREFILL = os.environ.get("PREFILL", "")  # text the kernel puts at the start of the answer (variant C: "WIKI:")
SYSTEM = "You are the agent of ringzero, an operating system. Answer briefly, in the user's language."


def key(m):
    return me.K.get(m, m)


def check():
    seen = {}
    for d, turns in S:
        for role, text, m, page in turns:
            k = key(m)
            assert role in "uta" and text, (d, text)
            seen.setdefault(k, set()).add(d)
            if k in me.DATE and not k.startswith("f:"):
                assert me.DATE[k] == d, (m, me.DATE[k], d)
    mems = {k for k, p, _ in rr.corpus() if p == "memory"}
    miss, extra = mems - set(seen), {k for k in seen if not k.startswith("f:")} - mems
    assert not miss and not extra, (sorted(miss), sorted(extra))
    assert all(len(v) == 1 for k, v in seen.items() if not k.startswith("f:")), "a memory in two sessions"
    assert all(key(a) in seen and key(b) in seen for a, b in PAIRS)
    assert [d for d, _ in S] == sorted(d for d, _ in S)
    print(f"ok: {len(S)} sessions carry all {len(mems)} memories; {len(PAIRS)} supersession pairs")


# ---- kernel rules (agentd.rs), ported ----

def norm_key(k):
    k = "".join("_" if c in " -." else c.lower() for c in k.strip())
    return k if 2 <= len(k) <= 32 and k[0].isascii() and k[0].islower() and all(
        c.isascii() and (c.islower() or c.isdigit() or c == "_") for c in k) else None


def norm_value(v):
    v = v.strip().rstrip(".").strip("\"'").strip()
    if not v or len(v.encode()) > LINE_MAX:
        return None, "empty or longer than %d bytes" % LINE_MAX
    if any(c in v for c in "<\n\t|"):
        return None, "markup"
    if v.lower() in ("unknown", "none", "n/a", "null", "?", "-", "nessuno", "sconosciuto"):
        return None, "placeholder"
    return v, ""


def grounded(v, said):
    """agentd::grounded: every number of v appears in said, and at least half of its words of 3+ characters."""
    said, sig, hit = said.lower(), 0, 0
    for w in (w for w in re.split(r"[^\w]|_", v.lower()) if w):
        dig = any(c.isdigit() for c in w)
        if dig and w not in said:
            return False
        if dig or len(w) >= 3:
            sig += 1
            hit += w in said
    return sig > 0 and hit * 2 >= sig


def ground_text(turns):
    return "\n".join(t for r, t, _, _ in turns if r in "ut")


def sig(s):
    return {w for w in re.split(r"[^\w]|_", s.lower()) if w and (any(c.isdigit() for c in w) or len(w) >= 3)} - STOPW


STOPW = {"the", "and", "user", "has", "his", "her", "with", "for", "che", "del", "della", "una", "per", "sono", "non",
         "mio", "mia", "miei", "his", "now", "adesso", "utente", "user's", "said", "are", "was", "is"}


def prov(line, turns):
    """Memory keys a written line comes from: the keys whose turns hold the largest share of its words (>= half)."""
    lw = sig(line)
    if not lw:
        return set()
    src = {}
    for r, t, m, _ in turns:
        src[key(m)] = src.get(key(m), "") + " " + t
    sc = {k: len(lw & sig(t)) / len(lw) for k, t in src.items()}
    best = max(sc.values())
    return {k for k, v in sc.items() if v == best} if best >= 0.5 else set()


# ---- the wiki store: the kernel appends and marks, the model never rewrites ----

class Wiki:
    def __init__(self):
        self.pages = {}  # name -> [{date, text, live, sup, keys, sess}]

    def live(self, p):
        return [l for l in self.pages.get(p, []) if l["live"]]

    def md(self, p):
        out = [f"# {p}"]
        for l in self.pages[p]:
            out.append(f"- {l['date']} {l['text']}" if l["live"] else f"- [superseded {l['sup']}] {l['date']} {l['text']}")
        return "\n".join(out) + "\n"

    def save(self, d):
        os.makedirs(d + "/wiki", exist_ok=True)
        for p in self.pages:
            open(f"{d}/wiki/{p}.md", "w").write(self.md(p))
        json.dump(self.pages, open(d + "/pages.json", "w"), indent=1, ensure_ascii=False, default=sorted)


def apply(w, out, shown, date, turns, sess):
    """Kernel validation of one answer. shown: {page: [line dicts as numbered in the prompt]}. Returns a log."""
    said, log, touched = ground_text(turns), [], set()
    pl = [l.strip() for l in out.splitlines() if l.strip()]
    n_page = 0
    for l in pl:
        m = re.match(r"PAGE\s+([^:]+):\s*(.+)$", l) or re.match(r"WIKI:\s*([^=]+?)\s*=\s*(.+)$", l) or \
            re.match(r"FACT:\s*([^=]+?)\s*=\s*(.+)$", l)
        if ASK == "K" and l.startswith("EPISODE:") and not any(x["kind"] == "EPISODE" for x in log):
            v = l[8:].strip()[:600]
            w.pages.setdefault("episodes", []).append({"date": date, "text": v, "live": True, "sup": "",
                                                       "keys": sorted(prov(v, turns)), "sess": sess})
            log.append({"kind": "EPISODE", "raw": l, "ok": True, "keys": sorted(prov(v, turns))})
            continue
        if not m:
            continue
        n_page += 1
        p, (v, why) = norm_key(m.group(1)), norm_value(m.group(2))
        if n_page > MAX_LINES:
            why = "over the line cap"
        elif not p:
            why = "page name not snake_case"
        elif v and not grounded(v, said):
            why = "not grounded"
        elif v and any(x["text"].lower() == v.lower() for x in w.live(p)):
            why = "duplicate of a live line"
        if why:
            log.append({"kind": "PAGE", "raw": l, "ok": False, "why": why, "keys": sorted(prov(m.group(2), turns))})
            continue
        ks = prov(v, turns)
        if ASK == "K":  # same key: the older live line is superseded by the kernel, not by the model
            for x in w.live(p):
                x["live"], x["sup"] = False, date
                log.append({"kind": "OLD", "raw": f"(same key) {p}", "ok": True, "page": p, "keys": x["keys"], "text": x["text"]})
        w.pages.setdefault(p, []).append({"date": date, "text": v, "live": True, "sup": "", "keys": sorted(ks), "sess": sess})
        touched.add(p)
        log.append({"kind": "PAGE", "raw": l, "ok": True, "page": p, "keys": sorted(ks)})
    for l in pl:
        m = re.match(r"OLD:?\s+([\w\- .]+?)\s+#?(\d+)\s*$", l)
        if not m:
            if not l.startswith(("PAGE", "WIKI", "FACT", "EPISODE")):
                log.append({"kind": "other", "raw": l, "ok": False, "why": "not PAGE or OLD"})
            continue
        p, n = norm_key(m.group(1)), int(m.group(2))
        why = ("page not shown" if p not in shown else "no such line" if not 1 <= n <= len(shown[p])
               else "no new line on that page" if p not in touched else "")
        if not why and not shown[p][n - 1]["live"]:
            why = "already superseded"
        if why:
            log.append({"kind": "OLD", "raw": l, "ok": False, "why": why})
            continue
        x = shown[p][n - 1]
        x["live"], x["sup"] = False, date
        log.append({"kind": "OLD", "raw": l, "ok": True, "page": p, "keys": x["keys"], "text": x["text"]})
    return log, touched


def render(turns):
    """The session as it sits in the context: chat turns, tool calls and results in Qwen's tool format."""
    out, last = [f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"], None
    for r, t, _, _ in turns:
        if r == "u":
            if last == "u":
                out.append("<|im_start|>assistant\nOk.<|im_end|>\n")
            out.append(f"<|im_start|>user\n{t}<|im_end|>\n")
        elif r == "t":
            cmd, res = t.split("\n", 1)
            out.append('<|im_start|>assistant\n<tool_call>\n{"name": "sh", "arguments": {"cmd": %s}}\n</tool_call><|im_end|>\n'
                       % json.dumps(cmd))
            out.append(f"<|im_start|>user\n<tool_response>\n{res}\n</tool_response><|im_end|>\n")
        else:
            out.append(f"<|im_start|>assistant\n{t}<|im_end|>\n")
        last = r
    if last == "u":
        out.append("<|im_start|>assistant\nOk.<|im_end|>\n")
    return "".join(out)


def ask(w, shown):
    names = ", ".join(sorted(p for p in w.pages if p != "episodes")) or "none yet"
    if ASK == "K":
        return ASKS["K"].format(names=names)
    sh = "".join(f"[{p}]\n" + "".join(f"{i + 1}. {l['date']} {l['text']}\n" for i, l in enumerate(ls)) for p, ls in shown.items())
    return ASKS[ASK].format(names=names, shown=("Lines of the closest pages:\n" + sh) if sh else "")


def page_text(w, p):
    return f"[wiki: {p}]\n" + "\n".join(f"- {l['date']} {l['text']}" for l in w.live(p))


def closest(w, emb, turns, vec):
    """The SHOW_PAGES pages nearest to the session (its user words and tool results as a query)."""
    if not w.pages:
        return {}, 0
    import torch
    q, nq = emb(f"Instruct: {rr.TASK}\nQuery:{ground_text(turns)[:600]}")
    for p in w.pages:
        if p not in vec:
            vec[p] = emb(page_text(w, p))
    sc = sorted(((float(vec[p][0] @ q), p) for p in w.pages if w.live(p)), reverse=True)[:SHOW_PAGES]
    return {p: w.live(p) for _, p in sc}, nq


def models(model):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, os.environ.get("DTYPE", "float32"))).eval()
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]

    def gen(full):
        ids = tok(full + PREFILL, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            o = lm.generate(ids, max_new_tokens=CONS_MAX, do_sample=False, eos_token_id=eos)
        g = o[0, ids.shape[1]:]
        return PREFILL + tok.decode(g, skip_special_tokens=True), ids.shape[1], len(g)
    return tok, gen, rr.embedder()


def probe(model, idx):
    """The ask on chosen sessions, each on top of the oracle pages of the sessions before it (format trials)."""
    tok, gen, emb = models(model)
    for i in map(int, idx.split(",")):
        w = Wiki()
        oracle_into(w, S[:i])
        shown, _ = closest(w, emb, S[i][1], {})
        full = render(S[i][1]) + f"<|im_start|>user\n{ask(w, shown)}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        text, n_in, n_out = gen(full)
        log, _ = apply(w, text, shown, S[i][0], S[i][1], i)
        print(f"== {i} {S[i][0]} shown {list(shown)} ({n_in} in, {n_out} out)\n{text}")
        for x in log:
            print("  ", "ok " if x["ok"] else "REJ", x["kind"], x.get("why", ""), x.get("keys", ""), x["raw"][:90])


def write(model, out):
    tok, gen, emb = models(model)
    w, vec, runs = Wiki(), {}, []
    os.makedirs(out, exist_ok=True)
    only = os.environ.get("ONLY")  # a subset of sessions (e.g. those holding a pair record), comma-separated
    sel = [int(x) for x in only.split(",")] if only else range(int(os.environ.get("LIMIT", len(S))))
    for i in sel:
        date, turns = S[i]
        t0 = time.time()
        shown, nq = closest(w, emb, turns, vec)
        a = ask(w, shown)
        prefix = render(turns)
        full = prefix + f"<|im_start|>user\n{a}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        n_pre = len(tok(prefix, add_special_tokens=False).input_ids)
        t1 = time.time()
        text, n_in, n_out = gen(full)
        t2 = time.time()
        log, touched = apply(w, text, shown, date, turns, i)
        for p in touched:
            vec.pop(p, None)
        re_tok = sum(len(tok(page_text(w, p), add_special_tokens=False).input_ids) for p in touched)
        runs.append({"sess": i, "date": date, "prompt_tok": n_in, "session_tok": n_pre,
                     "ask_tok": n_in - n_pre, "out_tok": n_out, "query_embed_tok": nq, "reembed_tok": re_tok,
                     "s_select": round(t1 - t0, 2), "s_gen": round(t2 - t1, 2), "shown": list(shown), "out": text, "log": log})
        print(f"[{i}/{len(S)}] {date} {n_in} tok in, {n_out} out, {t2 - t1:.1f}s; "
              f"{sum(x['ok'] for x in log)} ok / {len(log)}", flush=True)
        json.dump({"model": model, "ask": ASK, "prefill": PREFILL, "runs": runs}, open(out + "/runs.json", "w"), indent=1, ensure_ascii=False)
        w.save(out)


def oracle(out):
    """Upper bound: one line per memory (its user or tool words, the page a careful writer picks), and the older
    record of each pair marked superseded by the newer one's session."""
    w = Wiki()
    oracle_into(w, S)
    os.makedirs(out, exist_ok=True)
    w.save(out)


def oracle_into(w, sessions):
    newer = {key(b): key(a) for a, b in PAIRS}
    for i, (date, turns) in enumerate(sessions):
        by = {}
        for r, t, m, p in turns:
            if p and r in "ut":
                by.setdefault((key(m), p), []).append(t.split("\n", 1)[-1] if r == "t" else t)
        for (k, p), ts in by.items():
            if k in newer:
                for l in w.pages.get(p, []):
                    if newer[k] in l["keys"] and l["live"]:
                        l["live"], l["sup"] = False, date
            w.pages.setdefault(p, []).append({"date": date, "text": "; ".join(ts)[:400], "live": True, "sup": "",
                                              "keys": [k], "sess": i})


def load(out):
    w = Wiki()
    w.pages = json.load(open(out + "/pages.json"))
    return w


def chunks(w, how):
    """(key, text, memory keys of its live lines)."""
    if how == "line":
        return [(f"w:{p}#{i}", f"[wiki: {p}] {l['date']} {l['text']}", set(l["keys"]))
                for p, ls in w.pages.items() for i, l in enumerate(ls) if l["live"]]
    out = []
    for p in w.pages:
        lv = w.live(p)
        b, k, j = page_text(w, p).encode(), 0, 0
        while b and lv:
            n = rr.cut(b)
            part = b[:n].decode(errors="replace")
            ks = set()
            for l in lv:  # the lines whose text falls in this chunk
                if l["text"] in part:
                    ks |= set(l["keys"])
            out.append((f"w:{p}#{k}", part.strip(), ks))
            b, k = b[n:], k + 1
    return out


def embed(out):
    """Queries again (results.json keeps only cosines) and every wiki chunk of both layouts."""
    import torch
    emb = rr.embedder()
    w = load(out)
    Q = me.queries()
    qv = torch.stack([emb(f"Instruct: {rr.TASK}\nQuery:{r['q']}")[0] for r in Q])
    res = {"task": rr.TASK}
    for how in ("line", "page"):
        C = chunks(w, how)
        D = torch.stack([emb(t)[0] for _, t, _ in C])
        res[how] = {"keys": [k for k, _, _ in C], "texts": [t for _, t, _ in C], "map": [sorted(m) for _, _, m in C],
                    "cos": [[round(x, 5) for x in row] for row in (qv @ D.T).tolist()]}
        print(how, len(C), "chunks", flush=True)
    # sanity: the embedder reproduces the T69 cosines on two entries
    base = json.load(open(BASE))
    C0 = {k: t for k, _, t in rr.corpus()}
    for j in (0, 100):
        k = base["keys"][j]
        got = float(qv[0] @ emb(C0[k])[0])
        assert abs(got - base["rows"][0]["cos"][j]) < 2e-3, (k, got, base["rows"][0]["cos"][j])
    json.dump(res, open(out + "/cos.json", "w"))


BASE = os.path.join(rr.RZ, "docs/spikes/memeval/results.json")


def done_mem(runs, noise=True):
    """Memory keys of the sessions a run processed (all sessions for the hand-built pages)."""
    idx = {r["sess"] for r in runs} if runs else range(len(S))
    return {key(m) for i in idx for _, _, m, _ in S[i][1] if not key(m).startswith("f:") and (noise or m not in NOISE)}


def quality(w, runs=None):
    """kept / lost per memory, invented (rejected or untraced lines), supersessions right / wrong / missed."""
    mem = done_mem(runs, noise=False)
    lines = [l for p, ls in w.pages.items() if p != "episodes" for l in ls]
    eps = w.pages.get("episodes", [])
    kept = {k for l in lines for k in l["keys"]} & mem
    q = {"memories": len(mem), "kept": len(kept), "lost": len(mem - kept), "lines": len(lines),
         "untraced": sum(not l["keys"] for l in lines), "pages": len(w.pages) - bool(eps),
         "episode_lines": len(eps), "kept_with_episodes": len((kept | {k for l in eps for k in l["keys"]}) & mem),
         "noise_lines": sum(any(key(n) in l["keys"] for n in NOISE) for l in lines)}
    if runs:
        lg = [x for r in runs for x in r["log"]]
        q.update(proposed=sum(x["kind"] == "PAGE" for x in lg), rejected=sum(x["kind"] == "PAGE" and not x["ok"] for x in lg),
                 rejected_why={y: sum(x.get("why") == y for x in lg if x["kind"] == "PAGE")
                               for y in sorted({x.get("why") for x in lg if x["kind"] == "PAGE" and not x["ok"]})},
                 old_proposed=sum(x["kind"] == "OLD" for x in lg), old_rejected=sum(x["kind"] == "OLD" and not x["ok"] for x in lg),
                 other_lines=sum(x["kind"] == "other" for x in lg))
    old_of = {key(b): key(a) for a, b in PAIRS}
    sup_ok = sup_miss = sup_wrong = sup_na = 0
    for a, b in PAIRS:
        if not {key(a), key(b)} <= mem:
            continue
        la = [l for l in lines if key(a) in l["keys"]]
        lb = [l for l in lines if key(b) in l["keys"]]
        if not la or not lb:
            sup_na += 1
        elif all(not l["live"] and l["sup"] == lb[0]["date"] for l in la):
            sup_ok += 1
        else:
            sup_miss += 1
    for l in lines:
        if not l["live"]:
            sess_keys = {k for x in lines if x["date"] == l["sup"] for k in x["keys"]}
            if not any(old_of.get(k) in l["keys"] for k in sess_keys):
                sup_wrong += 1
    q.update(pairs=sum({key(a), key(b)} <= mem for a, b in PAIRS), superseded_right=sup_ok, superseded_missed=sup_miss, pair_not_written=sup_na,
             superseded_wrong=sup_wrong)
    return q


def scores(cosd, how, layout, gone=None):
    """memeval's kernel scorer over the T69 index plus (add) or instead of the memories (replace) the wiki chunks.
    gone: the memory keys the wiki stands in for (default: every memory)."""
    base = json.load(open(BASE))
    W = cosd[how]
    expand = dict(zip(W["keys"], map(set, W["map"])))
    C0 = rr.corpus()
    keep = [i for i, k in enumerate(base["keys"]) if layout == "add" or not (k in gone if gone is not None else re.fullmatch(r"m\d+", k))]
    C = [C0[i] for i in keep] + [(k, "wiki", t) for k, t in zip(W["keys"], W["texts"])]
    R = {"task": base["task"], "keys": [k for k, _, _ in C],
         "rows": [{"q": r["q"], "cos": [r["cos"][i] for i in keep] + w} for r, w in zip(base["rows"], W["cos"])]}
    ok0, corpus0 = rr.ok, rr.corpus
    rr.ok = lambda k, gold: any(ok0(x, gold) for x in expand.get(k, {k}))
    rr.corpus = lambda: C
    try:
        return me.scored(R, "kernel")
    finally:
        rr.ok, rr.corpus = ok0, corpus0


def table(S_):
    rows = {}
    for c in me.CLASSES + ("answerable",):
        g = [s for s in S_ if (s[0]["cls"] != "noanswer" if c == "answerable" else s[0]["cls"] == c)]
        ho = [s for s in g if s[0]["split"] == "held-out"]
        rows[c] = {"ho_t1": sum(s[1][0] for s in ho), "ho_n": len(ho), "ho_t3": sum(s[1][1] for s in ho),
                   "all_t1": sum(s[1][0] for s in g), "all_n": len(g), "all_t3": sum(s[1][1] for s in g)}
    return rows


def report(out):
    w = load(out)
    runs = json.load(open(out + "/runs.json"))["runs"] if os.path.exists(out + "/runs.json") else None
    res = {"quality": quality(w, runs)}
    print(json.dumps(res["quality"], indent=1))
    if runs:
        n = len(runs)
        s = lambda f: sum(r[f] for r in runs)
        # ring-0 [E]: prefill 215 tok/s batched (BATCHED_PREFILL_RESULT), decode 36 tok/s, embedder 6 ms/token
        ring0 = [r["ask_tok"] / 215 + r["out_tok"] / 36 + (r["query_embed_tok"] + r["reembed_tok"]) * 0.006 for r in runs]
        res["cost"] = {"sessions": n, "prompt_tok_mean": s("prompt_tok") / n, "ask_tok_mean": s("ask_tok") / n,
                       "out_tok_mean": s("out_tok") / n, "out_tok_max": max(r["out_tok"] for r in runs),
                       "embed_tok_mean": (s("query_embed_tok") + s("reembed_tok")) / n,
                       "host_s_gen_mean": s("s_gen") / n, "host_s_select_mean": s("s_select") / n,
                       "ring0_s_mean_E": sum(ring0) / n, "ring0_s_max_E": max(ring0)}
        print(json.dumps(res["cost"], indent=1))
    base = me.scored(json.load(open(BASE)), "kernel")
    res["baseline"] = table(base)
    if os.path.exists(out + "/cos.json"):
        cosd = json.load(open(out + "/cos.json"))
        for how in ("line", "page"):
            for layout in ("add", "replace"):
                res[f"{layout}_{how}"] = table(scores(cosd, how, layout, done_mem(runs) if runs else None))
        cols = ["baseline"] + [f"{l}_{h}" for h in ("line", "page") for l in ("add", "replace")]
        print("| class | " + " | ".join(cols) + " |\n" + "|---" * (len(cols) + 1) + "|")
        for c in me.CLASSES + ("answerable",):
            print(f"| {c} | " + " | ".join(f"{res[k][c]['ho_t1']}/{res[k][c]['ho_n']} ({res[k][c]['all_t1']}/{res[k][c]['all_n']})"
                                          for k in cols) + " |")
    json.dump(res, open(out + "/report.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    {"check": check, "oracle": lambda: oracle(a[2]), "write": lambda: write(a[2], a[3]),
     "probe": lambda: probe(a[2], a[3]), "embed": lambda: embed(a[2]),
     "report": lambda: report(a[2])}[a[1]]()
