#!/usr/bin/env python3
"""T69: the memory recall eval set (250+ queries, IT/EN) and a host scorer that ranks like today's kernel.

  memeval.py check            data self-check (gold keys exist, split, counts), no model
  memeval.py run OUT.json     host only (torch, transformers; CPU): embed corpus and queries, store every cosine
  memeval.py report OUT.json  top-1 / top-3 per class, pool and split; no-answer accuracy; the T63 check

Corpus: tools/rerankeval.py's (55 memories, 32 manual sections, 14 file chunks, agentd's formats) plus the
entries below. Dates are metadata (DATE): agentd stores no time in an index text, so today's ranking cannot
use them; they say which record of a pair is newer, for the gold and for later designs.

Scorer (kernel/src/memidx.rs search + kernel/src/agentd.rs recall/slot):
  - cosine of the embedder (rerankeval.embedder: the kernel's instruction format, last-token pooling, 1024 dims)
  - BM25 over every entry and the gate: identifier-like query, or the top BM25 entry has every query term and
    scores >= 1.5x the second; then RRF (k = 60) of the cosine ranks and the BM25 ranks
  - the top 24 split into manual and non-manual hits (hits_split)
  - mode "kernel": the answer list is the best manual section first when howto(q) and cos >= RECALL_MIN (the
    slot's order), then the non-manual hits; a non-how-to question never gets a manual section
  - mode "split": T66's reading, manual queries ranked over manual sections, the others over the rest
  - no-answer is right when nothing in the first RECALL_K of the list reaches its floor (RECALL_MIN = 0.55 for
    an embedding hit, RECALL_MIN_BM25 = 0.40 when BM25 brought it in), facts included
Differences left: fp32 vectors on the host, Q8_0 rows in the kernel; the queue order of entries (ties only).

Classes: para, exact; T66's miss classes pointer (P), sibling (S), hub (H), owner (E), raw (R), table (T); new
temporal (a newer record supersedes an older one: gold is the newer), contradiction (two records disagree on
one event: gold is the later), multihop (needs two entries: top-1 must be the answer, top-3 must hold both),
noanswer. Split: inside each class, queries sorted by md5 of the text, alternating tune / held-out."""
import hashlib, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rerankeval as rr  # noqa: E402

RECALL_MIN, RECALL_MIN_BM25, RECALL_K, HITS = 0.55, 0.40, 4, 24
OLD_KEYS = {k for k, _, _ in rr.corpus()}  # the T63 corpus, before the entries below

# (name, date, text in agentd's format). Key m<55+i>; gold refers to them as @name.
NEW_MEM = [
    # temporal: an older and a newer record of the same thing (facts of one key replace each other in agentd,
    # so a change arrives as a [said] note, an episode or a fact under another key)
    ("phone_old", "2026-02-10", "[fact] phone: Motorola g54, numero che finisce con 4411"),
    ("phone_new", "2026-11-02", "[said] ho cambiato telefono, adesso ho un iPhone 15"),
    ("car_old", "2025-09-01", "[fact] car: Fiat Panda grigia del 2015"),
    ("car_new", "2026-10-15", "[episode 21] The user said they sold the Panda and now drive a Toyota Yaris hybrid."),
    ("dentist_old", "2026-10-20", "[said] ho l'appuntamento dal dentista il 4 novembre alle 9"),
    ("dentist_new", "2026-11-01", "[said] il dentista ha spostato l'appuntamento al 18 novembre alle 15"),
    ("acme_old", "2026-11-10", "[episode 12] Acme call: the deadline for the CSV export is November 30."),
    ("isp_old", "2025-12-01", "[fact] isp: TIM, fibra 200 mega"),
    ("isp_new", "2026-09-12", "[said] sono passato a Fastweb, la fibra adesso e' da 1 giga"),
    ("boss_old", "2025-06-01", "[fact] head_teacher: il mio preside e' Paolo Ferri"),
    ("boss_new", "2026-09-20", "[episode 22] The user said Paolo Ferri retired and the new head teacher of the school is Elena Costa."),
    ("diet_old", "2025-03-01", "[said] sono vegetariano da due anni, niente carne e niente pesce"),
    ("diet_new", "2026-10-05", "[said] ho ricominciato a mangiare pesce, adesso sono pescetariano"),
    ("run_old", "2026-01-05", "[fact] running_goal: 10 km sotto i 50 minuti"),
    ("run_new", "2026-11-15", "[said] il mio nuovo obiettivo e' la mezza maratona di Bologna a marzo"),
    ("rent_old", "2025-07-01", "[fact] rent: 750 euro al mese"),
    ("rent_new", "2026-11-25", "[said] da gennaio l'affitto sale a 820 euro al mese"),
    ("piano_old", "2026-02-12", "[episode 14] The user said piano lessons are on Thursday at 18:00."),
    ("piano_new", "2026-10-08", "[episode 23] The user moved the piano lesson to Saturday morning at 10."),
    ("mail_old", "2024-05-01", "[fact] email: ada.moretti@libero.it"),
    ("mail_new", "2026-11-12", "[said] la mia nuova email e' ada.m@proton.me, quella di libero non la uso piu'"),
    # contradiction: two records of one event disagree; the later one is right
    ("backup_a", "2026-11-21", "[episode 30] The nightly backup of November 20 finished without errors."),
    ("backup_b", "2026-11-22", "[episode 31] Checking the log, the agent found that the November 20 backup had failed: the bucket was full."),
    ("parcel_a", "2026-11-20", "[said] il pacco di Amazon arriva giovedi'"),
    ("parcel_b", "2026-11-23", "[said] il corriere ha scritto che il pacco arriva solo lunedi' prossimo"),
    ("sister_a", "2026-11-18", "[said] sabato sera ceno da mia sorella"),
    ("sister_b", "2026-11-21", "[said] la cena da mia sorella salta, ci vediamo domenica a pranzo"),
    ("train_a", "2026-11-24", "[episode 34] The user said the train to Milano leaves at 7:40 on Friday."),
    ("train_b", "2026-11-25", "[episode 35] The user corrected the train time: it leaves at 8:10, not 7:40."),
    ("mom_bday_a", "2026-04-02", "[fact] mom_birthday: 12 maggio"),
    ("mom_bday_b", "2026-05-03", "[said] mi sbagliavo, il compleanno di mia madre e' il 21 maggio"),
    ("room_a", "2026-11-09", "[said] la riunione con i genitori degli studenti e' in aula 12"),
    ("room_b", "2026-11-16", "[said] hanno cambiato aula: la riunione con i genitori e' in aula magna"),
    ("marta_a", "2026-11-26", "[said] Marta arriva in stazione alle 18"),
    ("marta_b", "2026-11-26", "[said] Marta ha perso il treno, arriva alle 20:30"),
    ("book_a", "2026-11-04", "[said] devo restituire il libro in biblioteca il 2 dicembre"),
    ("book_b", "2026-11-28", "[said] ho rinnovato il prestito, il libro va restituito il 16 dicembre"),
    # multi-hop: a first hop that names someone or something, a second that holds the answer
    ("sister_city", "2026-03-15", "[fact] sister_city: mia sorella Chiara vive a Parma"),
    ("best_friend", "2025-11-01", "[fact] best_friend: il mio migliore amico e' Davide"),
    ("davide_city", "2026-08-30", "[said] Davide si e' trasferito a Lisbona per lavoro"),
    ("davide_bday", "2026-01-10", "[fact] davide_birthday: 3 febbraio"),
    ("giulia_bday", "2026-02-01", "[fact] giulia_birthday: 8 settembre"),
    ("acme_contact", "2026-10-02", "[fact] acme_contact: il referente di Acme e' Marco Bianchi"),
    ("mom_gift", "2026-05-01", "[said] per il compleanno di mia madre voglio regalarle un corso di ceramica"),
    ("rocco_vax", "2026-11-11", "[said] Rocco deve fare il vaccino a gennaio"),
    ("prescription", "2026-11-05", "[said] la dottoressa Bianchi mi ha prescritto la vitamina D"),
    ("giulia_job", "2025-12-12", "[said] Giulia lavora come architetta a Modena"),
    ("chiara_kids", "2026-06-06", "[said] Chiara ha due figli, Tommaso e Irene"),
    ("japan_with", "2026-10-30", "[said] in Giappone vado con Davide"),
    ("marta_who", "2026-09-15", "[fact] marta: Marta e' la mia collega di matematica"),
    ("council", "2026-11-19", "[said] Elena Costa ha fissato il collegio docenti per il 5 dicembre"),
    # wrong owner: relatives and partner next to the user's own
    ("brother_phone", "2026-07-07", "[fact] brother_phone: mio fratello ha un iPhone 13"),
    ("sister_bday", "2026-02-20", "[fact] sister_birthday: il compleanno di Chiara e' il 9 ottobre"),
    ("giulia_cat", "2026-04-04", "[said] il gatto di Giulia si chiama Mochi"),
    ("dad_doctor", "2026-03-03", "[fact] dad_doctor: il medico di mio padre e' il dottor Neri"),
    ("sister_laptop", "2026-06-01", "[fact] sister_laptop: mia sorella usa un MacBook Air"),
    ("brother_job", "2026-05-05", "[said] mio fratello lavora come ingegnere alla Ferrari"),
    ("mom_city", "2025-10-10", "[fact] mom_city: mia madre vive a Ravenna"),
    ("rocco_food", "2026-07-20", "[said] Rocco, il cane di mia sorella, non puo' mangiare cereali"),
    # pointers: a memory that names a file, next to the file
    ("p_japan", "2026-10-02", "[episode 41] The user asked about the Japan trip; the agent opened /home/viaggio_giappone.md."),
    ("p_carbonara", "2026-11-14", "[episode 42] The agent read /home/ricetta_carbonara.txt aloud while the user cooked dinner."),
    ("p_people", "2026-11-03", "[episode 43] The user asked the agent to show /demo/people.csv."),
    ("p_spesa", "2026-09-01", "[fact] shopping_list: la lista della spesa e' in /notes/spesa.txt"),
    ("p_todo", "2026-11-17", "[episode 44] The user added two items to /home/todo.md."),
    ("p_recipes", "2026-08-08", "[fact] recipes_dir: le ricette sono in /home"),
    ("p_lua", "2026-11-06", "[episode 45] The agent ran hello.lua from /demo and showed its output."),
    ("p_printer", "2026-09-25", "[episode 46] The agent wrote the printer details to /home/stampante.txt."),
    ("p_idee", "2026-07-15", "[said] ho messo le idee nel file idee.md"),
    ("p_json", "2026-11-08", "[episode 47] The agent printed /demo/data.json with jq."),
    # raw notes: a verbatim [said] sentence in words other than the question's
    ("r_coriander", "2026-06-10", "[said] non sopporto il coriandolo, non mettermelo mai nelle ricette"),
    ("r_sleep", "2026-04-11", "[said] di solito vado a letto verso mezzanotte e mi sveglio alle 7"),
    ("r_basil", "2026-07-01", "[said] I always forget to water the basil on the balcony"),
    ("r_team", "2026-09-03", "[said] la mia squadra del cuore e' il Bologna, ho l'abbonamento allo stadio"),
    ("r_penicillin", "2026-02-27", "[said] I'm allergic to penicillin, write it down"),
    ("r_fly", "2026-10-31", "[said] ho paura di volare, in aereo prendo sempre il posto corridoio"),
    ("r_keys", "2026-05-19", "[said] the spare house keys are with the neighbour on the second floor"),
    ("r_morning", "2026-03-30", "[said] mi piace correre la mattina presto prima del lavoro"),
    ("r_decaf", "2026-08-21", "[said] I only drink decaf after 4 pm"),
    ("r_book", "2026-11-10", "[said] sto leggendo Il nome della rosa, sono a meta'"),
    ("r_shoes", "2026-01-22", "[said] il mio numero di scarpe e' 43"),
    ("r_charger", "2026-10-12", "[said] my neighbour Franco has a spare charger for my e-bike"),
    # hubs: short generic entries close to many questions
    ("hub_hello", "2026-11-27", "[episode 49] The user said hello and the agent greeted back."),
    ("hub_qa", "2026-11-26", "[episode 50] The user asked a few questions and the agent answered them."),
]
NEW_FILES = {
    "/home/stampante.txt": "Stampante ufficio: Brother HL-L2350DW\nIP 10.0.0.50\ntoner TN-2420, comprato a settembre\n",
    "/home/spese_novembre.csv": "data,voce,euro\n2026-11-02,spesa Coop,54.20\n2026-11-05,benzina,60.00\n2026-11-09,cena fuori,38.50\n"
                                "2026-11-14,bolletta luce,72.10\n2026-11-21,libri,24.90\n2026-11-25,spesa Coop,61.35\n",
}
K = {n: f"m{len(rr.MEM) + i}" for i, (n, _, _) in enumerate(NEW_MEM)}
DATE = {K[n]: d for n, d, _ in NEW_MEM}
# dates of the older entries that take part in a pair (a file's date = its mtime)
DATE.update({"m27": "2026-10-20", "f:/home/todo.md": "2026-09-01", "f:/home/meeting_acme.md": "2026-11-24"})
rr.MEM += [t for _, _, t in NEW_MEM]
rr.FILES.update(NEW_FILES)

# T63's 93 queries keep their gold; class from T66's miss classes where the query is of that kind
OLD_CLS = {
    "come si chiama il mio animale domestico?": "owner", "come si chiama il mio gatto?": "owner",
    "come si chiama mio fratello?": "owner", "come si chiama la mia ragazza?": "owner",
    "come si chiama il mio medico?": "owner",
    "what's on my todo list?": "pointer", "che idee avevo per i progetti?": "pointer",
    "cosa abbiamo deciso alla riunione con Acme?": "pointer", "what did Acme ask for in the last call?": "pointer",
    "which script computes the mean age?": "pointer", "what's in /notes?": "pointer",
    "dove ho annotato il dentista?": "pointer",
    "come cambio la temperatura?": "sibling", "perche' mi chiede Run it y/N?": "sibling",
    "dove tiene ringzero i suoi dati?": "sibling", "what model runs this OS?": "sibling",
    "how do I switch the keyboard to Italian?": "sibling", "quanto e' grande il disco?": "sibling",
    "which cores are used for inference?": "sibling", "!set spec": "sibling",
    "che tipo di risposte preferisco?": "hub", "anna.rossi@example.org": "hub",
    "che lavoro faccio?": "raw", "what should I remember to do about the car?": "raw",
    "who lives in Milano?": "table", "which program counts the words of the quick brown fox?": "table",
}
FIX = {"what model runs this OS?": {"man:What ringzero is", "f:/demo/data.json"}}  # T66: data.json answers it too

M, F = "memory", "files"
# (query, pool, class, gold, need): gold = right at top-1; need = also in the top 3 (multi-hop first hop)
NEW_Q = [
    ("che telefono ho?", M, "temporal", "@phone_new", ""),
    ("what phone am I using now?", M, "temporal", "@phone_new", ""),
    ("che macchina ho?", M, "temporal", "@car_new", ""),
    ("what car do I drive?", M, "temporal", "@car_new", ""),
    ("quando ho il dentista?", M, "temporal", "@dentist_new", ""),
    ("when is my dentist appointment?", M, "temporal", "@dentist_new", ""),
    ("what's the deadline for the Acme export?", F, "temporal", "f:/home/meeting_acme.md", ""),
    ("entro quando devo consegnare l'export ad Acme?", F, "temporal", "f:/home/meeting_acme.md", ""),
    ("chi e' il mio provider internet?", M, "temporal", "@isp_new", ""),
    ("how fast is my home fibre?", M, "temporal", "@isp_new", ""),
    ("chi e' il mio preside?", M, "temporal", "@boss_new", ""),
    ("who is the head of my school now?", M, "temporal", "@boss_new", ""),
    ("do I eat fish?", M, "temporal", "@diet_new", ""),
    ("che dieta seguo?", M, "temporal", "@diet_new", ""),
    ("qual e' il mio obiettivo di corsa?", M, "temporal", "@run_new", ""),
    ("what race am I training for?", M, "temporal", "@run_new", ""),
    ("quanto pago di affitto?", M, "temporal", "@rent_new", ""),
    ("how much will my rent be?", M, "temporal", "@rent_new", ""),
    ("when is my piano lesson?", M, "temporal", "@piano_new", ""),
    ("a che ora ho lezione di piano?", M, "temporal", "@piano_new", ""),
    ("what's my email address?", M, "temporal", "@mail_new", ""),
    ("qual e' la mia email?", M, "temporal", "@mail_new", ""),
    ("have I updated the router firmware?", M, "temporal", "m27", ""),
    ("devo ancora aggiornare il firmware del router?", M, "temporal", "m27", ""),

    ("did last night's backup work?", M, "contradiction", "@backup_b", ""),
    ("il backup del 20 novembre e' andato a buon fine?", M, "contradiction", "@backup_b", ""),
    ("quando arriva il pacco?", M, "contradiction", "@parcel_b", ""),
    ("when is my Amazon package arriving?", M, "contradiction", "@parcel_b", ""),
    ("when am I seeing my sister?", M, "contradiction", "@sister_b", ""),
    ("quando vedo mia sorella questo weekend?", M, "contradiction", "@sister_b", ""),
    ("what time is my train on Friday?", M, "contradiction", "@train_b", ""),
    ("a che ora parte il treno per Milano?", M, "contradiction", "@train_b", ""),
    ("when is my mother's birthday?", M, "contradiction", "@mom_bday_b", ""),
    ("quando compie gli anni mia mamma?", M, "contradiction", "@mom_bday_b", ""),
    ("dove si tiene la riunione con i genitori?", M, "contradiction", "@room_b", ""),
    ("which room is the parents' meeting in?", M, "contradiction", "@room_b", ""),
    ("a che ora arriva Marta?", M, "contradiction", "@marta_b", ""),
    ("when does Marta get in?", M, "contradiction", "@marta_b", ""),
    ("when is the library book due?", M, "contradiction", "@book_b", ""),

    ("dove porto il gatto dal veterinario?", M, "multihop", "m51", "m11"),
    ("dove si trova il veterinario del gatto allergico al pollo?", M, "multihop", "m51", "m11"),
    ("cosa c'e' nella lista della spesa che mi ha mostrato l'agente?", F, "multihop", "f:/notes/spesa.txt", "m42"),
    ("quando riceve la dottoressa Bianchi?", M, "multihop", "m37", "m50"),
    ("in che citta' vive la padrona di Rocco?", M, "multihop", "@sister_city", "m40"),
    ("where does my best friend live now?", M, "multihop", "@davide_city", "@best_friend"),
    ("when is my best friend's birthday?", M, "multihop", "@davide_bday", "@best_friend"),
    ("when is my girlfriend's birthday?", M, "multihop", "@giulia_bday", "m25"),
    ("what's the email of my contact at Acme?", F, "multihop", "f:/demo/people.csv", "@acme_contact"),
    ("che ricetta posso fare con il guanciale della lista della spesa?", F, "multihop", "f:/home/ricetta_carbonara.txt", "f:/notes/spesa.txt"),
    ("che toner usa la stampante dell'ufficio?", F, "multihop", "f:/home/stampante.txt", "m18"),
    ("what gift did I plan for my mum's birthday?", M, "multihop", "@mom_gift", "@mom_bday_b"),
    ("quando fa il vaccino il cane di mia sorella?", M, "multihop", "@rocco_vax", "m40"),
    ("che cosa mi ha prescritto il medico di base?", M, "multihop", "@prescription", "m50"),
    ("which shell do I get when I ssh into buildbox?", M, "multihop", "m2", "m3"),
    ("what does my partner do for a living?", M, "multihop", "@giulia_job", "m25"),
    ("come si chiamano i figli di mia sorella?", M, "multihop", "@chiara_kids", "@sister_city"),
    ("with whom am I going to Kyoto?", M, "multihop", "@japan_with", "f:/home/viaggio_giappone.md"),
    ("when does the car I bought in October need its inspection?", M, "multihop", "m32", "@car_new"),
    ("did the build fail in my favourite programming language?", M, "multihop", "m24", "m52"),
    ("is the router in the hallway up to date?", M, "multihop", "m27", "m47"),
    ("in che lingua vuole le risposte Ada?", M, "multihop", "m1", "m0"),
    ("how old is the person in people.csv with my brother's name?", F, "multihop", "f:/demo/people.csv", "m41"),
    ("a che ora arriva la mia collega di matematica?", M, "multihop", "@marta_b", "@marta_who"),
    ("when is the staff meeting set by my new head teacher?", M, "multihop", "@council", "@boss_new"),

    ("come si chiama mio nonno?", M, "noanswer", "", ""),
    ("what's my blood type?", M, "noanswer", "", ""),
    ("quando scade il mio passaporto?", M, "noanswer", "", ""),
    ("what's my dog called?", M, "noanswer", "", ""),
    ("how tall am I?", M, "noanswer", "", ""),
    ("dove ho parcheggiato la macchina?", M, "noanswer", "", ""),
    ("what's the PIN of my bank card?", M, "noanswer", "", ""),
    ("quanti figli ho?", M, "noanswer", "", ""),
    ("what did I have for lunch yesterday?", M, "noanswer", "", ""),
    ("come si chiama il mio dentista?", M, "noanswer", "", ""),
    ("what's my flight number to Tokyo?", F, "noanswer", "", ""),
    ("how do I connect a bluetooth mouse?", "manual", "noanswer", "", ""),
    ("come mando una email dal terminale?", "manual", "noanswer", "", ""),
    ("how do I connect to the wifi?", "manual", "noanswer", "", ""),
    ("come alzo il volume dell'audio?", "manual", "noanswer", "", ""),
    ("what's the weather tomorrow?", M, "noanswer", "", ""),
    ("chi ha vinto la partita ieri sera?", M, "noanswer", "", ""),
    ("what's the license plate of my car?", M, "noanswer", "", ""),
    ("when is my brother's birthday?", M, "noanswer", "", ""),
    ("qual e' la password del NAS?", M, "noanswer", "", ""),
    ("what's in /notes/diario.txt?", F, "noanswer", "", ""),
    ("chi e' Federico?", M, "noanswer", "", ""),
    ("how many km did I run last week?", M, "noanswer", "", ""),
    ("what time does the pharmacy close?", M, "noanswer", "", ""),
    ("quanto e' costato il JR Pass?", F, "noanswer", "", ""),

    ("che telefono ha mio fratello?", M, "owner", "@brother_phone", ""),
    ("quando e' il compleanno di mia sorella?", M, "owner", "@sister_bday", ""),
    ("when's my birthday again?", M, "owner", "m20", ""),
    ("which city do I live in?", M, "owner", "m45", ""),
    ("come si chiama il gatto della mia ragazza?", M, "owner", "@giulia_cat", ""),
    ("what's my cat called?", M, "owner", "m11", ""),
    ("chi e' il medico di mio padre?", M, "owner", "@dad_doctor", ""),
    ("who's my GP?", M, "owner", "m50", ""),
    ("what laptop does my sister use?", M, "owner", "@sister_laptop", ""),
    ("che computer portatile uso io?", M, "owner", "m53", ""),
    ("what does my brother do?", M, "owner", "@brother_job", ""),
    ("what's my job?", M, "owner", "m44", ""),
    ("dove vive mia mamma?", M, "owner", "@mom_city", ""),
    ("what can't my cat eat?", M, "owner", "m11", ""),
    ("cosa non puo' mangiare il cane di mia sorella?", M, "owner", "@rocco_food", ""),

    ("quanti giorni sto a Tokyo?", F, "pointer", "f:/home/viaggio_giappone.md", ""),
    ("quanti tuorli ci vogliono per la carbonara?", F, "pointer", "f:/home/ricetta_carbonara.txt", ""),
    ("how old is Giulia Verdi?", F, "pointer", "f:/demo/people.csv", ""),
    ("devo comprare le uova?", F, "pointer", "f:/notes/spesa.txt", ""),
    ("is the photo backup still on my todo?", F, "pointer", "f:/home/todo.md", ""),
    ("ci va la panna nella carbonara?", F, "pointer", "f:/home/ricetta_carbonara.txt", ""),
    ("what does the lua demo write to lua-out.txt?", F, "pointer", "f:/demo/hello.lua", ""),
    ("what's the printer's IP address?", F, "pointer", "f:/home/stampante.txt", ""),
    ("avevo un'idea per un gioco?", F, "pointer", "f:/notes/idee.md", ""),
    ("what format does Acme want for the export?", F, "pointer", "f:/home/meeting_acme.md", ""),
    ("is sqlite3 in the list of apps?", F, "pointer", "f:/demo/data.json", ""),
    ("cosa devo fare per il report M4?", F, "pointer", "f:/notes/todo.txt m19", ""),

    ("come spengo il computer?", "manual", "sibling", "man:Power off and reboot", ""),
    ("how do I pick up where I left off after a reboot?", "manual", "sibling", "man:Hibernation", ""),
    ("come vedo quanta RAM ho?", "manual", "sibling", "man:CPU, cores, memory and uptime", ""),
    ("quali driver sono caricati?", "manual", "sibling", "man:System information in /proc", ""),
    ("how do I set the temperature with !set?", "manual", "sibling", "man:Temperature and thinking", ""),
    ("come attivo il thinking?", "manual", "sibling", "man:Temperature and thinking", ""),
    ("how do I clear the screen?", "manual", "sibling", "man:New conversation", ""),
    ("come annullo l'ultima risposta?", "manual", "sibling", "man:Undo and redo a turn", ""),
    ("how do I save a point in the chat to come back to later?", "manual", "sibling", "man:Branches and saved points", ""),
    ("dove sono salvati i miei ricordi?", "manual", "sibling", "man:Devices, /tmp and ringzero's own data", ""),
    ("how do I make it forget my address?", "manual", "sibling", "man:Saved facts", ""),
    ("come disattivo la memoria?", "manual", "sibling", "man:Memory slot and consolidation", ""),
    ("how do I edit a file?", "manual", "sibling", "man:SQLite, Lua, JavaScript, jq and the editor", ""),
    ("come lancio ls?", "manual", "sibling", "man:Running WASI programs", ""),

    ("how should you talk to me?", M, "hub", "m1 m15 m0", ""),
    ("cosa ho detto sulle spiegazioni lunghe?", M, "hub", "m15", ""),
    ("marco@example.com", F, "hub", "f:/demo/people.csv", ""),
    ("hai qualche promemoria per me?", M, "hub", "m7 m6 m32", ""),
    ("che lingua sto imparando?", M, "hub", "m36", ""),
    ("do I have a pet?", M, "hub", "m11", ""),
    ("che hardware ha questa macchina?", M, "hub", "m23", ""),
    ("is anything installed for serving web pages?", M, "hub", "m10", ""),
    ("sara@example.com", F, "hub", "f:/demo/people.csv", ""),
    ("what's my name?", M, "hub", "m0", ""),
    ("cosa sai di me?", M, "hub", "m0 m1 m15 m44 m45", ""),

    ("is there a herb I can't stand?", M, "raw", "@r_coriander", ""),
    ("what time do I go to sleep?", M, "raw", "@r_sleep", ""),
    ("che piante ho sul balcone?", M, "raw", "@r_basil", ""),
    ("what team do I support?", M, "raw", "@r_team", ""),
    ("ho allergie a qualche medicina?", M, "raw", "@r_penicillin", ""),
    ("which seat do I want on the flight to Tokyo?", M, "raw", "@r_fly", ""),
    ("dove sono le chiavi di riserva?", M, "raw", "@r_keys", ""),
    ("when do I usually go running?", M, "raw", "@r_morning", ""),
    ("posso prendere un caffe' normale stasera?", M, "raw", "@r_decaf", ""),
    ("what book am I reading?", M, "raw", "@r_book", ""),
    ("what's my shoe size?", M, "raw", "@r_shoes", ""),
    ("chi ha il caricabatterie della bici?", M, "raw", "@r_charger", ""),

    ("how much did I spend on petrol in November?", F, "table", "f:/home/spese_novembre.csv", ""),
    ("quanto ho pagato la bolletta della luce?", F, "table", "f:/home/spese_novembre.csv", ""),
    ("when did I eat out this month?", F, "table", "f:/home/spese_novembre.csv", ""),
    ("who in people.csv is from Torino?", F, "table", "f:/demo/people.csv", ""),
    ("quanti anni ha Marco Bianchi?", F, "table", "f:/demo/people.csv", ""),
    ("which people live in Roma?", F, "table", "f:/demo/people.csv", ""),
    ("what does fib compute in the JavaScript demo?", F, "table", "f:/demo/hello.js", ""),
    ("what does stats.py count by city?", F, "table", "f:/demo/stats.py", ""),
    ("come gestisce gli errori il demo lua?", F, "table", "f:/demo/hello.lua", ""),
    ("what quantization is the embedding model in data.json?", F, "table", "f:/demo/data.json", ""),
    ("quanto pesa il modello Qwen3.5-2B?", F, "table", "f:/demo/data.json", ""),
    ("which demo computes the average age of Anna, Luca and Sara?", F, "table", "f:/demo/hello.js", ""),
    ("dove scrive il report lo script python?", F, "table", "f:/demo/stats.py", ""),
]
CLASSES = ("para", "exact", "pointer", "sibling", "hub", "owner", "raw", "table",
           "temporal", "contradiction", "multihop", "noanswer")
POOLS = ("memory", "manual", "files")


def keys(s):
    return {K[w[1:]] if w.startswith("@") else w for w in re.findall(r"@\w+|man:[^@]+$|\S+", s.strip())} if s else set()


def queries():
    """[dict]: q, pool, cls, gold, need, old (T63 index or None), split."""
    out = [{"q": q, "pool": p, "cls": OLD_CLS.get(q, "exact" if kind == "exact" else "para"),
            "gold": FIX.get(q, g), "need": set(), "old": i} for i, (q, p, kind, g) in enumerate(rr.Q)]
    out += [{"q": q, "pool": p, "cls": c, "gold": keys(g), "need": keys(n), "old": None} for q, p, c, g, n in NEW_Q]
    for c in CLASSES:
        s = sorted((r for r in out if r["cls"] == c), key=lambda r: hashlib.md5(r["q"].encode()).hexdigest())
        for n, r in enumerate(s):
            r["split"] = ("tune", "held-out")[n % 2]
    return out


def check():
    C = rr.corpus()
    ks = {k for k, _, _ in C}
    Q = queries()
    for r in Q:
        assert r["cls"] in CLASSES and r["pool"] in POOLS, r
        assert (not r["gold"]) == (r["cls"] == "noanswer"), r["q"]
        for g in r["gold"] | r["need"]:
            assert any(rr.ok(k, {g}) for k in ks), (r["q"], g)
    assert len({r["q"] for r in Q}) == len(Q), "duplicate query"
    assert len(K) == len(set(K.values())) and all(any(rr.ok(k, {d}) for k in ks) for d in DATE)
    print(f"ok: {len(C)} entries ({len(C) - len(OLD_KEYS)} new), {len(Q)} queries")
    for c in CLASSES:
        s = [r for r in Q if r["cls"] == c]
        print(f"  {c:13} {len(s):3}  held-out {sum(r['split'] == 'held-out' for r in s):3}  " +
              "  ".join(f"{p} {sum(r['pool'] == p for r in s)}" for p in POOLS))
    print("  by pool: " + ", ".join(f"{p} {sum(r['pool'] == p for r in Q)}" for p in POOLS) +
          f"; held-out {sum(r['split'] == 'held-out' for r in Q)}")


def run(out):
    """Every entry and every query embedded once; the cosine matrix is stored (rows = queries)."""
    import torch
    assert rr.free_gb() >= 8, f"MemAvailable {rr.free_gb()} GB, want 8"
    C, Q = rr.corpus(), queries()
    emb = rr.embedder()
    D = torch.stack([emb(t)[0] for _, _, t in C])
    rows = []
    for n, r in enumerate(Q):
        v, nq = emb(f"Instruct: {rr.TASK}\nQuery:{r['q']}")
        rows.append({"q": r["q"], "q_tokens": nq, "cos": [round(x, 5) for x in (D @ v).tolist()]})
        if n % 25 == 0:
            print(n, f"free {rr.free_gb()} GB", r["q"][:50], flush=True)
    json.dump({"task": rr.TASK, "keys": [k for k, _, _ in C], "rows": rows}, open(out, "w"))


# ---- kernel ranking, ported ----

STOP = set(b"""the a an is are was were be to of in on at for and or my me i you your do does did what which when where
how who should this that it with from have has will can il lo la gli le un una e di da con su per che chi come quando
dove quale quanta quanto ho ha mio mia del della dei delle al alla col sono era cosa tipo""".split())


def words(s):
    """memidx::words: \\w runs (ASCII alnum, _, any byte >= 0x80), ASCII-lowercased, at most 64 bytes, no stopwords."""
    return [w for w in (m[:64].lower() for m in re.findall(rb"[A-Za-z0-9_\x80-\xff]+", s.encode())) if w not in STOP]


def bm25(q, texts):
    """memidx::bm25: scores per entry and whether the gate opens on score (top has every term, >= 1.5x second)."""
    qt = words(q)[:32]
    docs = [words(t) for t in texts]
    tf = [[min(d.count(t), 255) for t in qt] for d in docs]
    nd, avg = len(docs), max(sum(map(len, docs)) / max(len(docs), 1), 1.0)
    df = [sum(f[t] > 0 for f in tf) for t in range(len(qt))]
    idf = [math.log(1 + (nd - df[t] + 0.5) / (df[t] + 0.5)) if df[t] else 0.0 for t in range(len(qt))]
    bm = [sum(idf[t] * f[t] * 2.5 / (f[t] + 1.5 * (0.25 + 0.75 * len(d) / avg)) for t in range(len(qt)) if f[t])
          for d, f in zip(docs, tf)]
    top = max(range(nd), key=lambda d: (bm[d], -d))
    s1 = bm[top]
    s2 = max((b for d, b in enumerate(bm) if d != top), default=0.0)
    return bm, bool(qt) and s1 > 0 and all(tf[top]) and s1 >= 1.5 * s2


def identifier_like(q):
    for w in q.split():
        b = w.strip("?!,;\"'()")
        inner = any(b[i] in "/\\._-:@#" and b[i - 1].isascii() and b[i - 1].isalnum() and b[i + 1].isascii()
                    and b[i + 1].isalnum() for i in range(1, len(b) - 1))
        alpha, digit = any(c.isascii() and c.isalpha() for c in b), any(c.isascii() and c.isdigit() for c in b)
        if inner or (alpha and digit) or (len(b) >= 2 and all("A" <= c <= "Z" for c in b)) or b.startswith("/"):
            return True
    return False


def howto(q):
    P = ("how do i ", "how can i ", "how to ", "how do you ", "how should i ", "how does ", "what command",
         "which command", "is there a command", "quale comando", "che comando", "un comando per", "in che modo",
         "come si fa", "come si usa", "come posso")
    l = q.lower()
    w = re.findall(r"[^\W_]+", l)
    return any(p in l for p in P) or any(a == "come" and len(b) >= 3 and b.endswith("o") and b != "sono"
                                         for a, b in zip(w, w[1:]))


def search(q, keys, texts, cos, gate=True):
    """memidx::search: top HITS as (key, cos, via); via 'emb' or 'bm25' (BM25 had a say: the 0.40 floor)."""
    n = len(keys)
    ord_e = sorted(range(n), key=lambda d: (-cos[d], d))
    rank_e = {d: r for r, d in enumerate(ord_e)}
    bm, strong = bm25(q, texts)
    order = ord_e
    if gate == "always" or (gate and (identifier_like(q) or strong)):  # T70 B: the BM25 view always fused
        rb = sorted((d for d in range(n) if bm[d] > 0), key=lambda d: (-bm[d], d))
        rrf = {d: 1 / (61 + rank_e[d]) for d in range(n)}
        for r, d in enumerate(rb):
            rrf[d] += 1 / (61 + r)
        order = sorted(range(n), key=lambda d: (-rrf[d], rank_e[d]))
    else:
        bm = [0.0] * n
    return [(keys[d], cos[d], "bm25" if bm[d] > 0 else "emb") for d in order[:HITS]]


def answer(r, h, mode):
    """The ranked list the model would see first: kernel = slot order (manual line for how-to), split = T66."""
    man = [x for x in h if x[0].startswith("man:")]
    own = [x for x in h if not x[0].startswith("man:")]
    if mode == "split":
        return man if r["pool"] == "manual" else own
    return (man[:1] if man and howto(r["q"]) and man[0][1] >= RECALL_MIN else []) + own


def floor_ok(x):
    return x[1] >= (RECALL_MIN if x[2] == "emb" else RECALL_MIN_BM25)


def score(r, lst):
    """(top-1, top-3, top-1 above its floor); for noanswer all three = nothing in the first RECALL_K above a floor."""
    if r["cls"] == "noanswer":
        a = not any(floor_ok(x) for x in lst[:RECALL_K])
        return a, a, a
    t1 = bool(lst) and rr.ok(lst[0][0], r["gold"])
    t3 = any(rr.ok(x[0], r["gold"]) for x in lst[:3]) and all(any(rr.ok(x[0], {n}) for x in lst[:3]) for n in r["need"])
    return t1, t3, t1 and floor_ok(lst[0])


def scored(R, mode, gate=True, only=None):
    """Per query (row, (t1, t3, slot), answer list), ranked over the entries in `only` (all if None)."""
    C = {k: t for k, _, t in rr.corpus()}
    idx = [i for i, k in enumerate(R["keys"]) if only is None or k in only]
    ks, ts = [R["keys"][i] for i in idx], [C[R["keys"][i]] for i in idx]
    out = []
    for r, x in zip(queries(), R["rows"]):
        assert r["q"] == x["q"], "results are from another query set: rerun"
        lst = answer(r, search(r["q"], ks, ts, [x["cos"][i] for i in idx], gate), mode)
        out.append((r, score(r, lst), lst))
    return out


def report(path):
    R = json.load(open(path))
    assert R["task"] == rr.TASK
    pct = lambda a, n: f"{a}/{n} ({100 * a / n:.1f}%)" if n else "-"
    # 1. the T63 check: old corpus, old queries, old even/odd split, T66's pool split, no BM25 (as T66)
    print(f"T63 check: the 93 T63 queries, old even/odd split; old corpus = its {len(OLD_KEYS)} entries")
    for mode, gate, only in (("split", False, OLD_KEYS), ("split", True, OLD_KEYS), ("kernel", True, OLD_KEYS),
                             ("split", True, None), ("kernel", True, None)):
        S = [s for s in scored(R, mode, gate, only) if s[0]["old"] is not None]
        for name, sub in (("held-out", [s for s in S if s[0]["old"] % 2]), ("all", S)):
            print(f"  {mode} mode, bm25 gate {'on ' if gate else 'off'}, {'old' if only else 'full'} corpus, {name}: "
                  f"top-1 {pct(sum(s[1][0] for s in sub), len(sub))}, top-3 {pct(sum(s[1][1] for s in sub), len(sub))}")
    # 2. the new set
    for mode in ("kernel", "split"):
        S = scored(R, mode)
        print(f"\n== mode {mode}, full corpus ({len(R['keys'])} entries), BM25 gate on ==")
        print("| class | n | held-out top-1 | held-out top-3 | tune top-1 | tune top-3 | all top-1 | all top-3 | all top-1 >= floor |")
        print("|---|---|---|---|---|---|---|---|---|")
        groups = [(c, lambda s, c=c: s[0]["cls"] == c) for c in CLASSES] + [
            ("answerable", lambda s: s[0]["cls"] != "noanswer"), ("all", lambda s: True)]
        groups += [(f"pool {p}", lambda s, p=p: s[0]["pool"] == p and s[0]["cls"] != "noanswer") for p in POOLS]
        for name, f in groups:
            g = [s for s in S if f(s)]
            cell = lambda sp, j: (lambda h: f"{sum(s[1][j] for s in h)}/{len(h)}")([s for s in g if sp in ("all", s[0]["split"])])
            print(f"| {name} | {len(g)} | {cell('held-out', 0)} | {cell('held-out', 1)} | {cell('tune', 0)} | "
                  f"{cell('tune', 1)} | {cell('all', 0)} | {cell('all', 1)} | {cell('all', 2)} |")
        ho = [s for s in S if s[0]["split"] == "held-out" and s[0]["cls"] != "noanswer"]
        print(f"held-out answerable top-1: {pct(sum(s[1][0] for s in ho), len(ho))}")
        print("held-out misses (query -> top-1 key, cos):")
        for r, (t1, _, _), lst in S:
            if r["split"] == "held-out" and not t1:
                top = f"{lst[0][0]} {lst[0][1]:.3f}" if lst else "-"
                if r["cls"] == "noanswer":
                    top = "above floor: " + ", ".join(f"{x[0]} {x[1]:.2f}" for x in lst[:RECALL_K] if floor_ok(x))
                print(f"  [{r['cls']}] {r['q'][:58]:58} -> {top}")


# ---- T70: design A (ranking rules, no model call) and design B (write-time enrichment) ----
#
#   memeval.py ctx CTX.json                host only: the stock 2B writes a context line per entry, questions per section
#   memeval.py embed VEC.json [CTX.json]   host only: queries, probes and each corpus view embedded; cosines stored
#   memeval.py tune VEC.json CFG.json      rules and thresholds picked on the TUNE split only; writes the frozen config
#   memeval.py ab VEC.json CFG.json OUT    the frozen configs scored once on held-out and all; per-query JSON to OUT

# Generic questions for the hub score (CSLS-style): written for T70, none taken from the eval set.
PROBES = [
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
]
HUB_K = 10  # hub = mean cosine of an entry to its HUB_K nearest probes
# relation words (IT/EN), also read in fact keys (sister_dog, mom_city): whose record an entry is
REL = {r: set(w.split()) for r, w in {
    "sister": "sorella sister", "brother": "fratello brother", "mother": "madre mamma mother mum mom",
    "father": "padre papa father dad", "colleague": "collega colleague coworker", "friend": "amico amica friend",
    "partner": "ragazza fidanzata ragazzo fidanzato partner girlfriend boyfriend moglie marito wife husband"}.items()}
CAPSTOP = set("User User's The Il La Lo Le Gli Un Una Task Tool Past Promemoria Errore Default Checking Acme".split())
CSV_ROWS = True  # A.3: a .csv file is chunked by rows, the header repeated in each chunk


def corpus_rows():
    """rr.corpus() with every .csv file chunked by rows ("[PATH] header\\nrow"), keys f:PATH#r<i>."""
    out = [c for c in rr.corpus() if not (c[0].startswith("f:") and c[0].split("#")[0].endswith(".csv"))]
    for p, txt in rr.FILES.items():
        if p.endswith(".csv"):
            head, *rows = [l for l in txt.splitlines() if l.strip()]
            out += [(f"f:{p}#r{i}", "files", f"[{p}] {head}\n{row}") for i, row in enumerate(rows)]
    return out


def file_doc(key):
    return rr.FILES[key[2:].split("#")[0]]


def ctx(out):
    """The stock Qwen3.5-2B (thinking off, greedy) writes, as agentd would in idle time: one context line per
    memory (the known facts first, a shared prefix the kernel would snapshot once) and per file chunk (the whole
    file first, Contextual Retrieval), and 6 questions per manual section (offline, build time). Tokens and host
    seconds per call are stored; OUT.partial resumes a stop."""
    import copy, time, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    assert rr.free_gb() >= 12, f"MemAvailable {rr.free_gb()} GB, want 12"
    torch.set_num_threads(10)
    name = "Qwen/Qwen3.5-2B"
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    head, tail = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    C = {k: (p, t) for k, p, t in rr.corpus() + corpus_rows()}
    facts = [t for k, (p, t) in C.items() if p == "memory" and (k not in OLD_KEYS or int(k[1:]) < 40)
             and (t.startswith("[fact] ") or not t.startswith("["))]  # agentd's facts; T63's plain memories stand in
    pre_mem = ids(head + "Known facts about the user (Ada):\n" + "\n".join("- " + f for f in facts) + "\n\n")
    ask_mem = ("Write one line in English, at most 25 words, for a search index: what this record is about, and "
               "whose it is (the user's own, or which relative, friend or pet, by relation and name from the known "
               "facts). Start with the topic. Example: for the record \"[said] mio cugino Piero ha comprato una moto\" "
               "the line is \"Motorbike bought by the user's cousin Piero.\" Write the line only.")
    ask_file = ("Write one line in English, at most 30 words, that situates this chunk within the file for search: "
                "what the file is and what this chunk says or lists (names, values). Write the line only.")
    ask_man = ("Write 6 short questions a user could type that this section of the ringzero manual answers, 3 in "
               "Italian and 3 in English, one per line, nothing else.")
    done = json.load(open(out + ".partial")) if os.path.exists(out + ".partial") else {}
    with torch.no_grad():
        pc = lm(input_ids=torch.tensor([pre_mem]), use_cache=True).past_key_values
    for k, (pool, t) in C.items():
        if k in done:
            continue
        assert rr.free_gb() >= 3, "MemAvailable under 3 GB, stopped (rerun to resume)"
        if pool == "memory":
            d = DATE.get(k, "")
            pre, body, cache, n_new = pre_mem, ids(f"{ask_mem}\n\nRecord{' (' + d + ')' if d else ''}: {t}" + tail), pc, 48
        elif pool == "files":
            pre, cache, n_new = [], None, 56
            body = ids(head + f"<document>\n{file_doc(k)}</document>\nHere is a chunk of it:\n<chunk>\n{t}\n</chunk>\n\n{ask_file}" + tail)
        else:
            pre, cache, n_new = [], None, 120
            body = ids(head + f"{t}\n\n{ask_man}" + tail)
        t0 = time.perf_counter()
        with torch.no_grad():
            g = lm.generate(input_ids=torch.tensor([pre + body]), past_key_values=copy.deepcopy(cache) if cache else None,
                            max_new_tokens=n_new, do_sample=False)
        new = g[0, len(pre) + len(body):].tolist()
        txt = tok.decode(new, skip_special_tokens=True).strip()
        done[k] = {"pool": pool, "line": txt, "shared": len(pre), "prompt": len(body), "out": len(new),
                   "s": round(time.perf_counter() - t0, 2)}
        json.dump(done, open(out + ".partial", "w"))
        print(len(done), k, done[k]["prompt"], done[k]["out"], f"{done[k]['s']}s", f"free {rr.free_gb()} GB",
              txt.replace("\n", " | ")[:90], flush=True)
    json.dump({"model": name, "shared_prefix": len(pre_mem), "facts": len(facts), "entries": done}, open(out, "w"), indent=1)
    os.remove(out + ".partial")


def views(cx=None):
    """Corpus views as [(key, pool, index text, embedded/BM25 text)]. base = T69; rows = CSV by rows (A.3);
    ctx and ctx_rows = the same with the 2B's line before each memory and file chunk, plus one entry per generated
    manual question pointing at its section (several vectors, one section)."""
    V = {"base": [(k, p, t, t) for k, p, t in rr.corpus()], "rows": [(k, p, t, t) for k, p, t in corpus_rows()]}
    if cx:
        E = cx["entries"]
        for name in ("base", "rows"):
            v = [(k, p, t, f"{E[k]['line']}\n{t}" if p != "manual" else t) for k, p, t, _ in V[name]]
            for k, p, t, _ in V[name]:
                if p == "manual":
                    v += [(k, p, t, l.strip(" -*0123456789.")) for l in E[k]["line"].splitlines() if len(l.strip()) > 8]
            V["ctx" if name == "base" else "ctx_rows"] = v
    return V


def embed_views(out, ctx_path=None):
    """Host: every query, every probe and every entry of every view embedded once (fp32, CPU); stored are the
    query cosines, the entry-entry cosines (supersession) and the probe cosines (hub)."""
    import torch
    assert rr.free_gb() >= 8, f"MemAvailable {rr.free_gb()} GB, want 8"
    emb = rr.embedder()
    qv = lambda q: emb(f"Instruct: {rr.TASK}\nQuery:{q}")[0]
    Qv = torch.stack([qv(r["q"]) for r in queries()])
    Pv = torch.stack([qv(p) for p in PROBES])
    cache, R = {}, {"task": rr.TASK, "queries": [r["q"] for r in queries()], "views": {}}
    for name, v in views(json.load(open(ctx_path)) if ctx_path else None).items():
        for *_, e in v:
            if e not in cache:
                cache[e] = emb(e)
        D = torch.stack([cache[e][0] for *_, e in v])
        rnd = lambda M: [[round(x, 4) for x in row] for row in M.tolist()]
        R["views"][name] = {"keys": [x[0] for x in v], "tokens": [cache[e][1] for *_, e in v],
                            "cos": rnd(Qv @ D.T), "dd": rnd(D @ D.T), "probe": rnd(Pv @ D.T)}
        print(name, len(v), "entries", f"free {rr.free_gb()} GB", flush=True)
    json.dump(R, open(out, "w"))


def rel_words(s):
    """Relations named in s; a capitalized word is a name, not a relation (the printer brand Brother)."""
    return {r for w in re.findall(r"[A-Za-z]+", s) for r, ws in REL.items() if w in ws}


def aliases(texts):
    """Name -> relation: in a memory with a relation word, the capitalized word nearest to it (5 words either
    side) is that person's or pet's name ("mia sorella Chiara", "il cane di mia sorella si chiama Rocco")."""
    al = {}
    for t in texts:
        w = re.findall(r"[^\W\d_]+'?[^\W\d_]*", t.split("] ", 1)[-1])
        for i, x in enumerate(w):
            r = rel_words(x)
            if not r:
                continue
            near = [(abs(j - i), w[j]) for j in range(max(0, i - 5), min(len(w), i + 6))
                    if w[j][:1].isupper() and w[j] not in CAPSTOP and not rel_words(w[j])]
            if near:
                al.setdefault(min(near)[1].lower(), set()).update(r)
    return al


def owner(s, al):
    return rel_words(s) | {r for w in re.findall(r"[^\W\d_]+", s.lower()) for r in al.get(w, ())}


NAMESTOP = CAPSTOP | set("Instruct Query January February March April May June July August September October November "
                         "December Monday Tuesday Wednesday Thursday Friday Saturday Sunday CSV JSON Wi Fi".split())


def names(s):
    return {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", s.split("] ", 1)[-1]) if w not in NAMESTOP}


PATHS = re.compile(r"(?<![\w/])(/[\w./-]*\w|[\w-]+\.(?:txt|md|csv|py|lua|js|json))")


class View:
    """One corpus view, ready to rank: texts, pools, dates, the hub score, the alias table and the named files."""
    def __init__(self, V, name, cx=None):
        x = V["views"][name]
        v = views(cx)[name]
        self.keys, self.cos, self.dd = x["keys"], x["cos"], x["dd"]
        assert self.keys == [k for k, *_ in v], f"view {name}: stored keys differ, re-embed"
        self.n = len(self.keys)
        self.text, self.etext = [t for _, _, t, _ in v], [e for *_, e in v]
        self.line = [cx["entries"][k]["line"] if cx and p != "manual" else "" for k, p, _, _ in v]
        self.pool = ["manual" if k.startswith("man:") else "files" if k.startswith("f:") else "memory" for k in self.keys]
        self.hub = [sum(sorted(col, reverse=True)[:HUB_K]) / HUB_K for col in zip(*x["probe"])]
        base = lambda k: k.split("#")[0]
        self.date = [DATE.get(k, DATE.get(base(k), "")) for k in self.keys]
        self.order = [int(k[1:]) if re.fullmatch(r"m\d+", k) else 0 for k in self.keys]  # append order breaks a same-day tie
        self.files = {}
        for i, k in enumerate(self.keys):
            if self.pool[i] == "files":
                self.files.setdefault(base(k)[2:], []).append(i)
        self.named = []  # per entry: indices of the chunks of the files it names (full path or basename)
        for i, t in enumerate(self.text):
            hit = set()
            if self.pool[i] == "memory":
                for p in PATHS.findall(t):
                    hit |= {j for f, js in self.files.items() if f == p or f.rsplit("/", 1)[-1] == p for j in js}
            self.named.append(sorted(hit))
        mem = [t for t, p in zip(self.text, self.pool) if p == "memory"]
        self.al_text = aliases(mem)
        self.al_ctx = aliases(mem + [l for l, p in zip(self.line, self.pool) if p == "memory"])

    def own_src(self, d, use_ctx):
        """Text read for the owner: the index text (A), the 2B line only (B), or both (A+B)."""
        return {None: self.text[d], "only": self.line[d]}.get(use_ctx, self.text[d] + " " + self.line[d])


def rank(view, r, qi, cfg):
    """The answer list for query r under cfg (a dict of rules; missing = off): [(key, score, via, index)]."""
    g = cfg.get
    al = view.al_ctx if g("own_ctx") else view.al_text
    q_own = owner(r["q"], al)
    s = list(view.cos[qi])
    for d in range(view.n):
        s[d] -= g("hub", 0) * view.hub[d]
        if view.pool[d] == "memory" and (g("own1") or g("own2")):
            e_own = owner(view.own_src(d, g("own_ctx")), al)
            if e_own - q_own:
                s[d] -= g("own1", 0)
            if q_own and not e_own & q_own:
                s[d] -= g("own2", 0)
    h = search(r["q"], list(range(view.n)), view.etext, s, "always" if g("bm25") == "always" else True)
    seen, lst = set(), []
    for d, sc, via in h:  # several vectors of one manual section: the best one stands for it
        k = view.keys[d]
        if k in seen:
            continue
        seen.add(k)
        lst.append((k, sc, via, d))
    man = [x for x in lst if view.pool[x[3]] == "manual"]
    own = [x for x in lst if view.pool[x[3]] != "manual"]
    if g("ptr") is not None and own:  # A.1: a memory naming a file loses to that file's best chunk
        for i, x in enumerate(own[:3]):
            js = view.named[x[3]]
            if js:
                j = max(js, key=lambda j: s[j])
                if s[j] >= x[1] - g("ptr"):
                    rest = [y for y in own if y[3] != j]
                    own = rest[:i] + [(view.keys[j], s[j], "emb", j)] + rest[i:]
                break
    if g("sup") and own:  # supersession: of near-same records near the top, the newest wins
        tau, m = g("sup")
        t = own[0]
        same_file = lambda y: g("drop_sup") and view.pool[y[3]] == "files" and view.keys[y[3]].split("#")[0] == view.keys[t[3]].split("#")[0]
        grp = [y for y in own[:8] if view.date[y[3]] and view.date[t[3]] and y[1] >= t[1] - m
               and (y is t or (view.dd[t[3]][y[3]] >= tau and not same_file(y)))]  # T72: chunks of one file never
        if grp:
            w = max(grp, key=lambda y: (view.date[y[3]], view.order[y[3]]))
            own = [w] + [y for y in own if y is not w and not (g("drop_sup") and y in grp)]  # T72: older ones out
    if g("hop") and own:  # second hop: a name the top entry adds (not in the query) lifts entries holding it
        nm = names(view.text[own[0][3]]) - set(re.findall(r"[a-z]+", r["q"].lower()))
        cand = [y for y in own[1:8] if nm & set(re.findall(r"[a-z]+", view.text[y[3]].lower()))]
        if cand:
            b = max(cand, key=lambda y: y[1])
            if b[1] + g("hop") > own[0][1]:
                own = [b] + [y for y in own if y is not b]
    if g("sib") and len(man) > 1:  # sibling sections: within the margin, most query words in title + Asked as
        qw = set(words(r["q"]))
        near = [x for x in man if x[1] >= man[0][1] - g("sib")]
        lex = lambda x: len(qw & set(words(view.text[x[3]].split("] ", 1)[0] + " " + " ".join(
            l for l in view.text[x[3]].split("\n") if l.startswith("Asked as:")))))
        b = max(near, key=lambda x: (lex(x), x[1]))
        man = [b] + [x for x in man if x is not b]
    if g("mode") == "split":
        return man if r["pool"] == "manual" else own
    mc = view.cos[qi][man[0][3]] if man and g("raw_floor") else man[0][1] if man else 0  # T72: the floor on the raw cosine
    first = man[:1] if man and ((howto(r["q"]) and mc >= RECALL_MIN) or
                                (g("man") is not None and own and man[0][1] >= own[0][1] + g("man"))) else []
    return first + own



# ---- T70: tune on the tune split, then score the frozen configs once ----

FLOORS = (0.50, 0.55, 0.60, 0.65, 0.70)
# design -> (view, rules it may use, grid per rule); "None" = rule off
GRID = {
    "hub": (0, 0.25, 0.5, 1.0), "own1": (0, 0.03, 0.06, 0.1), "own2": (0, 0.03, 0.06, 0.1),
    "ptr": (None, 0.0, 0.03, 0.06, 0.1), "sup": (None, (0.5, 0.05), (0.5, 0.1), (0.5, 0.15), (0.5, 0.2), (0.6, 0.05), (0.6, 0.1), (0.6, 0.15),
            (0.6, 0.2), (0.7, 0.05), (0.7, 0.1), (0.7, 0.2)),
    "hop": (0, 0.03, 0.06), "sib": (0, 0.02, 0.04), "man": (None, 0.0, 0.03, 0.06, 0.1), "bm25": ("gate", "always"),
}
A_RULES = ("man", "sib", "hub", "own1", "own2", "ptr", "sup", "hop")
B_RULES = ("bm25", "own1", "own2")
DESIGNS = {  # baseline = T69's kernel ranking; B's owner view reads the 2B line only, A+B's both texts
    "baseline": ("base", (), {}),
    "A": ("rows", A_RULES, {}),
    "B": ("ctx", B_RULES, {"own_ctx": "only"}),
    "A+B": ("ctx_rows", A_RULES + ("bm25",), {"own_ctx": "both"}),
}


def evaluate(view, cfg, rows, floor=RECALL_MIN):
    """[(row, (t1, t3, slot))]: answerable rows by rank; no-answer by the floor on the first RECALL_K."""
    out = []
    for qi, r in rows:
        lst = [(k, s, v) for k, s, v, _ in rank(view, r, qi, cfg)]
        if r["cls"] == "noanswer":
            a = not any(x[1] >= (floor if x[2] == "emb" else RECALL_MIN_BM25) for x in lst[:RECALL_K])
            out.append((r, (a, a, a), lst))
        else:
            t1, t3, _ = score(r, lst)
            out.append((r, (t1, t3, t1 and lst[0][1] >= (floor if lst[0][2] == "emb" else RECALL_MIN_BM25)), lst))
    return out


def load_views(vec, cx_path=None):
    V = json.load(open(vec))
    assert V["queries"] == [r["q"] for r in queries()], "vectors are from another query set: re-embed"
    cx = json.load(open(cx_path)) if cx_path else None
    return {n: View(V, n, cx) for n in V["views"]}


def tune(vec, cfg_out, cx_path=None):
    """Greedy coordinate ascent over each design's rules, two passes, on the TUNE split only: objective =
    answerable top-1, then top-3, then fewer rules on (the first grid value wins a tie). Then the no-answer
    floor: most right abstentions + answerable top-1 hits above the floor. Held-out rows are never scored here."""
    Vs = load_views(vec, cx_path)
    rows = [(i, r) for i, r in enumerate(queries()) if r["split"] == "tune"]
    ans = [x for x in rows if x[1]["cls"] != "noanswer"]
    obj = lambda S: (sum(s[1][0] for s in S), sum(s[1][1] for s in S))
    out = {}
    for d, (vn, rules, fixed) in DESIGNS.items():
        if vn not in Vs:
            print(f"{d}: view {vn} missing (no CTX.json), skipped")
            continue
        cfg = dict(fixed, **{k: GRID[k][0] for k in rules})
        best = obj(evaluate(Vs[vn], cfg, ans))
        for _ in range(2):
            for k in rules:
                for v in GRID[k]:
                    c = dict(cfg, **{k: v})
                    o = obj(evaluate(Vs[vn], c, ans))
                    if o > best:
                        best, cfg = o, c
        fl = max(FLOORS, key=lambda f: (sum(s[1][2] for s in evaluate(Vs[vn], cfg, rows, f)), f == RECALL_MIN))
        out[d] = {"view": vn, "cfg": cfg, "floor": fl, "tune_top1": best[0], "tune_top3": best[1], "tune_n": len(ans)}
        print(f"{d}: tune top-1 {best[0]}/{len(ans)}, top-3 {best[1]}, floor {fl}, cfg {cfg}", flush=True)
    json.dump(out, open(cfg_out, "w"), indent=1)


def wilson(a, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = a / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def ab(vec, cfg_path, out, cx_path=None):
    """The frozen configs of CFG, scored once on every query: per class, split and design; raw per-query JSON."""
    Vs, F = load_views(vec, cx_path), json.load(open(cfg_path))
    for f in F.values():
        f["cfg"] = {k: tuple(v) if isinstance(v, list) else v for k, v in f["cfg"].items()}
    F = {"baseline": {"view": "base", "cfg": {}, "floor": RECALL_MIN}, **{k: v for k, v in F.items() if k != "baseline"}}
    rows = list(enumerate(queries()))
    S = {d: evaluate(Vs[f["view"]], f["cfg"], rows, f["floor"]) for d, f in F.items()}
    D = list(S)
    cnt = lambda d, f, j: (sum(s[1][j] for s in S[d] if f(s[0])), sum(1 for s in S[d] if f(s[0])))
    res = {"configs": F, "summary": {}, "queries": []}
    for sp in ("held-out", "all"):
        print(f"\n== {sp}: top-1 (top-3) per class ==")
        print("| class | n | " + " | ".join(D) + " |")
        print("|---|---|" + "---|" * len(D))
        for c in CLASSES + ("answerable",):
            f = lambda r, c=c: sp in ("all", r["split"]) and (r["cls"] == c or c == "answerable" and r["cls"] != "noanswer")
            cells = []
            for d in D:
                (a1, n), (a3, _) = cnt(d, f, 0), cnt(d, f, 1)
                cells.append(f"{a1}/{n}" + ("" if c == "noanswer" else f" ({a3})"))
                res["summary"][f"{sp}|{c}|{d}"] = {"top1": a1, "top3": a3, "n": n, "slot": cnt(d, f, 2)[0]}
            print(f"| {c} | {n} | " + " | ".join(cells) + " |")
        for d in D:
            a, n = cnt(d, lambda r: sp in ("all", r["split"]) and r["cls"] != "noanswer", 0)
            print(f"  {d}: answerable top-1 {a}/{n} = {100 * a / n:.1f}%, 95% Wilson CI {wilson(a, n)}; "
                  f"slot (top-1 above floor) {cnt(d, lambda r: sp in ('all', r['split']) and r['cls'] != 'noanswer', 2)[0]}")
    for i, r in rows:
        res["queries"].append({"q": r["q"], "cls": r["cls"], "split": r["split"], "gold": sorted(r["gold"]),
                               **{d: {"t1": S[d][i][1][0], "t3": S[d][i][1][1],
                                      "top3": [[k, round(s, 4)] for k, s, _ in S[d][i][2][:3]]} for d in D}})
    print("\nablation (frozen config minus one rule): answerable top-1 held-out / all")
    res["ablation"] = {}
    for d, f in F.items():
        for k, v in f["cfg"].items():
            if k in GRID and v != GRID[k][0]:
                A = evaluate(Vs[f["view"]], dict(f["cfg"], **{k: GRID[k][0]}), rows, f["floor"])
                h = sum(s[1][0] for s in A if s[0]["split"] == "held-out" and s[0]["cls"] != "noanswer")
                al = sum(s[1][0] for s in A if s[0]["cls"] != "noanswer")
                res["ablation"][f"{d} -{k}"] = {"held-out": h, "all": al}
                print(f"  {d} without {k}: {h} / {al}")
    print("\nheld-out misses of the last design:")
    for i, r in rows:
        if r["split"] == "held-out" and not S[D[-1]][i][1][0]:
            print(f"  [{r['cls']}] {r['q'][:56]:56} -> {S[D[-1]][i][2][0][0] if S[D[-1]][i][2] else '-'}")
    if cx_path:
        res["cost"] = cost(vec, cx_path)
    json.dump(res, open(out, "w"), indent=1)


PREFILL_2B, DECODE_2B, EMB_MS = 213.0, 36.0, 6.0  # tok/s, tok/s, ms/token: M45_RESULT.md, EMBEDDER_RESULT.md (ring 0)


def cost(vec, cx_path):
    """Write cost of B: tokens and host seconds of the 2B per entry, ring 0 [E] from the engine numbers, and the
    re-embedding of the longer texts. Returns a dict (also printed)."""
    import statistics as st
    E, V = json.load(open(cx_path)), json.load(open(vec))
    out = {"shared_prefix": E["shared_prefix"], "facts_in_prefix": E["facts"]}
    for pool in ("memory", "files", "manual"):
        x = [e for e in E["entries"].values() if e["pool"] == pool]
        ring = [e["prompt"] / PREFILL_2B + e["out"] / DECODE_2B for e in x]
        out[pool] = {"n": len(x), "prompt_med": st.median(e["prompt"] for e in x), "out_med": st.median(e["out"] for e in x),
                     "host_s_med": st.median(e["s"] for e in x), "host_s_sum": round(sum(e["s"] for e in x)),
                     "ring0_s_med_E": round(st.median(ring), 2), "ring0_s_sum_E": round(sum(ring))}
    out["prefix_once_ring0_s_E"] = round(E["shared_prefix"] / PREFILL_2B, 1)
    for a, b in (("base", "ctx"), ("rows", "ctx_rows")):
        ta = dict(zip(V["views"][a]["keys"], V["views"][a]["tokens"]))
        tb = [t for k, t, in zip(V["views"][b]["keys"], V["views"][b]["tokens"]) if not k.startswith("man:")]
        extra = sum(tb) - sum(t for k, t in ta.items() if not k.startswith("man:"))
        nq = sum(k.startswith("man:") for k in V["views"][b]["keys"]) - sum(k.startswith("man:") for k in ta)
        out[f"reembed_{b}"] = {"extra_tokens": extra, "per_entry": round(extra / len(tb), 1),
                               "ring0_s_per_entry_E": round(extra / len(tb) * EMB_MS / 1000, 2), "manual_question_rows": nq}
    print(json.dumps(out, indent=1))
    return out


# ---- T72: design A in the kernel (agentd.rs, memrank.rs) ----
#
#   memeval.py ported VEC CFG CTX      host: frozen A with the kernel's semantics (every entry dated, superseded records
#                                      out of the answer, floors on the raw cosine); the number to expect in the OS
#   memeval.py steps                   the shell lines that load the corpus into a booted kernel and search every query
#   memeval.py inos ISO OUTDIR         boot ISO headless without a disk (tools/boottest.py, MEM env), type the steps,
#                                      score the transcript (OUTDIR/timed.log)
#   memeval.py inos-score LOG [OUT]    score a transcript again; per-query JSON to OUT

def ported(vec, cfg_path, cx_path):
    """Frozen config A as agentd runs it. An entry without a date in DATE is dated by /index add at the time it is
    typed (after every date of the set), so it is the newest; floors read the raw cosine."""
    V = load_views(vec, cx_path)["rows"]
    dated = [d or ("9999" if V.pool[i] != "manual" else "") for i, d in enumerate(V.date)]
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in json.load(open(cfg_path))["A"]["cfg"].items()}
    out = {}
    for name, c in (("frozen A (host), floors on the raw cosine", cfg), ("A as ported", dict(cfg, raw_floor=1, drop_sup=1))):
        if c.get("drop_sup"):
            V.date = dated
        S = []
        for qi, r in enumerate(queries()):
            lst = rank(V, r, qi, c)
            raw = [(k, V.cos[qi][d], v) for k, _, v, d in lst]
            S.append((r, score(r, raw)))
        out[name] = S
        print(name)
        table(S)
    return out


def table(S):
    """top-1 (top-3) per class, held-out and all; no-answer = right abstentions."""
    print("| class | held-out | all |\n|---|---|---|")
    for c in CLASSES + ("answerable",):
        f = lambda s, sp: sp in ("all", s[0]["split"]) and (s[0]["cls"] == c or c == "answerable" and s[0]["cls"] != "noanswer")
        cell = lambda sp: (lambda g: f"{sum(s[1][0] for s in g)}/{len(g)}" + ("" if c == "noanswer" else f" ({sum(s[1][1] for s in g)})"))(
            [s for s in S if f(s, sp)])
        print(f"| {c} | {cell('held-out')} | {cell('all')} |")
    for sp in ("held-out", "all"):
        g = [s for s in S if sp in ("all", s[0]["split"]) and s[0]["cls"] != "noanswer"]
        a = sum(s[1][0] for s in g)
        print(f"  {sp} answerable top-1 {a}/{len(g)} = {100 * a / len(g):.1f}%, 95% Wilson CI {wilson(a, len(g))}")


def steps():
    """/index add for every non-manual entry of the rows view (dated ones with -d; the manual is in the kernel
    already, its vectors built in), then /search for every query. Line breaks go as \\n, a backslash as \\\\."""
    out = []
    for k, p, t in corpus_rows():
        if p == "manual":
            continue
        d = DATE.get(k, DATE.get(k.split("#")[0], ""))
        t = t.strip().replace("\\", "\\\\").replace("\n", "\\n")  # the shell's escapes: \\ and \n
        out.append(f"/index add {'-d ' + d + ' ' if d else ''}{t}")
    return out + [f"/search {r['q']}" for r in queries()]


HIT = re.compile(r"^(\d+)\. #(\d+)\s+cos (-?[\d.]+)\s+bm25\s+[\d.]+\s+(\S+)\s+score (-?[\d.]+)\s+(.*)$")


def inos_score(log, out=None):
    """Top-1 / top-3 of what /search printed (the answer list recall and the slot use), the ids mapped back to keys."""
    L = [re.sub(r"^\[\s*[\d.]+s\] ", "", l) for l in open(log, errors="replace").read().splitlines()]
    man = [k for k, p, _ in rr.corpus() if p == "manual"]
    added = [k for k, p, _ in corpus_rows() if p != "manual"]
    ids = [int(m.group(1)) for l in L for m in [re.match(r"\[index\] #(\d+) added", l)] if m]
    assert len(ids) == len(added), f"{len(ids)} entries added, want {len(added)}"
    key = dict(zip(ids, added))
    key.update({i: k for i, k in enumerate(man) if i not in key})
    blocks, cur = [], None
    for l in L:
        if re.match(r"ringzero(?:\[[^\]\n]*\])?> /search ", l):
            cur = []
            blocks.append(cur)
        elif cur is not None and (m := HIT.match(l)):
            cur.append((key[int(m.group(2))], float(m.group(3)), "emb" if m.group(4) == "emb" else "bm25", m.group(6)))
    Q = queries()
    assert len(blocks) == len(Q), f"{len(blocks)} /search blocks, want {len(Q)}"
    for b in blocks:  # the printed text must be the entry the id says (a wrong id map would score nonsense)
        for k, _, _, t in b:
            assert not k.startswith("man:") or f"[manual: {k[4:]}]" in t, (k, t[:60])
    S = [(r, score(r, [x[:3] for x in b])) for r, b in zip(Q, blocks)]
    table(S)
    print("held-out misses (query -> top-1):")
    for (r, (t1, _, _)), b in zip(S, blocks):
        if r["split"] == "held-out" and not t1:
            print(f"  [{r['cls']}] {r['q'][:56]:56} -> {', '.join(f'{k} {c:.2f}' for k, c, _, _ in b[:2]) or '-'}")
    if out:
        json.dump([{"q": r["q"], "cls": r["cls"], "split": r["split"], "t1": s[0], "t3": s[1],
                    "top": [[k, c, v] for k, c, v, _ in b]} for (r, s), b in zip(S, blocks)], open(out, "w"), indent=1)
    return S


def inos(iso, outdir):
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "DISK"}  # no disk: an index of the manual and the steps only
    subprocess.run([sys.executable, os.path.dirname(os.path.abspath(__file__)) + "/boottest.py", iso, "bios", outdir, "8",
                    *steps()], env=env, stdout=subprocess.DEVNULL, check=True)
    inos_score(f"{outdir}/timed.log", f"{outdir}/inos.json")


if __name__ == "__main__":
    a = sys.argv
    {"run": lambda: run(a[2]), "report": lambda: report(a[2]), "check": check, "ctx": lambda: ctx(a[2]),
     "embed": lambda: embed_views(a[2], a[3] if len(a) > 3 else None),
     "tune": lambda: tune(a[2], a[3], a[4] if len(a) > 4 else None),
     "ab": lambda: ab(a[2], a[3], a[4], a[5] if len(a) > 5 else None), "cost": lambda: cost(a[2], a[3]),
     "ported": lambda: ported(a[2], a[3], a[4]), "steps": lambda: print("\n".join(steps())),
     "inos": lambda: inos(a[2], a[3]), "inos-score": lambda: inos_score(a[2], a[3] if len(a) > 3 else None)}[a[1]]()
