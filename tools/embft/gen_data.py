#!/usr/bin/env python3
"""T74: synthetic (query, positive, hard negatives) triples for a contrastive fine-tune of Qwen3-Embedding-0.6B.

  gen_data.py OUT_DIR [--seed N] [--personas N]   writes OUT_DIR/train.jsonl, dev.jsonl, stats.json

Pure Python, no model. One record per line: {"cls", "lang", "q", "pos", "neg": [3 texts], "persona", "tpl"}; q is
the raw user question (train.py adds the kernel's "Instruct: ...\\nQuery:" prefix), pos and neg are index texts in
agentd's formats ([fact] key: value, [said] ..., [episode N] ..., [PATH] chunk, [manual: Title] ...).

Every T69 answerable class has its own generator (noanswer has no positive, so it is left to the gate):
  para, exact, pointer, sibling, hub, owner, raw, table, temporal, contradiction, multihop
Multihop emits two kinds: the first hop (question -> the record that names the person), and the second step in
the form tools/memhop.py's req_names uses (question + the name -> the record about that person).

The world is built from synthetic personas whose vocabulary (names, places, brands, files) is checked at import
against the T69 eval set (corpus, queries, files): no capitalized word and no file path of the eval set is used.
Records that share a word 6-gram with an eval query or an 8-gram with an eval entry are dropped here and counted;
tools/embft/check_disjoint.py proves the same independently. The dev split holds whole personas.
"""
import hashlib, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import check_disjoint as CD  # noqa: E402  (tools/embft/check_disjoint.py: the eval side and the n-gram tests)

# ---- vocabulary (asserted disjoint from the eval set below) ----
NAMES_F = ("Federica Valentina Martina Alessia Ilaria Beatrice Elisa Noemi Camilla Greta Viola Arianna Serena Debora "
           "Hannah Megan Chloe Zoe Olivia Emily Lucy Grace").split()
NAMES_M = ("Simone Lorenzo Stefano Matteo Riccardo Nicola Gabriele Filippo Emanuele Pietro Jacopo Alberto Michele "
           "Oliver Jacob Ethan Ryan Owen Samuel Daniel Henry").split()
SURNAMES = "Fontana Ricci Mancini Lombardi Barbieri Conti Marino Greco Leone Longo Serra Fabbri Caruso Ferraro Pellegrini Santoro".split()
CITIES = "Verona Padova Trieste Bari Lecce Perugia Ancona Genova Pisa Siena Udine Trento Bergamo Brescia Rimini Cagliari Palermo Catania Salerno Vienna Dublino Oslo Praga Berlino Lione Valencia Zurigo".split()
PETNAMES = "Biscotto Nuvola Zeus Briciola Ombra Pepe Tobia Kira Argo Fiocco Brina Oreo".split()
PETS = (("cane", "dog"), ("gatto", "cat"), ("coniglio", "rabbit"), ("pappagallo", "parrot"))
JOBS = (("infermiera", "nurse"), ("ingegnere", "engineer"), ("cuoco", "cook"), ("avvocato", "lawyer"),
        ("farmacista", "pharmacist"), ("elettricista", "electrician"), ("grafica", "graphic designer"),
        ("veterinaria", "vet"), ("pilota", "pilot"), ("bibliotecaria", "librarian"), ("idraulico", "plumber"),
        ("commercialista", "accountant"))
FOODS = (("lasagne", "lasagna"), ("risotto ai funghi", "mushroom risotto"), ("sushi", "sushi"), ("polpette", "meatballs"),
         ("tiramisu'", "tiramisu"), ("gnocchi", "gnocchi"), ("curry", "curry"), ("crostata", "jam tart"))
ALLERG = (("arachidi", "peanuts"), ("lattosio", "lactose"), ("crostacei", "shellfish"), ("polline", "pollen"),
          ("noci", "walnuts"), ("glutine", "gluten"))
CARS = ("Renault Clio", "Skoda Octavia", "Peugeot 208", "Volkswagen Polo", "Kia Picanto", "Hyundai Kona",
        "Dacia Sandero", "Mazda CX-30", "Opel Corsa", "Seat Ibiza")
PHONES = ("Samsung Galaxy S23", "Xiaomi 13", "OnePlus 12", "Nokia G42", "Sony Xperia 10")
HOBBIES = (("arrampicata", "climbing"), ("ceramica", "pottery"), ("scacchi", "chess"), ("fotografia", "photography"),
           ("nuoto", "swimming"), ("giardinaggio", "gardening"), ("chitarra", "guitar"), ("yoga", "yoga"))
MONTHS = (("gennaio", "January"), ("febbraio", "February"), ("marzo", "March"), ("aprile", "April"), ("maggio", "May"),
          ("giugno", "June"), ("luglio", "July"), ("agosto", "August"), ("settembre", "September"),
          ("ottobre", "October"), ("novembre", "November"), ("dicembre", "December"))
DAYS = (("lunedi'", "Monday"), ("martedi'", "Tuesday"), ("mercoledi'", "Wednesday"), ("giovedi'", "Thursday"),
        ("venerdi'", "Friday"), ("sabato", "Saturday"), ("domenica", "Sunday"))
# relations: IT possessive, EN, fact-key prefix, gender of the person
RELS = (("mia sorella", "my sister", "sister", "f"), ("mio fratello", "my brother", "brother", "m"),
        ("mia madre", "my mother", "mom", "f"), ("mio padre", "my father", "dad", "m"),
        ("il mio migliore amico", "my best friend", "friend", "m"), ("la mia collega", "my colleague", "colleague", "f"),
        ("mia nonna", "my grandmother", "grandma", "f"), ("mio zio", "my uncle", "uncle", "m"),
        ("mia cugina", "my cousin", "cousin", "f"), ("il mio vicino", "my neighbour", "neighbour", "m"))
DIRS = ("/data", "/srv/docs", "/home/archivio", "/mnt/usb", "/opt/lavoro")


def _vocab():
    out = set(NAMES_F + NAMES_M + SURNAMES + CITIES + PETNAMES)
    for x in CARS + PHONES:
        out |= set(x.split())
    return {w for w in out if w[:1].isupper()}


_clash = _vocab() & CD.eval_caps()
assert not _clash, f"vocabulary shared with the eval set: {sorted(_clash)}"


# ---- one persona: the user and the people around them, as records ----

class Persona:
    def __init__(self, rng, pid):
        self.rng, self.pid = rng, pid
        self.user = rng.choice(NAMES_F + NAMES_M)
        self.rels, used = [], {self.user}
        for it, en, key, sex in rng.sample(RELS, 6):
            nm = rng.choice([n for n in (NAMES_F if sex == "f" else NAMES_M) if n not in used])
            used.add(nm)
            self.rels.append({"it": it, "en": en, "key": key, "name": nm, "city": rng.choice(CITIES),
                              "job": rng.choice(JOBS), "bday": (rng.randint(1, 28), rng.choice(MONTHS))})
        self.city = rng.choice(CITIES)
        self.job = rng.choice(JOBS)
        self.bday = (rng.randint(1, 28), rng.choice(MONTHS))
        self.pet = (rng.choice(PETS), rng.choice(PETNAMES))
        self.ep = rng.randint(5, 60)

    def episode(self):
        self.ep += 1
        return f"[episode {self.ep}]"


def di(rel_it):
    """'di' + the relation, with the article joined: di mia sorella, del mio vicino, della mia collega."""
    return {"il ": "del ", "la ": "della "}.get(rel_it[:3], "di ") + (rel_it[3:] if rel_it[:3] in ("il ", "la ") else rel_it)


def pick(rng, seq, k):
    return rng.sample(list(seq), k)


def rec(cls, lang, q, pos, neg, p, tpl):
    return {"cls": cls, "lang": lang, "q": q, "pos": pos, "neg": neg, "persona": p.pid, "tpl": tpl}


def fillers(p, rng, k=3):
    """Generic records of the persona (used to pad hard negatives to 3)."""
    pool = [f"[fact] hobby: {rng.choice(HOBBIES)[0]}", f"[fact] city: vivo a {p.city}",
            f"{p.episode()} The user said hello and asked how the agent was doing.",
            f"[fact] job: lavoro come {p.job[0]}", f"[said] il mio piatto preferito: {rng.choice(FOODS)[0]}",
            f"{p.episode()} The user asked what time it was and the agent read the clock."]
    return pick(rng, pool, k)


def g_para(p, rng):
    j, (d, m) = p.job, p.bday
    food, hob = rng.choice(FOODS), rng.choice(HOBBIES)
    items = [
        (f"[fact] job: lavoro come {j[0]}", ["che mestiere faccio?", "di cosa mi occupo per vivere?"],
         ["what do I do for work?", "what's my profession again?"]),
        (f"[fact] birthday: il mio compleanno e' il {d} {m[0]}", ["quando sono nato?", "che giorno compio gli anni?"],
         ["when's my birthday?", "on which day was I born?"]),
        (f"[said] il mio piatto preferito: {food[0]}", ["qual e' il cibo che amo di piu'?", "cosa mi piace mangiare?"],
         ["what's my favourite dish?", "which food do I like best?"]),
        (f"[fact] hobby: {hob[0]}", ["che passatempo ho?", "cosa faccio nel tempo libero?"],
         ["what's my hobby?", "how do I spend my free time?"]),
    ]
    out = []
    for i, (pos, it, en) in enumerate(items):
        neg = [x[0] for k, x in enumerate(items) if k != i]
        out += [rec("para", "it", q, pos, neg, p, f"para{i}") for q in it]
        out += [rec("para", "en", q, pos, neg, p, f"para{i}") for q in en]
    return out


def g_exact(p, rng):
    out = []
    hosts = [f"{rng.choice(('nas', 'srv', 'cam', 'pi'))}-{rng.randint(10, 99)}" for _ in range(4)]
    ips = [f"10.{rng.randint(1, 250)}.{rng.randint(2, 250)}.{rng.randint(2, 250)}" for _ in hosts]
    recs = [f"[fact] host_{h.replace('-', '_')}: {h} ha indirizzo {ip}" for h, ip in zip(hosts, ips)]
    for i, h in enumerate(hosts):
        neg = [r for k, r in enumerate(recs) if k != i]
        out.append(rec("exact", "it", f"che IP ha {h}?", recs[i], neg, p, "exact_ip"))
        out.append(rec("exact", "en", f"what's the address of {h}?", recs[i], neg, p, "exact_ip"))
    scripts = [f"{rng.choice(('sync', 'deploy', 'pulizia', 'report'))}_v{rng.randint(2, 9)}.sh" for _ in range(3)]
    srec = [f"{p.episode()} The user wrote {s} to {rng.choice(('copy photos to the NAS', 'rotate the logs', 'renew the certificates'))}."
            for s in scripts]
    for i, s in enumerate(scripts):
        neg = [r for k, r in enumerate(srec) if k != i] + [recs[0]]
        out.append(rec("exact", "it", f"a cosa serve {s}?", srec[i], neg, p, "exact_script"))
        out.append(rec("exact", "en", f"what does {s} do?", srec[i], neg, p, "exact_script"))
    return out


def g_pointer(p, rng):
    out = []
    d = rng.choice(DIRS)
    files = {
        f"{d}/ricetta_{rng.choice(('lasagne', 'crostata', 'polpette'))}.txt":
            ("Ingredienti: 300 g di farina, 2 uova, 150 g di zucchero, un pizzico di sale. Forno a 180 gradi per 35 minuti.",
             ["quanta farina serve per la ricetta?", "a che temperatura va il forno?"],
             ["how much flour does the recipe need?", "what oven temperature does the recipe say?"]),
        f"{d}/garanzia_lavatrice.txt":
            (f"Lavatrice acquistata il 3 {rng.choice(MONTHS)[0]}, garanzia di 2 anni, assistenza al numero 800 55 {rng.randint(100, 999)}.",
             ["quanto dura la garanzia della lavatrice?", "che numero chiamo per l'assistenza della lavatrice?"],
             ["how long is the washing machine warranty?", "which number do I call for the washer service?"]),
        f"{d}/palestra.md":
            (f"# Scheda palestra\nLunedi': panca 4x8, squat 4x10. Giovedi': trazioni 3x6, stacchi 3x5. Istruttore: {rng.choice(NAMES_M)}.",
             ["cosa faccio il giovedi' in palestra?", "quante serie di squat ho in scheda?"],
             ["what's my Thursday workout?", "how many squat sets are on my plan?"]),
    }
    paths = list(files)
    for path, (body, it, en) in files.items():
        pos = f"[{path}] {body}"
        ptr = f"{p.episode()} The agent read {path} aloud for the user."
        ptr2 = f"[said] ho salvato le cose importanti in {path.rsplit('/', 1)[-1]}"
        other = f"[{paths[(paths.index(path) + 1) % 3]}] {files[paths[(paths.index(path) + 1) % 3]][0]}"
        out += [rec("pointer", "it", q, pos, [ptr, ptr2, other], p, "pointer") for q in it]
        out += [rec("pointer", "en", q, pos, [ptr, ptr2, other], p, "pointer") for q in en]
    return out


# synthetic help sections in the manual's format, for a generic Unix-like system (not ringzero's manual)
HELP = {
    "print": [("Print a file", "how do I print a document, come stampo un file, print this", "`lp FILE` sends a file to the default printer; `lp -n 2 FILE` prints two copies."),
              ("Cancel a print job", "how do I stop printing, come annullo una stampa, cancel the print", "`cancel ID` removes a job from the queue; `lpstat -o` lists the queued jobs and their ids."),
              ("Choose the default printer", "which printer is used, come cambio stampante predefinita, set default printer", "`lpoptions -d NAME` makes NAME the default printer for every later job.")],
    "archive": [("Create an archive", "how do I make a tar file, come comprimo una cartella, pack a folder", "`tar czf out.tar.gz DIR` writes DIR into a gzip-compressed archive."),
                ("Extract an archive", "how do I unpack a tar.gz, come estraggo un archivio, open a zip", "`tar xzf FILE.tar.gz` extracts into the current directory; `unzip FILE.zip` for zip files."),
                ("List an archive", "what is inside this archive, cosa c'e' nell'archivio, show the files in a tar", "`tar tzf FILE.tar.gz` lists the members without extracting them.")],
    "schedule": [("Run a job every day", "how do I run a script daily, come pianifico uno script, schedule a task", "`crontab -e` opens the table; a line `0 7 * * * CMD` runs CMD at 07:00 every day."),
                 ("List scheduled jobs", "which jobs are scheduled, quali attivita' sono pianificate, show my cron", "`crontab -l` prints the current table; empty output means no job."),
                 ("Run a job once later", "run this in an hour, esegui tra un'ora, delay a command", "`at now + 1 hour` reads commands to run once, at that time.")],
    "users": [("Add a user", "how do I create an account, come aggiungo un utente, new user", "`useradd -m NAME` creates the user and the home directory; set the password with `passwd NAME`."),
              ("Change a password", "how do I change my password, come cambio la password, new password", "`passwd` asks for the old password and then the new one twice."),
              ("Remove a user", "delete an account, come elimino un utente, remove user", "`userdel -r NAME` removes the user and the home directory.")],
    "packages": [("Install a package", "how do I install a program, come installo un pacchetto, add software", "`pkg install NAME` downloads and installs NAME and its dependencies."),
                 ("Update the system", "how do I update everything, come aggiorno il sistema, upgrade packages", "`pkg upgrade` updates every installed package to the newest version."),
                 ("Remove a package", "uninstall a program, come disinstallo un pacchetto, remove software", "`pkg remove NAME` uninstalls NAME; its configuration files stay.")],
    "disks": [("Show free disk space", "how full is the disk, quanto spazio libero ho, disk usage", "`df -h` prints size, used and free space per mounted file system."),
              ("Find big folders", "what takes the most space, cosa occupa piu' spazio, biggest directories", "`du -sh *` prints the size of each entry of the current directory."),
              ("Mount a USB stick", "how do I open a usb drive, come monto una chiavetta, attach a disk", "`mount /dev/sdb1 /mnt/usb` makes the stick's files appear under /mnt/usb.")],
    "time": [("Set the time zone", "change the time zone, come cambio fuso orario, wrong time zone", "`timedatectl set-timezone Europe/Rome` sets the zone; the clock itself does not change."),
             ("Show the date", "what day is it, che giorno e' oggi, print the date", "`date` prints the current date and time in the local zone."),
             ("Sync the clock", "the clock is wrong, l'orologio e' sbagliato, sync time", "`timedatectl set-ntp true` keeps the clock in sync with network time servers.")],
}
HELP_Q = {  # extra phrasings per section title, IT and EN; the "Asked as" line is the section's own
    "Print a file": (["mi stampi questo documento?", "voglio stampare un pdf"], ["can you print the report for me?", "send this file to the printer"]),
    "Cancel a print job": (["la stampante sta stampando la cosa sbagliata, fermala", "togli il lavoro dalla coda di stampa"], ["the printer is printing the wrong thing, stop it", "remove my job from the print queue"]),
    "Choose the default printer": (["voglio usare sempre la stampante del piano terra", "imposta la stampante di default"], ["always use the downstairs printer", "make the new printer the default"]),
    "Create an archive": (["fammi un file compresso della cartella foto", "come impacchetto i documenti in un solo file?"], ["zip up the photos folder", "bundle these documents into one file"]),
    "Extract an archive": (["ho scaricato un tar.gz, come lo apro?", "scompatta questo file"], ["I downloaded a tar.gz, how do I open it?", "unpack this file"]),
    "List an archive": (["vedere i file dentro un tar senza aprirlo", "elenca il contenuto del pacchetto"], ["peek inside the tarball without extracting", "list what's in the archive"]),
    "Run a job every day": (["fai partire il backup ogni mattina alle sette", "lancia lo script tutti i giorni"], ["run the backup every morning at seven", "start the script each day"]),
    "List scheduled jobs": (["cosa parte in automatico?", "elenca i job pianificati"], ["what runs automatically?", "list my scheduled jobs"]),
    "Run a job once later": (["spegni il computer tra un'ora", "fai partire il comando stasera una volta sola"], ["shut down in an hour", "run the command tonight, just once"]),
    "Add a user": (["crea un account per mio figlio", "aggiungi un nuovo utente al sistema"], ["make an account for my son", "add a new user to the system"]),
    "Change a password": (["voglio una password nuova", "come aggiorno la mia password?"], ["I want a new password", "how do I update my password?"]),
    "Remove a user": (["cancella l'account del vecchio stagista", "togli un utente"], ["delete the old intern's account", "get rid of a user"]),
    "Install a package": (["mi serve un editor di testo, installalo", "come metto un nuovo programma?"], ["I need a text editor, install one", "how do I add a new program?"]),
    "Update the system": (["aggiorna tutti i programmi", "ci sono aggiornamenti da fare?"], ["update all the programs", "bring the system up to date"]),
    "Remove a package": (["togli quel programma che non uso", "disinstalla il lettore musicale"], ["remove that program I never use", "uninstall the music player"]),
    "Show free disk space": (["il disco e' pieno?", "quanto posto rimane?"], ["is the disk full?", "how much room is left?"]),
    "Find big folders": (["quali cartelle sono le piu' grosse?", "trova cosa riempie il disco"], ["which folders are the largest?", "find what is filling the disk"]),
    "Mount a USB stick": (["ho attaccato una chiavetta, dove la trovo?", "collega il disco esterno"], ["I plugged in a stick, where is it?", "attach the external drive"]),
    "Set the time zone": (["mi sono trasferito all'estero, sistema il fuso", "imposta il fuso di Vienna"], ["I moved abroad, fix the time zone", "set the Rome time zone"]),
    "Show the date": (["che data abbiamo?", "dimmi la data di oggi"], ["what's today's date?", "tell me the date"]),
    "Sync the clock": (["l'ora e' indietro di cinque minuti", "sincronizza l'orologio"], ["the clock is five minutes behind", "sync the clock"]),
}


def g_sibling(p, rng):
    out = []
    for fam, secs in HELP.items():
        texts = [f"[manual: {t}] Asked as: {a}\n{b}" for t, a, b in secs]
        others = [f"[manual: {t}] Asked as: {a}\n{b}" for f2, s2 in HELP.items() if f2 != fam for t, a, b in s2]
        for i, (t, _, _) in enumerate(secs):
            neg = [x for k, x in enumerate(texts) if k != i] + [rng.choice(others)]
            it, en = HELP_Q[t]
            out += [rec("sibling", "it", q, texts[i], neg, p, f"sib:{t}") for q in it]
            out += [rec("sibling", "en", q, texts[i], neg, p, f"sib:{t}") for q in en]
    return out


def g_hub(p, rng):
    sur = rng.choice(SURNAMES)
    hubs = [f"{p.episode()} The user said hi and the agent greeted them back.",
            f"{p.episode()} The user asked a question and the agent answered it.",
            f"[/etc/motd] Welcome to {rng.choice(('box', 'casa', 'studio'))}-{rng.randint(1, 9)}",
            f"{p.episode()} Tool result: the command finished without output."]
    items = [(f"[fact] surname: il mio cognome e' {sur}", ["il mio cognome?", "come faccio di cognome?"], ["my surname?", "what's my last name?"]),
             (f"[fact] city: vivo a {p.city}", ["dove abito?", "la mia citta'?"], ["where do I live?", "my town?"]),
             (f"[fact] job: lavoro come {p.job[0]}", ["il mio lavoro?", "che lavoro?"], ["my job?", "what's my work?"])]
    out = []
    for pos, it, en in items:
        neg = pick(rng, hubs, 3)
        out += [rec("hub", "it", q, pos, neg, p, "hub") for q in it] + [rec("hub", "en", q, pos, neg, p, "hub") for q in en]
    return out


def g_owner(p, rng):
    out = []
    a, b = rng.sample(p.rels, 2)
    d, m = p.bday
    user_b = f"[fact] birthday: il mio compleanno e' il {d} {m[0]}"
    for r in (a, b):
        other = b if r is a else a
        rb = f"[fact] {r['key']}_birthday: il compleanno {di(r['it'])} e' il {r['bday'][0]} {r['bday'][1][0]}"
        ob = f"[fact] {other['key']}_birthday: il compleanno {di(other['it'])} e' il {other['bday'][0]} {other['bday'][1][0]}"
        out.append(rec("owner", "it", f"quando compie gli anni {r['it']}?", rb, [user_b, ob, fillers(p, rng, 1)[0]], p, "owner_bday"))
        out.append(rec("owner", "en", f"when is {r['en']}'s birthday?", rb, [user_b, ob, fillers(p, rng, 1)[0]], p, "owner_bday"))
        out.append(rec("owner", "it", "quando e' il mio compleanno?", user_b, [rb, ob, fillers(p, rng, 1)[0]], p, "owner_self"))
        rj = f"[said] {r['it']} lavora come {r['job'][0]}"
        uj = f"[fact] job: lavoro come {p.job[0]}"
        out.append(rec("owner", "en", f"what does {r['en']} do for a living?", rj, [uj, f"[said] {other['it']} lavora come {other['job'][0]}", user_b], p, "owner_job"))
        out.append(rec("owner", "it", "che lavoro faccio io?", uj, [rj, f"[said] {other['it']} lavora come {other['job'][0]}", user_b], p, "owner_self"))
    (pit, pen), pn = p.pet
    r = a
    mine = f"[fact] pet: il mio {pit} si chiama {pn}"
    theirs = f"[fact] {r['key']}_pet: il {pit} {di(r['it'])} si chiama {rng.choice([x for x in PETNAMES if x != pn])}"
    out.append(rec("owner", "it", f"come si chiama il mio {pit}?", mine, [theirs, fillers(p, rng, 1)[0], user_b], p, "owner_pet"))
    out.append(rec("owner", "en", f"what's the name of {r['en']}'s {pen}?", theirs, [mine, fillers(p, rng, 1)[0], user_b], p, "owner_pet"))
    return out


RAW = [  # (note, IT question, EN question, a record sharing the topic word)
    ("[said] non reggo il peperoncino, mi fa stare male", "c'e' qualcosa di piccante che evito?", "is there a spicy thing I avoid?", "[fact] food: mi piace la cucina messicana"),
    ("[said] la notte dormo con la finestra aperta anche d'inverno", "mi piace il freddo in camera?", "do I like a cold bedroom?", "[fact] heating: il riscaldamento si accende alle 6"),
    ("[said] il martedi' porto mia figlia a nuoto", "chi accompagno in piscina?", "who do I take to the pool?", "[fact] gym: la palestra chiude alle 22"),
    ("[said] odio le riunioni prima delle nove", "a che ora non voglio meeting?", "when don't I want meetings?", "[fact] calendar: le riunioni del team sono in sala blu"),
    ("[said] il caffe' lo bevo solo amaro", "metto lo zucchero nel caffe'?", "do I take sugar in my coffee?", "[fact] coffee_machine: la macchina del caffe' e' una Bialetti"),
    ("[said] ho paura dell'aereo, se posso prendo il treno", "come preferisco viaggiare?", "how do I prefer to travel?", "[fact] airport: l'aeroporto piu' vicino e' a 40 minuti"),
    ("[said] porto le lenti a contatto, non gli occhiali", "ho bisogno degli occhiali?", "do I wear glasses?", "[fact] optician: l'ottico e' in piazza"),
    ("[said] la sera leggo sempre un'ora prima di dormire", "cosa faccio prima di andare a letto?", "what do I do before bed?", "[fact] library: la biblioteca apre alle 9"),
    ("[said] sono mancino, il mouse lo uso con la sinistra", "con che mano scrivo?", "which hand do I write with?", "[fact] mouse: il mouse e' wireless"),
    ("[said] in vacanza preferisco la montagna al mare", "dove mi piace andare in ferie?", "where do I like to go on holiday?", "[fact] beach: la spiaggia piu' vicina e' a un'ora"),
    ("[said] il mio orologio e' sempre avanti di dieci minuti", "posso fidarmi dell'ora che segno?", "can I trust the time on my watch?", "[fact] watch: l'orologio l'ho comprato a Natale"),
    ("[said] dopo le sei di sera niente caffeina", "posso bere un espresso dopo cena?", "can I have an espresso after dinner?", "[fact] bar: il bar sotto casa fa un buon cappuccino"),
    ("[said] mi si gela il naso solo a pensare allo sci", "mi piacciono gli sport invernali?", "do I enjoy winter sports?", "[fact] skis: gli sci sono in cantina"),
    ("[said] ho smesso di fumare a marzo", "fumo ancora?", "am I still a smoker?", "[fact] balcony: sul balcone ci sono due sedie"),
    ("[said] le piante in salotto le annaffio la domenica", "quando do l'acqua ai fiori?", "when do I water the plants?", "[fact] living_room: il salotto ha due finestre"),
    ("[said] il gatto del vicino entra sempre dalla finestra del bagno", "da dove arriva l'animale che mi trovo in casa?", "how does the animal get into my flat?", "[fact] bathroom: il bagno e' stato rifatto"),
    ("[said] parlo bene il tedesco, ho vissuto due anni a Vienna", "che lingue conosco?", "which languages do I speak?", "[fact] course: il corso serale e' il mercoledi'"),
    ("[said] con la bici vado al lavoro solo se non piove", "come raggiungo l'ufficio col bel tempo?", "how do I commute when the weather is good?", "[fact] office: l'ufficio e' al terzo piano"),
    ("[said] la lavastoviglie la faccio partire di notte perche' costa meno", "quando conviene usare gli elettrodomestici?", "when is it cheaper to run appliances?", "[fact] kitchen: la cucina e' a gas"),
    ("[said] in macchina ascolto solo podcast di storia", "cosa sento mentre guido?", "what do I listen to while driving?", "[fact] radio: l'autoradio ha il bluetooth"),
]


def g_raw(p, rng):
    out = []
    for i, (note, it, en, near) in enumerate(RAW):
        neg = [near, RAW[(i + 1) % len(RAW)][0], fillers(p, rng, 1)[0]]
        out += [rec("raw", "it", it, note, neg, p, f"raw{i}"), rec("raw", "en", en, note, neg, p, f"raw{i}")]
    return out


def g_table(p, rng):
    out = []
    d = rng.choice(DIRS)
    path = f"{d}/clienti.csv"
    head = "nome,citta,piano,scadenza"
    people = pick(rng, NAMES_F + NAMES_M, 4)
    rows = [f"{n} {rng.choice(SURNAMES)},{rng.choice(CITIES)},{rng.choice(('base', 'pro', 'famiglia'))},{rng.randint(1, 28)}/{rng.randint(1, 12)}"
            for n in people]
    chunks = [f"[{path}] {head}\n{r}" for r in rows]
    ran = f"{p.episode()} The agent ran a count by city over {path} and printed the totals."
    for i, n in enumerate(people):
        neg = [c for k, c in enumerate(chunks) if k != i][:2] + [ran]
        out.append(rec("table", "it", f"che piano ha {n}?", chunks[i], neg, p, "table_row"))
        out.append(rec("table", "en", f"when does {n}'s plan expire?", chunks[i], neg, p, "table_row"))
    code = f"[{d}/conta.py] import csv\nrows = list(csv.reader(open('{path}')))[1:]\nby = {{}}\nfor r in rows:\n    by[r[2]] = by.get(r[2], 0) + 1\nprint(by)"
    ran2 = f"{p.episode()} The agent ran {d}/conta.py and showed the result."
    neg = [ran2, chunks[0], ran]
    out.append(rec("table", "en", "what does conta.py group the customers by?", code, neg, p, "table_code"))
    out.append(rec("table", "it", "lo script conta.py raggruppa i clienti per cosa?", code, neg, p, "table_code"))
    return out


def g_temporal(p, rng):
    out = []
    c_old, c_new = rng.sample(CARS, 2)
    ph_old, ph_new = rng.sample(PHONES, 2)
    g_old, g_new = rng.sample(HOBBIES, 2)
    da, db = rng.sample(DAYS, 2)
    pairs = [
        (f"[fact] car: {c_old}", f"[said] ho venduto la {c_old.split()[1]}, adesso guido una {c_new}",
         ["che macchina guido?", "che auto ho adesso?"], ["what car do I drive these days?", "which car do I have now?"]),
        (f"[fact] phone: {ph_old}", f"{p.episode()} The user said they switched to a {ph_new} last week.",
         ["che cellulare uso?", "che telefono ho ora?"], ["what phone do I use?", "which phone have I got now?"]),
        (f"[said] il corso di {g_old[0]} e' il {da[0]}", f"[said] ho spostato il corso di {g_old[0]} al {db[0]}",
         [f"che giorno ho {g_old[0]}?", f"quando c'e' il corso di {g_old[0]}?"], [f"which day is my {g_old[1]} class?", f"when is {g_old[1]} now?"]),
    ]
    for old, new, it, en in pairs:
        neg = [old] + fillers(p, rng, 2)
        out += [rec("temporal", "it", q, new, neg, p, "temporal") for q in it] + [rec("temporal", "en", q, new, neg, p, "temporal") for q in en]
    return out


def g_contra(p, rng):
    out = []
    h1, h2 = rng.sample(range(6, 22), 2)
    d1, d2 = rng.sample(DAYS, 2)
    r = rng.choice(p.rels)
    m = rng.choice(MONTHS)
    pairs = [
        (f"{p.episode()} The user said the train to {rng.choice(CITIES)} leaves at {h1}:15.",
         f"[said] mi ero sbagliato, il treno parte alle {h2}:40", ["a che ora parte il mio treno?"], ["what time does my train leave?"]),
        (f"[said] la cena da {r['it']} e' {d1[0]}", f"[said] correzione: la cena da {r['it']} e' {d2[0]}, non {d1[0]}",
         [f"quando ceno da {r['it']}?"], [f"which day is dinner at {r['en']}'s?"]),
        (f"[fact] {r['key']}_birthday: il compleanno {di(r['it'])} e' il {h1} {m[0]}",
         f"[said] mi sbagliavo, il compleanno {di(r['it'])} e' il {h2} {m[0]}", [f"quando e' il compleanno {di(r['it'])}?"],
         [f"when is {r['en']}'s birthday, really?"]),
    ]
    for old, new, it, en in pairs:
        neg = [old] + fillers(p, rng, 2)
        out += [rec("contradiction", "it", q, new, neg, p, "contra") for q in it] + [rec("contradiction", "en", q, new, neg, p, "contra") for q in en]
    return out


def g_multihop(p, rng):
    out = []
    for r in rng.sample(p.rels, 3):
        df = f"[fact] {r['key']}: {r['it']} si chiama {r['name']}"
        moved = f"[said] {r['name']} si e' trasferito a {r['city']}" if r["it"].startswith(("mio", "il")) else \
            f"[said] {r['name']} si e' trasferita a {r['city']}"
        job = f"[said] {r['name']} fa il lavoro di {r['job'][0]}"
        others = [f"[fact] {x['key']}: {x['it']} si chiama {x['name']}" for x in p.rels if x is not r][:2]
        # first hop: the question names the relation; the record that names the person
        out.append(rec("multihop", "it", f"dove vive adesso {r['it']}?", df, others + fillers(p, rng, 1), p, "hop1"))
        out.append(rec("multihop", "en", f"where does {r['en']} live now?", df, others + fillers(p, rng, 1), p, "hop1"))
        # second step (tools/memhop.py req_names): the question plus the name from the first hop
        out.append(rec("multihop", "it", f"dove vive adesso {r['it']}? {r['name']}", moved, [df, job, others[0]], p, "hop2"))
        out.append(rec("multihop", "en", f"what does {r['en']} do for a living? {r['name']}", job, [df, moved, others[0]], p, "hop2"))
    return out


GEN = {"para": g_para, "exact": g_exact, "pointer": g_pointer, "sibling": g_sibling, "hub": g_hub, "owner": g_owner,
       "raw": g_raw, "table": g_table, "temporal": g_temporal, "contradiction": g_contra, "multihop": g_multihop}
CAP = 400  # per class and split ratio below; shortcut: templates, not model-written paraphrases; upgrade with a
#            model paraphrase pass (on a fleet host) if the fine-tune overfits the template wording


def main(out_dir, seed=7, n_personas=120):
    rng = random.Random(seed)
    ev = CD.EvalSide()
    recs, dropped = [], {}
    for pid in range(n_personas):
        p = Persona(rng, pid)
        for cls, g in GEN.items():
            for x in g(p, rng):
                why = ev.leak(x)
                if why:
                    dropped[why] = dropped.get(why, 0) + 1
                    continue
                recs.append(x)
    seen, uniq = {}, []  # a (question, positive) pair appears at most 3 times, each with other hard negatives
    for x in recs:
        k = (x["q"], x["pos"])
        if seen.get(k, 0) < 3 and len(x["neg"]) == 3 and x["pos"] not in x["neg"] and \
                (k, tuple(x["neg"])) not in seen:
            seen[k] = seen.get(k, 0) + 1
            seen[(k, tuple(x["neg"]))] = 1
            uniq.append(x)
    rng.shuffle(uniq)
    by = {}
    for x in uniq:
        by.setdefault(x["cls"], []).append(x)
    os.makedirs(out_dir, exist_ok=True)
    stats = {"seed": seed, "personas": n_personas, "dropped_as_leak": dropped, "classes": {}}
    with open(f"{out_dir}/train.jsonl", "w") as tr, open(f"{out_dir}/dev.jsonl", "w") as dv:
        for cls in GEN:
            xs = by.get(cls, [])[:CAP]
            # whole personas held out; the shared help sections and raw notes by whole template instead
            held = set(sorted({x["tpl"] for x in xs})[3::7])
            dev = [x for x in xs if (x["tpl"] in held if cls in ("sibling", "raw") else x["persona"] % 10 == 9)]
            for x in xs:
                (dv if x in dev else tr).write(json.dumps(x, ensure_ascii=False) + "\n")
            stats["classes"][cls] = {"train": len(xs) - len(dev), "dev": len(dev),
                                     "it": sum(x["lang"] == "it" for x in xs), "en": sum(x["lang"] == "en" for x in xs),
                                     "templates": len({x["tpl"] for x in xs})}
    for f in ("train.jsonl", "dev.jsonl"):
        stats[f] = {"bytes": os.path.getsize(f"{out_dir}/{f}"),
                    "sha256": hashlib.sha256(open(f"{out_dir}/{f}", "rb").read()).hexdigest()}
    json.dump(stats, open(f"{out_dir}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if "--seed" in a:
        kw["seed"] = int(a.pop(a.index("--seed") + 1)); a.remove("--seed")
    if "--personas" in a:
        kw["n_personas"] = int(a.pop(a.index("--personas") + 1)); a.remove("--personas")
    main(a[0], **kw)
