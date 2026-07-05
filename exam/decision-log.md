# Decision Log — TINAT

> Registro delle decisioni tecniche e organizzative del progetto.
> Regola d'oro del corso: *"Se una decisione è importante, deve essere rintracciabile in un documento o in un artefatto nel repository."*

---

## [2026-07-05] Repo personale e indipendente (non il repo condiviso di classe)

**Contesto:** il progetto d'esame va versionato su GitHub. Esiste un repo di classe condiviso (`2025-27.ML---UFS14`) in cui ogni studente aggiunge una cartella; in alternativa si può usare un repo personale dedicato.
**Alternative considerate:**
- *Repo aziendale originale del tirocinio* — scartato: riservatezza, non è materiale mio da pubblicare.
- *Contribuire al repo condiviso di classe* — scartato: ogni modifica dipende da PR/issue accettati da terzi, rischio di conflitti con le cartelle dei compagni, e lo storico dei commit (che è l'artefatto da mostrare) non è pienamente sotto il mio controllo.
**Decisione:** creare un **repo personale, indipendente e privato** sul mio profilo GitHub, con nome generico **TINAT**.
**Motivazione:** controllo totale sullo storico dei commit, nessuna dipendenza da altri, libertà di sperimentare il flusso DevOps-Agile. La demo avviene comunque in classe dal mio computer; la docente può essere invitata come collaboratrice per rivedere lo storico e dare feedback.
**Conseguenze/rischi:** devo gestire io la disciplina dei commit; il repo non è visibile pubblicamente (privato) fino a invito esplicito.

## [2026-07-05] Repo "specchio" anonimizzato del progetto di tirocinio

**Contesto:** il progetto nasce da un contesto di tirocinio; non posso esporre nome reale, azienda, dati o codice proprietari.
**Alternative considerate:** pubblicare il progetto reale (scartato: riservatezza); non fare il progetto sul tema del tirocinio (scartato: perderei il know-how di dominio acquisito).
**Decisione:** creare una versione "specchio" **anonimizzata**, chiamata solo **TINAT**, che usa **esclusivamente dataset pubblici** (IP102) e codice riscritto da zero.
**Motivazione:** rispetta la riservatezza mantenendo il valore didattico e di dominio.
**Conseguenze/rischi:** prima di ogni commit va verificato che non compaiano riferimenti al nome reale, all'azienda, a URL interni o persone.

## [2026-07-05] Migrazione selettiva, un commit per pezzo

**Contesto:** esiste materiale locale da un progetto precedente; portarlo tutto in blocco sarebbe rischioso (riservatezza) e poco leggibile come processo.
**Alternative considerate:** clone/copia completa del progetto locale (scartato: rischio di introdurre materiale proprietario e storia poco tracciabile).
**Decisione:** migrazione **selettiva**, portando solo materiale mio/pubblico, **un commit dedicato per ogni pezzo** con messaggio descrittivo.
**Motivazione:** controllo su cosa entra nel repo + storico granulare come evidenza di processo.
**Conseguenze/rischi:** processo più lento ma molto più trasparente e verificabile.

## [2026-07-05] YOLOv8 come architettura di detection

**Contesto:** il compito è *object detection* (rilevare e localizzare insetti), non semplice classificazione.
**Alternative considerate:**
- *Classificazione a immagine intera (es. CNN/feature classiche)* — scartato: non fornisce la localizzazione (bounding box).
- *Altri detector (Faster R-CNN, SSD, YOLO precedenti)* — scartati per questa fase: setup più pesante o meno supporto/tooling rispetto a YOLOv8.
**Decisione:** usare **YOLOv8** (Ultralytics) con fine-tuning da pesi pre-addestrati.
**Motivazione:** buon compromesso velocità/accuratezza, tooling maturo, facile fine-tuning, ampia documentazione.
**Conseguenze/rischi:** dipendenza da PyTorch/Ultralytics; training pesante senza GPU.

## [2026-07-05] IP102 (classe Miridae) come base pubblica per il fine-tuning

**Contesto:** servono dati pubblici coerenti con il dominio (insetti parassiti), senza usare dati proprietari.
**Alternative considerate:** raccogliere/etichettare un dataset da zero (scartato: troppo costoso nei tempi dell'esame); usare dati del tirocinio (scartato: riservatezza).
**Decisione:** partire da **IP102**, concentrandosi sulla/e classe/i riconducibili alla famiglia **Miridae**, come base per il transfer verso Helopeltis.
**Motivazione:** dataset pubblico, ampio e citato in letteratura, coerente con il dominio.
**Conseguenze/rischi:** IP102 è un *proxy*; il transfer verso il target reale andrà validato; possibile sbilanciamento delle classi.

## [2026-07-05] Separazione notebooks/ vs src/

**Contesto:** serve sia sperimentazione rapida sia codice riutilizzabile e testabile.
**Alternative considerate:** tenere tutto in notebook (scartato: poco riutilizzabile/testabile); tutto in script (scartato: perde la comodità dell'esplorazione).
**Decisione:** `notebooks/` = laboratorio (codice "sporco" ammesso); `src/` = fabbrica (codice pulito, testato). Il codice che si stabilizza migra da notebooks a src, e la migrazione è essa stessa una decisione da loggare.
**Motivazione:** buona pratica DevOps/ML; separa esplorazione da produzione.
**Conseguenze/rischi:** occorre disciplina per non lasciare logica importante solo nei notebook.
