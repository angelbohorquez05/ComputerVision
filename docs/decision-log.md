# Decision Log — TINAT

> Registro delle decisioni tecniche e organizzative del progetto.
> Regola d'oro del corso: *"Se una decisione è importante, deve essere rintracciabile in un documento o in un artefatto nel repository."*

---

## [2026-07-05] Approccio a due fasi: classificazione ora, detection dopo

**Contesto:** le foto reali annotate degli insetti target non saranno disponibili per un paio di mesi. Serve però iniziare a lavorare su dati reali fin da subito. I dataset pubblici disponibili per Helopeltis/Miridae (es. `teaLeafBD`, `Tea_Leaf_Disease` su Kaggle) sono di **classificazione** (etichetta a livello di immagine: Helopeltis, mirid bug, foglia sana…), **senza bounding box**.
**Alternative considerate:**
- *Partire subito con YOLOv8 (detection)* — scartato per ora: richiede immagini annotate con bounding box, che non ho; su CPU è pesante; i dati pubblici disponibili non sono in formato detection.
- *Aspettare le foto reali per iniziare* — scartato: bloccherebbe il progetto per settimane.
**Decisione:** dividere il progetto in due fasi. **Fase 1 (ora):** classificazione di immagini con un pipeline multi-modello classico (scikit-learn: istogrammi di colore + RandomForest/SVC/KNN/LogisticRegression + cross-validation), riusando l'architettura già sperimentata in un progetto precedente. **Fase 2 (dopo):** object detection con YOLOv8 quando arriveranno le foto reali annotate.
**Motivazione:** consente di lavorare su dati reali e ottenere un pipeline funzionante su CPU fin da subito, con codice che conosco e so difendere; l'evoluzione verso la detection resta pianificata.
**Conseguenze/rischi:** questa scelta **sostituisce** la decisione precedente di partire direttamente da YOLOv8. La classificazione a livello di immagine non localizza l'insetto (niente bounding box): è un passo intermedio, non l'obiettivo finale.

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

> Nota: rivista dalla decisione "Approccio a due fasi" (vedi sopra). YOLOv8 resta la scelta per la **Fase 2** (detection su foto reali annotate), non per la Fase 1.

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

## [2026-07-05] Struttura minima ed essenziale

**Contesto:** una struttura con molte cartelle vuote (`notebooks/`, `models/`, `data/raw`, `data/processed`, `exam/`) rende il repo simile a uno scheletro d'esame più che a un progetto di modello reale.
**Alternative considerate:** mantenere lo scheletro completo "da manuale" con tutte le cartelle predisposte (scartato: cartelle vuote con `.gitkeep` comunicano "impalcatura", non lavoro reale).
**Decisione:** ridurre a `src/`, `tests/`, `data/` e `docs/` (con `docs/decision-log.md`). Cartelle aggiuntive verranno create solo quando conterranno materiale reale.
**Motivazione:** un repo credibile mostra solo ciò che serve; si evita di simulare struttura non ancora usata.
**Conseguenze/rischi:** alcune cartelle standard (es. per i notebook) andranno aggiunte più avanti, al momento del bisogno.
