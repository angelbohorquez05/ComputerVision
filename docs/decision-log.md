# Decision Log — TINAT

> Registro delle decisioni tecniche e organizzative del progetto.
> Regola d'oro del corso: *"Se una decisione è importante, deve essere rintracciabile in un documento o in un artefatto nel repository."*

---

## [2026-07-09] Containerizzazione con Docker (immagine slim per la demo)

**Contesto:** in un corso DevOps la riproducibilità dell'ambiente è centrale: il progetto deve poter girare allo stesso modo su qualsiasi macchina, senza dipendere dal setup locale. Mancava un artefatto di packaging.
**Alternative considerate:**
- *Nessun Docker, solo istruzioni di setup nel README* — scartato: il README aiuta ma non garantisce un ambiente identico e riproducibile.
- *Immagine Docker che installa tutto `requirements.txt` (torch, tensorflow, ultralytics)* — scartato: immagine multi-GB, build lenta; la demo e la logica di fusione usano solo la libreria standard di Python, quindi lo stack ML completo non serve per eseguirla.
- *Docker Compose multi-servizio* — scartato: non c'è ancora un secondo servizio (database, API); sarebbe complessità non giustificata.
**Decisione:** aggiungere un `Dockerfile` basato su `python:3.10-slim` che copia il progetto ed esegue la demo end-to-end (`python -m src.demo`) come comando predefinito, più un `.dockerignore` per mantenere l'immagine piccola (esclude `.git`, notebook, docs, dataset, ambienti).
**Motivazione:** fornisce un pacchetto riproducibile e verificabile (`docker build` + `docker run` mostrano la pipeline che gira) restando leggero e coerente con la scelta fatta per la CI. Lo stesso Python 3.10 della CI e del README.
**Conseguenze/rischi:** l'immagine esegue la demo/logica pura, non il training né l'inferenza reale con i pesi YOLOv8 (che richiederebbero lo stack ML pesante e i pesi del modello, non versionati). È un limite consapevole e documentato: per l'inferenza reale servirà un'immagine dedicata che installi `requirements.txt`.

## [2026-07-09] Integrazione continua (CI) con GitHub Actions

**Contesto:** il corso è DevOps-Agile e il repository ha già una suite di test (`pytest`, 10 casi), ma nulla li esegue in automatico: i test vengono lanciati solo a mano. Manca l'artefatto DevOps più tipico, la *continuous integration*.
**Alternative considerate:**
- *Nessuna CI, test solo a mano* — scartato: niente evidenza automatica di processo, il valore chiave per l'esame.
- *CI che installa tutto `requirements.txt` (torch, tensorflow, ultralytics)* — scartato: l'installazione è pesante, lenta e fragile, e i moduli coperti dai test (`who_where_when.py`, `fusion.py`) usano solo la libreria standard di Python; installare lo stack ML completo non serve per eseguire i test.
- *GitLab CI / altri runner* — scartato: il repo è su GitHub, GitHub Actions è integrato e senza configurazione esterna.
**Decisione:** aggiungere un workflow GitHub Actions (`.github/workflows/ci.yml`) che, a ogni push e pull request su `main`, su Python 3.10 (versione minima dichiarata) installa solo `pytest` ed esegue la suite di test.
**Motivazione:** fornisce evidenza automatica e tracciabile che il codice resta sano a ogni commit (cuore della CI), restando semplice, veloce e affidabile.
**Conseguenze/rischi:** la CI verifica la logica pura (metrica e fusione), non il training né le dipendenze ML pesanti; questo è un limite consapevole e documentato. Se in futuro i test copriranno codice che usa lo stack ML, il workflow andrà esteso per installare le dipendenze necessarie.

## [2026-07-06] Migrazione del detector YOLOv8 reale (anonimizzato)

**Contesto:** durante il lavoro era emerso il dubbio se partire da zero con una classificazione "tampone" (scikit-learn) in attesa dei dati. Rivedendo il materiale del tirocinio è emerso che **esisteva già un detector YOLOv8 di Helopeltis funzionante** (fine-tuning completato, metriche alte, export ONNX/TFLite, una metrica ecologica WHO×WHERE×WHEN e una capa di fusione). Il pipeline di classificazione con istogrammi era quindi un passo intermedio non necessario.
**Alternative considerate:**
- *Proseguire con la classificazione scikit-learn (istogrammi)* — scartato: non è il lavoro reale e non localizza l'insetto.
- *Riscrivere lo storico per nascondere il tentativo di classificazione* — scartato: quel tentativo è iterazione autentica (esplorato, poi corretto); lo si tiene come parte onesta del processo.
**Decisione:** migrare in TINAT, **in modo selettivo e anonimizzato**, il progetto reale di detection: codice proprio (`who_where_when.py`, `fusion.py`, `demo.py`), model card e dataset pubblico (Roboflow, CC BY 4.0). Rimosso il codice di classificazione abbandonato (`features.py`). Ripristinato lo stack YOLOv8.
**Motivazione:** il repo deve riflettere il lavoro reale, non un surrogato; la migrazione anonimizzata rispetta la riservatezza (nessun nome reale, azienda, persone, luoghi o infrastrutture).
**Conseguenze/rischi:** ogni file migrato va ripulito dai riferimenti sensibili (nome reale del progetto, colleghi, luogo, backend) e verificato con `grep` prima del commit. Questa decisione **sostituisce** quella dell'approccio a due fasi qui sotto.

## [2026-07-05] Approccio a due fasi: classificazione ora, detection dopo

> Nota: superata dalla decisione "Migrazione del detector YOLOv8 reale" (2026-07-06). La classificazione a istogrammi non viene proseguita; resta come tentativo esplorato nello storico dei commit.


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
