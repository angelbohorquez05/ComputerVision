# Decision Log — TINAT

> Registro delle decisioni tecniche e organizzative del progetto.
> Regola d'oro del corso: *"Se una decisione è importante, deve essere rintracciabile in un documento o in un artefatto nel repository."*

> Nota di lettura: le decisioni del **lavoro reale** (architettura, dati, metrica) risalgono a maggio-giugno 2026, quando il progetto è stato sviluppato durante il tirocinio. Le decisioni di **organizzazione del repository e DevOps** (repo "specchio" anonimizzato, CI, flusso Git) sono di luglio 2026, quando il lavoro è stato riportato in questo repository pubblico e anonimizzato. Le date indicano quando la decisione è stata presa, non quando è stata scritta qui.

---

## [2026-07-09] Riscrittura della cronologia per rimuovere riferimenti reali trapelati

**Contesto:** durante la migrazione, alcuni riferimenti reali (nome originale del progetto, colleghi, azienda) erano rimasti in file e messaggi di commit di alcune versioni già pubblicate sul repository pubblico, in violazione della regola di anonimizzazione. L'anonimizzazione era stata fatta sui file, ma quella correzione non era arrivata al remoto (la sua configurazione si era persa) e le versioni vecchie restavano nella cronologia e nelle Pull Request.
**Alternative considerate:**
- *Un nuovo commit che ripulisce solo i file attuali* — scartato: lascia i riferimenti reali visibili nella cronologia e nei diff delle PR.
- *Cancellare e ricreare il repository da una copia pulita* — è l'unica garanzia totale, ma scartato per ora: si perderebbero i metadati del repository; ripiegato su tenerlo privato come rete di sicurezza.
- *Riscrivere la cronologia con `git filter-repo` + force-push* — scelto.
**Decisione:** riscrivere l'intera cronologia con `git filter-repo` (tabella di sostituzioni testuali applicata a tutti i commit), force-push su `main`, cancellare la branch contaminata dal remoto e le branch *head* delle vecchie PR. Backup della cronologia precedente conservato localmente prima dell'operazione.
**Motivazione:** rispettare la riservatezza del tirocinio (regola di anonimizzazione) senza rinunciare allo storico dei commit, che è l'artefatto di processo richiesto dal corso.
**Conseguenze/rischi:** tutti gli hash dei commit cambiano (riscrittura completa). I commit vecchi possono restare raggiungibili tramite i ref delle PR (`refs/pull/*`) finché il repository non viene ricreato: mitigato mantenendo il repository **privato**. Come sempre, ciò che è stato pubblico può essere già stato indicizzato o clonato: la pulizia riduce l'esposizione futura, non cancella il passato.

## [2026-07-09] Rimozione di Docker (semplificazione del repository)

**Contesto:** il `Dockerfile` era stato aggiunto per la riproducibilità della demo (vedi la voce "Containerizzazione con Docker" più sotto). Rivalutando lo scope, il container non viene usato nel flusso di lavoro né nella presentazione: la demo gira già con `python -m src.demo`, e la riproducibilità dell'ambiente è già coperta da `requirements.txt` con versioni pinnate + la CI su Python 3.10.
**Alternative considerate:**
- *Tenere il `Dockerfile` anche se non usato* — scartato: un artefatto DevOps mai eseguito nel processo reale è "copertura di facciata"; il corso premia l'onestà del processo (onestà > copertura finta).
- *Trasformarlo in un'immagine per l'inferenza reale (stack ML completo + pesi)* — scartato: fuori scope adesso; richiederebbe un'immagine multi-GB e i pesi non versionati.
**Decisione:** rimuovere `Dockerfile` e `.dockerignore` dal repository e la relativa sezione dal README. La voce originale qui sotto resta come storico della decisione.
**Motivazione:** mantenere il repo minimale e onesto — solo artefatti realmente usati — senza perdere riproducibilità (garantita da `requirements.txt` pinnato + CI).
**Conseguenze/rischi:** se in futuro servirà un packaging containerizzato (es. deploy dell'inferenza reale), si ripartirà da un'immagine dedicata. La containerizzazione resta comunque documentata nello storico dei commit e in questa voce.

## [2026-07-09] Dataset e mappatura delle classi per il classificatore flush-stage (Workstream B)

**Contesto:** il Workstream B dell'outline richiede un **classificatore di stadio di flush** (dormant / new_flush / mature_leaf), a partire da un backbone mobile (transfer learning). Serviva una fonte pubblica di immagini di foglie di tè etichettate per stadio/età.
**Alternative considerate:**
- *Versione "Annotated" / "Annotated+Augmented" del dataset TeaLeafAgeQuality* — scartata: sono in formato detection (bounding box + labels), mentre il WS-B è **classificazione** (una classe per immagine).
- *Versionare le immagini in git* — scartata: il dataset completo è ~4 GB, incompatibile con un repo leggero.
- *Mappatura a 3 classi (A / B+C / D)* — scartata per ora: sbilanciata e comunque priva di `dormant`.
- *Mantenere le 4 categorie A/B/C/D così come sono* — scartata: sono per **età in giorni**, non allineate alla terminologia di flush dell'outline.
**Decisione:** usare la parte **Raw Data** del dataset **TeaLeafAgeQuality** (Mendeley, CC BY 4.0, ~2208 immagini, organizzato per età), con una mappatura **a 2 classi** (lo step "coarse: active flush vs not" dell'outline §5.2):
- `new_flush` = Categoria A (1-2 gg) + B (2-4 gg) → 1177 immagini
- `mature_leaf` = Categoria C (4-7 gg) + D (7+ gg, "Not for Tea") → 1031 immagini

Le immagini restano **fuori da git** (`.gitignore` su `data/raw/flush_stages/**`); si versionano solo la struttura e il `README.md` con le istruzioni. Lo `.zip` di origine è stato eliminato dopo l'estrazione per non duplicare ~4 GB.
**Motivazione:** fonte pubblica e coerente col dominio; l'organizzazione folder-per-classe è ideale per il transfer learning; le 2 classi sono bilanciate (~53/47%) e corrispondono al primo passo previsto dall'outline. La classe `dormant` è rimandata: non è presente in questo dataset e andrà raccolta da iNaturalist (come già annotato nel README).
**Conseguenze/rischi:** il dataset misura **età/qualità** della foglia, un *proxy* dello stadio botanico di flush; manca `dormant`, quindi il classificatore parte a 2 classi; chi clona il repo deve ri-estrarre le immagini seguendo `data/raw/flush_stages/README.md` (non sono versionate).

## [2026-07-09] Versionare il dataset pubblico e gli artefatti di risultato (non i pesi)

**Contesto:** per l'esame serve mostrare i **risultati** del training (metriche, curve, batch di esempio) e permettere la riproducibilità. Inizialmente `.gitignore` escludeva tutta la cartella `data/` e `models/`; ma il dataset è pubblico e piccolo, e i risultati sono proprio ciò che la docente vuole vedere.
**Alternative considerate:**
- *Non versionare nulla e scaricare tutto via script* — scartato: i risultati non sarebbero visibili nel repo, e per un dataset pubblico da ~5 MB la complessità non si giustifica.
- *Versionare anche i pesi del modello (`*.pt`, `*.onnx`, `*.tflite`)* — scartato: binari pesanti (47 MB i soli `.pt`), rigenerabili dal training, gonfiano git senza valore di processo.
**Decisione:** versionare **il dataset pubblico** (`data/`, ~5 MB, Roboflow CC BY 4.0) e **gli artefatti di risultato** del training (`models/helopeltis_phase1/`: `results.csv`, `args.yaml`, `labels.jpg`, `train_batch*.jpg`, ~1.2 MB). Restano ignorati solo i binari pesanti (`*.pt`/`*.onnx`/`*.tflite`, `runs/`).
**Motivazione:** i risultati diventano ispezionabili e riproducibili direttamente dal repository (evidenza di processo per l'esame), mantenendo git leggero. Dati e immagini sono pubblici (CC BY 4.0), quindi nessun problema di riservatezza.
**Conseguenze/rischi:** se in futuro si aggiungono dataset pesanti (es. il set di flush di Mendeley con migliaia di immagini) va ignorata quella specifica sottocartella. Gli `args.yaml` di Ultralytics contengono path assoluti locali (artefatti autentici del run), non riservati.

## [2026-07-09] Separazione in due livelli: repo anonimizzato + livello privato locale

**Contesto:** il tirocinio prosegue nel tempo e produce anche materiale **riservato** (documenti aziendali, note e conversazioni con il supervisore, versioni con nomi reali). Serviva un unico spazio di lavoro che valga sia per l'esame sia per la continuazione del tirocinio, senza che il materiale riservato finisca mai nel repository condiviso.
**Alternative considerate:**
- *Due repository separati (uno reale privato, uno anonimizzato)* — scartato: duplica il lavoro e disperde la storia; facile disallineare i due.
- *Rendere l'intero repo privato su GitHub e inserirvi i dati reali* — scartato: se poi il repo viene condiviso (es. con la docente) il materiale riservato e i nomi reali sarebbero comunque visibili.
- *Tenere il materiale riservato in una cartella fuori dal progetto* — scartato: si perde l'unità del progetto (l'obiettivo è "un solo progetto").
**Decisione:** un **unico progetto a due livelli**. Il livello pubblico/condivisibile è il repo **TINAT anonimizzato** (ciò che si committa e si pubblica: artefatto dell'esame). Il livello privato è la cartella locale **`_private/`**, inserita in `.gitignore` e quindi **mai versionata né pubblicata**; contiene il materiale reale del tirocinio. Verificato con `git check-ignore` e `git status` che `_private/` non è tracciata.
**Motivazione:** un solo posto in cui lavorare, la storia dei commit resta pulita e anonima (l'artefatto per la docente), e i nomi reali/materiale aziendale non lasciano mai la macchina. Coerente con la regola di anonimizzazione già adottata per il repo.
**Conseguenze/rischi:** la separazione dipende dalla disciplina nel mettere il materiale reale dentro `_private/` e nell'anonimizzare prima di spostare qualcosa verso il repo. Prima di ogni commit resta valido il controllo (`grep`) sui nomi reali. Nota operativa: il repo va tenuto fuori da cartelle sincronizzate (OneDrive) per non rischiare la corruzione di `.git`.

## [2026-07-09] Strategia di branching: feature-branch + Pull Request (GitHub Flow)

**Contesto:** finora i commit sono andati direttamente su `main` (accettabile per l'avvio di un progetto individuale). Ora che esiste una CI che gira sulle Pull Request, conviene adottare un flusso che mostri esplicitamente il ciclo DevOps-Agile: sviluppo su branch, revisione, merge controllato.
**Alternative considerate:**
- *Continuare a committare direttamente su `main`* — scartato d'ora in poi: semplice, ma non mostra il flusso di revisione né sfrutta la CI come gate prima del merge.
- *Git-flow completo (branch `develop`, `release/*`, `hotfix/*`)* — scartato: pensato per team e rilasci versionati, è complessità eccessiva per un progetto individuale d'esame.
**Decisione:** adottare un **GitHub Flow leggero**: `main` sempre "verde" e deployabile; ogni modifica su un branch a vita breve con prefisso di tipo (`feature/`, `fix/`, `docs/`, `ci/`, `build/`); merge su `main` tramite **Pull Request** con CI verde. Aggiunta una `.github/pull_request_template.md` per strutturare ogni PR (cosa cambia, test verdi, anonimizzazione verificata, decision-log aggiornato).
**Motivazione:** rende visibile e tracciabile il processo (il PR è un artefatto che la docente può rivedere), usa la CI come cancello di qualità e resta semplice per un singolo sviluppatore. Questa stessa modifica viene introdotta tramite il flusso che documenta.
**Conseguenze/rischi:** con un solo sviluppatore la revisione è auto-revisione (nessun secondo revisore); è un limite consapevole. I commit iniziali su `main` restano nello storico come avvio onesto del progetto.

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

## [2026-07-04] Approccio a due fasi: classificazione ora, detection dopo

> Nota: superata dalla decisione "Migrazione del detector YOLOv8 reale" (2026-07-06). La classificazione a istogrammi non viene proseguita; resta come tentativo esplorato nello storico dei commit.


**Contesto:** le foto reali annotate degli insetti target non saranno disponibili per un paio di mesi. Serve però iniziare a lavorare su dati reali fin da subito. I dataset pubblici disponibili per Helopeltis/Miridae (es. `teaLeafBD`, `Tea_Leaf_Disease` su Kaggle) sono di **classificazione** (etichetta a livello di immagine: Helopeltis, mirid bug, foglia sana…), **senza bounding box**.
**Alternative considerate:**
- *Partire subito con YOLOv8 (detection)* — scartato per ora: richiede immagini annotate con bounding box, che non ho; su CPU è pesante; i dati pubblici disponibili non sono in formato detection.
- *Aspettare le foto reali per iniziare* — scartato: bloccherebbe il progetto per settimane.
**Decisione:** dividere il progetto in due fasi. **Fase 1 (ora):** classificazione di immagini con un pipeline multi-modello classico (scikit-learn: istogrammi di colore + RandomForest/SVC/KNN/LogisticRegression + cross-validation), riusando l'architettura già sperimentata in un progetto precedente. **Fase 2 (dopo):** object detection con YOLOv8 quando arriveranno le foto reali annotate.
**Motivazione:** consente di lavorare su dati reali e ottenere un pipeline funzionante su CPU fin da subito, con codice che conosco e so difendere; l'evoluzione verso la detection resta pianificata.
**Conseguenze/rischi:** questa scelta **sostituisce** la decisione precedente di partire direttamente da YOLOv8. La classificazione a livello di immagine non localizza l'insetto (niente bounding box): è un passo intermedio, non l'obiettivo finale.

## [2026-07-03] Migrazione selettiva, un commit per pezzo

**Contesto:** esiste materiale locale da un progetto precedente; portarlo tutto in blocco sarebbe rischioso (riservatezza) e poco leggibile come processo.
**Alternative considerate:** clone/copia completa del progetto locale (scartato: rischio di introdurre materiale proprietario e storia poco tracciabile).
**Decisione:** migrazione **selettiva**, portando solo materiale mio/pubblico, **un commit dedicato per ogni pezzo** con messaggio descrittivo.
**Motivazione:** controllo su cosa entra nel repo + storico granulare come evidenza di processo.
**Conseguenze/rischi:** processo più lento ma molto più trasparente e verificabile.

## [2026-07-03] Struttura minima ed essenziale

**Contesto:** una struttura con molte cartelle vuote (`notebooks/`, `models/`, `data/raw`, `data/processed`, `exam/`) rende il repo simile a uno scheletro d'esame più che a un progetto di modello reale.
**Alternative considerate:** mantenere lo scheletro completo "da manuale" con tutte le cartelle predisposte (scartato: cartelle vuote con `.gitkeep` comunicano "impalcatura", non lavoro reale).
**Decisione:** ridurre a `src/`, `tests/`, `data/` e `docs/` (con `docs/decision-log.md`). Cartelle aggiuntive verranno create solo quando conterranno materiale reale.
**Motivazione:** un repo credibile mostra solo ciò che serve; si evita di simulare struttura non ancora usata.
**Conseguenze/rischi:** alcune cartelle standard (es. per i notebook) andranno aggiunte più avanti, al momento del bisogno.

## [2026-07-02] Repo personale e indipendente (non il repo condiviso di classe)

**Contesto:** il progetto d'esame va versionato su GitHub. Esiste un repo di classe condiviso (`2025-27.ML---UFS14`) in cui ogni studente aggiunge una cartella; in alternativa si può usare un repo personale dedicato.
**Alternative considerate:**
- *Repo aziendale originale del tirocinio* — scartato: riservatezza, non è materiale mio da pubblicare.
- *Contribuire al repo condiviso di classe* — scartato: ogni modifica dipende da PR/issue accettati da terzi, rischio di conflitti con le cartelle dei compagni, e lo storico dei commit (che è l'artefatto da mostrare) non è pienamente sotto il mio controllo.
**Decisione:** creare un **repo personale, indipendente e privato** sul mio profilo GitHub, con nome generico **TINAT**.
**Motivazione:** controllo totale sullo storico dei commit, nessuna dipendenza da altri, libertà di sperimentare il flusso DevOps-Agile. La demo avviene comunque in classe dal mio computer; la docente può essere invitata come collaboratrice per rivedere lo storico e dare feedback.
**Conseguenze/rischi:** devo gestire io la disciplina dei commit; il repo non è visibile pubblicamente (privato) fino a invito esplicito.

## [2026-07-02] Repo "specchio" anonimizzato del progetto di tirocinio

**Contesto:** il progetto nasce da un contesto di tirocinio; non posso esporre nome reale, azienda, dati o codice proprietari.
**Alternative considerate:** pubblicare il progetto reale (scartato: riservatezza); non fare il progetto sul tema del tirocinio (scartato: perderei il know-how di dominio acquisito).
**Decisione:** creare una versione "specchio" **anonimizzata**, chiamata solo **TINAT**, che usa **esclusivamente dataset pubblici** (IP102) e codice riscritto da zero.
**Motivazione:** rispetta la riservatezza mantenendo il valore didattico e di dominio.
**Conseguenze/rischi:** prima di ogni commit va verificato che non compaiano riferimenti al nome reale, all'azienda, a URL interni o persone.

## [2026-06-16] IP102 (classe Miridae) come base pubblica per il fine-tuning

**Contesto:** servono dati pubblici coerenti con il dominio (insetti parassiti), senza usare dati proprietari.
**Alternative considerate:** raccogliere/etichettare un dataset da zero (scartato: troppo costoso nei tempi dell'esame); usare dati del tirocinio (scartato: riservatezza).
**Decisione:** partire da **IP102**, concentrandosi sulla/e classe/i riconducibili alla famiglia **Miridae**, come base per il transfer verso Helopeltis.
**Motivazione:** dataset pubblico, ampio e citato in letteratura, coerente con il dominio.
**Conseguenze/rischi:** IP102 è un *proxy*; il transfer verso il target reale andrà validato; possibile sbilanciamento delle classi.

## [2026-06-10] YOLOv8 come architettura di detection

> Nota: rivista dalla decisione "Approccio a due fasi" (vedi sopra). YOLOv8 resta la scelta per la **Fase 2** (detection su foto reali annotate), non per la Fase 1.

**Contesto:** il compito è *object detection* (rilevare e localizzare insetti), non semplice classificazione.
**Alternative considerate:**
- *Classificazione a immagine intera (es. CNN/feature classiche)* — scartato: non fornisce la localizzazione (bounding box).
- *Altri detector (Faster R-CNN, SSD, YOLO precedenti)* — scartati per questa fase: setup più pesante o meno supporto/tooling rispetto a YOLOv8.
**Decisione:** usare **YOLOv8** (Ultralytics) con fine-tuning da pesi pre-addestrati.
**Motivazione:** buon compromesso velocità/accuratezza, tooling maturo, facile fine-tuning, ampia documentazione. La scelta è stata confermata con il supervisore del tirocinio.
**Conseguenze/rischi:** dipendenza da PyTorch/Ultralytics; training pesante senza GPU.

## [2026-05-20] Avvio del progetto e studio della documentazione

**Contesto:** inizio del tirocinio. Prima di scrivere codice serve capire il dominio (coltura del tè, insetti dannosi del genere Helopeltis / famiglia Miridae) e la documentazione del progetto esistente.
**Alternative considerate:** iniziare subito a programmare (scartato: senza aver compreso dominio e requisiti si rischia di costruire la cosa sbagliata).
**Decisione:** dedicare le prime settimane allo studio della documentazione e del dominio prima di prendere decisioni tecniche.
**Motivazione:** decisioni tecniche informate richiedono prima la comprensione del problema; è la base del ciclo di vita del progetto.
**Conseguenze/rischi:** avvio più lento sulla parte di codice, ma fondamenta più solide per le scelte successive.
