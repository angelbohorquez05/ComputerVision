# TINAT — Riconoscimento di insetti dannosi (Helopeltis/Miridae)

Sistema di riconoscimento su immagini di insetti dannosi delle colture, con focus sul genere **Helopeltis** (famiglia **Miridae**).

Il progetto è organizzato in due fasi:

- **Fase 1 — classificazione (attuale):** dato un'immagine, prevedere la classe (es. presenza di Helopeltis / mirid bug / foglia sana) con un pipeline multi-modello classico (scikit-learn). Leggero, gira su CPU, usa dati pubblici già disponibili.
- **Fase 2 — detection (futura):** localizzare gli insetti con *bounding box* usando YOLOv8, quando saranno disponibili le foto reali annotate.

## Descrizione

Il monitoraggio manuale degli insetti dannosi sulle colture è lento e soggetto a errore. TINAT automatizza il riconoscimento a partire da un'immagine, così da supportare decisioni agronomiche più tempestive.

Nella Fase 1 il modello viene addestrato e confrontato su più classificatori (Random Forest, SVC, K-Neighbors, Logistic Regression) tramite *cross-validation*, e viene scelto automaticamente il migliore.

## Requisiti e installazione

Requisiti: **Python 3.10+** e `git`. Il progetto gira **su CPU** e non richiede una GPU.

```bash
git clone https://github.com/angelbohorquez05/TINAT.git
cd TINAT

python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Per lanciare i test:

```bash
pytest
```

## Dataset

- **Origine (Fase 1):** dataset pubblici di immagini di foglie di tè che includono le classi di interesse, come **teaLeafBD** e **Tea_Leaf_Disease** (Kaggle), con le classi **Helopeltis** (tea mosquito bug) e **Green mirid bug** (Miridae), oltre a foglia sana e altre condizioni.
- **Perché:** sono pubblici e contengono immagini reali proprio degli insetti target, quindi permettono di iniziare a lavorare su dati reali fin da subito.
- **Qualità dei dati:** le immagini corrotte o non apribili vengono scartate in fase di caricamento (vedi `src/features.py`).
- I dati non sono versionati nel repository (vedi `.gitignore`): vanno scaricati localmente nella cartella `data/`, con una sotto-cartella per classe.

## Struttura

```
TINAT/
├── src/            codice del progetto
│   └── features.py  estrazione feature (istogrammi di colore) e caricamento dataset
├── tests/          test con pytest
├── data/           dataset locale (non versionato)
├── docs/           documentazione e log delle decisioni di progetto
├── requirements.txt
└── pytest.ini
```

## Ciclo di vita del modello

- **Dati:** download del dataset pubblico e organizzazione in `data/` (una cartella per classe).
- **Feature:** estrazione di istogrammi di colore RGB per immagine.
- **Training e selezione:** confronto di più classificatori in cross-validation, scelta automatica del migliore.
- **Validazione:** metriche di classificazione (accuratezza, e in seguito precision/recall per classe).
- **Inferenza:** predizione della classe su nuove immagini.

## MLOps

- **Monitoraggio:** distribuzione delle immagini in ingresso (*drift* stagionale/di dominio), errori segnalati dagli utenti, andamento dell'accuratezza nel tempo.
- **Criteri di re-training:** calo dell'accuratezza sotto soglia, nuovi errori segnalati, disponibilità di nuovi dati etichettati (incluse le foto reali della Fase 2), cambio delle condizioni di ripresa.

## Limiti e note

- La classificazione a livello di immagine **non localizza** l'insetto (niente bounding box): è un passo intermedio verso la detection della Fase 2.
- Gli istogrammi di colore sono feature semplici: robuste e veloci, ma meno potenti di feature apprese (CNN). Scelta voluta per restare leggeri su CPU.
- Le classi dei dataset pubblici possono essere sbilanciate; da valutare *augmentation* e ribilanciamento.

## Decisioni di progetto

Le scelte tecniche e organizzative sono tracciate in [`docs/decision-log.md`](docs/decision-log.md).
