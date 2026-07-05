# TINAT — Rilevamento di insetti (Helopeltis/Miridae) con YOLOv8

Sistema di *object detection* per individuare e localizzare insetti della famiglia **Miridae** (in particolare il genere **Helopeltis**) all'interno di immagini, basato su **YOLOv8**.

## Descrizione

Il monitoraggio manuale degli insetti dannosi sulle colture è lento e soggetto a errore. TINAT automatizza il riconoscimento: data un'immagine, individua la presenza degli insetti target e ne restituisce le *bounding box*, così da supportare decisioni agronomiche più tempestive.

Il modello viene addestrato per *fine-tuning* a partire da pesi YOLOv8 pre-addestrati, usando esclusivamente dati pubblici.

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

## Dataset

- **Origine:** [IP102](https://github.com/xpwu95/IP102), dataset pubblico di *insect pest recognition* (oltre 75.000 immagini, 102 classi di parassiti agricoli).
- **Sottoinsieme usato:** le classi riconducibili alla famiglia **Miridae**, come base per il *transfer* verso *Helopeltis*.
- **Perché:** è pubblico, ampiamente usato in letteratura e coerente con il dominio (parassiti delle colture).
- **Qualità dei dati:** le immagini corrotte o non apribili vengono scartate in fase di caricamento; le annotazioni incoerenti vengono ignorate e registrate a log.

I dati non sono versionati nel repository (vedi `.gitignore`): vanno scaricati localmente nella cartella `data/`.

## Struttura

```
TINAT/
├── src/     codice del progetto (preparazione dati, training, inferenza)
├── tests/   test con pytest
├── data/    dataset locale (non versionato)
├── docs/    documentazione e log delle decisioni di progetto
└── requirements.txt
```

## Ciclo di vita del modello

- **Dati:** download del sottoinsieme IP102 e organizzazione in `data/`.
- **Training:** fine-tuning di YOLOv8 sulle classi target.
- **Validazione:** split train/val e metriche di detection (mAP@0.5, precision, recall).
- **Inferenza:** predizione su singola immagine o cartella, con salvataggio delle immagini annotate.

## MLOps

- **Monitoraggio:** distribuzione delle immagini in ingresso (*drift* stagionale/di dominio), tasso di falsi negativi segnalati, andamento di mAP nel tempo.
- **Criteri di re-training:** calo di mAP sotto soglia, accumulo di nuovi errori segnalati, disponibilità di nuovi dati etichettati, cambio delle condizioni di ripresa (luce/coltura).

## Limiti e note

- Il training su CPU è più lento rispetto a una GPU: per il fine-tuning conviene usare i modelli YOLOv8 leggeri (`n`/`s`) e immagini ridimensionate.
- IP102 è usato come base pubblica: il transfer verso il target reale va validato su dati specifici.
- Le classi possono essere sbilanciate; da valutare *augmentation* e ribilanciamento.

## Decisioni di progetto

Le scelte tecniche e organizzative sono tracciate in [`docs/decision-log.md`](docs/decision-log.md).
