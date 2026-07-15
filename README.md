# TINAT — Rilevamento di Helopeltis (Miridae) con YOLOv8

[![CI](https://github.com/angelbohorquez05/TINAT/actions/workflows/ci.yml/badge.svg)](https://github.com/angelbohorquez05/TINAT/actions/workflows/ci.yml)

Sistema di *object detection* per individuare e localizzare insetti dannosi del genere **Helopeltis** (famiglia **Miridae**) su colture di tè, con una metrica ecologica che arricchisce ogni rilevamento.

> Il progetto nasce in un contesto di **tirocinio** ed è qui pubblicato in versione **anonimizzata** (nessun riferimento a nome reale, azienda, persone o luoghi), usando esclusivamente **dati pubblici**. L'obiettivo del repo è mostrare il processo di sviluppo (organizzazione, decisioni, ciclo di vita ML), non un modello di produzione.

## Descrizione

Il monitoraggio manuale degli insetti dannosi sul tè è lento e soggetto a errore. TINAT automatizza il riconoscimento a partire da un'immagine e, oltre a rilevare l'insetto, ne qualifica l'**interazione ecologica** con la pianta tramite una metrica a tre dimensioni:

| Dimensione | Domanda | Esempio |
|-----------|---------|---------|
| **WHO** | Quale insetto, con quale ruolo? | *Helopeltis* spp. — parassita |
| **WHERE** | In quale zona della pianta? | Zona di efficienza (foglie/steli) |
| **WHEN** | In quale stadio della coltura? | `new_flush` — germogli teneri |

Un parassita su un germoglio tenero è un evento molto diverso dallo stesso parassita su una foglia matura: questa struttura trasforma le rilevazioni grezze in dati azionabili. La logica è in [`src/who_where_when.py`](src/who_where_when.py).

## Requisiti e installazione

Requisiti: **Python 3.10+** e `git`. Tutto gira **su CPU** (nessuna GPU richiesta).

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

- **Origine:** dataset pubblico su **Roboflow Universe** (licenza **CC BY 4.0**), con immagini di *Helopeltis* raccolte da fonti pubbliche (GBIF, iNaturalist, letteratura sui parassiti del tè).
- **Composizione:** ~483 immagini annotate con *bounding box* (split 80/10/10 train/val/test), classe singola `Helopeltis spp.`.
- **Perché:** è pubblico, annotato in formato detection e coerente con il target, quindi permette di lavorare fin da subito senza dati proprietari.
- **Qualità/limiti dei dati:** immagini da internet (illuminazione, sfondi e angolazioni vari), soprattutto insetti adulti su foglie. Le immagini reali dal campo non sono ancora disponibili (previste tra qualche mese): c'è quindi un *domain gap* da validare.
- **Versionamento:** essendo un dataset pubblico (CC BY 4.0) e leggero (~5 MB), è versionato nel repository sotto `data/`, così i risultati sono riproducibili e ispezionabili. Restano fuori da git solo i binari pesanti dei modelli (pesi `*.pt`, export `*.onnx`/`*.tflite`), rigenerabili dal training (vedi `.gitignore`).

## Struttura

```
TINAT/
├── src/                  codice del progetto
│   ├── who_where_when.py  metrica ecologica WHO×WHERE×WHEN
│   ├── fusion.py          motore di fusione (detector + zona + stadio → Observation)
│   └── demo.py            demo end-to-end su dati sintetici
├── tests/                test con pytest
├── notebooks/            fine-tuning del detector (01) e classificatore di stadio (02)
├── data/                 dataset pubblico versionato (~5 MB)
├── models/               artefatti dei risultati del training (metriche, plot); pesi esclusi
├── docs/                 documentazione, model card e log delle decisioni
├── requirements.txt
└── pytest.ini
```

## Il modello

- **Architettura:** YOLOv8-n (Ultralytics), pre-addestrato su COCO e poi fine-tuned sul dataset pubblico di Helopeltis.
- **Input/Output:** immagine RGB → *bounding box* + confidenza.
- **Export:** ONNX e TFLite (deploy offline su mobile).
- **Risultati (test set, 49 immagini da internet):** mAP@0.5 ≈ 0.995, precision ≈ 0.99, recall = 1.0. *Da leggere con cautela:* test set piccolo e stesso dominio "internet" del training; il vero banco di prova saranno le foto reali dal campo.

Dettagli completi (metriche, training, limiti, considerazioni etiche) nella [model card](docs/model-card.md).

## Ciclo di vita del modello

- **Dati:** download del dataset pubblico (Roboflow) e organizzazione in `data/`.
- **Training:** fine-tuning di YOLOv8-n su CPU (mantiene la classe target Helopeltis).
- **Validazione:** metriche di detection (mAP@0.5, mAP@0.5:0.95, precision, recall per classe).
- **Export/Deploy:** conversione in ONNX/TFLite per l'inferenza su CPU/mobile.
- **Monitoring:** vedi MLOps.

## MLOps

- **Cosa monitorare:** distribuzione delle immagini in ingresso (*domain/seasonal drift*), tasso di falsi negativi segnalati, andamento di mAP nel tempo.
- **Criteri di re-training:** calo di mAP sotto soglia; nuovi falsi negativi/positivi; arrivo delle foto reali dal campo; cambio delle condizioni di ripresa.
- **Coda "Needs Review":** le rilevazioni a bassa confidenza (< 0.65) vengono instradate per revisione e futuro ri-addestramento.

## Roadmap

- **Terza classe `dormant`** per il classificatore di stadio (oggi a 2 classi: `new_flush` / `mature_leaf`); richiede un dataset dedicato (es. iNaturalist).
- **Livello di fusione:** combina detection, zona della pianta e stadio in un unico record `Observation`, con aggregazione temporale su più frame.
- **Validazione su foto reali dal campo** quando disponibili → fine-tuning con dati di dominio.

## Rischi, assunzioni e limiti

**Assunzioni**

- Le immagini pubbliche da internet sono un *proxy* accettabile per iniziare, in attesa delle foto reali dal campo (previste tra qualche mese).
- Il classificatore di stadio (WHEN) copre 2 classi (`new_flush` / `mature_leaf`); lo stadio `dormant` non è ancora classificabile. Se il modello non è presente, la demo usa un fallback esplicito (`new_flush`).
- Il deploy di destinazione è offline su CPU/mobile (da cui l'export ONNX/TFLite).

**Rischi**

- **Domain gap:** il modello è addestrato su foto da internet, non da campo reale: le metriche alte non garantiscono le stesse prestazioni sul campo.
- **Classe singola:** rileva Helopeltis, ma non lo distingue ancora da altri Miridae simili.
- **Frame singolo:** l'inferenza è per-immagine; la stabilità su frame sfocati/parziali dipende dall'aggregazione temporale (roadmap).

**Stato onesto del progetto**

- **Funziona end-to-end:** la logica di fusione e la metrica WHO×WHERE×WHEN (`Observation` → export JSON). Verificata da `pytest` (13 test) e dalla CI a ogni commit; eseguibile con `python -m src.demo`.
- **Inferenza reale del detector (WHO) integrata:** `HelopeltisDetector` esegue il modello YOLOv8-n via ONNX Runtime su CPU (coordinate normalizzate + *non-max suppression*). La `demo_real` di [`src/demo.py`](src/demo.py) gira sulle immagini reali del test set (`data/raw/helopeltis/test/`): detector reale → classificatore di stadio reale → `Observation`. Se il file ONNX (binario pesante, fuori da git) o `onnxruntime` mancano, la demo salta questo passo senza errori.
- **Classificatore di stadio (WHEN) addestrato e integrato:** MobileNetV2 a 2 classi (`new_flush` / `mature_leaf`), accuratezza ~88% sul test set, con recall di `new_flush` (classe ad alto rischio) portato a **0.86** grazie a una soglia di decisione dedicata. Caricato da `FlushClassifier` quando `models/flush_stage.tflite` è presente; codice riproducibile in [`src/train_flush_stage.py`](src/train_flush_stage.py). La terza classe `dormant` è rimandata (manca il dataset).
- **Non ancora nel repo:** il segmentatore di zona della pianta (WHERE) — oggi la `demo_real` usa una zona `Efficiency` a frame intero come *placeholder* esplicito. È il prossimo passo per chiudere il ciclo su dati reali.

## Ulteriori informazioni

- **Decisioni di progetto:** tracciate in [`docs/decision-log.md`](docs/decision-log.md) (contesto, alternative, motivazione di ogni scelta).
- **Model card:** metriche, training, limiti e considerazioni etiche del detector in [`docs/model-card.md`](docs/model-card.md) e del classificatore di stadio in [`docs/model-card-flush-stage.md`](docs/model-card-flush-stage.md).
- **Notebook:** fine-tuning del detector e classificatore di stadio in [`notebooks/`](notebooks/).
- **CI:** stato dei test a ogni commit nella tab *Actions* del repository (badge in cima a questo README).
