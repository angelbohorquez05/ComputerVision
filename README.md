# TINAT — Rilevamento di insetti (Helopeltis/Miridae) con YOLOv8

> Progetto d'esame — UFS14 *Processo e Sviluppo del Software* (DevOps-Agile), ITS Angelo Rizzoli.
> Lavoro individuale. Documentazione in italiano, codice in inglese.

---

## 1. Project name / idea

**TINAT** è un sistema di *object detection* per individuare insetti della famiglia **Miridae** (in particolare il genere **Helopeltis**) all'interno di immagini, usando **YOLOv8**.

È la versione "specchio" e **anonimizzata** di un progetto nato in contesto di tirocinio: qui vengono usati **solo dataset pubblici** e codice riscritto da zero, senza alcun dato, peso o riferimento proprietario. Lo scopo del repo è mostrare il **processo di sviluppo** (organizzazione, decisioni tracciate, ciclo di vita ML), non ottenere un modello perfetto.

## 2. Setup / How to run this project

> Scritto per chi non ha mai visto il progetto. Requisiti minimi: **Python 3.10+** e `git`.

```bash
# 1. Clonare il repository
git clone <url-del-repo-personale>
cd TINAT

# 2. Creare e attivare un ambiente virtuale
python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

# 3. Installare le dipendenze
pip install -r requirements.txt

# 4. (Quando disponibili) lanciare i test
pytest tests/
```

**Requisiti di sistema:** una GPU NVIDIA con CUDA velocizza molto il training di YOLOv8, ma non è obbligatoria (il training può girare su CPU, più lentamente). Le librerie esterne principali sono elencate in [`requirements.txt`](requirements.txt) con versioni pinnate.

> **Stato attuale:** il repository è in fase di *scaffolding*. La struttura, la documentazione e il tracciamento delle decisioni sono impostati; il codice di training/inferenza e i dati verranno aggiunti in modo incrementale (un commit per pezzo). Vedi la sezione *Rischi, assunzioni e limiti*.

## 3. Spiegazione del progetto

**Obiettivo:** dato un insieme di immagini, rilevare e localizzare (bounding box) la presenza di insetti target.

**Problema che risolve:** il monitoraggio manuale degli insetti su colture è lento e soggetto a errore umano. Un rilevatore automatico può segnalare precocemente la presenza dei parassiti, supportando decisioni agronomiche più tempestive.

## 4. Dati

- **Origine:** [IP102](https://github.com/xpwu95/IP102) — dataset pubblico di *insect pest recognition* (>75.000 immagini, 102 classi). Da qui si parte dalla/e classe/i riconducibili alla famiglia **Miridae** come base per il fine-tuning verso Helopeltis.
- **Perché quello:** è pubblico, ampiamente citato in letteratura, e copre parassiti agricoli — coerente con il dominio senza usare dati proprietari.
- **Caratteristiche:** immagini reali eterogenee (illuminazione, sfondo, scala variabili) — utile per testare la robustezza.
- **Gestione dati mancanti/malformattati:** immagini corrotte o non apribili vengono scartate in fase di caricamento; le annotazioni mancanti/incoerenti vengono loggate e non usate per il training. *(Dettaglio implementativo da completare nello script di preparazione dati.)*

## 5. Ciclo di vita ML

- **Raccolta dati:** download selettivo di IP102 (solo classi rilevanti), organizzazione in `data/raw` → `data/processed`.
- **Training:** fine-tuning di un modello YOLOv8 pre-addestrato sulle classi target.
- **Validazione:** split train/val, metriche di detection (mAP@0.5, precision, recall) su un set di validazione tenuto separato.
- **Deploy:** script di inferenza su singola immagine o cartella, con salvataggio delle immagini annotate.
- **Monitoring:** vedi sezione MLOps.

## 6. MLOps

- **Cosa monitorare:** distribuzione delle immagini in ingresso (drift stagionale/di dominio), tasso di falsi negativi segnalati dagli utenti, andamento di mAP sul set di validazione nel tempo.
- **Criteri per il re-training:**
  - calo di mAP sotto una soglia concordata;
  - accumulo di nuovi falsi negativi/positivi segnalati;
  - **drift stagionale** delle immagini (condizioni di luce/coltura diverse);
  - disponibilità di nuovi dati etichettati.

## 7. Rischi, assunzioni e limiti

- **Stato onesto:** il progetto **non è ancora funzionante end-to-end**. Al momento è impostato lo scheletro (struttura, docs, decision-log, dipendenze). Codice di training/inferenza, dati e modelli sono da aggiungere.
- **Assunzione:** IP102 (classe Miridae) è una base sufficiente come *proxy* pubblico per Helopeltis; il transfer verso il target reale andrà validato.
- **Limiti:** risorse di calcolo limitate (training su CPU lento); dataset target reale non disponibile per riservatezza; possibile sbilanciamento delle classi.
- **Come ampliarlo:** aggiungere dati etichettati specifici, provare augmentation, confrontare varianti di YOLOv8 (n/s/m), aggiungere un piccolo servizio di inferenza.

## 8. Ulteriori informazioni

- Le decisioni tecniche e organizzative sono tracciate in [`exam/decision-log.md`](exam/decision-log.md).
- Il contesto per l'assistente di sviluppo è in [`CLAUDE.md`](CLAUDE.md).
