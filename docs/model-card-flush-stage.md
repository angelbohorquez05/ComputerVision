# Model Card — Classificatore di stadio di flush (Workstream B, dimensione WHEN)

**Versione:** 0.1 (2 classi)
**Data:** luglio 2026
**Autore:** Angel Bohorquez
**Parte del progetto:** TINAT

## Panoramica del modello

Classificatore di immagini che stima lo **stadio di flush** di una foglia di tè,
per popolare la dimensione **WHEN** della metrica WHO×WHERE×WHEN. Alimenta
`FlushClassifier` in [`src/fusion.py`](../src/fusion.py); se assente, la demo usa
un fallback esplicito.

- **Architettura:** MobileNetV2 (transfer learning da ImageNet) con testa lineare a 2 classi.
- **Input:** immagine RGB 224×224, normalizzazione ImageNet (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Classi (ordine dei logit):** `new_flush` (germoglio tenero, **alto rischio** Helopeltis), `mature_leaf` (foglia matura, basso rischio).
- **Stadio `dormant`:** **fuori scope** — è la fase con meno attività di parassiti e non esiste un dataset pubblico (vedi decision-log). Le 2 classi sono lo scope definitivo del pilot.

## Metriche di performance

### Risultati sul test set (332 immagini, held-out, seed fisso)

Punto operativo **deployato**: `FlushClassifier` usa una **soglia di decisione di 0.35**
su `new_flush` (invece di argmax = 0.5), scelta sul *validation set* per alzare il
recall della classe ad alto rischio (vedi decision-log).

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `new_flush` | 0.916 | **0.859** | 0.886 | 177 |
| `mature_leaf` | 0.849 | 0.910 | 0.879 | 155 |
| **Accuracy** | | **0.883** | | 332 |

Matrice di confusione (righe = vero `[new_flush, mature_leaf]`): `[[152, 25], [14, 141]]`.

Per riferimento, con **argmax puro** (soglia 0.5) il recall di `new_flush` era 0.791
(accuratezza 0.864): la soglia recupera ~7 punti di recall sulla classe che conta,
a fronte di poca precision persa.

**Interpretazione (con cautela):** il **recall di `new_flush` (0.86)** supera ora
l'obiettivo di 0.85. Resta comunque il tipo di errore più costoso (25 germogli teneri
persi come foglia matura); migliorabile ulteriormente con più dati o re-training con
loss pesata verso `new_flush`.

## Dataset di training

- **Fonte:** TeaLeafAgeQuality (Mendeley Data, licenza CC BY 4.0), 2208 immagini di *Camellia sinensis*.
- **Mappatura età → stadio:** categorie giovani → `new_flush` (1177), categorie mature → `mature_leaf` (1031). Dettagli in [`data/raw/flush_stages/README.md`](../data/raw/flush_stages/README.md).
- **Split:** stratificato 70/15/15 deterministico → train 1545, val 331, test 332.

### Limiti dei dati

- Il dataset è per **età/qualità** della foglia, un *proxy* dello stadio di flush.
- Immagini di dominio pubblico, non foto reali dal campo: possibile *domain gap*.
- Nessuna classe `dormant` (fuori scope: fase a bassissima attività di parassiti, senza dataset pubblico).

## Procedura di training (riproducibilità)

- **Fasi:** testa a backbone congelato (5 epoche, Adam lr 1e-3) → fine-tuning completo (2 epoche, lr 1e-4).
- **Loss:** cross-entropy con pesi di classe bilanciati. Seed 42, CPU.
- **Comando:**
  ```bash
  python -m src.train_flush_stage --epochs-head 5 --epochs-ft 2 --batch 32
  ```

## Artefatti del modello

Non versionati (vedi `.gitignore`); si rigenerano con lo script:

- `models/flush_stage.pt` — pesi PyTorch.
- `models/flush_stage.onnx` — grafo ONNX (1×3×224×224).
- `models/flush_stage.tflite` — modello TFLite (NHWC 1×224×224×3 → 2 logit), convertito con `onnx2tf`.
- `models/flush_stage_metrics.json` — riepilogo delle metriche.

I risultati (report, matrice di confusione, griglia di predizioni) si visualizzano **inline nel notebook** [`notebooks/02_flush_stage_classifier.ipynb`](../notebooks/02_flush_stage_classifier.ipynb), senza salvare immagini.

## Integrazione

`FlushClassifier` carica `models/flush_stage.tflite` se presente e restituisce lo
stadio applicando una **soglia di 0.35** su `new_flush` (`NEW_FLUSH_THRESHOLD`); con
un numero di classi diverso da 2 ricadrebbe su `argmax`. Il preprocessing del wrapper
(normalizzazione ImageNet, NHWC) combacia con quello di training. Verificato con
`python -m src.demo`.

## Limiti noti e prossimi passi

- **Recall `new_flush`**: portato a **0.86** (> obiettivo 0.85) tramite la soglia di decisione; migliorabile ancora con più dati o re-training con loss pesata verso `new_flush`.
- **Validazione su foto reali dal campo** quando disponibili.

**Fuori scope:** una classe `dormant` (fase a bassissima attività di parassiti, senza dataset pubblico — vedi decision-log). Le 2 classi sono lo scope definitivo.

**Ultimo aggiornamento:** luglio 2026
**Stato:** 2 classi (scope definitivo), addestrato e integrato end-to-end; miglioramento del recall come prossimo passo.
