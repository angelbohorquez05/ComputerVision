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
- **Stadio `dormant`:** rimandato — nessun dataset disponibile (vedi decision-log).

## Metriche di performance

### Risultati sul test set (332 immagini, held-out, seed fisso)

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `new_flush` | 0.960 | **0.808** | 0.877 | 177 |
| `mature_leaf` | 0.814 | 0.961 | 0.882 | 155 |
| **Accuracy** | | **0.880** | | 332 |

Matrice di confusione (righe = vero `[new_flush, mature_leaf]`): `[[143, 34], [6, 149]]`.

**Interpretazione (con cautela):** l'accuratezza globale è ~88%, ma il **recall di
`new_flush` (0.81)** è sotto l'obiettivo di 0.85 per la classe ad alto rischio: 34
germogli teneri sono stati classificati come foglia matura (falsi negativi, il tipo
di errore più costoso). Migliorabile pesando di più `new_flush` nella loss o
abbassando la soglia di decisione a inferenza.

## Dataset di training

- **Fonte:** TeaLeafAgeQuality (Mendeley Data, licenza CC BY 4.0), 2208 immagini di *Camellia sinensis*.
- **Mappatura età → stadio:** categorie giovani → `new_flush` (1177), categorie mature → `mature_leaf` (1031). Dettagli in [`data/raw/flush_stages/README.md`](../data/raw/flush_stages/README.md).
- **Split:** stratificato 70/15/15 deterministico → train 1545, val 331, test 332.

### Limiti dei dati

- Il dataset è per **età/qualità** della foglia, un *proxy* dello stadio di flush.
- Immagini di dominio pubblico, non foto reali dal campo: possibile *domain gap*.
- Nessuna classe `dormant`.

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
- `models/flush_stage_predictions.png`, `models/flush_stage_confusion.png` — visualizzazioni dei risultati sul test set (rigenerabili con `python -m src.viz_flush_stage`).

## Integrazione

`FlushClassifier` carica `models/flush_stage.tflite` se presente e restituisce lo
stadio via `argmax`. Il preprocessing del wrapper (normalizzazione ImageNet, NHWC)
combacia con quello di training. Verificato con `python -m src.demo`.

## Limiti noti e prossimi passi

- **Recall `new_flush` da migliorare** (classe ad alto rischio): re-training con loss pesata verso `new_flush` o soglia dedicata.
- **Aggiungere `dormant`** quando esisterà un dataset (iNaturalist), passando a 3 classi (aggiornare anche `FlushClassifier.CLASSES`).
- **Validazione su foto reali dal campo** quando disponibili.

**Ultimo aggiornamento:** luglio 2026
**Stato:** 2 classi, addestrato e integrato end-to-end; `dormant` e miglioramento del recall come prossimi passi.
