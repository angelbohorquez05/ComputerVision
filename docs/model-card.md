# Model Card — Detector di Helopeltis (Fase 1)

**Versione:** 1.0
**Data:** giugno 2026
**Autore:** Angel Bohorquez
**Parte del progetto:** TINAT

> Documento anonimizzato: nessun riferimento a nome reale del progetto, azienda, persone, luoghi o infrastrutture. Dati di training esclusivamente pubblici.

---

## Panoramica del modello

| Attributo | Valore |
|-----------|--------|
| **Tipo** | Object detection (YOLO) |
| **Framework** | YOLOv8-n (Ultralytics) |
| **Input** | 320 × 320 (training a 640 × 640) |
| **Output** | Bounding box + punteggi di confidenza |
| **Formati** | ONNX (~12 MB), TFLite (~12 MB) |
| **Insetto target** | *Helopeltis* spp. (parassita) |
| **Coltura target** | Tè |

---

## Metriche di performance

### Risultati sul test set (49 immagini)

| Metrica | Valore | Target |
|---------|--------|--------|
| **mAP@0.5** | **0.995** | ≥ 0.70 |
| **Precision** | **0.992** | ≥ 0.90 |
| **Recall** | **1.000** | ≥ 0.75 |
| **mAP@0.5:0.95** | 0.789 | — |

**Interpretazione (con cautela):**
- Sul test set il modello non ha mancato alcun insetto (recall = 1.0) — importante per il monitoraggio dei parassiti.
- Precision alta (0.992): pochi falsi allarmi.
- **Attenzione:** il test set è piccolo (49 immagini) e appartiene allo stesso dominio "internet" del training. Le metriche NON garantiscono le stesse prestazioni su foto reali dal campo.

---

## Dataset di training

- **Origine:** dataset pubblico su **Roboflow Universe**, licenza **CC BY 4.0**. Immagini raccolte da fonti pubbliche (GBIF, iNaturalist, letteratura sui parassiti del tè).
- **Composizione:** ~483 immagini totali (train 386 / val 48 / test 49), classe singola `Helopeltis spp.`, annotazioni in formato YOLO (bounding box).
- **Caratteristiche:** condizioni varie (illuminazione, angolazioni, sfondi), soprattutto insetti adulti su foglie.

### Limiti dei dati
- **Nessuna foto dal campo:** tutte le immagini provengono da internet, non da coltivazioni reali.
- **Diversità limitata:** prevalgono insetti adulti su sfondo fogliare.
- **Domain gap:** servirà un ri-addestramento con foto reali dal campo per il deploy in produzione.

---

## Procedura di training

- **Base model:** YOLOv8-n pre-addestrato su COCO (~3.0M parametri, leggero e adatto a CPU).
- **Iperparametri:** 50 epoche (early stopping @38), batch 16, learning rate 1e-3, augmentation (mosaic 1.0, HSV shift, rotazione ±15°), optimizer SGD.
- **Hardware:** CPU di un laptop (nessuna GPU). Durata ~3–4 ore.
- **Versioni:** PyTorch (CPU), Ultralytics 8.x, ONNX 1.x.

### Comando di training (riproducibilità)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="data/helopeltis/helopeltis.yaml",
    epochs=50, imgsz=640, batch=16, lr0=1e-3,
    patience=10, degrees=15, hsv_v=0.5, mosaic=1.0,
    project="models", name="helopeltis_phase1", exist_ok=True,
)
```

### Dataset YAML
```yaml
path: data/helopeltis
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  - Helopeltis spp.
```

---

## Artefatti del modello

I pesi sono file pesanti e **non sono versionati** nel repository (vedi `.gitignore`):

```
models/helopeltis_phase1/
├── weights/
│   ├── best.pt          # checkpoint PyTorch
│   ├── best.onnx        # export ONNX (usato da fusion.py)
│   └── best.tflite      # export TFLite (deploy mobile)
```

Caricamento (ONNX, usato dal livello di fusione):
```python
import onnxruntime as ort
session = ort.InferenceSession("models/helopeltis_phase1.onnx",
                               providers=["CPUExecutionProvider"])
```

---

## Integrazione

Il detector è incapsulato nella classe `HelopeltisDetector` in [`src/fusion.py`](../src/fusion.py):

```python
from src.fusion import HelopeltisDetector
det = HelopeltisDetector()
insects = det.detect(image_rgb, imgsz=320)   # -> list[InsectDetection]
```

Le rilevazioni vengono poi combinate dal `FusionEngine` con la posizione (centro del box) e lo stadio di flush per produrre `Observation` (metrica WHO×WHERE×WHEN); il verdetto spaziale periferia/nucleo è calcolato da `spatial.py`. Le rilevazioni a bassa confidenza (< 0.65) vanno nella coda "Needs Review".

---

## Limiti noti e prossimi passi

### Limiti
1. **Domain gap:** addestrato su foto da internet, non da campo reale.
2. **Classe singola:** rileva Helopeltis, ma non lo distingue da altri Miridae simili.
3. **Frame singolo:** l'inferenza è per-immagine; la stabilità dipende dall'aggregazione temporale.

### Prossimi passi
- [ ] Validare su foto reali dal campo (quando disponibili) → fine-tuning di dominio.
- [ ] Misurare la confusione con altri Miridae.
- [ ] Estendere a più classi e integrare una baseline multi-classe (IP102) quando disponibile.

---

## Considerazioni etiche

- **Uso previsto:** assistere nel rilevamento di focolai di Helopeltis sul tè, per un uso mirato dei trattamenti.
- **Uso NON previsto:** sorveglianza di massa o tracciamento oltre il contesto agricolo.
- **Dati:** immagini di training pubbliche; eventuali output (GPS + rilevazioni) memorizzati in un database con consenso dell'utente.
- **Bias:** le foto da internet sovra-rappresentano alcune aree geografiche; da validare su varietà e condizioni locali.

---

**Ultimo aggiornamento:** giugno 2026
**Stato:** Fase 1 completata (baseline su dati pubblici) — validazione su campo reale ancora da fare.
