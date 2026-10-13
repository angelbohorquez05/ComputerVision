# Model Card — Helopeltis Damage Detector


## Model overview

| Attribute | Value |
|-----------|-------|
| **Type** | Object detection (YOLO) |
| **Framework** | YOLOv8-n (Ultralytics) |
| **Input** | 320 × 320 (trained at 640 × 640) |
| **Output** | Bounding boxes + confidence scores |
| **Formats** | ONNX (~12 MB) |
| **What it detects** | **Feeding damage (marks)** of *Helopeltis* spp. on leaves — **not** the insect |
| **Target crop** | Tea |

> **Important — what it really detects.** This model **does not detect the
> insect** Helopeltis, but the **feeding damage** (lesions on the leaves) the
> pest leaves behind. The dataset's annotations enclose damaged tea leaves,
> not the insect. This is a choice consistent with field reality: the pest is
> tiny (~6–8 mm) and mobile, while the damage is visible and persistent on the
> leaf.

---

## Performance metrics

### Results on the mixed test set (123 images: 49 positive + 74 background)

Current model (third training run) — trained and measured on a **capture-session split**: no test photo is
within 120 seconds, or 10 sequence positions, of any training photo.

| Metric | Value | Target |
|--------|-------|--------|
| **AP@0.5** | **1.000** | ≥ 0.70 |
| **AP@0.5:0.95** | **0.886** | — |
| **Precision** | **1.000** | ≥ 0.90 |
| **Recall** | **1.000** | ≥ 0.75 |
| **Specificity** | **1.000** (0/74 backgrounds fired) | ≥ 0.90 |
| *Constant-box floor (AP@0.5)* | *0.514* | — |

Measured by `python -m src.evaluate detector` on the integrated ONNX path
(→ `models/detector_eval.json`). The 74 backgrounds are held-out iNaturalist
images — other insects, spiders, butterflies, and 22 insect-free tea/foliage
photos — with no overlap against the training negatives.

**Read the floor before the headline.** The ground-truth boxes cover ~40% of the
image and the subject is centred, so a **constant centred box — no model —
scores 0.514 AP@0.5** on this set (it scored 0.755 on the old split). AP@0.5 is
not a metric whose floor is zero. **AP@0.5:0.95 = 0.886** is the number that
judges localisation, and it is the one to quote.

**Interpretation (with caution):**
- These are the numbers *after* the test set stopped being contaminated. Until
  2026-07-24 the split scattered burst photos across train and test — 45 of the
  49 test images sat next to a training image in capture order, one 5 seconds
  away. The detector was re-trained and re-measured, and **held**: it was not
  living on the leak (unlike the flush-stage classifier, which lost 15 recall
  points on the same correction).
- Precision 1.0 is a real measurement. Until 2026-07-23 the test split was
  **all-positive**, so a background false positive was impossible by
  construction and the figure meant nothing. The first model, re-measured on backgrounds,
  fired on **100%** of them; the current one fires on none,
  and scores 0.0 on them even at a 0.01 confidence threshold.
- **Remaining warning:** the backgrounds come from the same source (iNaturalist)
  and the same taxa list as the training negatives — different observations, but
  not a different domain. Specificity 1.0 says the unconditional-firing bug is
  fixed; it does **not** predict behaviour on real field photos. The domain gap
  (internet images vs. plantation photos) is untested and remains the main risk.

---

## Training dataset

- **Source:** public dataset on **Roboflow Universe**, **CC BY 4.0** licence. Images collected from public sources (GBIF, iNaturalist, tea-pest literature).
- **Composition:** ~483 images total (train 385 / val 49 / test 49 under the capture-session split), single class `Helopeltis spp.`, YOLO-format annotations (bounding boxes). **The boxes enclose tea leaves with feeding damage**, not the insect. Plus ~180 background negatives (other insects, healthy leaves) added in the second training run.
- **Characteristics:** varied conditions (lighting, angles, backgrounds); recurring subject: **leaves with lesions/feeding damage** on a light background.

### Data limits
- **No field photos:** all images come from the internet, not from real plantations.
- **Limited diversity:** damaged leaves on a light background dominate.
- **Domain gap:** re-training with real field photos will be needed for production deployment.

---

## Training procedure

- **Base model:** YOLOv8-n pre-trained on COCO (~3.0M parameters, lightweight and CPU-friendly).
- **Hyperparameters:** 60 epochs, batch 16, image size 640, AdamW (auto), Ultralytics default augmentation.
- **Split:** capture session — photos taken in one burst stay on one side of the split.
- **Hardware:** laptop CPU (no GPU). Duration ~3–4 hours.
- **Versions:** PyTorch (CPU), Ultralytics 8.x, ONNX 1.x.

### Training command (reproducibility)

```bash
python -m src.data download              # positives + negatives (exact files), then the split
python -m src.train detector --epochs 60 # fine-tune on positives + negatives, re-export ONNX
python -m src.evaluate detector          # re-measure on the mixed test set
```

`src/train.py` generates the dataset YAML (positives + negatives) automatically.

---

## Model artifacts

The weights are heavy files and are **not versioned** in the repository (see `.gitignore`); `python -m src.data download` installs the ONNX:

```
models/
├── helopeltis_damage.onnx        # the ONNX loaded by src/pipeline.py
└── helopeltis_phase3/weights/
    └── best.pt                    # PyTorch checkpoint (regenerates the ONNX)
```

Loading (ONNX, used by the fusion layer):
```python
import onnxruntime as ort
session = ort.InferenceSession("models/helopeltis_damage.onnx",
                               providers=["CPUExecutionProvider"])
```

---

## Integration

The detector is wrapped by the `HelopeltisDamageDetector` class in [`src/pipeline.py`](../src/pipeline.py):

```python
from src.pipeline import HelopeltisDamageDetector
det = HelopeltisDamageDetector()
insects = det.detect(image_rgb, imgsz=320)   # -> list[DamageDetection]
```

Detections are then combined by the `FusionEngine` with the position (box centre) and the flush stage to produce an `Observation` (WHO×WHERE×WHEN metric); the periphery/core spatial verdict is computed by `assess_infestation` in the same module. Low-confidence detections (< 0.65) go to the "Needs Review" queue.

The ONNX is re-exported inside `train_detector` in the same run that produces the checkpoint, so the deployed artifact cannot silently lag the weights (the failure `warn_if_stale` guards against for the flush-stage classifier).

---

## Known limits and next steps

### Limits
1. **Detects the damage, not the insect:** it recognises feeding marks on the leaf, not the pest itself (dataset labelling).
2. **Domain gap:** trained on internet photos, not real field data.
3. **Single class:** it does not distinguish Helopeltis damage from damage of other similar pests.
4. **Single frame:** inference is per-image; stability depends on temporal aggregation.

### Next steps
- [ ] Decide whether to stay on a "damage detector" or aim for a true insect detector (different dataset). A classifier on the insect already exists as a separate entry point — see [`model-card-insect-id.md`](model-card-insect-id.md); merging the two needs boxed, species-level annotations.
- [ ] Validate on real field photos (when available) → domain fine-tuning.
- [ ] Measure confusion with damage from other pests.

---

## Ethical considerations

- **Intended use:** assist in detecting Helopeltis outbreaks on tea, for targeted treatment.
- **NOT intended use:** mass surveillance or tracking beyond the agricultural context.
- **Data:** public training images only. Any real deployment that stores photo locations should do so with the photographer's consent.
- **Bias:** internet photos over-represent some geographic areas; to be validated on local varieties and conditions.

---

**Last updated:** 2026-07-24
**Status:** third training run — re-trained on a capture-session split and re-measured; the
result held. Field validation still to be done, and it remains the main risk.
