# Model Card — Flush-Stage Classifier (WHEN dimension)

## Model overview

Image classifier that estimates the **flush stage** of a tea leaf, to populate
the **WHEN** dimension of the WHO×WHERE×WHEN metric. It feeds
`FlushClassifier` in [`src/pipeline.py`](../src/pipeline.py); if the model file is
absent, `predict` returns `None` rather than failing.

- **Architecture:** MobileNetV2 (transfer learning from ImageNet) with a 2-class linear head.
- **Input:** RGB image 224×224, ImageNet normalisation (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Classes (logit order):** `new_flush` (tender shoot, **high risk** for Helopeltis), `mature_leaf` (mature leaf, low risk).
- **`dormant` stage:** **out of scope** — it is the lowest-pest-activity stage and no public dataset exists.

## Performance metrics

> **These numbers replaced much better ones on 2026-07-24, and the drop is the
> result.** The previous split was random by image, on a dataset where 100% of
> the timestamped photos have another photo within 60 seconds — bursts of the
> same leaf, scattered across train and test. The old test accuracy of 0.883
> was largely recall of images the model had all but seen.

### Results on the test set (490 images: 109 `new_flush` + 381 `mature_leaf`)

Split by **capture session**, sessions kept whole, stratified by class.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `new_flush` | **0.429** | 0.642 | 0.515 | 109 |
| `mature_leaf` | 0.881 | 0.756 | 0.814 | 381 |
| **Accuracy** | | **0.731** | | 490 |

Confusion matrix (rows = true `[new_flush, mature_leaf]`): `[[70, 39], [93, 288]]`.

**What changed, measured honestly.** Recall is the fair comparison — precision
moves with class prevalence, and the new test split is 22/78 where the old was
53/47:

| | Contaminated split | Capture-session split |
|---|---|---|
| `new_flush` recall | 0.791 | **0.642** |
| `mature_leaf` recall | 0.948 | **0.756** |
| accuracy | 0.864 | **0.731** |

**Interpretation.** The model is weak, and now honestly so. `new_flush`
precision of 0.429 means that when it calls a shoot tender it is **wrong more
often than right** — unusable as a standalone trigger for a treatment decision.
Training accuracy ended at 0.968 against validation 0.565: it memorises.

**The cause is the dataset, not the training.** 2208 images, but only **20
capture sessions**, the five largest holding 84% of them. The effective sample
size is about twenty scenes. No number of epochs, no augmentation and no loss
weighting manufactures diversity that is not there — only more sources will.

### Operating point: 0.50 (argmax), re-calibrated 2026-07-24

`FlushClassifier` applied a **0.35 threshold** on `new_flush` until 2026-07-24,
to buy recall on the high-risk class. That value had been tuned on the
contaminated split. Re-swept from 0.20 to 0.80 on the **new validation split**,
the macro-F1 optimum is **0.50** — plain argmax — and the curve is flat and
monotone around it: no threshold beats argmax.

Confirmed on the test set:

| Threshold | Accuracy | macro-F1 | `new_flush` P / R |
|---|---|---|---|
| **0.50 (deployed)** | **0.731** | **0.664** | 0.429 / 0.642 |
| 0.35 (retired) | 0.686 | 0.629 | 0.381 / 0.661 |
| 0.25 | 0.673 | — | 0.376 / 0.706 |

The old value bought 2 points of `new_flush` recall for 5 of precision. The
`NEW_FLUSH_THRESHOLD` constant stays in the code — it is a calibration result,
to be re-swept whenever the classifier is re-trained, not a default nobody
examined.

## Training dataset

- **Source:** TeaLeafAgeQuality (Mendeley Data, CC BY 4.0 licence), 2208 images of *Camellia sinensis*.
- **Age → stage mapping:** young categories → `new_flush` (1177), mature categories → `mature_leaf` (1031). Details in [`data/raw/flush_stages/README.md`](../data/raw/flush_stages/README.md).
- **Split:** **capture session**, 70/15/15, sessions kept whole and stratified by class → train 1267, val 451, test 490. Not a random split by image.
- **Effective diversity:** 2208 images, but only **20 capture sessions** — the five largest hold 84%.

### Data limits

- The dataset is about leaf **age/quality**, a *proxy* of the flush stage.
- Public-domain images, not real field photos: possible *domain gap*.
- No `dormant` class (out of scope: lowest-pest-activity stage, no public dataset).

## Training procedure (reproducibility)

- **Stages:** head with frozen backbone (5 epochs, Adam lr 1e-3) → full fine-tuning (2 epochs, lr 1e-4).
- **Loss:** cross-entropy with balanced class weights. Seed 42, CPU.
- **Command:**
  ```bash
  python -m src.train flush --epochs-head 5 --epochs-ft 2 --batch 32
  ```

## Model artifacts

Weights are not versioned (see `.gitignore`). `python -m src.data download`
installs the ONNX model; training regenerates both files:

- `models/flush_stage.pt` — PyTorch weights.
- `models/flush_stage.onnx` — ONNX export (NCHW 1×3×224×224 → 2 logits), loaded by the pipeline.
- `models/flush_stage_metrics.json` — metrics summary (versioned).

The results are shown **inline** in [`notebooks/03_the_moment.ipynb`](../notebooks/03_the_moment.ipynb), and the full report and confusion matrix stay machine-readable in `models/flush_stage_metrics.json`.

## Integration

`FlushClassifier` loads `models/flush_stage.onnx` with ONNX Runtime if present
and applies `NEW_FLUSH_THRESHOLD` (0.50, see Operating point); with a number of
classes other than 2 it falls back to `argmax`. The wrapper's preprocessing
(ImageNet normalisation, channels-first) matches training. On load it calls
`warn_if_stale`, which reports if the ONNX predates the checkpoint it was
exported from, so the pipeline never silently serves a model the metrics do not
describe. Verified with `notebooks/03_the_moment.ipynb`.

## Known limits and next steps

1. **Not usable as a standalone trigger.** `new_flush` precision 0.429. As one
   input among several it still carries signal; as the thing that decides
   whether to treat a plot, it does not.
2. **Twenty scenes, not two thousand photos.** The ceiling here is data
   diversity. The next useful step is *more sources* — different plantations,
   devices, seasons — not more training on this set.
3. **Re-sweep the threshold** whenever this model is re-trained (see Operating
   point above) — the constant exists to be re-derived, not inherited.
4. **Validation on real field photos** when available.

**Out of scope:** a `dormant` class (lowest-pest-activity stage, no public dataset).

**Last updated:** 2026-07-24
**Status:** re-trained on a capture-session split. The honest numbers are much
weaker than the ones this card carried until 2026-07-24, and the earlier figures
should not be quoted from anywhere.
