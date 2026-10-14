# LeafTrace

### Detecting *Helopeltis* feeding damage on tea leaves with YOLOv8, and turning it into a risk reading

[![CI](https://github.com/angelbohorquez05/LeafTrace/actions/workflows/ci.yml/badge.svg)](https://github.com/angelbohorquez05/LeafTrace/actions/workflows/ci.yml)

An *object detection* system that locates, on tea leaves, the **feeding damage left by *Helopeltis*** (family **Miridae**), i.e. the pest's **traces**, not the insect itself, and enriches every detection with the leaf's growth stage and its position.

> **About this project.** A personal re-implementation, on public data only, of a problem I first worked on during an internship. All code here is my own and every dataset is public (CC BY 4.0 / open iNaturalist licences). It is meant to show the development process (data, evaluation, decisions, ML lifecycle), not a production model.

## What it does

*Helopeltis* is ~6–8 mm and mobile, so it is almost never captured in a photo: what stays visible and persistent is the **damage it leaves on the leaf**. LeafTrace detects that damage, in two parts that run end-to-end ([`src/pipeline.py`](src/pipeline.py)):

1. **Detects the pest's damage**: localises *Helopeltis* **feeding marks** on a leaf with a *bounding box* (YOLOv8-n, run with ONNX Runtime).
2. **Classifies the leaf stage**: **young leaf** (tender shoot, high risk) vs **mature leaf**.

From these two signals, each detection becomes an `Observation` organised as **WHO×WHERE×WHEN**:

| Dimension | Question | Status |
|-----------|----------|--------|
| **WHO** | Is the pest present? (read from its damage) | Damage detector, works on the available data |
| **WHEN** | Which leaf stage? | 2 stages: young / mature, trained but weak (see below) |
| **WHERE** | Edge or centre of the monitored area? | Logic built and tested; on today's photos the verdict is an artefact (see below) |

**WHERE** aggregates many detections into an edge-vs-core verdict: is the damage only at the border of the area, or has it reached the middle? It needs a real position for each photo, which internet photos don't carry. Run on the current data it returns `breach=True` with **every single detection in the core**, and that is not a finding, it is an artefact of photography: people centre their subject, so every box centre lands near the middle of the frame. Today the verdict means "edge vs centre of the *frame*". The logic does not change when real coordinates exist, only the coordinate space. Shown, with the plot, in [`notebooks/04_the_verdict.ipynb`](notebooks/04_the_verdict.ipynb).

### A third model, on the insect itself

The detector never names anything: it has one class and stamps a fixed label on every box. A **separate classifier** answers *is this Helopeltis?* (vs another mirid, vs another insect), and a second variant sorts ten insect groups into ecological roles.

It is a **second entry point, not a pipeline stage**: the detector boxes *lesions on a leaf*, and a lesion crop contains no insect to classify. One model looks at a leaf, the other at an animal. Metrics, and the geographic confound it exposed, in the [insect model card](docs/model-card-insect-id.md).

## Quick start

Requirements: **Python 3.10+** and `git`. Everything runs **on CPU**.

```bash
git clone https://github.com/angelbohorquez05/LeafTrace.git
cd LeafTrace

python -m venv venv
venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source venv/bin/activate       # macOS / Linux

# optional, much smaller download: CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

No images are stored in git. One command downloads the exact datasets and trained models the results below were measured on (checksummed, ~60 MB), then rebuilds the capture-session split:

```bash
python -m src.data download
```

Then:

```bash
python -m src.evaluate detector   # reproduces the detector numbers below
pytest                            # 51 tests, including an end-to-end photo -> Observation run
```

The notebooks in [`notebooks/`](notebooks/) are saved with their figures, so they read without being run.

### Rebuilding or retraining

| Goal | Command |
|---|---|
| Retrain the detector (~3–4 h on a laptop CPU) | `python -m src.train detector --epochs 60` |
| Leaf-stage images (Mendeley, ~4 GB) | `python -m src.data flush "<Raw Data folder>"` |
| Retrain the leaf-stage classifier | `python -m src.train flush` |
| Insect images (iNaturalist, exact set) | `python -m src.data rebuild` |
| Retrain the insect classifier | `python -m src.train insects --task helopeltis` |
| Geographic control for the insect classifier | `python -m src.evaluate geo` |

## Data

- **Detector positives:** public dataset from **Roboflow Universe** (**CC BY 4.0**), *Helopeltis* images collected from public sources. ~483 images with *bounding boxes*, single class. **Important:** the boxes enclose **tea leaves with feeding damage** (lesions), not the insect, so the model learns to recognise the *damage*, not the pest itself.
- **Detector backgrounds:** 251 iNaturalist photos (other insects, healthy foliage) with empty labels, 74 of them held out for testing. Provenance per image in [`manifest.csv`](data/raw/helopeltis_negatives/manifest.csv).
- **Leaf stage:** TeaLeafAgeQuality (Mendeley Data, CC BY 4.0), 2208 images by leaf age.
- **Insects:** iNaturalist, open licences, rebuildable exactly from a manifest.

Each set has its own README under [`data/raw/`](data/raw/) covering provenance, licences and how to rebuild it. The detector sets and the two deployed models are published as checksummed [release assets](https://github.com/angelbohorquez05/LeafTrace/releases), so cloning stays light and every download is the same set of bytes.

## Structure

```
LeafTrace/
├── src/
│   ├── data.py            every dataset: download, rebuild, capture-session split
│   ├── train.py           the three models: detector, flush stage, insect ID
│   ├── evaluate.py        the honest evaluations (mixed test set, geographic control)
│   └── pipeline.py        runtime: models -> Observation -> field verdict
├── tests/                 pytest (unit tests + end-to-end with the real models)
├── notebooks/             the visual story, in order: 01 damage · 02 insect · 03 moment · 04 verdict
├── data/raw/              READMEs, manifests and the session split (images are downloaded)
├── models/                metrics and training histories (weights are downloaded)
├── docs/                  model cards
├── .github/workflows/     CI: unit tests, then download + reproduce + end-to-end
└── requirements.txt
```

## The models

All three are measured on splits that keep **capture sessions whole**; see below for why that sentence is the most important one on this page.

| Model | Architecture | Result | Card |
|---|---|---|---|
| Damage detector | YOLOv8-n (COCO) → ONNX | AP@0.5 1.0, **AP@0.5:0.95 0.886**, specificity 1.0 | [card](docs/model-card.md) |
| Leaf stage (WHEN) | MobileNetV2 → ONNX | accuracy 0.731, `new_flush` P 0.43 / R 0.64 | [card](docs/model-card-flush-stage.md) |
| Insect classifier | MobileNetV2 | 0.86 / 0.87 precision-recall on Helopeltis | [card](docs/model-card-insect-id.md) |

### The split, and what it cost

My first numbers were measured on a test set that was a near-duplicate of the training set. Both public datasets are **burst photography** (someone walking a plantation, shooting continuously), and a random split scatters photos of the same leaf, taken seconds apart, across train and test. Measured: **45 of the 49 detector test images sat immediately beside a training image** in capture order, the closest 5 seconds away; in the leaf-stage set, **100%** of the timestamped photos have another within 60 seconds.

The fix ([`src/data.py`](src/data.py), `split`) reconstructs the capture sessions from the filenames and sends each one whole to a single split. Both models were re-trained. They answered differently, and that is the result:

| | Contaminated split | Capture-session split |
|---|---|---|
| Detector AP@0.5 | 1.000 | **1.000** |
| Leaf stage, `new_flush` recall | 0.791 | **0.642** |
| Leaf stage, `mature_leaf` recall | 0.948 | **0.756** |

**The detector held; the leaf-stage classifier did not.** The detector learned the texture of the damage. The classifier was memorising: 2208 images, but only **20 capture sessions**, five of which hold 84% of them.

### Reading the detector's numbers

Precision 1.0 is a real measurement: the test set is **mixed**, 49 positives plus 74 held-out backgrounds, so a false positive is possible. The first detector, trained without backgrounds, fired on **100%** of those images; the current model fires on none, scoring 0.0 even at a 0.01 confidence threshold.

But **AP@0.5 is not a metric whose floor is zero.** The boxes cover ~40% of the image and subjects are centred, so a constant centred box with no model at all scores **0.514**. That floor is printed under every evaluation run. **AP@0.5:0.95 = 0.886** is the number that judges localisation, and the one to quote.

What none of this measures: every image is from the internet. Specificity 1.0 says a bug is fixed, not that the model works in a plantation.

## Model lifecycle

- **Data:** `python -m src.data download` (or rebuild from source), then the capture-session split, which training and evaluation refuse to run without.
- **Training:** fine-tune YOLOv8-n on CPU, positives + background negatives.
- **Validation:** detection metrics (AP@0.5, AP@0.5:0.95, precision, recall) + specificity on held-out backgrounds, with the constant-box floor alongside.
- **Export:** ONNX for both deployed models; inference needs only ONNX Runtime, NumPy and Pillow. Machine paths are scrubbed from the model metadata, and a test fails if one comes back.
- **Monitoring (if deployed):** input drift (domain, season), reported false negatives, AP over time; low-confidence detections (< 0.65) go to a review queue instead of being reported.

## Limits

**Assumptions**

- Public internet images are an acceptable *proxy* to develop and evaluate the method; photos taken on the bush would be the real test.
- The stage classifier covers 2 classes (`new_flush` / `mature_leaf`); a `dormant` stage is out of scope (lowest pest activity, no public dataset). If its model file is absent, `FlushClassifier.predict` returns `None` instead of failing.

**Risks**

- **Domain gap:** everything is trained on internet photos. The high metrics do not carry over to a plantation, and out of domain the detector draws frame-sized boxes, which also makes the position WHERE depends on unreliable.
- **Geographic confound:** the insect classifier partly reads the *region* rather than the animal: 11.1% of same-region mirids called Helopeltis, against 6.1% on the ordinary test split. Halved by re-training, not removed.
- **The leaf stage (WHEN) is weak:** `new_flush` precision 0.43; when it calls a shoot tender it is wrong more often than right. Usable as one input among several, not as a treatment trigger. The ceiling is data diversity (20 capture sessions), not training.
- **Single class, single frame:** the detector cannot separate Helopeltis damage from another pest's damage, and inference is per image (`FusionEngine.push_frame` adds temporal aggregation for video).

## Next steps

1. Validate and fine-tune the detector on photos taken in the field.
2. Attach a real position to each photo so the edge-vs-core verdict means something.
3. Widen the leaf-stage dataset (more plantations, devices, seasons): its limit is diversity, not training.

## Further information

- **Model cards:** [detector](docs/model-card.md) · [leaf stage](docs/model-card-flush-stage.md) · [insect classifier](docs/model-card-insect-id.md).
- **Notebooks:** [`notebooks/`](notebooks/): 01 the damage · 02 the insect · 03 the moment · 04 the verdict.
- **Licences:** datasets as listed above; YOLOv8 by Ultralytics (AGPL-3.0); MobileNetV2 from torchvision.
