# helopeltis_negatives — Background images for the detector (negatives)

**Negative** images (background, in YOLO format) to cure the first detector's
flaw: trained on a **single-class, all-positive** dataset (every image contained
Helopeltis, no background images), the model had learned to emit a high-confidence
"Helopeltis" box on **almost any image** — 100% false positives on other insects
and even on insect-free leaves.

Adding **background** images (with an empty label) teaches the model what is
**not** Helopeltis.

## What it contains

Photos of clearly **non-Helopeltis** subjects from iNaturalist (open licences),
chosen to exclude the **Miridae** family (so a real Helopeltis is never
mislabelled as background):

- **Easy negatives:** bees (*Apis mellifera*), ladybirds, hoverflies, ants,
  butterflies, spiders.
- **Hard negatives:** other Hemiptera that look bug-like but are not Miridae
  (*Cicadellidae*, *Coreidae*).
- **Pure backgrounds (no insect):** tea leaves (*Camellia sinensis*) and
  generic foliage (*Plantae*).

Structure (YOLO format; the labels are **empty** files → background images):

```
helopeltis_negatives/
├── train/{images,labels}/   # 137 images  (training)
├── valid/{images,labels}/   # 40 images   (training)
├── test/{images,labels}/    # 74 images   (HELD OUT — evaluation only)
└── manifest.csv             # id, taxon, licence, attribution for each photo
```

The `test/` split is **never trained on**. It exists so that precision can be
measured at all: the positive test split has a ground-truth box in every image,
so a background false positive is impossible there by construction. These 74
images are what turn `models/detector_eval.json` into a real measurement. They are disjoint from `train`/`valid` — the builder
skips every observation id already listed in `manifest.csv`.

## Getting the images (exact set)

Re-downloading from iNaturalist would not return the exact same set (the
sort-by-votes order changes, some observations get removed), so the images the
published results were measured on are kept as a checksummed release asset
rather than in git:

```bash
python -m src.data download            # exact set, verified by SHA-256 (~38 MB)
```

To build a fresh set or extend it from iNaturalist (a *different* sample):

```bash
python -m src.data negatives           # train + valid (training negatives)
python -m src.data negatives --test    # held-out test negatives (evaluation)
```

## Licences and attribution

Each image comes from iNaturalist with a `cc0`, `cc-by` or `cc-by-nc` licence.
Author and observation link are in `manifest.csv`. **Academic / non-commercial**
use, consistent with `cc-by-nc`.

## How they are used in training

`src/train.py` generates a `data.yaml` that lists both the positives
(`helopeltis/`) and these negatives. Ultralytics reads labels by swapping
`/images/` → `/labels/`: finding empty labels, it treats these images as
background. After training, re-measure with the specificity test.
