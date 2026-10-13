# Model Card — Insect Classifier

## Model overview

Image classifier on the **insect itself**, answering the question the damage
detector cannot: the detector has one class and stamps a fixed
`"Helopeltis spp."` string on every box — it never names anything.

Two tasks, one architecture (MobileNetV2, ImageNet transfer, 224×224 RGB):

| Task | Classes | Artifact |
|---|---|---|
| `helopeltis` | `helopeltis` · `other_miridae` · `other_insects` | `models/helopeltis_id.pt` |
| `groups` | ten ecological insect groups | weights not kept — see below |

> **Not a pipeline stage.** This is a **second entry point**, not a step after
> the detector. The detector boxes *lesions on a leaf*; a lesion crop contains no
> insect, so its output cannot be fed here. One model looks at a leaf, the other
> at an animal.

## Performance — `helopeltis` task

2795 images, 420 held-out test. **Accuracy 0.819.**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `helopeltis` | **0.862** | **0.872** | 0.867 | 86 |
| `other_miridae` | 0.709 | 0.817 | 0.759 | 131 |
| `other_insects` | 0.890 | 0.798 | 0.842 | 203 |

Confusion matrix in `models/helopeltis_id_metrics.json`. The errors that matter
are the ones inside Miridae: **7** Helopeltis missed as another mirid, **8**
mirids called Helopeltis. Distinguishing Helopeltis from a bee is free;
distinguishing it from its own family is the task.

## Performance — `groups` task

1498 images, 225 held-out test. **Group accuracy 0.627, role accuracy 0.760.**

**Role accuracy is the honest headline.** Confusing Coreidae with Cicadellidae
costs a grower nothing — both pests, same action. Confusing a pest with a
pollinator means spraying something that was helping. Group accuracy penalises
the first error as heavily as the second; role accuracy does not.

Weakest classes: `aphididae` (0.35 precision — aphids are small and photographed
in clusters) and `araneae` (0.36 recall, often confused with bees). Strongest:
`formicidae` (1.00 precision), `cicadellidae` (0.74/0.77).

**Its weights are not kept.** Nothing loads them — the model answered a question
(can a general insect classifier be useful at role level? yes, 0.76) and the
answer is this section plus `models/insect_families_metrics.json`. Keeping 17 MB
of binaries that nothing reads would be storing the receipt instead of the
result. `python -m src.train insects --task groups` rebuilds it.

## Geographic control — the number that matters

The training classes are split by geography: Helopeltis is tropical-Asian, and
the mirid negatives came mostly from the USA and Europe. A model can score well
on that split by learning vegetation and light instead of the insect — and the
test split, carrying the same bias, cannot detect it.

`python -m src.evaluate geo` measures it against **208 mirids (Helopeltis
excluded) photographed in Helopeltis's own ten regions**, with every image the
model trained on removed by observation id.

| | Same-region mirids | Ordinary test split |
|---|---|---|
| First run | **22.8%** called Helopeltis | 9.4% |
| After adding regional negatives | **11.1%** | 6.1% |

The shortcut was real, the fix halved it, and the residual gap is real too.
Full result in `models/helopeltis_geo_control.json`.

## Training dataset

iNaturalist, open licences (`cc0` / `cc-by` / `cc-by-nc`), provenance per image in
`data/raw/insect_families/manifest.csv`. Classes, the ranks-are-mixed caveat and
the rebuild procedure are in
[`data/raw/insect_families/README.md`](../data/raw/insect_families/README.md).

**Not affected by the capture-session contamination** that hit the other two models,
and this was measured rather than assumed before deciding not to re-train:
one photo per observation, excluded by `obs_id`, from **1497 distinct
contributors**. Only **5.5%** of images sit within 3 observation ids of another
by the same person, against 92% burst-adjacency in the detector's Roboflow set.
The number that matters here — the geographic control — already drops every
image the model trained on, by id.

**Data limits**

- One insect, large and centred — nothing here measures a scouting photo, where
  the bug is small in a wide frame of foliage.
- Cannot separate Helopeltis from other mirids below family level without
  species-level boxed annotations.
- Images are **not versioned** (~390 MB); `manifest.csv` is, and rebuilds the
  exact set.

## Training procedure (reproducibility)

Frozen backbone (8 epochs, Adam lr 1e-3) → full fine-tune (8 epochs, lr 1e-4),
early stopping on validation loss, patience 3. Best: fine-tune epoch 5
(`helopeltis`, val loss 0.450) / epoch 3 (`groups`, val loss 1.075).
Augmentation: `RandomResizedCrop(0.6–1.0)` + flip + rotation 15° + jitter 0.3.
Seed 42, CPU. Deliberately identical to the flush-stage classifier so the two
sets of numbers are comparable.

```bash
python -m src.data helopeltis            # the two taxa
python -m src.data regional              # same-region mirids (geographic fix)
python -m src.train insects --task helopeltis
python -m src.evaluate geo               # the control above
```

## Integration

**Not wired into `src/pipeline.py`.** The pipeline runs the damage detector and
the flush classifier; this model is reached through `notebooks/02_the_insect.ipynb`.
Wiring it would require a detector that boxes insects rather than lesions — see
"Next steps" below.

The `ECOLOGICAL_ROLE` map in `src/pipeline.py` already resolves a species label
to pest / beneficial / pollinator / neutral, so the contract this model would
plug into exists; what is missing is the crop to hand it.

## Known limits and next steps

1. **Geographic confound, halved not removed** — 11.1% vs 6.1%. Only field
   photos close the rest.
2. **Family, not species** — `other_miridae` is the whole family; Helopeltis has
   close relatives inside it.
3. **No detection** — classification assumes a cropped insect. Field use needs
   boxes plus tiled high-resolution inference.
4. **Next:** a boxed insect dataset would merge this model and the damage
   detector into one multi-class detector, and make the WHO dimension a
   prediction rather than a hard-coded string.

**Last updated:** July 2026
**Status:** trained and measured, including the geographic control; a separate
entry point, not integrated into the runtime pipeline.
