# models/ — models and results

**The weights are not in git** (`.gitignore`: `*.pt`, `*.onnx`, heavy and
regenerable). A fresh clone gets the metrics, not the binaries. The two models
the pipeline loads come with `python -m src.data download`; everything can also
be rebuilt with `src/train.py`, exact commands in each model card.

## Weights

| File | Model | Loaded by |
|---|---|---|
| `helopeltis_damage.onnx` | damage detector (YOLOv8-n) | `HelopeltisDamageDetector`, `src/pipeline.py` |
| `flush_stage.onnx` | growth stage, WHEN | `FlushClassifier`, `src/pipeline.py` |
| `helopeltis_id.pt` | Helopeltis vs other mirids vs other insects | `src/evaluate.py geo` |

The pipeline loads **two** models. `helopeltis_id.pt` is a separate entry point,
not a stage after the detector: the detector boxes lesions, and a lesion crop
contains no insect. Reached through `notebooks/02_the_insect.ipynb`.

The ten-group model's weights are **not kept** — nothing loads them. It was an
experiment, its result is in `insect_families_metrics.json`, and
`python -m src.train insects --task groups` rebuilds it.

## Metrics (versioned — just open them)

| File | What it measures | Card |
|---|---|---|
| `detector_eval.json` | detector on the **mixed** set: 49 positives + 74 held-out backgrounds | [detector](../docs/model-card.md) |
| `flush_stage_metrics.json` | WHEN classifier + confusion matrix | [flush stage](../docs/model-card-flush-stage.md) |
| `helopeltis_id_metrics.json` | Helopeltis vs other mirids | [insect](../docs/model-card-insect-id.md) |
| `insect_families_metrics.json` | ten groups, plus **role accuracy** | [insect](../docs/model-card-insect-id.md) |
| `helopeltis_geo_control.json` | whether the classifier reads the insect or the region | [insect](../docs/model-card-insect-id.md) |

Both evaluations are built so they *can* come out badly: a number that cannot
fail is not evidence.

## Machine-specific paths

`helopeltis_damage.onnx` carries its training metadata, and Ultralytics stamps
that with the absolute path of the machine that trained it. `scrub_onnx_paths`
removes it on every export, and `tests/test_no_local_paths.py` fails if it ever
comes back. Generated lists (`split_*.txt`, `detector_*.txt`) and `args.yaml`
hold absolute paths by design and are gitignored.

## Subfolders

- `helopeltis_phase3/` — the current detector: first run trained on a
  capture-session split. `results.csv` holds the per-epoch history.
- `helopeltis_phase2/` — the previous run (contaminated split), kept on purpose
  as the evidence for the before/after comparison in the model card.

`weights/` is not versioned in either.

## Visual results

The notebooks carry their figures inline, so they read without being run:
`01_the_damage` (the detector, and why its old precision figure meant nothing) ·
`02_the_insect` (Helopeltis vs other mirids, and the geographic trap) ·
`03_the_moment` (growth stage, the risk window) · `04_the_verdict` (the readings
combined).
