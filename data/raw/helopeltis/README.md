# helopeltis — Detector positives

483 photos of tea leaves with *Helopeltis* feeding damage, one bounding box per
damaged leaf, YOLO format (`<split>/images/*.jpg` + `<split>/labels/*.txt`, each
line `class cx cy w h` normalised to [0, 1], single class).

- **Source:** public dataset on Roboflow Universe, **CC BY 4.0** (export notes in
  `README.roboflow.txt`). Images collected from public sources.
- **Get it:** `python -m src.data download` (checksummed release asset, ~3 MB).
  The images are not in git.
- **Split:** the Roboflow `train/valid/test` folders are kept as delivered, but
  they scatter burst photos of the same leaf across splits. Training and
  evaluation use the **capture-session split** instead: `splits/sessions.csv`
  (versioned) records which session and split every image belongs to.
  `python -m src.data split` rebuilds it deterministically from the filenames
  (no random seed) and writes the path lists training reads.
