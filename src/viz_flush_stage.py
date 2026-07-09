"""Visualize flush-stage classifier results on the held-out test split.

Produces two artifacts under models/ (result plots, in the same spirit as the
detector's sample batches):
  * flush_stage_predictions.png  — a grid of test images with predicted vs true
    label (green = correct, red = wrong) and the model's confidence;
  * flush_stage_confusion.png    — the confusion matrix as a heatmap.

Reuses the exact deterministic split from train_flush_stage so the images shown
are genuinely held-out (never seen in training).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.train_flush_stage import (
    CLASSES, MODELS_DIR, SEED, build_model, eval_tf, gather_samples,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_model():
    model = build_model(num_classes=len(CLASSES), freeze_backbone=False)
    state = torch.load(MODELS_DIR / "flush_stage.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def test_split() -> list[tuple[Path, int]]:
    samples = gather_samples()
    labels = [y for _, y in samples]
    _, temp = train_test_split(samples, test_size=0.30, stratify=labels, random_state=SEED)
    _, test = train_test_split(temp, test_size=0.50,
                               stratify=[y for _, y in temp], random_state=SEED)
    return test


@torch.no_grad()
def predict(model, path: Path) -> tuple[int, float]:
    img = Image.open(path).convert("RGB")
    logits = model(eval_tf(img).unsqueeze(0))
    probs = torch.softmax(logits, 1)[0]
    idx = int(probs.argmax())
    return idx, float(probs[idx])


def grid(model, test, n=16, cols=4, seed=0):
    rng = np.random.default_rng(seed)
    # Show a representative mix: keep a few misclassified ones so errors are visible.
    preds = [(p, y, *predict(model, p)) for p, y in test]
    wrong = [r for r in preds if r[1] != r[2]]
    right = [r for r in preds if r[1] == r[2]]
    rng.shuffle(wrong); rng.shuffle(right)
    chosen = (wrong[:max(4, n // 4)] + right)[:n]
    rng.shuffle(chosen)

    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    for ax, (path, true, pred, conf) in zip(axes.ravel(), chosen):
        ax.imshow(Image.open(path).convert("RGB").resize((224, 224)))
        ok = pred == true
        ax.set_title(f"pred: {CLASSES[pred]} ({conf:.2f})\ntrue: {CLASSES[true]}",
                     color="#1b7a1b" if ok else "#c0271e", fontsize=10)
        for s in ax.spines.values():
            s.set_edgecolor("#1b7a1b" if ok else "#c0271e"); s.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Flush-stage classifier — predictions on held-out test images",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out = MODELS_DIR / "flush_stage_predictions.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def confusion(model, test):
    ys = [y for _, y in test]
    ps = [predict(model, p)[0] for p, _ in test]
    cm = confusion_matrix(ys, ps)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test, n={len(test)})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    out = MODELS_DIR / "flush_stage_confusion.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    model = load_model()
    test = test_split()
    g = grid(model, test)
    c = confusion(model, test)
    print(f"Saved {g}")
    print(f"Saved {c}")


if __name__ == "__main__":
    main()
