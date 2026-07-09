"""Train the flush-stage classifier (Workstream B — the WHEN dimension).

Transfer learning on MobileNetV2 (ImageNet) with a 2-class head:
``new_flush`` vs ``mature_leaf``. The third stage ``dormant`` is deferred until
a dedicated dataset exists (see data/raw/flush_stages/README.md).

The script:
  1. builds a deterministic, stratified train/val/test split (fixed seed);
  2. fine-tunes MobileNetV2 (frozen backbone, then a short full fine-tune);
  3. reports accuracy and per-class recall (recall on ``new_flush`` is the
     one that matters — it is the high-risk stage);
  4. saves the weights and exports an ONNX graph (1, 3, 224, 224).

TFLite conversion is a separate step (``onnx2tf``) so the fragile part is
isolated; see ``export_to_tflite`` and the README. Runs on CPU.

Class order MUST stay aligned with ``FlushClassifier.CLASSES`` in src/fusion.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252; the ONNX exporter and some libs print
# non-latin glyphs. Force UTF-8 so those prints never crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# --- Constants ---------------------------------------------------------------
# Index order defines the output logit order of the exported model; it must
# match FlushClassifier.CLASSES in src/fusion.py.
CLASSES = ["new_flush", "mature_leaf"]

_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data" / "raw" / "flush_stages"
MODELS_DIR = _ROOT / "models"

IMG_SIZE = 224
# ImageNet normalization — must match the preprocessing in FlushClassifier.predict.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42
_IMG_EXT = {".jpg", ".jpeg", ".png"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- Data --------------------------------------------------------------------
def gather_samples(limit_per_class: int = 0) -> list[tuple[Path, int]]:
    """Return (path, label) pairs; label is the index of the class in CLASSES."""
    samples: list[tuple[Path, int]] = []
    for label, cls in enumerate(CLASSES):
        files = sorted(p for p in (DATA_DIR / cls).glob("*") if p.suffix.lower() in _IMG_EXT)
        if limit_per_class:
            files = files[:limit_per_class]
        samples.extend((p, label) for p in files)
    if not samples:
        raise SystemExit(f"No images found under {DATA_DIR}. Extract the dataset first.")
    return samples


class FlushDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], tf: transforms.Compose):
        self.items = items
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# --- Model -------------------------------------------------------------------
def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


# --- Train / eval loops ------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    for x, y in loader:
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            out = model(x)
            loss = criterion(out, y)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def predict_all(model, loader) -> tuple[list[int], list[int]]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.extend(model(x).argmax(1).tolist())
            ys.extend(y.tolist())
    return ys, ps


# --- Export ------------------------------------------------------------------
def export_onnx(model, path: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        opset_version=13, dynamo=False,
    )


# --- Main --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs-head", type=int, default=5, help="epochs with frozen backbone")
    ap.add_argument("--epochs-ft", type=int, default=2, help="epochs of full fine-tuning")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="cap images/class (quick smoke run)")
    ap.add_argument("--no-export", action="store_true", help="skip ONNX export")
    args = ap.parse_args()

    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, (torch.get_num_threads())))

    # Stratified 70/15/15 split, deterministic.
    samples = gather_samples(args.limit)
    labels = [y for _, y in samples]
    train, temp = train_test_split(samples, test_size=0.30, stratify=labels, random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50,
                                 stratify=[y for _, y in temp], random_state=SEED)
    print(f"Split: train={len(train)}  val={len(val)}  test={len(test)}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = {c: sum(1 for _, y in split if y == i) for i, c in enumerate(CLASSES)}
        print(f"  {name}: {counts}")

    dl = lambda items, tf, shuffle: DataLoader(
        FlushDataset(items, tf), batch_size=args.batch, shuffle=shuffle, num_workers=0)
    train_loader = dl(train, train_tf, True)
    val_loader = dl(val, eval_tf, False)
    test_loader = dl(test, eval_tf, False)

    model = build_model(num_classes=len(CLASSES), freeze_backbone=True)

    # Balanced class weights (mild imbalance ~53/47).
    counts = np.bincount([y for _, y in train], minlength=len(CLASSES))
    weights = torch.tensor((counts.sum() / (len(CLASSES) * counts)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights: {weights.tolist()}")

    def train_phase(epochs, lr, label):
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        for e in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt)
            va_loss, va_acc = run_epoch(model, val_loader, criterion)
            print(f"[{label} {e}/{epochs}] "
                  f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} | "
                  f"val_loss={va_loss:.3f} val_acc={va_acc:.3f} "
                  f"({time.time() - t0:.0f}s)")

    print("\n== Phase 1: head only (frozen backbone) ==")
    train_phase(args.epochs_head, args.lr, "head")

    if args.epochs_ft > 0:
        print("\n== Phase 2: full fine-tuning ==")
        set_backbone_trainable(model, True)
        train_phase(args.epochs_ft, args.ft_lr, "ft")

    # Final evaluation on the held-out test split.
    print("\n== Test set evaluation ==")
    ys, ps = predict_all(model, test_loader)
    report = classification_report(ys, ps, target_names=CLASSES, digits=3)
    cm = confusion_matrix(ys, ps).tolist()
    print(report)
    print(f"Confusion matrix (rows=true {CLASSES}): {cm}")

    # Persist weights + a small metrics summary for the model card.
    torch.save(model.state_dict(), MODELS_DIR / "flush_stage.pt")
    metrics = {
        "classes": CLASSES,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "report": report, "confusion_matrix": cm,
    }
    (MODELS_DIR / "flush_stage_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved weights -> {MODELS_DIR / 'flush_stage.pt'}")

    if not args.no_export:
        onnx_path = MODELS_DIR / "flush_stage.onnx"
        export_onnx(model, onnx_path)
        print(f"Exported ONNX -> {onnx_path}")
        print("Next: convert to TFLite with "
              "`onnx2tf -i models/flush_stage.onnx -o models/flush_stage_tf` "
              "then copy the NHWC float32 model to models/flush_stage.tflite")


if __name__ == "__main__":
    main()
