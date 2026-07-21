"""
evaluate_detector.py — Validate the integrated ONNX detector on the test set

The mAP figures quoted in the README come from Ultralytics' training-time
validation. This script instead measures the *integrated* inference path — the
same ``HelopeltisDetector`` (ONNX Runtime + NMS) the fusion engine uses — against
the test-set ground truth, so the reported numbers reflect what actually runs in
the pipeline, not just what training logged.

Metric: single-class detection at IoU 0.5.
  * precision / recall at the detector's operating confidence threshold,
  * AP@0.5 (area under the precision-recall curve, all-point interpolation —
    the same definition Ultralytics uses for mAP@0.5).

Run (from the repository root):
    python -m src.evaluate_detector

Needs the ONNX detector (models/helopeltis_phase1.onnx) plus onnxruntime,
Pillow and numpy. Writes a JSON artifact to models/detector_eval.json and prints
a summary. Skips cleanly if the model or its dependencies are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.fusion import BBox, HelopeltisDetector

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEST_DIR = _REPO_ROOT / "data" / "raw" / "helopeltis" / "test"
_METRICS_PATH = _REPO_ROOT / "models" / "detector_eval.json"

IOU_MATCH = 0.5   # a prediction counts as a hit when IoU with a GT box >= this


def _load_ground_truth(label_path: Path) -> list:
    """Parse a YOLO label file into a list of BBox (normalised corners)."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        # class cx cy w h — all normalised to [0, 1]
        _, cx, cy, w, h = (float(p) for p in parts[:5])
        boxes.append(BBox(cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    return boxes


def _match_predictions(predictions: list, ground_truth: list) -> list:
    """Greedily label each prediction TP (1) or FP (0) against the GT boxes.

    Predictions are consumed highest-confidence first; each GT box can be
    matched at most once (extra detections on the same insect are false
    positives, which is what NMS is meant to prevent).
    """
    unmatched = list(ground_truth)
    flags = []
    for _conf, box in sorted(predictions, key=lambda cb: cb[0], reverse=True):
        best_iou, best_i = 0.0, -1
        for i, gt in enumerate(unmatched):
            iou = box.iou(gt)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= IOU_MATCH:
            flags.append(1)
            unmatched.pop(best_i)
        else:
            flags.append(0)
    return flags


def _average_precision(sorted_flags: list, n_ground_truth: int) -> float:
    """All-point-interpolated AP from TP/FP flags sorted by descending conf."""
    if n_ground_truth == 0:
        return 0.0
    tp = fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for flag in sorted_flags:
        tp += flag
        fp += 1 - flag
        recalls.append(tp / n_ground_truth)
        precisions.append(tp / (tp + fp))

    # Monotonically decreasing precision envelope, then integrate over recall.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def evaluate(iou_match: float = IOU_MATCH) -> dict | None:
    """Run the integrated detector over the test set and return metrics."""
    detector = HelopeltisDetector()
    if not detector.available:
        print("[eval] Detector model not available — skipping.")
        print("       (expected models/helopeltis_phase1.onnx + onnxruntime)")
        return None

    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"[eval] Missing dependency ({e}) — skipping.")
        return None

    images = sorted((_TEST_DIR / "images").glob("*.jpg"))
    if not images:
        print(f"[eval] No test images in {_TEST_DIR / 'images'} — skipping.")
        return None

    all_flags = []          # (confidence, tp_flag) across every image
    n_gt = 0
    n_pred = 0
    for img_path in images:
        gt = _load_ground_truth(_TEST_DIR / "labels" / f"{img_path.stem}.txt")
        n_gt += len(gt)

        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        preds = [(d.confidence, d.bbox) for d in detector.detect(image_rgb)]
        n_pred += len(preds)

        flags = _match_predictions(preds, gt)
        confs = [c for c, _ in sorted(preds, key=lambda cb: cb[0], reverse=True)]
        all_flags.extend(zip(confs, flags))

    # Precision / recall at the operating point (every returned detection is
    # already above the detector's confidence threshold).
    tp = sum(f for _, f in all_flags)
    fp = len(all_flags) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0

    sorted_flags = [f for _, f in sorted(all_flags, key=lambda cf: cf[0], reverse=True)]
    ap = _average_precision(sorted_flags, n_gt)

    return {
        "path": "integrated ONNX (HelopeltisDetector + NMS)",
        "n_images": len(images),
        "n_ground_truth": n_gt,
        "n_predictions": n_pred,
        "conf_threshold": detector.CONF_THRESHOLD,
        "iou_match": iou_match,
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "ap50": round(ap, 4),
    }


def main():
    metrics = evaluate()
    if metrics is None:
        return

    _METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print("=" * 60)
    print("Integrated detector — test-set evaluation (IoU 0.5)")
    print("=" * 60)
    print(f"Images        : {metrics['n_images']}")
    print(f"Ground truth  : {metrics['n_ground_truth']} boxes")
    print(f"Predictions   : {metrics['n_predictions']} (conf >= {metrics['conf_threshold']})")
    print(f"TP / FP       : {metrics['true_positives']} / {metrics['false_positives']}")
    print(f"Precision     : {metrics['precision']:.3f}")
    print(f"Recall        : {metrics['recall']:.3f}")
    print(f"AP@0.5        : {metrics['ap50']:.3f}")
    print(f"\nSaved -> {_METRICS_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
