"""Honest evaluation — the numbers this project is willing to publish.

Both checks here exist because an earlier metric was a mirage, and both are
built so they CAN come out badly. That is the whole point: a number that cannot
fail is not evidence.

  ``detector``  measures the damage detector on a MIXED test set — the 49
                positive images plus held-out backgrounds. On positives alone
                every image contains a box, so a background false positive is
                impossible and the precision figure means nothing. It reported
                1.0 for weeks that way.
  ``geo``       measures whether the Helopeltis classifier reads the insect or
                the continent. Its training classes are split by geography
                (Helopeltis from tropical Asia, other mirids from the USA and
                Europe), so a model can score well by learning the vegetation.
                The control is other mirids from Helopeltis's OWN regions.

Commands (from the repository root):

    python -m src.evaluate detector    -> models/detector_eval.json
    python -m src.evaluate geo         -> models/helopeltis_geo_control.json
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import (fetch, resolve_place, HELOPELTIS_REGIONS,
                      HELOPELTIS_TAXON, MIRIDAE_TAXON, FAM, UA, MODELS_DIR)
from src.pipeline import BBox, HelopeltisDamageDetector


# ============================================================================
# 1. Damage detector on a mixed set
# ============================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPLIT_TEST = _REPO_ROOT / "models" / "split_test.txt"
_NEG_TEST_DIR = _REPO_ROOT / "data" / "raw" / "helopeltis_negatives" / "test"
_METRICS_PATH = _REPO_ROOT / "models" / "detector_eval.json"

IOU_MATCH = 0.5   # a prediction counts as a hit when IoU with a GT box >= this
# COCO-style sweep. AP@0.5 alone rewards a box that merely lands on the right
# leaf; the strict end asks whether it is the right box.
IOU_SWEEP = [round(0.5 + 0.05 * i, 2) for i in range(10)]


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


def _match_predictions(predictions: list, ground_truth: list,
                       iou_match: float = IOU_MATCH) -> list:
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
        if best_i >= 0 and best_iou >= iou_match:
            flags.append(1)
            unmatched.pop(best_i)
        else:
            flags.append(0)
    return flags


def _ap_at(per_image: list, n_gt: int, iou_match: float) -> float:
    """AP over the whole set at one IoU threshold."""
    scored = []
    for preds, gt in per_image:
        flags = _match_predictions(preds, gt, iou_match)
        confs = sorted((c for c, _ in preds), reverse=True)
        scored.extend(zip(confs, flags))
    ranked = [f for _, f in sorted(scored, key=lambda cf: cf[0], reverse=True)]
    return _average_precision(ranked, n_gt)


def _trivial_box_ap(per_image: list, n_gt: int) -> dict:
    """What a detector that learned NOTHING would score on this test set.

    The ground-truth boxes here cover ~40% of the image, one per photo, and the
    subject is centred because people centre subjects. So a constant box —
    same size, same place, every image, no model involved — already clears the
    IoU 0.5 bar on most of them. Publishing AP@0.5 without this number invites
    the reader to assume the floor is zero. It is not.

    Measured on the positive images only: this baseline fires on every
    background too, so its specificity is 0 and it is not a detector at all.
    """
    sides = []
    for _preds, gt in per_image:
        for b in gt:
            sides.append(((b.x2 - b.x1) + (b.y2 - b.y1)) / 2)
    if not sides:
        return {}
    side = sorted(sides)[len(sides) // 2]           # median GT side
    const = BBox(0.5 - side/2, 0.5 - side/2, 0.5 + side/2, 0.5 + side/2)
    positives = [([(1.0, const)], gt) for _preds, gt in per_image if gt]
    return {
        "box_side": round(side, 3),
        "ap50": round(_ap_at(positives, n_gt, IOU_MATCH), 4),
        "note": "constant centred box, no model; fires on every background "
                "(specificity 0). The floor AP@0.5 has to clear to mean anything.",
    }


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
    detector = HelopeltisDamageDetector()
    if not detector.available:
        print("[eval] Detector model not available — skipping.")
        print("       (expected models/helopeltis_damage.onnx + onnxruntime)")
        return None

    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"[eval] Missing dependency ({e}) — skipping.")
        return None

    # Positives come from the capture-session split, not the Roboflow folders:
    # those scatter photos taken seconds apart across train and test, so the
    # numbers below would describe near-duplicates of training images.
    if not _SPLIT_TEST.exists():
        raise SystemExit(
            f"Missing {_SPLIT_TEST.name}. Run `python -m src.data split` first — "
            "the Roboflow test split leaks near-duplicates from training "
            "(see README, 'The split, and what it cost').")
    # splitlines(), not split(): repository paths can contain spaces.
    images = [Path(ln) for ln in _SPLIT_TEST.read_text(encoding="utf-8").splitlines()
              if ln.strip()]
    missing = [p for p in images if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} listed test images not found, e.g. {missing[0]}")
    images = sorted(images)
    if not images:
        print(f"[eval] No test images listed in {_SPLIT_TEST} — skipping.")
        return None

    # Held-out backgrounds. Missing is not fatal — the run degrades to the old
    # positives-only measurement — but the precision figure is then not one to
    # quote, so say so loudly.
    backgrounds = sorted((_NEG_TEST_DIR / "images").glob("*.jpg"))
    if not backgrounds:
        print(f"[eval] WARNING: no background images in {_NEG_TEST_DIR / 'images'}.")
        print("       Precision below is measured on positives only and cannot")
        print("       detect false positives. Run: python -m src.data negatives --test")

    backgrounds_set = set(backgrounds)
    per_image = []          # (predictions, ground truth) for every image
    n_gt = 0
    n_pred = 0
    bg_fp = 0               # false positives raised on insect-free images
    bg_dirty = 0            # background images with at least one detection
    for img_path in images + backgrounds:
        is_background = img_path in backgrounds_set
        # Labels sit beside the images: .../<split>/images/x.jpg -> .../labels/x.txt
        gt = _load_ground_truth(img_path.parent.parent / "labels" / f"{img_path.stem}.txt")
        n_gt += len(gt)

        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        preds = [(d.confidence, d.bbox) for d in detector.detect(image_rgb)]
        n_pred += len(preds)
        per_image.append((preds, gt))

        # A background has no GT, so every prediction on it is a false positive.
        if is_background:
            bg_fp += len(preds)
            bg_dirty += 1 if preds else 0

    # Precision / recall at the operating point (every returned detection is
    # already above the detector's confidence threshold).
    flags = []
    for preds, gt in per_image:
        flags.extend(_match_predictions(preds, gt, iou_match))
    tp = sum(flags)
    fp = len(flags) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0

    # AP@0.5 is the headline; the sweep is the honest version of it. A box that
    # covers the right leaf clears 0.5 easily — the strict end is where sloppy
    # localisation shows up.
    ap_by_iou = {f"{t:.2f}": round(_ap_at(per_image, n_gt, t), 4) for t in IOU_SWEEP}
    ap = ap_by_iou[f"{iou_match:.2f}"]
    ap_50_95 = sum(ap_by_iou.values()) / len(ap_by_iou)

    # Specificity: share of insect-free images the detector leaves clean. This
    # is the number that moves when the model stops firing unconditionally.
    specificity = (len(backgrounds) - bg_dirty) / len(backgrounds) if backgrounds else None

    return {
        "path": "integrated ONNX (HelopeltisDamageDetector + NMS)",
        "split": "capture session — no test photo within 120s or 10 sequence "
                 "positions of a training photo",
        "test_set": "mixed (positives + held-out backgrounds)" if backgrounds
                    else "positives only — precision NOT meaningful",
        "n_images": len(images) + len(backgrounds),
        "n_positive_images": len(images),
        "n_background_images": len(backgrounds),
        "n_ground_truth": n_gt,
        "n_predictions": n_pred,
        "conf_threshold": detector.CONF_THRESHOLD,
        "iou_match": iou_match,
        "true_positives": tp,
        "false_positives": fp,
        "background_false_positives": bg_fp,
        "background_images_with_detection": bg_dirty,
        "specificity": round(specificity, 4) if specificity is not None else None,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "ap50": round(ap, 4),
        "ap50_95": round(ap_50_95, 4),
        "ap_by_iou": ap_by_iou,
        "trivial_baseline": _trivial_box_ap(per_image, n_gt),
    }


def evaluate_detector_cli():
    metrics = evaluate()
    if metrics is None:
        return

    _METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print("=" * 60)
    print("Integrated detector — test-set evaluation (IoU 0.5)")
    print("=" * 60)
    print(f"Test set      : {metrics['test_set']}")
    print(f"Images        : {metrics['n_images']} "
          f"({metrics['n_positive_images']} positive + "
          f"{metrics['n_background_images']} background)")
    print(f"Ground truth  : {metrics['n_ground_truth']} boxes")
    print(f"Predictions   : {metrics['n_predictions']} (conf >= {metrics['conf_threshold']})")
    print(f"TP / FP       : {metrics['true_positives']} / {metrics['false_positives']}")
    print(f"Precision     : {metrics['precision']:.3f}")
    print(f"Recall        : {metrics['recall']:.3f}")
    print(f"AP@0.5        : {metrics['ap50']:.3f}")
    print(f"AP@0.5:0.95   : {metrics['ap50_95']:.3f}   <- the one that judges localisation")
    baseline = metrics.get("trivial_baseline") or {}
    if baseline:
        print(f"  floor       : {baseline['ap50']:.3f} AP@0.5 for a constant centred "
              f"box ({baseline['box_side']:.2f} side, no model)")
    print(f"Split         : {metrics['split']}")
    if metrics["specificity"] is not None:
        print("-" * 60)
        print(f"Background FPs: {metrics['background_false_positives']} boxes on "
              f"{metrics['background_images_with_detection']}/"
              f"{metrics['n_background_images']} insect-free images")
        print(f"Specificity   : {metrics['specificity']:.3f} "
              f"(share of backgrounds left clean)")
    print(f"\nSaved -> {_METRICS_PATH.relative_to(_REPO_ROOT)}")


# ============================================================================
# 2. Geographic control for the Helopeltis classifier
# ============================================================================

CONTROL_DIR = FAM / "_geo_control"
METRICS_PATH = MODELS_DIR / "helopeltis_geo_control.json"

# Same region list the training download uses, imported rather than copied so
# the control can never drift away from what the negatives cover.
PLACES = HELOPELTIS_REGIONS
N_PER_PLACE = 30


def download_control():
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    have = len(list(CONTROL_DIR.glob("*.jpg")))
    if have >= 150:
        print(f"{have} control images already on disk — skipping download")
        return

    for place in PLACES:
        pid, resolved = resolve_place(place)
        if pid is None:
            print(f"[warn] could not resolve place '{place}'")
            continue
        got = 0
        for rec in fetch(MIRIDAE_TAXON, N_PER_PLACE,
                         without_taxon_id=HELOPELTIS_TAXON, place_id=pid):
            path = CONTROL_DIR / f"geo_{rec['obs_id']}.jpg"
            if path.exists():
                continue
            try:
                path.write_bytes(urllib.request.urlopen(
                    urllib.request.Request(rec["url"], headers=UA), timeout=60).read())
                got += 1
            except Exception as e:
                print(f"  [skip] {rec['obs_id']}: {e}")
        print(f"{resolved:24s} +{got:3d}")


def drop_seen_images():
    """Delete control images that also live in a training folder.

    The first `other_miridae` pull had no place filter, so a handful of its
    globally top-voted observations are the same ones the control later fetched
    from Asian places. A control image the model trained on measures memory, not
    generalisation — and the leak is silent, which is what makes it dangerous.
    Pruning the control (rather than the training set) keeps the measurement
    honest at the cost of a few test images.
    """
    trained = {p.stem.split("_")[-1]
               for folder in ("other_miridae", "miridae", "helopeltis")
               for p in (FAM / folder).glob("*.jpg")}
    leaked = [p for p in CONTROL_DIR.glob("*.jpg") if p.stem.split("_")[-1] in trained]
    for p in leaked:
        p.unlink()
    if leaked:
        print(f"[control] removed {len(leaked)} images also present in training folders")
    return len(leaked)


def evaluate_geo_cli():
    # PyTorch is only needed here. Importing it inside the function keeps the
    # detector evaluation (and its tests) runnable with the light inference
    # install: ONNX Runtime, NumPy, Pillow.
    import torch
    from src.train import TASK_CLASSES, build_model, eval_tf

    download_control()
    drop_seen_images()
    images = sorted(CONTROL_DIR.glob("*.jpg"))
    if not images:
        print("[eval] No control images — nothing to measure.")
        return

    weights = MODELS_DIR / "helopeltis_id.pt"
    if not weights.exists():
        print(f"[eval] {weights} not found — run: "
              f"python -m src.train insects --task helopeltis")
        return

    classes = TASK_CLASSES["helopeltis"]
    model = build_model(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    counts = {c: 0 for c in classes}
    with torch.no_grad():
        for path in images:
            x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0)
            counts[classes[int(model(x).argmax(1))]] += 1

    n = len(images)
    fp_rate = counts["helopeltis"] / n

    # The number to compare against: false positives on the ordinary test split,
    # where the negatives are mostly American and European.
    reference = None
    metrics_path = MODELS_DIR / "helopeltis_id_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        cm, cls = m["confusion_matrix"], m["classes"]
        i, j = cls.index("other_miridae"), cls.index("helopeltis")
        reference = cm[i][j] / sum(cm[i])

    print("\n" + "=" * 62)
    print("Geographic control — mirids from Helopeltis's own regions")
    print("=" * 62)
    print(f"Control images        : {n}")
    for c in classes:
        print(f"  called {c:16s}: {counts[c]:4d}  ({counts[c]/n:5.1%})")
    print(f"\nFalse-positive rate   : {fp_rate:.3f}  (same-region mirids)")
    if reference is not None:
        print(f"Reference (test split): {reference:.3f}  (mixed-geography mirids)")
        ratio = fp_rate / reference if reference else float("inf")
        print(f"Ratio                 : {ratio:.1f}x")
        verdict = ("geography explains much of the score — the headline numbers "
                   "are inflated" if ratio >= 2
                   else "holds up: the model is not simply reading the region")
        print(f"\nVerdict: {verdict}")

    METRICS_PATH.write_text(json.dumps({
        "n_control_images": n,
        "places": PLACES,
        "predicted": counts,
        "false_positive_rate_same_region": round(fp_rate, 4),
        "false_positive_rate_test_split": round(reference, 4) if reference else None,
        "note": ("Mirids (Helopeltis excluded) photographed in the same regions as "
                 "the Helopeltis training images. Measures whether the classifier "
                 "keys on the insect or on regional background."),
    }, indent=2) + "\n")
    print(f"\nSaved -> {METRICS_PATH.relative_to(MODELS_DIR.parent)}")


# ============================================================================
# Command line
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("detector", help="damage detector, mixed test set")
    sub.add_parser("geo", help="Helopeltis classifier, geographic control")
    args = ap.parse_args()
    if args.command == "detector":
        evaluate_detector_cli()
    else:
        evaluate_geo_cli()


if __name__ == "__main__":
    main()
