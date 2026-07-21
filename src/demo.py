"""
demo.py — End-to-end pipeline demo

Simulates the full server-side inference pipeline:
  detector output → FusionEngine → Observations → spatial verdict → JSON export

Run (from the repository root):
    python -m src.demo

Demos 1-3 need no models or dataset (hand-crafted detections). Demo 4 runs the
real detector + flush classifier on dataset images when they are available.
"""

import json
from datetime import datetime
from pathlib import Path

from src.who_where_when import FLUSH_STAGES
from src.fusion import (
    FusionEngine, FlushClassifier, HelopeltisDetector,
    InsectDetection, BBox,
)
from src.spatial import FieldExtent, assess_infestation

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_IMAGE_DIR = _REPO_ROOT / "data" / "raw" / "helopeltis" / "test" / "images"

_FALLBACK_STAGE = "new_flush"

# Built on first use: constructing it imports TensorFlow, which this demo
# never needs when it runs on synthetic detections (no image to classify).
_flush_clf = None


def resolve_stage(image_rgb=None) -> str:
    """Return flush stage from classifier if available, else fallback constant."""
    global _flush_clf

    if image_rgb is None:
        print(f"[demo] No image to classify — using hardcoded stage='{_FALLBACK_STAGE}'")
        return _FALLBACK_STAGE

    if _flush_clf is None:
        _flush_clf = FlushClassifier()
    if _flush_clf.available:
        return _flush_clf.predict(image_rgb)

    print(f"[demo] FlushClassifier not loaded — using hardcoded stage='{_FALLBACK_STAGE}'")
    return _FALLBACK_STAGE


def where_verdict(observations, extent=None):
    """Run the spatial WHERE (periphery vs core) over pest observations.

    Uses the detection positions carried by each Observation. ``extent`` is the
    monitored area; for a single image the frame ([0,1]²) is used, so the
    verdict reads "edge vs centre of the frame". With field-level GPS positions
    the same function gives true field periphery/core.
    """
    if extent is None:
        extent = FieldExtent(0.0, 0.0, 1.0, 1.0)   # the frame is the monitored area
    points = [o.position for o in observations if o.role == "pest" and o.position]
    report = assess_infestation(points, extent=extent)
    print(f"\nWHERE (spatial): {report.verdict}")
    print(f"  periphery={report.n_periphery}  core={report.n_core}  (of {report.total} pest)")
    return report


# ── Mock model outputs ───────────────────────────────────────────────────────

def mock_insects() -> list[InsectDetection]:
    """Simulates YOLOv8-n predictions on a tea-garden image."""
    return [
        InsectDetection("Helopeltis spp.", BBox(0.42, 0.45, 0.58, 0.60), confidence=0.82),  # centre → core
        InsectDetection("bees",            BBox(0.60, 0.10, 0.80, 0.35), confidence=0.91),
        InsectDetection("Thrips",          BBox(0.10, 0.70, 0.25, 0.85), confidence=0.58),  # → review
    ]


# ── Demo 1: single image ─────────────────────────────────────────────────────

def demo_image():
    print("=" * 60)
    print("DEMO 1 — Single image fusion")
    print("=" * 60)

    engine = FusionEngine()
    stage = resolve_stage()

    observations = engine.fuse_frame(
        insects=mock_insects(),
        stage=stage,
        timestamp=datetime(2026, 6, 16, 9, 30),
    )

    print(f"\nFlush stage : {stage} — {FLUSH_STAGES[stage]}")
    print(f"Observations: {len(observations)}\n")

    for obs in observations:
        print(obs.summary())

    where_verdict(observations)

    review = engine.drain_review_queue()
    if review:
        print(f"\nQueued for review ({len(review)} item/s):")
        for item in review:
            print(f"  [{item['label']}] conf={item['confidence']:.2f} — {item['reason']}")

    return observations


# ── Demo 2: video sliding window ─────────────────────────────────────────────

def demo_video():
    print("\n" + "=" * 60)
    print("DEMO 2 — Video sliding-window fusion (5 frames)")
    print("=" * 60)

    engine = FusionEngine()
    stage = resolve_stage()

    # Helopeltis appears in every frame; bee appears only in frame 0.
    # After the window, bee should be filtered out (single-frame hit).
    frames = []
    for i in range(5):
        insects = [InsectDetection("Helopeltis spp.", BBox(0.42, 0.45, 0.58, 0.60), 0.80 + i * 0.01)]
        if i == 0:
            insects.append(InsectDetection("bees", BBox(0.60, 0.10, 0.80, 0.35), 0.88))
        frames.append(insects)

    stable = []
    for insects in frames:
        stable = engine.push_frame(insects, stage, window=5)

    print(f"\nStable observations after {len(frames)} frames:\n")
    for obs in stable:
        print(obs.summary())
    print("(Bee suppressed — appeared in only 1 of 5 frames)")


# ── Demo 3: JSON export ───────────────────────────────────────────────────────

def demo_export(observations):
    print("\n" + "=" * 60)
    print("DEMO 3 — JSON export (database-ready payload)")
    print("=" * 60)
    payload = [obs.to_dict() for obs in observations]
    print(json.dumps(payload, indent=2))


# ── Demo 4: real inference on a dataset image ────────────────────────────────

def demo_real(max_images: int = 3):
    """Run the real YOLOv8-n detector + flush-stage classifier end-to-end.

    Needs the ONNX detector (``models/helopeltis_phase1.onnx``) and the optional
    heavy deps (onnxruntime, Pillow, numpy). When they are missing the function
    skips cleanly instead of failing, so the stdlib-only demos above still run.
    """
    print("\n" + "=" * 60)
    print("DEMO 4 — Real inference on dataset images")
    print("=" * 60)

    detector = HelopeltisDetector()
    if not detector.available:
        print("[demo] Detector model not available — skipping real inference.")
        print("       (expected models/helopeltis_phase1.onnx + onnxruntime)")
        return

    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"[demo] Missing dependency for real inference ({e}) — skipping.")
        return

    images = sorted(_REAL_IMAGE_DIR.glob("*.jpg"))[:max_images]
    if not images:
        print(f"[demo] No images found in {_REAL_IMAGE_DIR} — skipping.")
        return

    engine = FusionEngine()

    for path in images:
        image_rgb = np.array(Image.open(path).convert("RGB"))
        insects = detector.detect(image_rgb)
        stage = resolve_stage(image_rgb)
        observations = engine.fuse_frame(insects, stage=stage)

        print(f"\n{path.name}")
        print(f"  detector : {len(insects)} insect(s), stage={stage}")
        for obs in observations:
            print(f"  -> {obs.summary()}")
        if not observations and insects:
            print("  (detections below publish threshold — routed to review)")

    review = engine.drain_review_queue()
    if review:
        print(f"\nQueued for review ({len(review)} item/s):")
        for item in review:
            print(f"  [{item['label']}] conf={item['confidence']:.2f} — {item['reason']}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Windows consoles default to cp1252; the demo prints em-dashes and box
    # rules, so force UTF-8 output and degrade gracefully rather than crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    obs = demo_image()
    demo_video()
    demo_export(obs)
    demo_real()
