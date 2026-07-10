"""
demo.py — End-to-end pipeline demo with synthetic data

Simulates the full server-side inference pipeline:
  mock detector output → FusionEngine → Observations → JSON export

Run (from the repository root):
    python -m src.demo

No models or dataset needed — all detections are hand-crafted to cover the
interesting cases (alert, non-alert, low-confidence review).
"""

import json
from datetime import datetime

from src.who_where_when import FLUSH_STAGES
from src.fusion import FusionEngine, FlushClassifier, InsectDetection, ZoneDetection, BBox

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


# ── Mock model outputs ───────────────────────────────────────────────────────

def mock_insects() -> list[InsectDetection]:
    """Simulates YOLOv8-n predictions on a tea-garden image."""
    return [
        InsectDetection("Helopeltis spp.", BBox(0.30, 0.35, 0.55, 0.65), confidence=0.82),
        InsectDetection("bees",            BBox(0.60, 0.10, 0.80, 0.35), confidence=0.91),
        InsectDetection("Thrips",          BBox(0.10, 0.70, 0.25, 0.85), confidence=0.58),  # → review
    ]


def mock_zones() -> list[ZoneDetection]:
    """Simulates plant-zone segmentation on the same image."""
    return [
        ZoneDetection("Efficiency", BBox(0.20, 0.25, 0.65, 0.75), confidence=0.88),
        ZoneDetection("Attraction", BBox(0.55, 0.05, 0.90, 0.40), confidence=0.79),
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
        zones=mock_zones(),
        stage=stage,
        timestamp=datetime(2026, 6, 16, 9, 30),
        gps=(0.0, 0.0),   # example coordinates
    )

    print(f"\nFlush stage : {stage} — {FLUSH_STAGES[stage]}")
    print(f"Observations: {len(observations)}\n")

    for obs in observations:
        print(obs.summary())

    review = engine.drain_review_queue()
    if review:
        print(f"Queued for review ({len(review)} item/s):")
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
        insects = [InsectDetection("Helopeltis spp.", BBox(0.30, 0.35, 0.55, 0.65), 0.80 + i * 0.01)]
        if i == 0:
            insects.append(InsectDetection("bees", BBox(0.60, 0.10, 0.80, 0.35), 0.88))
        frames.append((insects, mock_zones()))

    stable = []
    for insects, zones in frames:
        stable = engine.push_frame(insects, zones, stage, window=5)

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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    obs = demo_image()
    demo_video()
    demo_export(obs)
