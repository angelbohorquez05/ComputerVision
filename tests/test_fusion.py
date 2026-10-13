import math

import pytest

from src.pipeline import (
    warn_if_stale,
    BBox,
    FlushClassifier,
    FusionEngine,
    HelopeltisDamageDetector,
    DamageDetection,
    LOW_CONF_THRESHOLD,
)


def _decode(rows, imgsz=320, conf=0.25):
    return HelopeltisDamageDetector._decode_rows(rows, imgsz, conf)


def test_high_confidence_pest_becomes_observation():
    engine = FusionEngine()
    detections = [DamageDetection("Helopeltis spp.", BBox(0.4, 0.4, 0.6, 0.6), 0.9)]
    observations = engine.fuse_frame(detections, stage="new_flush")
    assert len(observations) == 1
    assert observations[0].is_alert()
    assert engine.drain_review_queue() == []


def test_observation_carries_bbox_centre_as_position():
    engine = FusionEngine()
    detections = [DamageDetection("Helopeltis spp.", BBox(0.4, 0.4, 0.6, 0.6), 0.9)]
    obs = engine.fuse_frame(detections, stage="new_flush")[0]
    assert obs.position == (0.5, 0.5)


def test_low_confidence_goes_to_review_queue():
    engine = FusionEngine()
    detections = [DamageDetection("Helopeltis spp.", BBox(0.4, 0.4, 0.6, 0.6), LOW_CONF_THRESHOLD - 0.1)]
    observations = engine.fuse_frame(detections, stage="new_flush")
    assert observations == []
    queue = engine.drain_review_queue()
    assert len(queue) == 1 and queue[0]["reason"] == "low_confidence"


def test_sliding_window_suppresses_single_frame_hit():
    engine = FusionEngine()
    helo = DamageDetection("Helopeltis spp.", BBox(0.4, 0.4, 0.6, 0.6), 0.9)
    bee = DamageDetection("bees", BBox(0.1, 0.1, 0.2, 0.2), 0.9)
    stable = []
    for i in range(5):
        detections = [helo] + ([bee] if i == 0 else [])
        stable = engine.push_frame(detections, stage="new_flush", window=5)
    labels = {o.insect for o in stable}
    assert "Helopeltis spp." in labels
    assert "bees" not in labels


def test_flush_decision_follows_the_calibrated_threshold():
    # The threshold is a calibration result (0.50 since 2026-07-24, re-swept on
    # the capture-session split), so the test asserts against the constant
    # rather than a hard-coded value: it must keep tracking a re-calibration.
    thr = FlushClassifier.NEW_FLUSH_THRESHOLD
    just_over = math.log(thr + 0.02) - math.log(1 - (thr + 0.02))
    just_under = math.log(thr - 0.02) - math.log(1 - (thr - 0.02))
    assert FlushClassifier._decide([just_over, 0.0]) == "new_flush"
    assert FlushClassifier._decide([just_under, 0.0]) == "mature_leaf"
    assert FlushClassifier._decide([5.0, 0.0]) == "new_flush"    # unambiguous
    assert FlushClassifier._decide([-5.0, 0.0]) == "mature_leaf"


def test_nms_collapses_overlapping_boxes():
    # The detector emits one raw prediction per anchor, so many near-identical
    # boxes cover the same damage region. NMS must keep only the highest-confidence one.
    overlapping = [
        (BBox(0.30, 0.30, 0.60, 0.60), 0.70),
        (BBox(0.31, 0.31, 0.61, 0.61), 0.90),  # best — should survive
        (BBox(0.29, 0.29, 0.59, 0.59), 0.80),
    ]
    kept = HelopeltisDamageDetector._nms(overlapping, iou_threshold=0.45)
    assert len(kept) == 1
    assert kept[0][1] == 0.90


def test_nms_keeps_distinct_boxes():
    # Two detections far apart must both survive NMS.
    distinct = [
        (BBox(0.05, 0.05, 0.20, 0.20), 0.90),
        (BBox(0.70, 0.70, 0.90, 0.90), 0.85),
    ]
    kept = HelopeltisDamageDetector._nms(distinct, iou_threshold=0.45)
    assert len(kept) == 2


# --- Decoding the raw model output -------------------------------------------
# The model emits centre-form boxes, and the coordinate units depend on how the
# ONNX was exported. Guessing wrong does not raise anything: it silently moves
# every box into a corner and takes AP@0.5 to ~0. These tests are the guard.

def test_decode_converts_centre_form_to_corners():
    (bbox, conf), = _decode([(0.5, 0.5, 0.2, 0.2, 0.9)])
    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == pytest.approx((0.4, 0.4, 0.6, 0.6))
    assert conf == 0.9


def test_decode_leaves_a_normalised_export_untouched():
    # Every coordinate already in [0, 1] — dividing again would shrink the box.
    (bbox, _), = _decode([(0.5, 0.5, 0.2, 0.2, 0.9)], imgsz=320)
    assert bbox.x2 == pytest.approx(0.6)


def test_decode_normalises_a_pixel_space_export():
    # The same box as above, in input-pixel units at imgsz=320.
    (bbox, _), = _decode([(160.0, 160.0, 64.0, 64.0, 0.9)], imgsz=320)
    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == pytest.approx((0.4, 0.4, 0.6, 0.6))


def test_decode_agrees_across_export_conventions():
    # The regression that matters: one box, two ways of writing it, one answer.
    (norm, _), = _decode([(0.5, 0.5, 0.2, 0.2, 0.9)], imgsz=640)
    (pixel, _), = _decode([(320.0, 320.0, 128.0, 128.0, 0.9)], imgsz=640)
    assert (norm.x1, norm.y1, norm.x2, norm.y2) == pytest.approx(
        (pixel.x1, pixel.y1, pixel.x2, pixel.y2))


# --- Deployed-model freshness -------------------------------------------------
# Training writes a .pt; a separate conversion produces the file the pipeline
# loads. Nothing forces the second step, and when it is skipped the pipeline
# serves an old model while the metrics describe a new one.

def test_stale_deployed_model_is_reported(tmp_path):
    deployed = tmp_path / "flush_stage.onnx"
    trained = tmp_path / "flush_stage.pt"
    deployed.write_bytes(b"old")
    trained.write_bytes(b"new")
    import os
    os.utime(deployed, (1000, 1000))
    os.utime(trained, (99000, 99000))       # trained much later
    assert warn_if_stale(deployed, trained) is True


def test_fresh_deployed_model_is_not_reported(tmp_path):
    deployed = tmp_path / "flush_stage.onnx"
    trained = tmp_path / "flush_stage.pt"
    trained.write_bytes(b"new")
    deployed.write_bytes(b"converted")
    import os
    os.utime(trained, (1000, 1000))
    os.utime(deployed, (1030, 1030))        # converted right after training
    assert warn_if_stale(deployed, trained) is False
    # A missing counterpart is not a staleness claim.
    assert warn_if_stale(deployed, tmp_path / "absent.pt") is False


def test_decode_drops_rows_below_the_confidence_gate():
    rows = [(0.5, 0.5, 0.2, 0.2, 0.90),   # keep
            (0.2, 0.2, 0.1, 0.1, 0.10)]   # below 0.25
    assert len(_decode(rows)) == 1
    assert _decode([]) == []
