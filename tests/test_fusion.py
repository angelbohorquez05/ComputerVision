from src.fusion import (
    BBox,
    FlushClassifier,
    FusionEngine,
    HelopeltisDetector,
    InsectDetection,
    ZoneDetection,
    LOW_CONF_THRESHOLD,
)


def _full_zone():
    return [ZoneDetection("Efficiency", BBox(0.0, 0.0, 1.0, 1.0), 0.9)]


def test_high_confidence_pest_becomes_observation():
    engine = FusionEngine()
    insects = [InsectDetection("Helopeltis spp.", BBox(0.4, 0.4, 0.6, 0.6), 0.9)]
    observations = engine.fuse_frame(insects, _full_zone(), stage="new_flush")
    assert len(observations) == 1
    assert observations[0].is_alert()
    assert engine.drain_review_queue() == []


def test_low_confidence_goes_to_review_queue():
    engine = FusionEngine()
    insects = [InsectDetection("Helopeltis spp.", BBox(0.4, 0.4, 0.6, 0.6), LOW_CONF_THRESHOLD - 0.1)]
    observations = engine.fuse_frame(insects, _full_zone(), stage="new_flush")
    assert observations == []
    queue = engine.drain_review_queue()
    assert len(queue) == 1 and queue[0]["reason"] == "low_confidence"


def test_detection_without_zone_match_goes_to_review():
    engine = FusionEngine()
    insects = [InsectDetection("Helopeltis spp.", BBox(0.4, 0.4, 0.6, 0.6), 0.9)]
    observations = engine.fuse_frame(insects, [], stage="new_flush")
    assert observations == []
    queue = engine.drain_review_queue()
    assert len(queue) == 1 and queue[0]["reason"] == "no_zone_match"


def test_sliding_window_suppresses_single_frame_hit():
    engine = FusionEngine()
    zones = _full_zone()
    helo = InsectDetection("Helopeltis spp.", BBox(0.4, 0.4, 0.6, 0.6), 0.9)
    bee = InsectDetection("bees", BBox(0.1, 0.1, 0.2, 0.2), 0.9)
    stable = []
    for i in range(5):
        insects = [helo] + ([bee] if i == 0 else [])
        stable = engine.push_frame(insects, zones, stage="new_flush", window=5)
    labels = {o.insect for o in stable}
    assert "Helopeltis spp." in labels
    assert "bees" not in labels


def test_flush_threshold_favors_new_flush():
    # The high-risk class new_flush wins whenever P(new_flush) >= 0.35, even when
    # plain argmax would pick mature_leaf.
    assert FlushClassifier._decide([-0.405, 0.0]) == "new_flush"   # P(new_flush)=0.40
    assert FlushClassifier._decide([-1.386, 0.0]) == "mature_leaf"  # P(new_flush)=0.20
    assert FlushClassifier._decide([2.0, 0.0]) == "new_flush"       # clear new_flush


def test_nms_collapses_overlapping_boxes():
    # The detector emits one raw prediction per anchor, so many near-identical
    # boxes cover the same insect. NMS must keep only the highest-confidence one.
    overlapping = [
        (BBox(0.30, 0.30, 0.60, 0.60), 0.70),
        (BBox(0.31, 0.31, 0.61, 0.61), 0.90),  # best — should survive
        (BBox(0.29, 0.29, 0.59, 0.59), 0.80),
    ]
    kept = HelopeltisDetector._nms(overlapping, iou_threshold=0.45)
    assert len(kept) == 1
    assert kept[0][1] == 0.90


def test_nms_keeps_distinct_boxes():
    # Two insects far apart must both survive NMS.
    distinct = [
        (BBox(0.05, 0.05, 0.20, 0.20), 0.90),
        (BBox(0.70, 0.70, 0.90, 0.90), 0.85),
    ]
    kept = HelopeltisDetector._nms(distinct, iou_threshold=0.45)
    assert len(kept) == 2
