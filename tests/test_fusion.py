from src.fusion import (
    BBox,
    FusionEngine,
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
