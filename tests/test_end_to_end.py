"""Photo in, Observation out, with the real models.

Skipped until `python -m src.data download` has installed the models and the
test images, so the unit tests still run on a bare clone (and in CI).
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
TEST_IMAGES = REPO / "data" / "raw" / "helopeltis" / "test" / "images"

pytestmark = pytest.mark.skipif(
    not ((MODELS / "helopeltis_damage.onnx").exists()
         and (MODELS / "flush_stage.onnx").exists()
         and TEST_IMAGES.is_dir()),
    reason="run `python -m src.data download` first")


def test_photo_becomes_an_observation_with_stage_and_position():
    np = pytest.importorskip("numpy")
    from PIL import Image

    from src.pipeline import FlushClassifier, FusionEngine, HelopeltisDamageDetector

    detector, flush = HelopeltisDamageDetector(), FlushClassifier()
    assert detector.available and flush.available

    image = np.array(Image.open(sorted(TEST_IMAGES.glob("*.jpg"))[0]).convert("RGB"))
    detections = detector.detect(image)
    stage = flush.predict(image)
    assert stage in FlushClassifier.CLASSES
    assert detections, "the detector should find the damage on a positive test image"

    observations = FusionEngine().fuse_frame(detections, stage=stage)
    assert observations
    obs = observations[0]
    assert obs.role == "pest"
    assert all(0.0 <= v <= 1.0 for v in obs.position)
    assert set(obs.to_dict()) >= {"insect", "stage", "position", "is_alert"}
