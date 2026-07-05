"""
fusion.py — WHO×WHERE×WHEN Fusion Engine

Combines insect detections (WHO) with plant-zone detections (WHERE) and the
current flush stage (WHEN) into Observation events.

Architecture note: this runs server-side in the inference service. A mobile app
uploads media → the service runs the two YOLO/classifier models → this module
fuses the raw predictions → Observations are stored in a database.

Pipeline:
    [YOLOv8-n insect detector]  → list[InsectDetection]
    [Plant zone segmenter]       → list[ZoneDetection]
    [Flush-stage classifier]     → str  (dormant | new_flush | mature_leaf)
                                    ↓
                            FusionEngine.fuse_frame()
                                    ↓
                             list[Observation]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.who_where_when import Observation, ZONES, FLUSH_STAGES

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ── Helopeltis detector wrapper (ONNX) ───────────────────────────────────────

class HelopeltisDetector:
    """
    Wraps the YOLOv8-n Helopeltis ONNX model for CPU inference.

    Usage:
        det = HelopeltisDetector()
        insects = det.detect(image_rgb)   # returns list[InsectDetection]

    Falls back to an empty list if the model file is not yet available.
    """

    _MODEL_PATH = _MODELS_DIR / "helopeltis_phase1.onnx"
    CONF_THRESHOLD = 0.25

    def __init__(self):
        self._session = None
        if self._MODEL_PATH.exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    str(self._MODEL_PATH),
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
            except Exception as e:
                print(f"[HelopeltisDetector] Could not load model: {e}")

    @property
    def available(self) -> bool:
        return self._session is not None

    def detect(self, image_rgb, imgsz: int = 320) -> list:
        """Return list[InsectDetection] from an RGB numpy array."""
        if not self.available:
            return []
        import numpy as np
        from PIL import Image

        img = Image.fromarray(image_rgb).resize((imgsz, imgsz))
        tensor = np.array(img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        raw = self._session.run(None, {self._input_name: tensor})[0]  # (1, 5, anchors) or (1, anchors, 5)
        # YOLOv8 ONNX exports vary: transpose to (anchors, 5) regardless of shape
        out = raw[0]
        if out.shape[0] == 5:      # shape (5, anchors) → transpose to (anchors, 5)
            out = out.T
        detections = []
        for pred in out:           # each row: cx cy w h conf
            conf = float(pred[4])
            if conf < self.CONF_THRESHOLD:
                continue
            cx, cy, w, h = pred[:4] / imgsz
            detections.append(InsectDetection(
                label="Helopeltis spp.",
                bbox=BBox(cx - w/2, cy - h/2, cx + w/2, cy + h/2),
                confidence=conf,
            ))
        return detections


# ── Flush-stage classifier wrapper ───────────────────────────────────────────

class FlushClassifier:
    """
    Wraps the MobileNetV2 flush-stage TFLite model.

    Usage:
        clf = FlushClassifier()          # loads models/flush_stage.tflite
        stage = clf.predict(image_rgb)   # numpy array HxWx3, uint8

    Falls back to None if the model file is not yet available, so the
    FusionEngine can still run in demo mode with a hardcoded stage.
    """

    CLASSES = ["dormant", "new_flush", "mature_leaf"]
    _MODEL_PATH = _MODELS_DIR / "flush_stage.tflite"

    def __init__(self):
        self._interpreter = None
        if self._MODEL_PATH.exists():
            try:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(model_path=str(self._MODEL_PATH))
                self._interpreter.allocate_tensors()
                self._input_idx  = self._interpreter.get_input_details()[0]["index"]
                self._output_idx = self._interpreter.get_output_details()[0]["index"]
            except Exception as e:
                print(f"[FlushClassifier] Could not load model: {e}")

    @property
    def available(self) -> bool:
        return self._interpreter is not None

    def predict(self, image_rgb) -> Optional[str]:
        """Return stage string or None if model not loaded."""
        if not self.available:
            return None
        import numpy as np
        from PIL import Image
        img = Image.fromarray(image_rgb).resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((arr - mean) / std)[np.newaxis]    # (1, 224, 224, 3)
        self._interpreter.set_tensor(self._input_idx, tensor)
        self._interpreter.invoke()
        logits = self._interpreter.get_tensor(self._output_idx)[0]
        return self.CLASSES[int(logits.argmax())]

# Detections below this confidence are routed to the review queue instead of
# being published as Observations. 0.65 is the Phase 2 review-queue threshold.
LOW_CONF_THRESHOLD = 0.65


# ── Geometry primitives ──────────────────────────────────────────────────────

@dataclass
class BBox:
    """Normalised bounding box, all coords in [0, 1]."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


# ── Detection dataclasses ────────────────────────────────────────────────────

@dataclass
class InsectDetection:
    """One bounding-box prediction from the YOLOv8-n insect detector."""
    label: str           # e.g. "Helopeltis spp.", "bees"
    bbox: BBox
    confidence: float    # [0, 1]


@dataclass
class ZoneDetection:
    """One predicted plant-zone region (from segmentation or rule-based mapping)."""
    zone: str            # "Attraction" | "Retention" | "Efficiency"
    bbox: BBox
    confidence: float


# ── Fusion engine ────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Fuses InsectDetections + ZoneDetections + flush stage → Observations.

    Usage (single image):
        engine = FusionEngine()
        obs = engine.fuse_frame(insects, zones, stage="new_flush", gps=(...))

    Usage (video / frame stream):
        for frame in frames:
            stable_obs = engine.push_frame(frame.insects, frame.zones, stage)
        review = engine.drain_review_queue()
    """

    def __init__(self):
        self._review_queue: list[dict] = []
        self._frame_buffer: list[list[Observation]] = []

    # ── public API ───────────────────────────────────────────────────────────

    def fuse_frame(
        self,
        insects: list[InsectDetection],
        zones: list[ZoneDetection],
        stage: str,
        timestamp: Optional[datetime] = None,
        gps: Optional[tuple] = None,
    ) -> list[Observation]:
        """
        Fuse a single frame's detections into Observation records.

        Low-confidence insects (< LOW_CONF_THRESHOLD) are not published;
        they are appended to the review queue for human labelling / retraining.
        """
        if timestamp is None:
            timestamp = datetime.now()

        observations: list[Observation] = []

        for ins in insects:
            if ins.confidence < LOW_CONF_THRESHOLD:
                self._review_queue.append({
                    "reason":     "low_confidence",
                    "label":      ins.label,
                    "confidence": round(ins.confidence, 4),
                    "timestamp":  timestamp.isoformat(),
                })
                continue

            matched = self._best_zone(ins.bbox, zones)
            if matched is None:
                self._review_queue.append({
                    "reason":     "no_zone_match",
                    "label":      ins.label,
                    "confidence": round(ins.confidence, 4),
                    "timestamp":  timestamp.isoformat(),
                })
                continue

            try:
                obs = Observation(
                    insect=ins.label,
                    zone=matched.zone,
                    stage=stage,
                    confidence=ins.confidence,
                    timestamp=timestamp,
                    gps=gps,
                )
                observations.append(obs)
            except ValueError:
                # Unknown zone or stage — skip silently; caller should validate inputs.
                pass

        return observations

    def push_frame(
        self,
        insects: list[InsectDetection],
        zones: list[ZoneDetection],
        stage: str,
        window: int = 5,
        **kwargs,
    ) -> list[Observation]:
        """
        Accumulate frames in a sliding window and return *stable* observations.

        A detection must appear in at least 2 of the last `window` frames to be
        published — this suppresses spurious hits from blurry or partial frames.
        """
        frame_obs = self.fuse_frame(insects, zones, stage, **kwargs)
        self._frame_buffer.append(frame_obs)
        if len(self._frame_buffer) > window:
            self._frame_buffer.pop(0)

        counts: dict[tuple, list[Observation]] = {}
        for frame in self._frame_buffer:
            for obs in frame:
                key = (obs.insect, obs.zone, obs.stage)
                counts.setdefault(key, []).append(obs)

        stable: list[Observation] = []
        seen: set[tuple] = set()
        for key, obs_list in counts.items():
            if len(obs_list) >= 2 and key not in seen:
                stable.append(obs_list[-1])   # most recent version of this event
                seen.add(key)

        return stable

    def drain_review_queue(self) -> list[dict]:
        """Return and clear all low-confidence detections queued for review."""
        queue = self._review_queue.copy()
        self._review_queue.clear()
        return queue

    def reset(self):
        """Clear frame buffer and review queue (call between unrelated video clips)."""
        self._frame_buffer.clear()
        self._review_queue.clear()

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _best_zone(
        insect_box: BBox,
        zones: list[ZoneDetection],
    ) -> Optional[ZoneDetection]:
        """
        Return the zone that best explains where the insect is.

        Priority:
          1. Zone whose bbox contains the insect centroid (spatial containment).
          2. If no containment match, fall back to highest IoU zone.
        """
        cx, cy = insect_box.cx, insect_box.cy
        contained = [z for z in zones if z.bbox.contains_point(cx, cy)]
        if contained:
            return max(contained, key=lambda z: z.confidence)

        if zones:
            return max(zones, key=lambda z: insect_box.iou(z.bbox))

        return None
