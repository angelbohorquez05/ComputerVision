"""The runtime — everything that turns a photo into a verdict.

Three layers, in the order a photo passes through them:

  1. **Models**  ``HelopeltisDamageDetector`` reads the leaf (is there feeding
     damage?), ``FlushClassifier`` reads the growth stage (tender shoot or
     mature leaf — the WHEN, and the risk window).
  2. **Observation**  the WHO x WHERE x WHEN record, with the ecological role
     attached. ``FusionEngine`` builds these from raw predictions, and
     ``push_frame`` requires a detection to persist across several frames
     before it counts, which is what suppresses single-frame false positives
     in video.
  3. **Field verdict**  ``assess_infestation`` aggregates many Observations
     into periphery-vs-core, the spatial question the grower actually asks.

Naming, because it has misled people before: the detector finds the *damage*
Helopeltis leaves on a leaf, NOT the insect. Identifying the insect is a
separate model (see src/train.py, the ``helopeltis`` task).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ============================================================================
# 1. Observation — WHO x WHERE x WHEN
# ============================================================================

# ─── WHO: insect classes mapped to ecological role ───
# NOTE: here WHO is really "the pest is present", inferred from the *feeding
# damage* the detector finds on the leaf, not from identifying the insect (the
# dataset labels damage, not the bug; see docs/model-card.md).
# This role table is kept for when a true multi-species detector exists.
ECOLOGICAL_ROLE: dict[str, list[str]] = {
    "pollinator": [
        "bees", "Papilio xuthus", "Ampelophaga"
    ],
    "pest": [
        "Helopeltis", "Miridae", "aphids", "rice leaf caterpillar",
        "corn borer", "army worm", "Thrips"
    ],
    "beneficial": [
        "Apolygus lucorum"
    ],
}

# Flat lookup: insect name → role
_INSECT_TO_ROLE: dict[str, str] = {
    insect: role
    for role, insects in ECOLOGICAL_ROLE.items()
    for insect in insects
}

def get_role(insect_name: str) -> str:
    """Return the ecological role for a known insect, or 'unknown'."""
    for known_name, role in _INSECT_TO_ROLE.items():
        if known_name.lower() in insect_name.lower():
            return role
    return "unknown"


# ─── WHEN: tea crop flush stages ───
FLUSH_STAGES: dict[str, str] = {
    "new_flush":   "Tender new shoot (young leaf) — HIGH RISK for Helopeltis",
    "mature_leaf": "Mature leaf — lower risk",
}


# ─── The full metric ───
@dataclass
class Observation:
    """A single detection enriched with WHO / WHERE / WHEN.

    WHO   = insect + ecological role (assigned from ECOLOGICAL_ROLE).
    WHEN  = flush stage (young vs mature leaf).
    WHERE = ``position`` (x, y) of the detection in the monitored area. The
            periphery-vs-core verdict is a field-level aggregate computed from
            the positions of many Observations — see ``assess_infestation`` below.
    """
    # WHO
    insect: str
    role: str = field(init=False)           # assigned automatically from ECOLOGICAL_ROLE

    # WHEN
    stage: str                              # new_flush / mature_leaf

    # Metadata
    confidence: float

    # WHERE — detection position (frame-normalised coords, or field GPS)
    position: Optional[tuple] = None

    timestamp: datetime = field(default=None)

    def __post_init__(self):
        # Assign role from the registry; falls back to "unknown"
        self.role = get_role(self.insect)

        # Validate stage and confidence against known values
        if self.stage not in FLUSH_STAGES:
            raise ValueError(f"Unknown stage '{self.stage}'. Valid: {list(FLUSH_STAGES)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

        if self.timestamp is None:
            self.timestamp = datetime.now()

    def is_alert(self) -> bool:
        """True when a pest is detected on a young leaf (highest risk)."""
        return self.role == "pest" and self.stage == "new_flush"

    def summary(self) -> str:
        status = "[ALERT]" if self.is_alert() else "[OK]"
        where = f"({self.position[0]:.2f}, {self.position[1]:.2f})" if self.position else "n/a"
        return (
            f"{status}\n"
            f"  WHO  : {self.insect} ({self.role})\n"
            f"  WHEN : {self.stage} — {FLUSH_STAGES[self.stage]}\n"
            f"  WHERE: position {where}\n"
            f"  Conf : {self.confidence:.1%}\n"
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON export / database insert."""
        return {
            "insect":     self.insect,
            "role":       self.role,
            "stage":      self.stage,
            "confidence": round(self.confidence, 4),
            "is_alert":   self.is_alert(),
            "position":   list(self.position) if self.position else None,
            "timestamp":  self.timestamp.isoformat(),
        }


# ─── Smoke tests ───

# ============================================================================
# 2. Field verdict — periphery vs core
# ============================================================================

PERIPHERY = "periphery"
CORE = "core"

# Fraction of the normalised edge→centre depth below which a point counts as
# periphery. 0.25 → the outer quarter of the way in from any side is "edge".
DEFAULT_MARGIN = 0.25


@dataclass(frozen=True)
class FieldExtent:
    """Axis-aligned bounds of the monitored field (the 'system')."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self):
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("FieldExtent requires xmax > xmin and ymax > ymin")

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @classmethod
    def from_points(cls, points: list, pad: float = 0.0) -> "FieldExtent":
        """Derive an extent from sample points, optionally padded outward.

        Use when the field boundary is not known explicitly; pass the reference
        sampling/monitoring points (not only the pest hits) so the extent
        represents the whole surveyed area.
        """
        if not points:
            raise ValueError("Cannot derive a FieldExtent from zero points")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        w = (max(xs) - min(xs)) or 1.0
        h = (max(ys) - min(ys)) or 1.0
        return cls(
            min(xs) - pad * w, min(ys) - pad * h,
            max(xs) + pad * w, max(ys) + pad * h,
        )


def penetration(x: float, y: float, extent: FieldExtent) -> float:
    """How deep a point sits inside the field, in [0, 1].

    0.0 on (or outside) the boundary, 1.0 only at the exact centre. Defined as
    the normalised distance to the nearest edge, so it reads as "how far from
    the periphery toward the core".
    """
    dx = min(x - extent.xmin, extent.xmax - x)
    dy = min(y - extent.ymin, extent.ymax - y)
    if dx < 0 or dy < 0:
        return 0.0        # outside the field → treat as edge
    nx = dx / (extent.width / 2)
    ny = dy / (extent.height / 2)
    return max(0.0, min(1.0, min(nx, ny)))


def classify_location(
    x: float, y: float, extent: FieldExtent, margin: float = DEFAULT_MARGIN
) -> str:
    """Label a single position as PERIPHERY or CORE."""
    return PERIPHERY if penetration(x, y, extent) < margin else CORE


@dataclass
class InfestationReport:
    """Field-level verdict aggregated over all pest positions."""
    total: int
    n_periphery: int
    n_core: int
    breach: bool                       # True once any pest reaches the core
    locations: list = field(default_factory=list)   # per-point (x, y, label, penetration)

    @property
    def verdict(self) -> str:
        if self.total == 0:
            return "no pests detected"
        if self.breach:
            return "breach — pest reached the core of the system"
        return "contained — pests only at the periphery"

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "n_periphery": self.n_periphery,
            "n_core": self.n_core,
            "breach": self.breach,
            "verdict": self.verdict,
        }


def assess_infestation(
    points: list,
    extent: FieldExtent | None = None,
    margin: float = DEFAULT_MARGIN,
) -> InfestationReport:
    """Classify every pest position and return a field-level report.

    ``points`` is a list of (x, y). When ``extent`` is None it is derived from
    the points themselves — convenient for a demo, but in production pass the
    real field boundary so "edge" means the edge of the plot, not of the swarm.
    """
    if extent is None:
        extent = FieldExtent.from_points(points) if points else None

    locations = []
    n_core = 0
    for x, y in points:
        depth = penetration(x, y, extent)
        label = PERIPHERY if depth < margin else CORE
        if label == CORE:
            n_core += 1
        locations.append((x, y, label, round(depth, 4)))

    total = len(points)
    return InfestationReport(
        total=total,
        n_periphery=total - n_core,
        n_core=n_core,
        breach=n_core > 0,
        locations=locations,
    )

# ============================================================================
# 3. Models and fusion
# ============================================================================

# ── Helopeltis detector wrapper (ONNX) ───────────────────────────────────────

def warn_if_stale(deployed, trained, tolerance_seconds: int = 60) -> bool:
    """Warn when the file the pipeline loads is older than the weights it came from.

    Deployment is two steps: training writes a ``.pt``, an export produces the
    ``.onnx`` that actually gets loaded, and nothing forces the second step to
    happen. When it does not, the pipeline silently keeps serving the previous
    model while the metrics files describe the new one. That is not
    hypothetical: an interrupted training run once left a stale export
    installed, and the only symptom would have been numbers that did not match
    the code.

    Returns True when the deployed artifact looks stale. Mtime-based on purpose:
    stdlib only, so this runs in CI and costs nothing at import time.
    """
    if not (deployed.exists() and trained.exists()):
        return False
    lag = trained.stat().st_mtime - deployed.stat().st_mtime
    if lag > tolerance_seconds:
        print(f"[stale model] {deployed.name} is {lag/60:.0f} min older than "
              f"{trained.name}. The pipeline is loading a model that predates the "
              f"latest training run — re-run the export/conversion step.")
        return True
    return False


class HelopeltisDamageDetector:
    """
    Wraps the YOLOv8-n ONNX model for CPU inference.

    Detects the *feeding damage* Helopeltis leaves on tea leaves, NOT the insect
    itself — the training dataset labels damaged leaves, not the bug (see
    docs/model-card.md).

    Usage:
        det = HelopeltisDamageDetector()
        detections = det.detect(image_rgb)   # returns list[DamageDetection]

    Falls back to an empty list if the model file is not yet available.
    """

    _MODEL_PATH = _MODELS_DIR / "helopeltis_damage.onnx"
    CONF_THRESHOLD = 0.25
    # IoU above which two boxes are treated as the same object during NMS.
    IOU_THRESHOLD = 0.45
    # Fallback square input size if the model does not declare a static one.
    _DEFAULT_IMGSZ = 320

    def __init__(self):
        self._session = None
        self._imgsz = self._DEFAULT_IMGSZ
        if self._MODEL_PATH.exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    str(self._MODEL_PATH),
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                # Read the export's static input side (…, H, W); keep the default
                # only if the model was exported with a dynamic axis.
                shape = self._session.get_inputs()[0].shape
                if isinstance(shape[-1], int):
                    self._imgsz = shape[-1]
            except Exception as e:
                print(f"[HelopeltisDamageDetector] Could not load model: {e}")

    @property
    def available(self) -> bool:
        return self._session is not None

    def detect(self, image_rgb, imgsz: Optional[int] = None) -> list:
        """Return list[DamageDetection] from an RGB numpy array.

        The model exports normalised box coordinates (cx, cy, w, h all in
        [0, 1]) and one raw prediction per anchor, so overlapping anchors on the
        same insect are collapsed with non-max suppression before returning.
        """
        if not self.available:
            return []
        import numpy as np
        from PIL import Image

        if imgsz is None:
            imgsz = self._imgsz

        img = Image.fromarray(image_rgb).resize((imgsz, imgsz))
        tensor = np.array(img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        raw = self._session.run(None, {self._input_name: tensor})[0]  # (1, 5, anchors) or (1, anchors, 5)
        # YOLOv8 ONNX exports vary: transpose to (anchors, 5) regardless of shape
        out = raw[0]
        if out.shape[0] == 5:      # shape (5, anchors) → transpose to (anchors, 5)
            out = out.T

        boxes = self._decode_rows(out, imgsz, self.CONF_THRESHOLD)
        return [
            DamageDetection(label="Helopeltis spp.", bbox=bbox, confidence=conf)
            for bbox, conf in self._nms(boxes, self.IOU_THRESHOLD)
        ]

    @staticmethod
    def _decode_rows(rows, imgsz: int, conf_threshold: float) -> list:
        """Turn raw model rows into (BBox, confidence) pairs, coordinates in [0, 1].

        ``rows`` is any iterable of 5-value sequences — cx, cy, w, h, confidence.
        Deliberately free of numpy and of the ONNX session so the test suite can
        reach it: the scale branch below is the one piece of decoding that fails
        *silently*. Get it wrong and nothing raises — every box lands in a corner
        and AP@0.5 drops to ~0, which is exactly what happened once already.
        """
        rows = [tuple(float(v) for v in row[:5]) for row in rows]
        if not rows:
            return []

        # Coordinate convention differs by export: the original model emitted
        # xywh already normalised to [0, 1]; a standard Ultralytics export emits
        # xywh in input-pixel units. Normalise to [0, 1] either way, so the rest
        # of the pipeline (positions, spatial WHERE) is scale-consistent.
        widest = max(max(row[:4]) for row in rows)
        scale = float(imgsz) if widest > 2.0 else 1.0

        boxes = []                 # (bbox, confidence) above the confidence gate
        for cx, cy, w, h, conf in rows:
            if conf < conf_threshold:
                continue
            cx, cy, w, h = cx / scale, cy / scale, w / scale, h / scale
            boxes.append((
                BBox(cx - w/2, cy - h/2, cx + w/2, cy + h/2),
                conf,
            ))
        return boxes

    @staticmethod
    def _nms(boxes: list, iou_threshold: float) -> list:
        """Greedy non-max suppression over (BBox, confidence) pairs."""
        kept = []
        for bbox, conf in sorted(boxes, key=lambda bc: bc[1], reverse=True):
            if all(bbox.iou(kbox) <= iou_threshold for kbox, _ in kept):
                kept.append((bbox, conf))
        return kept


# ── Flush-stage classifier wrapper ───────────────────────────────────────────

class FlushClassifier:
    """
    Wraps the MobileNetV2 flush-stage model, exported to ONNX.

    Usage:
        clf = FlushClassifier()          # loads models/flush_stage.onnx
        stage = clf.predict(image_rgb)   # numpy array HxWx3, uint8

    Falls back to None if the model file is not yet available, so the
    FusionEngine can still run with a hardcoded stage.
    """

    # Two classes: a ``dormant`` stage is out of scope (least pest activity,
    # and no public dataset exists for it).
    # The order MUST match FLUSH_CLASSES in src/train.py.
    CLASSES = ["new_flush", "mature_leaf"]

    # Decision threshold for the high-risk class: predict ``new_flush`` when
    # P(new_flush) >= this value. 0.50 is argmax, and it is set here explicitly
    # because it is a *calibration result*, not a default nobody looked at.
    #
    # It used to be 0.35, on the reasoning that missing a tender-shoot window is
    # the costly error so recall is worth buying. That value was tuned on a
    # contaminated validation split. Re-swept on the capture-session split,
    # every threshold from 0.20 to 0.80 scores worse than 0.50 on macro-F1, and
    # 0.35 costs more precision than it buys recall on the test set too
    # (accuracy 0.686 vs 0.731). The trade the old value assumed does not exist
    # in this model. Re-sweep when the classifier is re-trained on better data.
    NEW_FLUSH_THRESHOLD = 0.50
    _MODEL_PATH = _MODELS_DIR / "flush_stage.onnx"

    def __init__(self):
        self._session = None
        if self._MODEL_PATH.exists():
            warn_if_stale(self._MODEL_PATH, self._MODEL_PATH.with_suffix(".pt"))
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    str(self._MODEL_PATH), providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
            except Exception as e:
                print(f"[FlushClassifier] Could not load model: {e}")

    @property
    def available(self) -> bool:
        return self._session is not None

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
        # (224, 224, 3) -> (1, 3, 224, 224): the PyTorch export is channels-first
        tensor = ((arr - mean) / std).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        return self._decide(logits)

    @classmethod
    def _decide(cls, logits) -> str:
        """Map raw logits to a stage label.

        For the 2-class model the high-risk class ``new_flush`` is predicted
        whenever ``P(new_flush) >= NEW_FLUSH_THRESHOLD`` (not plain argmax). For
        any other number of classes it falls back to argmax.
        """
        import math
        logits = [float(x) for x in logits]
        if len(cls.CLASSES) == 2 and "new_flush" in cls.CLASSES:
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            total = sum(exps)
            p_new = exps[cls.CLASSES.index("new_flush")] / total
            return "new_flush" if p_new >= cls.NEW_FLUSH_THRESHOLD else "mature_leaf"
        return cls.CLASSES[max(range(len(logits)), key=lambda i: logits[i])]

# Detections below this confidence are routed to the review queue instead of
# being published as Observations, for a person to check and possibly add to
# the training data.
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
class DamageDetection:
    """One bounding-box prediction from the detector.

    Represents a region of Helopeltis feeding *damage* on a leaf (the dataset
    labels damage, not the insect).
    """
    label: str           # e.g. "Helopeltis spp.", "bees"
    bbox: BBox
    confidence: float    # [0, 1]


# ── Fusion engine ────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Fuses DamageDetections + flush stage → Observations (each with a position).

    Usage (single image):
        engine = FusionEngine()
        obs = engine.fuse_frame(detections, stage="new_flush")

    Usage (video / frame stream):
        for frame in frames:
            stable_obs = engine.push_frame(frame.detections, stage)
        review = engine.drain_review_queue()
    """

    def __init__(self):
        self._review_queue: list[dict] = []
        self._frame_buffer: list[list[Observation]] = []

    # ── public API ───────────────────────────────────────────────────────────

    def fuse_frame(
        self,
        detections: list[DamageDetection],
        stage: str,
        timestamp: Optional[datetime] = None,
    ) -> list[Observation]:
        """
        Fuse a single frame's detections into Observation records.

        Each Observation's WHERE is the centre of the detection's bounding box.
        Low-confidence detections (< LOW_CONF_THRESHOLD) are not published; they
        are appended to the review queue for human labelling / retraining.
        """
        if timestamp is None:
            timestamp = datetime.now()

        observations: list[Observation] = []

        for det in detections:
            if det.confidence < LOW_CONF_THRESHOLD:
                self._review_queue.append({
                    "reason":     "low_confidence",
                    "label":      det.label,
                    "confidence": round(det.confidence, 4),
                    "timestamp":  timestamp.isoformat(),
                })
                continue

            try:
                obs = Observation(
                    insect=det.label,
                    stage=stage,
                    confidence=det.confidence,
                    position=(det.bbox.cx, det.bbox.cy),
                    timestamp=timestamp,
                )
                observations.append(obs)
            except ValueError:
                # Unknown stage — skip silently; caller should validate inputs.
                pass

        return observations

    def push_frame(
        self,
        detections: list[DamageDetection],
        stage: str,
        window: int = 5,
        **kwargs,
    ) -> list[Observation]:
        """
        Accumulate frames in a sliding window and return *stable* observations.

        A detection must appear in at least 2 of the last `window` frames to be
        published — this suppresses spurious hits from blurry or partial frames.
        Detections are grouped by label, stage and (coarse) position, so two
        detections in different parts of the frame stay distinct events.
        """
        frame_obs = self.fuse_frame(detections, stage, **kwargs)
        self._frame_buffer.append(frame_obs)
        if len(self._frame_buffer) > window:
            self._frame_buffer.pop(0)

        counts: dict[tuple, list[Observation]] = {}
        for frame in self._frame_buffer:
            for obs in frame:
                px, py = obs.position if obs.position else (None, None)
                key = (obs.insect, obs.stage,
                       round(px, 1) if px is not None else None,
                       round(py, 1) if py is not None else None)
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