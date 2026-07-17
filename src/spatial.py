"""
spatial.py — WHERE as periphery-vs-core infestation mapping

The plant-zone framing of WHERE (Attraction / Retention / Efficiency) answers
"which part of the plant". This module answers the question the field actually
needs: **is the pest only at the edge of the monitored area, or has it reached
the core?** A pest kept to the periphery means the surrounding ecosystem design
is holding; a pest deep inside the system is a real breach.

It works on the *aggregate* of detections (their positions across the field),
not on a single frame — the spatial counterpart of the temporal aggregation in
the fusion engine. Positions are plain (x, y) coordinates in the same units as
the field extent (e.g. normalised plot coordinates or projected GPS).

Pure standard-library geometry: no models, no heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
