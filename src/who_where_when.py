from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ─── WHO: insect classes mapped to ecological role ───
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
            the positions of many Observations — see ``src/spatial.py``.
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
if __name__ == "__main__":

    # Scenario 1: Helopeltis on a young leaf — dangerous
    obs1 = Observation(insect="Helopeltis spp.", stage="new_flush",   confidence=0.82, position=(0.5, 0.5))
    # Scenario 2: bee — healthy
    obs2 = Observation(insect="bees",            stage="new_flush",   confidence=0.91, position=(0.7, 0.2))
    # Scenario 3: Helopeltis on a mature leaf — lower risk
    obs3 = Observation(insect="Helopeltis spp.", stage="mature_leaf", confidence=0.76, position=(0.1, 0.9))

    assert obs1.is_alert(), "obs1 should trigger an alert"
    assert not obs2.is_alert(), "obs2 should not trigger an alert"
    assert not obs3.is_alert(), "obs3 should not trigger an alert"
    assert obs1.role == "pest"
    assert obs2.role == "pollinator"

    print("=== TINAT OBSERVATIONS ===\n")
    for obs in [obs1, obs2, obs3]:
        print(obs.summary())
