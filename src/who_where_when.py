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


# ─── WHERE: plant zones (A/R/E framework) ───
ZONES: dict[str, str] = {
    "Attraction":  "Reproductive zone — flowers, tassels",
    "Retention":   "Resource zone — nectaries, axils, shelter",
    "Efficiency":  "Defense zone — leaves, stems, trichomes",
}

# ─── WHEN: tea crop flush stages ───
FLUSH_STAGES: dict[str, str] = {
    "dormant":     "Months 0-2  — vegetative, no flush",
    "new_flush":   "Months 3-4  — tender new shoots, HIGH RISK for Helopeltis",
    "mature_leaf": "Months 5+   — mature leaf, lower risk",
}


# ─── The full metric ───
@dataclass
class Observation:
    # WHO
    insect: str
    role: str = field(init=False)           # assigned automatically from ECOLOGICAL_ROLE

    # WHERE
    zone: str                               # Attraction / Retention / Efficiency

    # WHEN
    stage: str                              # dormant / new_flush / mature_leaf

    # Metadata
    confidence: float
    timestamp: datetime = field(default=None)
    gps: Optional[tuple] = None

    def __post_init__(self):
        # Assign role from the registry; falls back to "unknown"
        self.role = get_role(self.insect)

        # Validate zone and stage against known values
        if self.zone not in ZONES:
            raise ValueError(f"Unknown zone '{self.zone}'. Valid: {list(ZONES)}")
        if self.stage not in FLUSH_STAGES:
            raise ValueError(f"Unknown stage '{self.stage}'. Valid: {list(FLUSH_STAGES)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

        if self.timestamp is None:
            self.timestamp = datetime.now()

    def is_alert(self) -> bool:
        """True when a pest is detected during the new-flush stage (highest risk)."""
        return self.role == "pest" and self.stage == "new_flush"

    def summary(self) -> str:
        status = "[ALERT]" if self.is_alert() else "[OK]"
        return (
            f"{status}\n"
            f"  WHO  : {self.insect} ({self.role})\n"
            f"  WHERE: {self.zone} — {ZONES[self.zone]}\n"
            f"  WHEN : {self.stage} — {FLUSH_STAGES[self.stage]}\n"
            f"  Conf : {self.confidence:.1%}\n"
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON export / database insert."""
        return {
            "insect":     self.insect,
            "role":       self.role,
            "zone":       self.zone,
            "stage":      self.stage,
            "confidence": round(self.confidence, 4),
            "is_alert":   self.is_alert(),
            "timestamp":  self.timestamp.isoformat(),
            "gps":        list(self.gps) if self.gps else None,
        }


# ─── Smoke tests ───
if __name__ == "__main__":

    # Scenario 1: Helopeltis on new flush — dangerous
    obs1 = Observation(insect="Helopeltis spp.", zone="Efficiency", stage="new_flush",   confidence=0.82)
    # Scenario 2: bee in attraction zone — healthy
    obs2 = Observation(insect="bees",            zone="Attraction", stage="new_flush",   confidence=0.91)
    # Scenario 3: Helopeltis on mature leaf — lower risk
    obs3 = Observation(insect="Helopeltis spp.", zone="Efficiency", stage="mature_leaf", confidence=0.76)

    assert obs1.is_alert(), "obs1 should trigger an alert"
    assert not obs2.is_alert(), "obs2 should not trigger an alert"
    assert not obs3.is_alert(), "obs3 should not trigger an alert"
    assert obs1.role == "pest"
    assert obs2.role == "pollinator"

    print("=== TINAT OBSERVATIONS ===\n")
    for obs in [obs1, obs2, obs3]:
        print(obs.summary())
