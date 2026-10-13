import pytest

from src.pipeline import Observation, get_role


def test_pest_on_new_flush_triggers_alert():
    obs = Observation(insect="Helopeltis spp.", stage="new_flush", confidence=0.82, position=(0.5, 0.5))
    assert obs.is_alert()
    assert obs.role == "pest"


def test_pollinator_does_not_trigger_alert():
    obs = Observation(insect="bees", stage="new_flush", confidence=0.91, position=(0.7, 0.2))
    assert not obs.is_alert()
    assert obs.role == "pollinator"


def test_pest_on_mature_leaf_is_not_alert():
    obs = Observation(insect="Helopeltis spp.", stage="mature_leaf", confidence=0.76, position=(0.1, 0.9))
    assert not obs.is_alert()


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        Observation(insect="Helopeltis", stage="whenever", confidence=0.5)


def test_confidence_out_of_range_raises():
    with pytest.raises(ValueError):
        Observation(insect="Helopeltis", stage="new_flush", confidence=1.5)


def test_position_is_serialised():
    d = Observation(insect="Helopeltis", stage="new_flush", confidence=0.8, position=(0.3, 0.4)).to_dict()
    assert d["position"] == [0.3, 0.4]
    assert "zone" not in d


def test_get_role_falls_back_to_unknown():
    assert get_role("some unlisted bug") == "unknown"
