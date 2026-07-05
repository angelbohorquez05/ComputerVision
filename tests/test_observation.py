import pytest

from src.who_where_when import Observation, get_role


def test_pest_on_new_flush_triggers_alert():
    obs = Observation(insect="Helopeltis spp.", zone="Efficiency", stage="new_flush", confidence=0.82)
    assert obs.is_alert()
    assert obs.role == "pest"


def test_pollinator_does_not_trigger_alert():
    obs = Observation(insect="bees", zone="Attraction", stage="new_flush", confidence=0.91)
    assert not obs.is_alert()
    assert obs.role == "pollinator"


def test_pest_on_mature_leaf_is_not_alert():
    obs = Observation(insect="Helopeltis spp.", zone="Efficiency", stage="mature_leaf", confidence=0.76)
    assert not obs.is_alert()


def test_unknown_zone_raises():
    with pytest.raises(ValueError):
        Observation(insect="Helopeltis", zone="Nowhere", stage="new_flush", confidence=0.5)


def test_confidence_out_of_range_raises():
    with pytest.raises(ValueError):
        Observation(insect="Helopeltis", zone="Efficiency", stage="new_flush", confidence=1.5)


def test_get_role_falls_back_to_unknown():
    assert get_role("some unlisted bug") == "unknown"
