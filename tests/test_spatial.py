import pytest

from src.pipeline import (
    CORE,
    PERIPHERY,
    FieldExtent,
    assess_infestation,
    classify_location,
    penetration,
)


# ── FieldExtent ──────────────────────────────────────────────────────────────

def test_field_extent_rejects_degenerate_bounds():
    with pytest.raises(ValueError):
        FieldExtent(0, 0, 0, 1)      # xmax == xmin
    with pytest.raises(ValueError):
        FieldExtent(0, 0, 1, 0)      # ymax == ymin


def test_field_extent_from_points_covers_all_points():
    ext = FieldExtent.from_points([(0, 0), (10, 4)])
    assert (ext.xmin, ext.ymin, ext.xmax, ext.ymax) == (0, 0, 10, 4)


def test_field_extent_from_points_pads_outward():
    ext = FieldExtent.from_points([(0, 0), (10, 10)], pad=0.1)
    assert ext.xmin == pytest.approx(-1.0)
    assert ext.xmax == pytest.approx(11.0)


def test_field_extent_from_zero_points_raises():
    with pytest.raises(ValueError):
        FieldExtent.from_points([])


# ── penetration ──────────────────────────────────────────────────────────────

def test_penetration_is_max_at_centre():
    ext = FieldExtent(0, 0, 10, 10)
    assert penetration(5, 5, ext) == pytest.approx(1.0)


def test_penetration_is_zero_on_the_edge():
    ext = FieldExtent(0, 0, 10, 10)
    assert penetration(0, 5, ext) == pytest.approx(0.0)


def test_point_outside_field_reads_as_edge():
    ext = FieldExtent(0, 0, 10, 10)
    assert penetration(-3, 5, ext) == 0.0


# ── classify_location ────────────────────────────────────────────────────────

def test_edge_point_is_periphery():
    ext = FieldExtent(0, 0, 10, 10)
    assert classify_location(1, 5, ext) == PERIPHERY


def test_centre_point_is_core():
    ext = FieldExtent(0, 0, 10, 10)
    assert classify_location(5, 5, ext) == CORE


# ── assess_infestation ───────────────────────────────────────────────────────

def test_no_pests_reports_no_detection():
    report = assess_infestation([], extent=FieldExtent(0, 0, 10, 10))
    assert report.total == 0
    assert not report.breach
    assert report.verdict == "no pests detected"


def test_pests_only_at_edge_are_contained():
    ext = FieldExtent(0, 0, 10, 10)
    report = assess_infestation([(0.5, 5), (9.5, 5), (5, 0.5)], extent=ext)
    assert report.total == 3
    assert report.n_core == 0
    assert not report.breach
    assert report.verdict == "contained — pests only at the periphery"


def test_a_single_core_hit_is_a_breach():
    ext = FieldExtent(0, 0, 10, 10)
    report = assess_infestation([(0.5, 5), (5, 5)], extent=ext)
    assert report.n_periphery == 1
    assert report.n_core == 1
    assert report.breach
    assert report.verdict.startswith("breach")


def test_report_to_dict_is_serialisable():
    ext = FieldExtent(0, 0, 10, 10)
    d = assess_infestation([(5, 5)], extent=ext).to_dict()
    assert d["breach"] is True
    assert d["n_core"] == 1
    assert set(d) == {"total", "n_periphery", "n_core", "breach", "verdict"}


def test_extent_defaults_to_bounding_box_of_points():
    # With no explicit extent the outermost points sit on the derived edge,
    # so they are periphery and only genuinely interior points count as core.
    report = assess_infestation([(0, 0), (10, 10), (5, 5)])
    assert report.n_core == 1
    assert report.breach
