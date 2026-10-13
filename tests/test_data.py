"""Dataset setup: verified downloads and the flush-stage preparation step."""
import hashlib
import io
import zipfile

import pytest

from src import data


def _zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


def _fake_release(tmp_path, monkeypatch, tamper=False):
    """A local 'release' folder with one asset, and RELEASE_ASSETS pointed at it."""
    blob = _zip_bytes({"demo/test/images/a.jpg": b"x"})
    assets = tmp_path / "release"
    assets.mkdir()
    (assets / "demo.zip").write_bytes(blob + (b"!" if tamper else b""))
    dest = tmp_path / "raw"
    monkeypatch.setattr(data, "RELEASE_ASSETS", {
        "demo.zip": (hashlib.sha256(blob).hexdigest(), dest, dest / "demo" / "test" / "images"),
    })
    monkeypatch.setattr(data, "split_detector", lambda: None)
    return assets, dest


def test_download_verifies_and_extracts(tmp_path, monkeypatch):
    assets, dest = _fake_release(tmp_path, monkeypatch)
    data.download(base=str(assets))
    assert (dest / "demo" / "test" / "images" / "a.jpg").read_bytes() == b"x"


def test_download_refuses_a_corrupted_asset(tmp_path, monkeypatch):
    assets, dest = _fake_release(tmp_path, monkeypatch, tamper=True)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        data.download(base=str(assets))
    assert not dest.exists()


def test_download_skips_what_is_already_installed(tmp_path, monkeypatch):
    assets, dest = _fake_release(tmp_path, monkeypatch)
    data.download(base=str(assets))
    (assets / "demo.zip").unlink()          # would fail if it tried again
    data.download(base=str(assets))


def test_prepare_flush_maps_four_ages_to_two_classes(tmp_path):
    raw = tmp_path / "Raw Data"
    for folder in ("Catagory A", "Category B", "Category C", "Category D (Not for Tea)"):
        (raw / folder).mkdir(parents=True)
        (raw / folder / "leaf.jpg").write_bytes(b"x")
    out = tmp_path / "flush"
    counts = data.prepare_flush(raw, out)
    assert counts == {"new_flush": 2, "mature_leaf": 2}
    assert sorted(p.name for p in (out / "new_flush").iterdir()) == ["A_leaf.jpg", "B_leaf.jpg"]


def test_prepare_flush_reports_missing_categories(tmp_path):
    (tmp_path / "Category A").mkdir()
    with pytest.raises(SystemExit, match="not found"):
        data.prepare_flush(tmp_path, tmp_path / "out")
