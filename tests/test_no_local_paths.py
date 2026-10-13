"""No machine-specific absolute paths in versioned files.

An absolute home-directory path is useless to anyone who clones the repo, and it
leaks the local username. The check is structural, so it catches a path from any
machine. It exists because it happened once: re-executed notebooks printed the
project root into their saved output, and nothing complained.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# A home directory on Windows, macOS or Linux, followed by the user's name.
# The trailing class excludes < > [ ] ^ | * ? ( ) — characters a real path
# cannot contain but a regex literal can. That is what keeps this pattern from
# matching its own definition, so no file has to be exempted from the scan and
# a genuine leak inside these very files would still be caught.
HOME_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]Users[\\/]|/Users/|/home/)[^\\/\s\"'<>\[\]^|*?()]+")

# Generated files hold absolute paths by design and are gitignored for it.
SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", ".pytest_cache", "runs",
             "flush_stage_tf", ".ipynb_checkpoints"}
# Ultralytics writes args.yaml with the absolute paths of the run that produced
# it. Regenerable, and gitignored for exactly this reason — the hyperparameters
# that matter are in the model card.
SKIP_NAMES = {"helopeltis_v2.yaml", "args.yaml"}
SKIP_PREFIXES = ("split_", "detector_")


def _versioned_text_files():
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".ipynb", ".json",
                                                     ".yml", ".yaml", ".csv", ".txt"}:
            continue
        if SKIP_DIRS & set(path.relative_to(REPO).parts):
            continue
        if path.name in SKIP_NAMES or path.name.startswith(SKIP_PREFIXES):
            continue
        yield path


def test_no_home_directory_paths_in_versioned_files():
    leaks = []
    for path in _versioned_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in HOME_PATH.finditer(text):
            leaks.append(f"{path.relative_to(REPO)}: {match.group()[:40]}...")
    assert not leaks, (
        "Absolute home paths found in versioned files (machine-specific, and "
        "they leak the local username):\n  "
        + "\n  ".join(sorted(set(leaks))[:10]))


def test_deployed_model_metadata_carries_no_absolute_paths():
    """The model file travels furthest and is the last place anyone greps.

    Ultralytics stamps the export with the absolute path of the data yaml it
    trained on. `scrub_onnx_paths` strips it, and this asserts the scrub
    actually ran. Skipped when the binary is absent — it is not versioned.
    """
    model = REPO / "models" / "helopeltis_damage.onnx"
    if not model.exists():
        return
    blob = model.read_bytes()
    # Search the raw bytes: metadata is stored as plain UTF-8 inside the file,
    # so this catches a path wherever it hides, not only where we expect it.
    text = blob.decode("utf-8", errors="ignore")
    found = HOME_PATH.findall(text)
    assert not found, f"Absolute path inside {model.name}: {found[:2]}"


def test_notebook_outputs_carry_no_absolute_paths():
    """Notebook *outputs* are the easy leak: code review reads the source."""
    leaks = []
    for path in (REPO / "notebooks").glob("*.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb.get("cells", [])):
            for out in cell.get("outputs", []):
                text = "".join(out.get("text", "")) + "".join(
                    out.get("data", {}).get("text/plain", ""))
                if HOME_PATH.search(text):
                    leaks.append(f"{path.name} cell {i}")
    assert not leaks, f"Absolute paths in saved notebook output: {sorted(set(leaks))}"
