"""Feature extraction for the TINAT image-classification pipeline (Phase 1).

While the real annotated photos are not yet available, TINAT starts as an
image *classification* problem using public data (tea-leaf images that include
the Helopeltis and mirid-bug classes). We rely on classic computer-vision
features -- per-channel colour histograms -- which are cheap enough to train on
CPU and easy to reason about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = (128, 128)
DEFAULT_BINS = 16
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def extract_color_histogram(image: Image.Image, bins: int = DEFAULT_BINS) -> np.ndarray:
    """Return a normalised per-channel RGB colour histogram.

    The image is converted to RGB and resized to a fixed size, then each colour
    channel is summarised as a ``bins``-length density histogram. The channels
    are concatenated, so the feature vector has length ``bins * 3``.
    """
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    pixels = np.asarray(resized, dtype=np.uint8)

    features: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(
            pixels[:, :, channel], bins=bins, range=(0, 255), density=True
        )
        features.extend(hist.tolist())
    return np.asarray(features, dtype=np.float64)


def load_image_dataset(root: str | Path, bins: int = DEFAULT_BINS):
    """Load an image dataset organised as one sub-folder per class.

    ``root`` must contain one sub-directory per class, each holding the images
    of that class. Corrupted or unreadable images are skipped rather than
    raising. Returns ``(features, labels, class_names)`` where ``features`` has
    shape ``(n_images, bins * 3)``.
    """
    root = Path(root)
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    features: list[np.ndarray] = []
    labels: list[int] = []
    for class_name in class_names:
        for path in sorted((root / class_name).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    features.append(extract_color_histogram(img, bins))
                    labels.append(class_to_idx[class_name])
            except Exception:
                continue  # skip corrupted / unreadable images

    return np.asarray(features), np.asarray(labels), class_names
