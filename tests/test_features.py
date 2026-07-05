import numpy as np
from PIL import Image

from src.features import DEFAULT_BINS, extract_color_histogram


def _solid_image(color, size=(64, 64)):
    return Image.new("RGB", size, color)


def test_histogram_length_matches_bins():
    feat = extract_color_histogram(_solid_image((10, 20, 30)), bins=8)
    assert feat.shape == (8 * 3,)


def test_histogram_is_normalised_per_channel():
    # A density histogram integrates to 1 over each channel's value range.
    feat = extract_color_histogram(_solid_image((128, 128, 128)), bins=DEFAULT_BINS)
    bin_width = 255 / DEFAULT_BINS
    per_channel = feat.reshape(3, DEFAULT_BINS)
    integrals = per_channel.sum(axis=1) * bin_width
    assert np.allclose(integrals, 1.0, atol=1e-6)
