"""
Tests for src/loader.py — image loading, resizing, and matrix conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.loader import (
    ImageLoadError,
    ImagePair,
    DimensionMismatchError,
    load_image,
    load_and_prepare_images,
    resize_to_match,
    flatten_image,
)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    """Tests for the load_image function."""

    def test_load_grayscale(self, temp_image_a: Path):
        """Should load a grayscale image as float64 in [0, 1]."""
        img = load_image(temp_image_a)
        assert img.dtype == np.float64
        assert img.ndim == 2
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_load_color_converts_to_grayscale(self, temp_color_image: Path):
        """Should convert color image to grayscale."""
        img = load_image(temp_color_image)
        assert img.dtype == np.float64
        assert img.ndim == 2

    def test_load_nonexistent_file(self):
        """Should raise ImageLoadError for missing file."""
        with pytest.raises(ImageLoadError, match="not found"):
            load_image("nonexistent_path_12345.png")

    def test_load_invalid_file(self, temp_invalid_file: Path):
        """Should raise ImageLoadError for non-image file."""
        with pytest.raises(ImageLoadError, match="Failed to decode"):
            load_image(temp_invalid_file)

    def test_load_directory_path(self, temp_dir: Path):
        """Should raise ImageLoadError when path is a directory."""
        with pytest.raises(ImageLoadError, match="not a file"):
            load_image(temp_dir)


# ---------------------------------------------------------------------------
# resize_to_match
# ---------------------------------------------------------------------------

class TestResizeToMatch:
    """Tests for the resize_to_match function."""

    def test_same_size_unchanged(self, small_image: np.ndarray):
        """Images already same size should not change."""
        a, b = resize_to_match(small_image, small_image.copy())
        assert a.shape == b.shape == (64, 64)

    def test_different_sizes_resized(self):
        """Different-sized images should be resized to match."""
        a = np.ones((100, 80))
        b = np.ones((50, 40))
        ra, rb = resize_to_match(a, b)
        assert ra.shape == rb.shape

    def test_explicit_target_size(self):
        """Should resize to explicit target_size."""
        a = np.ones((100, 80))
        b = np.ones((50, 40))
        ra, rb = resize_to_match(a, b, target_size=(32, 32))
        assert ra.shape == (32, 32)
        assert rb.shape == (32, 32)


# ---------------------------------------------------------------------------
# flatten_image
# ---------------------------------------------------------------------------

class TestFlattenImage:
    """Tests for the flatten_image function."""

    def test_flatten_shape(self):
        """Should produce 1D array of size H*W."""
        img = np.ones((10, 20))
        vec = flatten_image(img)
        assert vec.shape == (200,)

    def test_flatten_preserves_values(self):
        """Should preserve all values."""
        img = np.arange(12, dtype=np.float64).reshape(3, 4)
        vec = flatten_image(img)
        assert np.allclose(vec, np.arange(12, dtype=np.float64))


# ---------------------------------------------------------------------------
# ImagePair dataclass
# ---------------------------------------------------------------------------

class TestImagePair:
    """Tests for the ImagePair dataclass."""

    def test_attributes(self, small_image: np.ndarray):
        """Should store all attributes correctly."""
        pair = ImagePair(
            image_a=small_image,
            image_b=small_image,
            vector_a=small_image.flatten(),
            vector_b=small_image.flatten(),
            height=64,
            width=64,
            path_a="a.png",
            path_b="b.png",
        )
        assert pair.height == 64
        assert pair.width == 64
        assert pair.path_a == "a.png"
        assert pair.path_b == "b.png"
        assert pair.vector_a.shape == (4096,)


# ---------------------------------------------------------------------------
# load_and_prepare_images (integration)
# ---------------------------------------------------------------------------

class TestLoadAndPrepareImages:
    """Integration tests for load_and_prepare_images."""

    def test_load_pair(self, temp_image_a: Path, temp_image_b: Path):
        """Should load and prepare a valid image pair."""
        pair = load_and_prepare_images(temp_image_a, temp_image_b)
        assert isinstance(pair, ImagePair)
        assert pair.image_a.shape == pair.image_b.shape
        assert pair.vector_a.shape == pair.vector_b.shape

    def test_load_with_color(self, temp_color_image: Path, temp_image_a: Path):
        """Should handle mixed color/grayscale pair."""
        pair = load_and_prepare_images(temp_color_image, temp_image_a)
        assert pair.image_a.shape == pair.image_b.shape

    def test_nonexistent_raises(self, temp_dir: Path):
        """Should raise ImageLoadError for missing file."""
        with pytest.raises(ImageLoadError):
            load_and_prepare_images(
                temp_dir / "missing_a.png",
                temp_dir / "missing_b.png",
            )

    def test_custom_target_size(self, temp_image_a: Path, temp_image_b: Path):
        """Should resize to explicit target_size."""
        pair = load_and_prepare_images(
            temp_image_a, temp_image_b, target_size=(32, 32)
        )
        assert pair.height == 32
        assert pair.width == 32
