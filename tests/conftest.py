"""
Shared pytest fixtures for PicPicComparison tests.

Provides synthetic images, temporary file helpers, and common test data.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.RandomState:
    """Reproducible random number generator."""
    return np.random.RandomState(42)


@pytest.fixture
def small_image() -> np.ndarray:
    """64x64 random grayscale image, float64 in [0, 1]."""
    rng = np.random.RandomState(42)
    return rng.rand(64, 64).astype(np.float64)


@pytest.fixture
def small_image_b() -> np.ndarray:
    """64x64 random grayscale image (different seed), float64 in [0, 1]."""
    rng = np.random.RandomState(99)
    return rng.rand(64, 64).astype(np.float64)


@pytest.fixture
def identical_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Two identical 64x64 images."""
    rng = np.random.RandomState(7)
    img = rng.rand(64, 64).astype(np.float64)
    return img.copy(), img.copy()


@pytest.fixture
def different_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Two different 64x64 images."""
    rng_a = np.random.RandomState(1)
    rng_b = np.random.RandomState(2)
    return rng_a.rand(64, 64).astype(np.float64), rng_b.rand(64, 64).astype(np.float64)


@pytest.fixture
def single_pixel_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Two 1x1 images with different values."""
    return np.array([[0.3]]), np.array([[0.7]])


@pytest.fixture
def all_zeros_image() -> np.ndarray:
    """64x64 image of all zeros."""
    return np.zeros((64, 64), dtype=np.float64)


@pytest.fixture
def all_ones_image() -> np.ndarray:
    """64x64 image of all ones."""
    return np.ones((64, 64), dtype=np.float64)


@pytest.fixture
def uniform_image() -> np.ndarray:
    """64x64 image with uniform value 0.5."""
    return np.full((64, 64), 0.5, dtype=np.float64)


# ---------------------------------------------------------------------------
# Temporary file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory and clean up after."""
    with tempfile.TemporaryDirectory(prefix="picpic_test_") as d:
        yield Path(d)


@pytest.fixture
def temp_image_a(temp_dir) -> Path:
    """Create a temporary grayscale PNG image and return its path."""
    rng = np.random.RandomState(42)
    img = (rng.rand(64, 64) * 255).astype(np.uint8)
    path = temp_dir / "test_image_a.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def temp_image_b(temp_dir) -> Path:
    """Create a temporary grayscale PNG image (different) and return its path."""
    rng = np.random.RandomState(99)
    img = (rng.rand(64, 64) * 255).astype(np.uint8)
    path = temp_dir / "test_image_b.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def temp_color_image(temp_dir) -> Path:
    """Create a temporary color (BGR) PNG image and return its path."""
    rng = np.random.RandomState(55)
    img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
    path = temp_dir / "test_color.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def temp_invalid_file(temp_dir) -> Path:
    """Create a temporary file with invalid image data."""
    path = temp_dir / "not_an_image.txt"
    path.write_text("this is not an image file")
    return path


# ---------------------------------------------------------------------------
# Known-value fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def known_vectors():
    """Pairs of vectors with known mathematical relationships."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])  # orthogonal to a
    c = np.array([1.0, 0.0, 0.0])  # identical to a
    d = np.array([3.0, 4.0, 0.0])  # known norm = 5
    return {
        "a": a, "b": b, "c": c, "d": d,
        "dot_ab": 0.0,
        "dot_ac": 1.0,
        "dot_ad": 3.0,
        "cosine_ab": 0.0,  # orthogonal
        "cosine_ac": 1.0,  # identical
        "norm_a": 1.0,
        "norm_d": 5.0,
    }


@pytest.fixture
def known_images():
    """Pairs of images with known metric values."""
    # 4x4 images for easy hand-calculation
    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    B = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    C = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float64)

    return {
        "zeros": A,
        "single_diff": B,
        "ones": C,
        "frobenius_zeros_B": 1.0,  # sqrt(1^2) = 1
        "l1_zeros_B": 1.0,         # |1| = 1
        "l_inf_zeros_B": 1.0,      # max(|1|) = 1
        "mse_zeros_B": 1.0 / 16,   # 1/16
    }
