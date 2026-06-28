"""
Algorithmic image comparison metrics.

Implements standard algorithmic comparison techniques: MSE, PSNR,
and histogram-based intensity distribution comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# Maximum pixel value for float64 images normalized to [0, 1]
_MAX_PIXEL_VALUE: float = 1.0


def mse(A: np.ndarray, B: np.ndarray) -> float:
    """Compute Mean Squared Error between two image matrices.

    MSE = (1 / M*N) * sum_ij (a_ij - b_ij)^2

    A fundamental algorithmic baseline for image quality assessment.
    Lower values indicate greater similarity (0 = identical).

    Args:
        A: First image matrix (H x W), float64 in [0, 1].
        B: Second image matrix (H x W), float64 in [0, 1].

    Returns:
        Non-negative scalar MSE value.

    Raises:
        ValueError: If A and B have different shapes.
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A={A.shape}, B={B.shape}"
        )
    diff = A - B
    return float(np.mean(diff ** 2))


def psnr(A: np.ndarray, B: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two image matrices.

    PSNR = 10 * log10(MAX^2 / MSE)

    Where MAX is the maximum possible pixel value (1.0 for normalized images).
    Higher PSNR indicates better quality / greater similarity.

    If MSE is 0 (identical images), returns float('inf').

    Args:
        A: First image matrix (H x W), float64 in [0, 1].
        B: Second image matrix (H x W), float64 in [0, 1].

    Returns:
        PSNR in decibels (dB), or inf if images are identical.

    Raises:
        ValueError: If A and B have different shapes.
    """
    mse_val = mse(A, B)
    if mse_val == 0.0:
        return float("inf")
    return float(10.0 * np.log10(_MAX_PIXEL_VALUE ** 2 / mse_val))


def _compute_histogram(
    img: np.ndarray, bins: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized histogram for a grayscale image.

    Args:
        img: 2D image matrix, float64 in [0, 1].
        bins: Number of histogram bins.

    Returns:
        Tuple of (histogram, bin_edges).
        Histogram is normalized to sum to 1.0 (probability distribution).
    """
    # Scale to [0, 255] integer range for binning
    img_scaled = (img * 255.0).astype(np.uint8).flatten()
    hist, bin_edges = np.histogram(img_scaled, bins=bins, range=(0, 256))
    # Normalize to probability distribution
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist, bin_edges


def histogram_intersection(
    A: np.ndarray, B: np.ndarray, bins: int = 256
) -> float:
    """Compute histogram intersection between two grayscale images.

    Histogram intersection measures the overlap between the intensity
    distributions of two images. Values range from 0 (no overlap) to
    1 (identical distributions).

        Intersection = sum_i min(hist_A[i], hist_B[i])

    This is an algorithmic metric that compares color/intensity
    distributions regardless of spatial arrangement.

    Args:
        A: First image matrix (H x W), float64 in [0, 1].
        B: Second image matrix (H x W), float64 in [0, 1].
        bins: Number of histogram bins (default 256 for full 8-bit range).

    Returns:
        Histogram intersection in [0.0, 1.0].

    Raises:
        ValueError: If A and B have different shapes.
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A={A.shape}, B={B.shape}"
        )

    hist_a, _ = _compute_histogram(A, bins=bins)
    hist_b, _ = _compute_histogram(B, bins=bins)

    # Intersection = sum of element-wise minima
    return float(np.sum(np.minimum(hist_a, hist_b)))


@dataclass
class HistogramResult:
    """Container for histogram comparison results.

    Attributes:
        intersection: Histogram intersection score [0, 1].
        hist_a: Normalized histogram of image A.
        hist_b: Normalized histogram of image B.
        bin_edges: Bin edges for the histograms.
    """

    intersection: float
    hist_a: np.ndarray
    hist_b: np.ndarray
    bin_edges: np.ndarray


def histogram_intersection_detailed(
    A: np.ndarray, B: np.ndarray, bins: int = 256
) -> HistogramResult:
    """Compute histogram intersection with full histogram data.

    Same as histogram_intersection but also returns the individual
    histograms for visualization purposes.

    Args:
        A: First image matrix (H x W), float64 in [0, 1].
        B: Second image matrix (H x W), float64 in [0, 1].
        bins: Number of histogram bins.

    Returns:
        HistogramResult with intersection score and histogram data.
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A={A.shape}, B={B.shape}"
        )

    hist_a, bin_edges = _compute_histogram(A, bins=bins)
    hist_b, _ = _compute_histogram(B, bins=bins)

    intersection = float(np.sum(np.minimum(hist_a, hist_b)))

    return HistogramResult(
        intersection=intersection,
        hist_a=hist_a,
        hist_b=hist_b,
        bin_edges=bin_edges,
    )


def difference_image(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the absolute difference image between two matrices.

    Useful for visualizing where pixel-level differences occur.

    Args:
        A: First image matrix (H x W), float64.
        B: Second image matrix (H x W), float64.

    Returns:
        Absolute difference matrix (H x W), float64 in [0, max_diff].
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A={A.shape}, B={B.shape}"
        )
    return np.abs(A - B)


# ---------------------------------------------------------------------------
# Edge detection (extracted from gui_similarity.py)
# ---------------------------------------------------------------------------

_EPS: float = 1e-9


def sobel_edge_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute Sobel edge magnitude using pure NumPy convolution.

    Applies 3x3 Sobel kernels in X and Y directions, then computes
    the gradient magnitude. The result is normalized by mean + 3*std.

    Args:
        img: 2D grayscale image matrix, float64.

    Returns:
        Edge magnitude matrix of same shape, float64, normalized.
    """
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    def _conv_pad(a: np.ndarray, k: np.ndarray) -> np.ndarray:
        kh, kw = k.shape
        ph, pw = kh // 2, kw // 2
        ap = np.pad(a, ((ph, ph), (pw, pw)), mode="edge")
        H, W = a.shape
        out = np.empty_like(a, dtype=np.float64)
        for i in range(H):
            for j in range(W):
                out[i, j] = float((ap[i : i + kh, j : j + kw] * k).sum())
        return out

    gx = _conv_pad(img, Kx)
    gy = _conv_pad(img, Ky)
    mag = np.sqrt(gx * gx + gy * gy)
    denom = float(mag.mean() + 3.0 * mag.std() + _EPS)
    return (mag / denom).astype(np.float64)


# ---------------------------------------------------------------------------
# Normalized Cross-Correlation (extracted from gui_similarity.py)
# ---------------------------------------------------------------------------

def normalized_cross_correlation(
    A: np.ndarray,
    B: np.ndarray,
    M: Optional[np.ndarray] = None,
    max_shift: int = 8,
) -> float:
    """Compute shift-robust Normalized Cross-Correlation (NCC).

    Searches over small translations (dy, dx) in [-max_shift, max_shift]
    to find the alignment that maximizes the cosine similarity between
    the masked pixel vectors.

    Args:
        A: First image matrix (H x W), float64.
        B: Second image matrix (H x W), float64.
        M: Optional boolean mask (H x W). If None, uses all pixels.
        max_shift: Maximum pixel shift to search in each direction.

    Returns:
        Best NCC score as a percentage in [0, 100].
    """
    from src.loader import apply_mask_flat

    H, W = A.shape
    if M is None:
        M = np.ones((H, W), dtype=bool)

    best = -1.0

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            # Overlap region for A
            y0a = max(0, dy)
            y1a = min(H, H + dy)
            x0a = max(0, dx)
            x1a = min(W, W + dx)
            # Corresponding region for B
            y0b = max(0, -dy)
            y1b = min(H, H - dy)
            x0b = max(0, -dx)
            x1b = min(W, W - dx)

            if y1a - y0a <= 0 or x1a - x0a <= 0:
                continue

            Aov = A[y0a:y1a, x0a:x1a]
            Bov = B[y0b:y1b, x0b:x1b]
            Mov = M[y0a:y1a, x0a:x1a] & M[y0b:y1b, x0b:x1b]

            va = apply_mask_flat(Aov, Mov, center=True)
            vb = apply_mask_flat(Bov, Mov, center=True)
            na = np.linalg.norm(va)
            nb = np.linalg.norm(vb)

            if na < _EPS or nb < _EPS:
                continue

            s = float(np.dot(va, vb) / (na * nb))
            if s > best:
                best = s

    return 100.0 * max(0.0, best if best > -2 else -1.0)
