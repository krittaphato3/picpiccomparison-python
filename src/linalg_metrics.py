"""
Linear algebra-based image comparison metrics.

Implements mathematically rigorous comparison functions using matrix
norms, vector space geometry, and singular value decomposition as
defined in the project specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import norm as np_norm


# ---------------------------------------------------------------------------
# Norm-based distances (pixel-wise divergence)
# ---------------------------------------------------------------------------

def frobenius_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Compute the Frobenius (L2) norm of the difference matrix.

    The Frobenius norm measures the Euclidean distance between two image
    matrices treated as vectors in R^(m*n):

        ||A - B||_F = sqrt( sum_ij |a_ij - b_ij|^2 )

    Args:
        A: First image matrix (H x W), float64.
        B: Second image matrix (H x W), float64. Same shape as A.

    Returns:
        Non-negative scalar representing the Frobenius distance.

    Raises:
        ValueError: If A and B have different shapes.
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A={A.shape}, B={B.shape}"
        )
    return float(np_norm(A - B, ord="fro"))


def l1_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Compute the L1 (Manhattan / Nuclear) norm of the difference matrix.

    The L1 norm sums the absolute pixel-wise differences:

        ||A - B||_1 = sum_ij |a_ij - b_ij|

    This is more robust to salt-and-pepper noise than the Frobenius norm.

    Args:
        A: First image matrix (H x W), float64.
        B: Second image matrix (H x W), float64. Same shape as A.

    Returns:
        Non-negative scalar representing the Manhattan distance.

    Raises:
        ValueError: If A and B have different shapes.
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A={A.shape}, B={B.shape}"
        )
    return float(np.sum(np.abs(A - B)))


def l_inf_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Compute the L-infinity (Chebyshev) norm of the difference matrix.

    The L-infinity norm is the maximum absolute pixel-wise difference:

        ||A - B||_inf = max_ij |a_ij - b_ij|

    Args:
        A: First image matrix (H x W), float64.
        B: Second image matrix (H x W), float64. Same shape as A.

    Returns:
        Non-negative scalar representing the maximum absolute difference.

    Raises:
        ValueError: If A and B have different shapes.
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A={A.shape}, B={B.shape}"
        )
    return float(np.max(np.abs(A - B)))


# ---------------------------------------------------------------------------
# Vector space geometry
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened image vectors.

    Measures the cosine of the angle between vectors a and b in N-dimensional
    pixel space. Values range from -1 to 1, where 1 indicates identical
    structural orientation regardless of brightness/contrast scaling.

        cos(theta) = (a . b) / (||a||_2 * ||b||_2)

    Args:
        a: Flattened image vector, shape (N,).
        b: Flattened image vector, shape (N,).

    Returns:
        Cosine similarity in [-1.0, 1.0].

    Raises:
        ValueError: If either vector has zero norm (undefined direction).
    """
    norm_a = np_norm(a)
    norm_b = np_norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            "Cosine similarity is undefined for zero-norm vectors."
        )

    return float(np.dot(a, b) / (norm_a * norm_b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product of two flattened image vectors.

    Args:
        a: Flattened image vector, shape (N,).
        b: Flattened image vector, shape (N,).

    Returns:
        Scalar dot product value.
    """
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# SVD Energy Comparison (Matrix Factorization)
# ---------------------------------------------------------------------------

@dataclass
class SVDEnergyResult:
    """Result container for SVD energy comparison.

    Attributes:
        singular_values_a: Singular values of image A, descending order.
        singular_values_b: Singular values of image B, descending order.
        energy_a: Cumulative energy (sum of squared singular values) for A.
        energy_b: Cumulative energy (sum of squared singular values) for B.
        energy_ratio: Ratio of energy_a to energy_b (1.0 = equal energy).
        cosine_distance_sv: Cosine distance between singular value vectors.
        top_k_singular_cosine: Cosine similarity of top-k singular values.
    """

    singular_values_a: np.ndarray
    singular_values_b: np.ndarray
    energy_a: float
    energy_b: float
    energy_ratio: float
    cosine_distance_sv: float
    top_k_singular_cosine: float


def svd_energy_comparison(
    A: np.ndarray,
    B: np.ndarray,
    top_k: int = 50,
) -> SVDEnergyResult:
    """Compare the SVD energy spectra of two image matrices.

    Decomposes both images via SVD (A = U Sigma V^T) and compares the
    distribution of singular values. The singular values represent the
    "energy" or importance of each structural feature.

    We compare:
    1. Total energy: sum(sigma_i^2) — proportional to Frobenius norm squared.
    2. Energy ratio: how the total energies relate.
    3. Cosine distance between singular value vectors.
    4. Cosine similarity of the top-k singular values (dominant features).

    Args:
        A: First image matrix (H x W), float64.
        B: Second image matrix (H x W), float64.
        top_k: Number of top singular values to compare for structural
                similarity. Defaults to 50.

    Returns:
        SVDEnergyResult with all comparison data.
    """
    # Compute SVD for both matrices
    # full_matrices=False is more efficient; we only need singular values
    _, sigma_a, _ = np.linalg.svd(A, full_matrices=False)
    _, sigma_b, _ = np.linalg.svd(B, full_matrices=False)

    # Total energy = sum of squared singular values = ||A||_F^2
    energy_a = float(np.sum(sigma_a ** 2))
    energy_b = float(np.sum(sigma_b ** 2))

    # Energy ratio (handle zero-energy edge case)
    if energy_b == 0.0:
        energy_ratio = float("inf") if energy_a > 0.0 else 1.0
    else:
        energy_ratio = energy_a / energy_b

    # Cosine distance between full singular value vectors
    # Pad shorter vector with zeros if lengths differ
    max_len = max(len(sigma_a), len(sigma_b))
    sv_a_padded = np.zeros(max_len, dtype=np.float64)
    sv_b_padded = np.zeros(max_len, dtype=np.float64)
    sv_a_padded[: len(sigma_a)] = sigma_a
    sv_b_padded[: len(sigma_b)] = sigma_b

    norm_a_sv = np_norm(sv_a_padded)
    norm_b_sv = np_norm(sv_b_padded)

    if norm_a_sv > 0.0 and norm_b_sv > 0.0:
        cosine_sim_sv = float(
            np.dot(sv_a_padded, sv_b_padded) / (norm_a_sv * norm_b_sv)
        )
    else:
        cosine_sim_sv = 0.0

    cosine_distance_sv = 1.0 - cosine_sim_sv  # distance = 1 - similarity

    # Top-k singular value cosine similarity
    k = min(top_k, len(sigma_a), len(sigma_b))
    if k > 0:
        top_a = sigma_a[:k]
        top_b = sigma_b[:k]
        norm_top_a = np_norm(top_a)
        norm_top_b = np_norm(top_b)
        if norm_top_a > 0.0 and norm_top_b > 0.0:
            top_k_cosine = float(
                np.dot(top_a, top_b) / (norm_top_a * norm_top_b)
            )
        else:
            top_k_cosine = 0.0
    else:
        top_k_cosine = 0.0

    return SVDEnergyResult(
        singular_values_a=sigma_a,
        singular_values_b=sigma_b,
        energy_a=energy_a,
        energy_b=energy_b,
        energy_ratio=energy_ratio,
        cosine_distance_sv=cosine_distance_sv,
        top_k_singular_cosine=top_k_cosine,
    )


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

_EPS: float = 1e-9


def zscore_normalize(vec: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization (z-score).

    Transforms the vector so that mean=0 and std=1:
        z = (v - mean(v)) / (std(v) + eps)

    Args:
        vec: Input vector (1D or any shape).

    Returns:
        Z-score normalized vector of same shape.
    """
    m = float(vec.mean())
    s = float(vec.std() + _EPS)
    return (vec - m) / s


# ---------------------------------------------------------------------------
# Masked metrics (extracted from gui_similarity.py)
# ---------------------------------------------------------------------------

def apply_mask_flat(
    img: np.ndarray,
    mask: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    """Extract masked pixels and optionally z-score normalize.

    Args:
        img: 2D image matrix.
        mask: Boolean mask, same shape as img.
        center: If True, apply z-score normalization to extracted values.

    Returns:
        1D array of masked (and optionally centered) pixel values.
    """
    v = img[mask].astype(np.float64)
    if v.size == 0:
        return np.zeros(1, dtype=np.float64)
    return zscore_normalize(v) if center else v


def cosine_masked(
    A: np.ndarray, B: np.ndarray, M: np.ndarray
) -> float:
    """Compute cosine similarity on masked, z-scored pixel vectors.

    Extracts pixels where mask M is True from both images, applies
    z-score normalization, then computes cosine similarity.

    Args:
        A: First image matrix (H x W).
        B: Second image matrix (H x W).
        M: Boolean mask (H x W).

    Returns:
        Cosine similarity in [-1, 1].
    """
    va = apply_mask_flat(A, M, center=True)
    vb = apply_mask_flat(B, M, center=True)
    na, nb = np_norm(va), np_norm(vb)
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def frobenius_norm_masked(
    A: np.ndarray, B: np.ndarray, M: np.ndarray
) -> float:
    """Compute normalized masked Frobenius distance as a similarity score.

    Returns a similarity score in [0, 100] where 100 = identical.

    Args:
        A: First image matrix (H x W).
        B: Second image matrix (H x W).
        M: Boolean mask (H x W).

    Returns:
        Similarity score in [0, 100].
    """
    W = M.astype(np.float64)
    num = float(((A - B) ** 2 * W).sum())
    den = float(((A * A + B * B) * W).sum() + _EPS)
    dist = np.sqrt(num) / np.sqrt(den)
    return 100.0 * (1.0 - min(1.0, dist))


# ---------------------------------------------------------------------------
# Patch-wise and PCA metrics (extracted from gui_similarity.py)
# ---------------------------------------------------------------------------

def patch_cosine_similarity(
    A: np.ndarray,
    B: np.ndarray,
    M: np.ndarray,
    patch_size: int = 16,
    stride: int = 8,
    min_coverage: float = 0.1,
) -> float:
    """Compute average patch-wise cosine similarity on masked images.

    Slides a window across the images, computes cosine similarity on
    each patch (masked and z-scored), and returns the mean.

    Args:
        A: First image matrix (H x W).
        B: Second image matrix (H x W).
        M: Boolean mask (H x W).
        patch_size: Size of each square patch.
        stride: Step size between patches.
        min_coverage: Minimum fraction of mask pixels in a patch
                      to include it in the average.

    Returns:
        Mean cosine similarity across patches, in [-1, 1].
    """
    H, W = A.shape
    if patch_size > H or patch_size > W:
        return 0.0

    scores: list[float] = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            m = M[i : i + patch_size, j : j + patch_size]
            if m.mean() < min_coverage:
                continue
            va = apply_mask_flat(A[i : i + patch_size, j : j + patch_size], m, True)
            vb = apply_mask_flat(B[i : i + patch_size, j : j + patch_size], m, True)
            na, nb = np_norm(va), np_norm(vb)
            if na < _EPS or nb < _EPS:
                continue
            scores.append(float(np.dot(va, vb) / (na * nb)))

    if not scores:
        return 0.0
    return float(np.mean(scores))


def pca_projection_similarity(
    A: np.ndarray,
    B: np.ndarray,
    M: np.ndarray,
    k: int = 20,
) -> float:
    """Compute PCA projection cosine similarity on masked images.

    Stacks the masked z-scored vectors from both images, performs PCA
    via SVD, then computes cosine similarity of the projections onto
    the top-k principal components.

    Args:
        A: First image matrix (H x W).
        B: Second image matrix (H x W).
        M: Boolean mask (H x W).
        k: Number of top principal components to use.

    Returns:
        Cosine similarity of PCA projections, in [-1, 1].
    """
    va = apply_mask_flat(A, M, center=True)
    vb = apply_mask_flat(B, M, center=True)

    # If vectors are identical, similarity is 1.0
    if float(np_norm(va - vb)) < 1e-8:
        return 1.0

    X = np.stack([va, vb], axis=1)
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu

    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    r = int((S > 1e-8).sum())

    if r == 0:
        # Degenerate case: fall back to direct cosine
        na, nb = np_norm(va), np_norm(vb)
        if na < _EPS or nb < _EPS:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    k = int(max(1, min(k, r)))
    Uk = U[:, :k]
    ak = Uk.T @ (va - mu[:, 0])
    bk = Uk.T @ (vb - mu[:, 0])
    na, nb = np_norm(ak), np_norm(bk)

    if na < _EPS or nb < _EPS:
        # Fall back to full-vector cosine
        na0, nb0 = np_norm(va), np_norm(vb)
        if na0 < _EPS or nb0 < _EPS:
            return 0.0
        return float(np.dot(va, vb) / (na0 * nb0))

    return float(np.dot(ak, bk) / (na * nb))


def svd_directional_similarity(
    A: np.ndarray,
    B: np.ndarray,
    M: np.ndarray,
    k: int = 20,
) -> float:
    """Compute SVD spectral + directional similarity (masked).

    Combines singular value similarity with the alignment of the
    principal left/right singular vectors (U and V columns).

    Returns a similarity score in [0, 100].

    Args:
        A: First image matrix (H x W).
        B: Second image matrix (H x W).
        M: Boolean mask (H x W).
        k: Number of top singular values to compare.

    Returns:
        Combined similarity score in [0, 100].
    """
    from src.loader import bbox_from_mask, crop_to_foreground

    # Crop to mask bounding box for efficient SVD
    bbox = bbox_from_mask(M)
    x0, y0, x1, y1 = bbox
    Ac = A[y0:y1, x0:x1]
    Bc = B[y0:y1, x0:x1]

    Ua, Sa, Va = np.linalg.svd(Ac, full_matrices=False)
    Ub, Sb, Vb = np.linalg.svd(Bc, full_matrices=False)

    k = int(max(1, min(k, len(Sa), len(Sb))))
    sa = Sa[:k].astype(np.float64)
    sb = Sb[:k].astype(np.float64)

    # Spectral similarity: 1 - normalized distance of singular values
    num = float(np_norm(sa - sb))
    den = float(np_norm(sa) + np_norm(sb) + _EPS)
    s_sigma = max(0.0, 1.0 - num / den)

    # Directional similarity: alignment of principal vectors
    cu = abs(float(np.dot(Ua[:, 0], Ub[:, 0])) /
             (np_norm(Ua[:, 0]) * np_norm(Ub[:, 0]) + _EPS))
    cv = abs(float(np.dot(Va[0, :], Vb[0, :])) /
             (np_norm(Va[0, :]) * np_norm(Vb[0, :]) + _EPS))
    s_vec = 0.5 * (cu + cv)

    return 100.0 * max(0.0, s_sigma * s_vec)
