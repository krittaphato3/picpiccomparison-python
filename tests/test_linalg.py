"""
Tests for src/linalg_metrics.py — linear algebra comparison metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.linalg_metrics import (
    SVDEnergyResult,
    cosine_similarity,
    dot_product,
    frobenius_norm,
    l1_norm,
    l_inf_norm,
    svd_energy_comparison,
)


# ---------------------------------------------------------------------------
# frobenius_norm
# ---------------------------------------------------------------------------

class TestFrobeniusNorm:
    """Tests for the Frobenius (L2) norm."""

    def test_identical_images(self, identical_pair):
        """Identical images should have Frobenius norm of 0."""
        a, b = identical_pair
        assert frobenius_norm(a, b) == 0.0

    def test_different_images(self, different_pair):
        """Different images should have positive norm."""
        a, b = different_pair
        assert frobenius_norm(a, b) > 0.0

    def test_known_value(self, known_images):
        """Should match hand-calculated Frobenius norm."""
        zeros, single_diff = known_images["zeros"], known_images["single_diff"]
        result = frobenius_norm(zeros, single_diff)
        assert abs(result - known_images["frobenius_zeros_B"]) < 1e-10

    def test_symmetry(self, different_pair):
        """frobenius(A, B) == frobenius(B, A)."""
        a, b = different_pair
        assert abs(frobenius_norm(a, b) - frobenius_norm(b, a)) < 1e-10

    def test_shape_mismatch_raises(self):
        """Should raise ValueError for different shapes."""
        a = np.ones((10, 10))
        b = np.ones((20, 20))
        with pytest.raises(ValueError, match="Shape mismatch"):
            frobenius_norm(a, b)

    def test_single_pixel(self):
        """Should work with 1x1 images."""
        a = np.array([[0.3]])
        b = np.array([[0.7]])
        assert abs(frobenius_norm(a, b) - 0.4) < 1e-10

    def test_triangle_inequality(self, different_pair):
        """Frobenius norm should satisfy triangle inequality."""
        a, b = different_pair
        c = np.ones_like(a) * 0.5
        # ||A-C|| <= ||A-B|| + ||B-C||
        lhs = frobenius_norm(a, c)
        rhs = frobenius_norm(a, b) + frobenius_norm(b, c)
        assert lhs <= rhs + 1e-10

    @pytest.mark.parametrize("scale", [0.1, 0.5, 2.0, 10.0])
    def test_scaling(self, scale):
        """||scale*A - scale*B|| = scale * ||A - B||."""
        a = np.random.RandomState(42).rand(8, 8)
        b = np.random.RandomState(43).rand(8, 8)
        assert abs(
            frobenius_norm(scale * a, scale * b) - scale * frobenius_norm(a, b)
        ) < 1e-10


# ---------------------------------------------------------------------------
# l1_norm
# ---------------------------------------------------------------------------

class TestL1Norm:
    """Tests for the L1 (Manhattan) norm."""

    def test_identical_images(self, identical_pair):
        a, b = identical_pair
        assert l1_norm(a, b) == 0.0

    def test_different_images(self, different_pair):
        a, b = different_pair
        assert l1_norm(a, b) > 0.0

    def test_known_value(self, known_images):
        zeros, single_diff = known_images["zeros"], known_images["single_diff"]
        assert abs(l1_norm(zeros, single_diff) - known_images["l1_zeros_B"]) < 1e-10

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            l1_norm(np.ones((5, 5)), np.ones((10, 10)))

    def test_symmetry(self, different_pair):
        a, b = different_pair
        assert abs(l1_norm(a, b) - l1_norm(b, a)) < 1e-10

    def test_known_all_ones(self):
        """L1 norm of all-zeros vs all-ones (4x4) should be 16."""
        a = np.zeros((4, 4))
        b = np.ones((4, 4))
        assert abs(l1_norm(a, b) - 16.0) < 1e-10


# ---------------------------------------------------------------------------
# l_inf_norm
# ---------------------------------------------------------------------------

class TestLInfNorm:
    """Tests for the L-infinity (Chebyshev) norm."""

    def test_identical_images(self, identical_pair):
        a, b = identical_pair
        assert l_inf_norm(a, b) == 0.0

    def test_positive_for_different(self, different_pair):
        a, b = different_pair
        assert l_inf_norm(a, b) > 0.0

    def test_known_value(self, known_images):
        zeros, single_diff = known_images["zeros"], known_images["single_diff"]
        assert abs(l_inf_norm(zeros, single_diff) - known_images["l_inf_zeros_B"]) < 1e-10

    def test_range_normalized(self, different_pair):
        """For [0,1] images, L-inf should be in [0, 1]."""
        a, b = different_pair
        val = l_inf_norm(a, b)
        assert 0.0 <= val <= 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            l_inf_norm(np.ones((3, 3)), np.ones((5, 5)))


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Tests for cosine similarity."""

    def test_identical_vectors(self, identical_pair):
        """Identical images should have cosine similarity ~1.0."""
        a, b = identical_pair
        sim = cosine_similarity(a.flatten(), b.flatten())
        assert abs(sim - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have cosine similarity ~0.0."""
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 1e-10

    def test_known_value(self, known_vectors):
        """Should match known cosine similarity values."""
        a, b = known_vectors["a"], known_vectors["b"]
        assert abs(cosine_similarity(a, b) - known_vectors["cosine_ab"]) < 1e-10

    def test_range(self, different_pair):
        """Cosine similarity should be in [-1, 1]."""
        a, b = different_pair
        sim = cosine_similarity(a.flatten(), b.flatten())
        assert -1.0 <= sim <= 1.0

    def test_zero_norm_raises(self):
        """Should raise ValueError for zero-norm vector."""
        a = np.zeros(10)
        b = np.ones(10)
        with pytest.raises(ValueError, match="zero-norm"):
            cosine_similarity(a, b)

    def test_both_zero_norm_raises(self):
        """Should raise ValueError when both vectors are zero."""
        a = np.zeros(10)
        b = np.zeros(10)
        with pytest.raises(ValueError, match="zero-norm"):
            cosine_similarity(a, b)

    def test_anti_parallel(self):
        """Opposite vectors should have cosine similarity -1."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-10


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------

class TestDotProduct:
    """Tests for dot product."""

    def test_known_value(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert abs(dot_product(a, b) - 32.0) < 1e-10

    def test_zero_for_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(dot_product(a, b)) < 1e-10

    def test_symmetry(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert abs(dot_product(a, b) - dot_product(b, a)) < 1e-10


# ---------------------------------------------------------------------------
# svd_energy_comparison
# ---------------------------------------------------------------------------

class TestSVDEnergyComparison:
    """Tests for SVD energy comparison."""

    def test_identical_images(self, identical_pair):
        """Identical images should have energy ratio ~1.0 and distance ~0."""
        a, b = identical_pair
        result = svd_energy_comparison(a, b)
        assert isinstance(result, SVDEnergyResult)
        assert abs(result.energy_ratio - 1.0) < 1e-10
        assert abs(result.cosine_distance_sv) < 1e-10

    def test_energy_positive(self, different_pair):
        """Energy should be positive for non-zero images."""
        a, b = different_pair
        result = svd_energy_comparison(a, b)
        assert result.energy_a > 0
        assert result.energy_b > 0

    def test_singular_values_descending(self, different_pair):
        """Singular values should be in descending order."""
        a, b = different_pair
        result = svd_energy_comparison(a, b)
        assert np.all(np.diff(result.singular_values_a) <= 0)
        assert np.all(np.diff(result.singular_values_b) <= 0)

    def test_cosine_distance_range(self, different_pair):
        """Cosine distance should be in [0, 1]."""
        a, b = different_pair
        result = svd_energy_comparison(a, b)
        assert 0.0 <= result.cosine_distance_sv <= 1.0

    def test_top_k_parameter(self, different_pair):
        """Should respect top_k parameter."""
        a, b = different_pair
        result = svd_energy_comparison(a, b, top_k=5)
        # top_k_cosine should still be valid
        assert -1.0 <= result.top_k_singular_cosine <= 1.0

    def test_energy_ratio_identity(self):
        """Energy ratio of A vs 2*A should be 0.25."""
        rng = np.random.RandomState(42)
        a = rng.rand(16, 16)
        b = a * 2.0
        result = svd_energy_comparison(a, b)
        # Energy of b = 4 * energy of a, so ratio = 0.25
        assert abs(result.energy_ratio - 0.25) < 1e-10

    def test_zero_energy_handling(self):
        """Should handle zero-energy images gracefully."""
        a = np.zeros((8, 8))
        b = np.zeros((8, 8))
        result = svd_energy_comparison(a, b)
        assert result.energy_a == 0.0
        assert result.energy_b == 0.0
        assert result.energy_ratio == 1.0  # 0/0 -> 1.0 by design
