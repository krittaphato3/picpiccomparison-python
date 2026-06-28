"""
Tests for src/algo_metrics.py — algorithmic comparison metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo_metrics import (
    HistogramResult,
    difference_image,
    histogram_intersection,
    histogram_intersection_detailed,
    mse,
    psnr,
)


# ---------------------------------------------------------------------------
# mse
# ---------------------------------------------------------------------------

class TestMSE:
    """Tests for Mean Squared Error."""

    def test_identical_images(self, identical_pair):
        a, b = identical_pair
        assert mse(a, b) == 0.0

    def test_different_images(self, different_pair):
        a, b = different_pair
        assert mse(a, b) > 0.0

    def test_known_value(self, known_images):
        """Should match hand-calculated MSE."""
        zeros, single_diff = known_images["zeros"], known_images["single_diff"]
        result = mse(zeros, single_diff)
        assert abs(result - known_images["mse_zeros_B"]) < 1e-10

    def test_known_all_ones(self):
        """MSE of zeros vs ones (4x4) should be 1.0."""
        a = np.zeros((4, 4))
        b = np.ones((4, 4))
        assert abs(mse(a, b) - 1.0) < 1e-10

    def test_symmetry(self, different_pair):
        a, b = different_pair
        assert abs(mse(a, b) - mse(b, a)) < 1e-10

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            mse(np.ones((5, 5)), np.ones((10, 10)))

    def test_single_pixel(self):
        assert abs(mse(np.array([[0.5]]), np.array([[1.0]])) - 0.25) < 1e-10

    @pytest.mark.parametrize("offset", [0.1, 0.25, 0.5, 1.0])
    def test_uniform_offset(self, offset):
        """MSE of constant-offset images should be offset^2."""
        a = np.full((8, 8), 0.3)
        b = np.full((8, 8), 0.3 + offset)
        assert abs(mse(a, b) - offset ** 2) < 1e-10


# ---------------------------------------------------------------------------
# psnr
# ---------------------------------------------------------------------------

class TestPSNR:
    """Tests for Peak Signal-to-Noise Ratio."""

    def test_identical_images(self, identical_pair):
        a, b = identical_pair
        assert psnr(a, b) == float("inf")

    def test_different_images(self, different_pair):
        a, b = different_pair
        assert psnr(a, b) > 0.0

    def test_known_relationship(self):
        """PSNR should decrease as MSE increases."""
        a = np.full((16, 16), 0.5)
        b_low_noise = a + 0.01
        b_high_noise = a + 0.1
        assert psnr(a, b_low_noise) > psnr(a, b_high_noise)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            psnr(np.ones((5, 5)), np.ones((10, 10)))

    def test_psnr_formula(self):
        """Verify PSNR = 10 * log10(MAX^2 / MSE)."""
        a = np.full((8, 8), 0.5)
        b = a.copy()
        b[0, 0] = 0.6
        mse_val = mse(a, b)
        expected_psnr = 10.0 * np.log10(1.0 ** 2 / mse_val)
        assert abs(psnr(a, b) - expected_psnr) < 1e-10


# ---------------------------------------------------------------------------
# histogram_intersection
# ---------------------------------------------------------------------------

class TestHistogramIntersection:
    """Tests for histogram intersection."""

    def test_identical_images(self, identical_pair):
        """Identical images should have intersection ~1.0."""
        a, b = identical_pair
        hi = histogram_intersection(a, b)
        assert abs(hi - 1.0) < 1e-6

    def test_range(self, different_pair):
        """Result should be in [0, 1]."""
        a, b = different_pair
        hi = histogram_intersection(a, b)
        assert 0.0 <= hi <= 1.0

    def test_uniform_images(self):
        """Uniform images should have high intersection."""
        a = np.full((32, 32), 0.5)
        b = np.full((32, 32), 0.5)
        assert abs(histogram_intersection(a, b) - 1.0) < 1e-6

    def test_orthogonal_distributions(self):
        """Completely non-overlapping distributions should have low intersection."""
        a = np.zeros((32, 32))
        b = np.ones((32, 32))
        hi = histogram_intersection(a, b)
        # Should be very low (only edge bins might overlap due to binning)
        assert hi < 0.1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            histogram_intersection(np.ones((5, 5)), np.ones((10, 10)))

    def test_bins_parameter(self, different_pair):
        """Should accept custom bin count."""
        a, b = different_pair
        hi_64 = histogram_intersection(a, b, bins=64)
        hi_128 = histogram_intersection(a, b, bins=128)
        assert 0.0 <= hi_64 <= 1.0
        assert 0.0 <= hi_128 <= 1.0


# ---------------------------------------------------------------------------
# histogram_intersection_detailed
# ---------------------------------------------------------------------------

class TestHistogramIntersectionDetailed:
    """Tests for detailed histogram intersection."""

    def test_returns_result(self, different_pair):
        a, b = different_pair
        result = histogram_intersection_detailed(a, b)
        assert isinstance(result, HistogramResult)
        assert 0.0 <= result.intersection <= 1.0

    def test_histogram_shapes(self, different_pair):
        """Histograms should have correct shape."""
        a, b = different_pair
        result = histogram_intersection_detailed(a, b, bins=128)
        assert result.hist_a.shape == (128,)
        assert result.hist_b.shape == (128,)
        assert result.bin_edges.shape == (129,)  # edges = bins + 1

    def test_histograms_sum_to_one(self, different_pair):
        """Normalized histograms should sum to ~1.0."""
        a, b = different_pair
        result = histogram_intersection_detailed(a, b)
        assert abs(result.hist_a.sum() - 1.0) < 1e-10
        assert abs(result.hist_b.sum() - 1.0) < 1e-10

    def test_consistency_with_simple(self, different_pair):
        """Detailed result intersection should match simple function."""
        a, b = different_pair
        simple = histogram_intersection(a, b)
        detailed = histogram_intersection_detailed(a, b)
        assert abs(simple - detailed.intersection) < 1e-10


# ---------------------------------------------------------------------------
# difference_image
# ---------------------------------------------------------------------------

class TestDifferenceImage:
    """Tests for difference image computation."""

    def test_identical_images(self, identical_pair):
        """Identical images should produce all-zero difference."""
        a, b = identical_pair
        diff = difference_image(a, b)
        assert np.allclose(diff, 0.0)

    def test_known_value(self):
        """Should compute element-wise absolute difference."""
        a = np.array([[0.0, 0.5], [1.0, 0.3]])
        b = np.array([[0.1, 0.5], [0.8, 0.3]])
        diff = difference_image(a, b)
        expected = np.array([[0.1, 0.0], [0.2, 0.0]])
        assert np.allclose(diff, expected)

    def test_output_shape(self, different_pair):
        a, b = different_pair
        diff = difference_image(a, b)
        assert diff.shape == a.shape

    def test_non_negative(self, different_pair):
        a, b = different_pair
        diff = difference_image(a, b)
        assert np.all(diff >= 0.0)

    def test_symmetry(self, different_pair):
        """|A-B| should equal |B-A|."""
        a, b = different_pair
        assert np.allclose(difference_image(a, b), difference_image(b, a))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            difference_image(np.ones((5, 5)), np.ones((10, 10)))
