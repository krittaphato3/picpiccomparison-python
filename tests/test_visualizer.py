"""
Tests for src/visualizer.py — matplotlib visualization functions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualizer import (
    plot_difference_heatmap,
    plot_full_comparison,
    plot_histogram_comparison,
    plot_original_images,
    plot_svd_spectrum,
)


# ---------------------------------------------------------------------------
# plot_original_images
# ---------------------------------------------------------------------------

class TestPlotOriginalImages:
    """Tests for the original images side-by-side plot."""

    def test_returns_figure(self, small_image, small_image_b):
        fig = plot_original_images(small_image, small_image_b)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_titles(self, small_image, small_image_b):
        fig = plot_original_images(
            small_image, small_image_b,
            title_a="Custom A", title_b="Custom B"
        )
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, small_image, small_image_b, temp_dir):
        save_path = str(temp_dir / "originals.png")
        fig = plot_original_images(small_image, small_image_b, save_path=save_path)
        assert Path(save_path).exists()
        plt.close(fig)

    def test_custom_figsize(self, small_image, small_image_b):
        fig = plot_original_images(small_image, small_image_b, figsize=(8, 4))
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_difference_heatmap
# ---------------------------------------------------------------------------

class TestPlotDifferenceHeatmap:
    """Tests for the difference heatmap plot."""

    def test_returns_figure(self, different_pair):
        a, b = different_pair
        diff = np.abs(a - b)
        fig = plot_difference_heatmap(diff)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_to_file(self, different_pair, temp_dir):
        a, b = different_pair
        diff = np.abs(a - b)
        save_path = str(temp_dir / "heatmap.png")
        fig = plot_difference_heatmap(diff, save_path=save_path)
        assert Path(save_path).exists()
        plt.close(fig)

    def test_custom_colormap(self, different_pair):
        a, b = different_pair
        diff = np.abs(a - b)
        fig = plot_difference_heatmap(diff, cmap="viridis")
        assert fig is not None
        plt.close(fig)

    def test_zero_difference(self):
        """Zero difference should still produce a valid figure."""
        diff = np.zeros((32, 32))
        fig = plot_difference_heatmap(diff)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_svd_spectrum
# ---------------------------------------------------------------------------

class TestPlotSVDSpectrum:
    """Tests for the SVD singular value spectrum plot."""

    def test_returns_figure(self):
        sigma_a = np.sort(np.random.RandomState(42).rand(20))[::-1]
        sigma_b = np.sort(np.random.RandomState(43).rand(15))[::-1]
        fig = plot_svd_spectrum(sigma_a, sigma_b)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_top_k_filter(self):
        sigma_a = np.sort(np.random.RandomState(42).rand(50))[::-1]
        sigma_b = np.sort(np.random.RandomState(43).rand(50))[::-1]
        fig = plot_svd_spectrum(sigma_a, sigma_b, top_k=10)
        assert fig is not None
        plt.close(fig)

    def test_different_lengths(self):
        """Should handle different-length singular value arrays."""
        sigma_a = np.array([5.0, 3.0, 1.0])
        sigma_b = np.array([4.0, 2.0])
        fig = plot_svd_spectrum(sigma_a, sigma_b)
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, temp_dir):
        sigma_a = np.array([5.0, 3.0, 1.0])
        sigma_b = np.array([4.0, 2.0])
        save_path = str(temp_dir / "svd.png")
        fig = plot_svd_spectrum(sigma_a, sigma_b, save_path=save_path)
        assert Path(save_path).exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_histogram_comparison
# ---------------------------------------------------------------------------

class TestPlotHistogramComparison:
    """Tests for the histogram comparison plot."""

    def test_returns_figure(self, different_pair):
        a, b = different_pair
        hist_a = np.random.rand(256)
        hist_a /= hist_a.sum()
        hist_b = np.random.rand(256)
        hist_b /= hist_b.sum()
        bin_edges = np.linspace(0, 256, 257)
        fig = plot_histogram_comparison(hist_a, hist_b, bin_edges, 0.85)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_to_file(self, temp_dir):
        hist_a = np.random.rand(256)
        hist_a /= hist_a.sum()
        hist_b = np.random.rand(256)
        hist_b /= hist_b.sum()
        bin_edges = np.linspace(0, 256, 257)
        save_path = str(temp_dir / "hist.png")
        fig = plot_histogram_comparison(
            hist_a, hist_b, bin_edges, 0.85, save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_full_comparison
# ---------------------------------------------------------------------------

class TestPlotFullComparison:
    """Tests for the full comparison dashboard."""

    def test_returns_figure(self, different_pair):
        a, b = different_pair
        diff = np.abs(a - b)
        sigma_a = np.sort(np.random.RandomState(42).rand(20))[::-1]
        sigma_b = np.sort(np.random.RandomState(43).rand(20))[::-1]
        fig = plot_full_comparison(a, b, diff, sigma_a, sigma_b)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_to_file(self, different_pair, temp_dir):
        a, b = different_pair
        diff = np.abs(a - b)
        sigma_a = np.sort(np.random.RandomState(42).rand(20))[::-1]
        sigma_b = np.sort(np.random.RandomState(43).rand(20))[::-1]
        save_path = str(temp_dir / "dashboard.png")
        fig = plot_full_comparison(a, b, diff, sigma_a, sigma_b, save_path=save_path)
        assert Path(save_path).exists()
        plt.close(fig)
