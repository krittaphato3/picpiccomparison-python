"""
Tests for src/comparator.py — ImageComparator orchestrator and ComparisonReport.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.comparator import ComparisonReport, ImageComparator
from src.loader import ImageLoadError


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------

class TestComparisonReport:
    """Tests for the ComparisonReport dataclass."""

    def _make_report(self) -> ComparisonReport:
        """Create a minimal valid ComparisonReport."""
        from src.linalg_metrics import SVDEnergyResult
        return ComparisonReport(
            path_a="a.png",
            path_b="b.png",
            dimensions=(64, 64),
            frobenius=1.5,
            l1=10.0,
            l_inf=0.5,
            cosine_sim=0.95,
            dot_prod=1000.0,
            mse=0.01,
            psnr_db=20.0,
            hist_intersection=0.85,
            svd=SVDEnergyResult(
                singular_values_a=np.array([10.0, 5.0, 1.0]),
                singular_values_b=np.array([9.0, 4.5, 0.9]),
                energy_a=126.0,
                energy_b=101.61,
                energy_ratio=1.24,
                cosine_distance_sv=0.01,
                top_k_singular_cosine=0.99,
            ),
        )

    def test_to_dict_structure(self):
        """to_dict should have correct nested structure."""
        report = self._make_report()
        d = report.to_dict()
        assert "path_a" in d
        assert "path_b" in d
        assert "dimensions" in d
        assert "linear_algebra_metrics" in d
        assert "algorithmic_metrics" in d
        assert "svd_analysis" in d
        assert "frobenius_norm_l2" in d["linear_algebra_metrics"]
        assert "mse" in d["algorithmic_metrics"]
        assert "energy_a" in d["svd_analysis"]

    def test_to_json_valid(self):
        """to_json should produce valid JSON."""
        report = self._make_report()
        j = report.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_to_json_custom_indent(self):
        """to_json should respect indent parameter."""
        report = self._make_report()
        j = report.to_json(indent=4)
        assert "    " in j  # 4-space indent

    def test_summary_content(self):
        """summary should contain key metric names."""
        report = self._make_report()
        s = report.summary()
        assert "Frobenius" in s
        assert "L1" in s or "Manhattan" in s
        assert "Cosine" in s
        assert "MSE" in s
        assert "PSNR" in s
        assert "Histogram" in s
        assert "SVD" in s or "Energy" in s

    def test_summary_contains_paths(self):
        report = self._make_report()
        s = report.summary()
        assert "a.png" in s
        assert "b.png" in s


# ---------------------------------------------------------------------------
# ImageComparator.compare_images (array-based)
# ---------------------------------------------------------------------------

class TestComparatorCompareImages:
    """Tests for ImageComparator.compare_images (no file I/O)."""

    def test_returns_report(self, different_pair):
        a, b = different_pair
        comp = ImageComparator()
        report = comp.compare_images(a, b)
        assert isinstance(report, ComparisonReport)

    def test_identical_images_metrics(self, identical_pair):
        a, b = identical_pair
        comp = ImageComparator()
        report = comp.compare_images(a, b)
        assert report.mse == 0.0
        assert report.frobenius == 0.0
        assert report.l1 == 0.0
        assert abs(report.cosine_sim - 1.0) < 1e-10

    def test_shape_mismatch_raises(self):
        comp = ImageComparator()
        with pytest.raises(ValueError, match="must match"):
            comp.compare_images(np.ones((10, 10)), np.ones((20, 20)))

    def test_custom_labels(self, different_pair):
        a, b = different_pair
        comp = ImageComparator()
        report = comp.compare_images(a, b, path_a="custom_a", path_b="custom_b")
        assert report.path_a == "custom_a"
        assert report.path_b == "custom_b"

    def test_dimensions_stored(self, different_pair):
        a, b = different_pair
        comp = ImageComparator()
        report = comp.compare_images(a, b)
        assert report.dimensions == (64, 64)

    def test_svd_result_populated(self, different_pair):
        a, b = different_pair
        comp = ImageComparator()
        report = comp.compare_images(a, b)
        assert len(report.svd.singular_values_a) > 0
        assert len(report.svd.singular_values_b) > 0
        assert report.svd.energy_a > 0


# ---------------------------------------------------------------------------
# ImageComparator.compare (file-based)
# ---------------------------------------------------------------------------

class TestComparatorCompare:
    """Tests for ImageComparator.compare (file I/O)."""

    def test_compare_from_files(self, temp_image_a, temp_image_b):
        comp = ImageComparator()
        report = comp.compare(temp_image_a, temp_image_b)
        assert isinstance(report, ComparisonReport)
        assert report.dimensions == (64, 64)

    def test_nonexistent_file_raises(self, temp_dir):
        comp = ImageComparator()
        with pytest.raises(ImageLoadError):
            comp.compare(temp_dir / "missing.png", temp_dir / "missing2.png")

    def test_json_serializable(self, temp_image_a, temp_image_b):
        """Report from file comparison should be JSON-serializable."""
        comp = ImageComparator()
        report = comp.compare(temp_image_a, temp_image_b)
        j = report.to_json()
        parsed = json.loads(j)
        assert "linear_algebra_metrics" in parsed


# ---------------------------------------------------------------------------
# ImageComparator.save_report
# ---------------------------------------------------------------------------

class TestComparatorSaveReport:
    """Tests for saving reports to disk."""

    def test_save_report(self, different_pair, temp_dir):
        a, b = different_pair
        comp = ImageComparator()
        report = comp.compare_images(a, b)
        report_path = str(temp_dir / "report.json")
        saved = comp.save_report(report, report_path)
        assert Path(saved).exists()

        with open(saved) as f:
            data = json.load(f)
        assert "linear_algebra_metrics" in data

    def test_save_report_creates_dir(self, different_pair, temp_dir):
        """Should create parent directories if needed."""
        a, b = different_pair
        comp = ImageComparator()
        report = comp.compare_images(a, b)
        nested_path = str(temp_dir / "subdir" / "report.json")
        comp.save_report(report, nested_path)
        assert Path(nested_path).exists()


# ---------------------------------------------------------------------------
# ImageComparator.save_visualizations
# ---------------------------------------------------------------------------

class TestComparatorSaveVisualizations:
    """Tests for saving visualization plots to disk."""

    def test_saves_plots(self, temp_image_a, temp_image_b, temp_dir):
        comp = ImageComparator()
        report = comp.compare(temp_image_a, temp_image_b)
        output_dir = temp_dir / "plots"
        saved = comp.save_visualizations(report, output_dir)
        assert len(saved) == 5
        for p in saved:
            assert Path(p).exists()

    def test_plot_names(self, temp_image_a, temp_image_b, temp_dir):
        comp = ImageComparator()
        report = comp.compare(temp_image_a, temp_image_b)
        output_dir = temp_dir / "plots"
        saved = comp.save_visualizations(report, output_dir)
        filenames = [Path(p).name for p in saved]
        assert "original_images.png" in filenames
        assert "difference_heatmap.png" in filenames
        assert "svd_spectrum.png" in filenames
        assert "histogram_comparison.png" in filenames
        assert "full_dashboard.png" in filenames
