"""
Main orchestrator class for image comparison.

Ties together the loader, linear algebra metrics, algorithmic metrics,
and visualizer modules into a single high-level API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.loader import (
    ImagePair,
    ImageLoadError,
    DimensionMismatchError,
    load_and_prepare_images,
)
from src.linalg_metrics import (
    frobenius_norm,
    l1_norm,
    l_inf_norm,
    cosine_similarity,
    dot_product,
    svd_energy_comparison,
    SVDEnergyResult,
)
from src.algo_metrics import (
    mse,
    psnr,
    histogram_intersection,
    histogram_intersection_detailed,
    difference_image,
    HistogramResult,
)
from src.visualizer import (
    plot_original_images,
    plot_difference_heatmap,
    plot_svd_spectrum,
    plot_histogram_comparison,
    plot_full_comparison,
)


@dataclass
class ComparisonReport:
    """Complete comparison report containing all metrics and results.

    Attributes:
        path_a: Path to image A.
        path_b: Path to image B.
        dimensions: (height, width) of processed images.
        frobenius: Frobenius (L2) norm of difference.
        l1: L1 (Manhattan) norm of difference.
        l_inf: L-infinity (Chebyshev) norm of difference.
        cosine_sim: Cosine similarity between flattened vectors.
        dot_prod: Dot product of flattened vectors.
        mse: Mean Squared Error.
        psnr_db: Peak Signal-to-Noise Ratio in dB.
        hist_intersection: Histogram intersection score.
        svd: SVDEnergyResult with full SVD comparison data.
    """

    path_a: str
    path_b: str
    dimensions: Tuple[int, int]
    frobenius: float
    l1: float
    l_inf: float
    cosine_sim: float
    dot_prod: float
    mse: float
    psnr_db: float
    hist_intersection: float
    svd: SVDEnergyResult = field(default_factory=lambda: SVDEnergyResult(
        singular_values_a=np.array([]),
        singular_values_b=np.array([]),
        energy_a=0.0,
        energy_b=0.0,
        energy_ratio=0.0,
        cosine_distance_sv=0.0,
        top_k_singular_cosine=0.0,
    ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a JSON-serializable dictionary.

        Returns:
            Dictionary with all scalar metrics. SVD arrays are converted
            to lists for serialization.
        """
        svd_dict = {
            "energy_a": self.svd.energy_a,
            "energy_b": self.svd.energy_b,
            "energy_ratio": self.svd.energy_ratio,
            "cosine_distance_sv": self.svd.cosine_distance_sv,
            "top_k_singular_cosine": self.svd.top_k_singular_cosine,
            "num_singular_values_a": len(self.svd.singular_values_a),
            "num_singular_values_b": len(self.svd.singular_values_b),
            "top_10_singular_values_a": self.svd.singular_values_a[:10].tolist(),
            "top_10_singular_values_b": self.svd.singular_values_b[:10].tolist(),
        }

        return {
            "path_a": self.path_a,
            "path_b": self.path_b,
            "dimensions": {"height": self.dimensions[0], "width": self.dimensions[1]},
            "linear_algebra_metrics": {
                "frobenius_norm_l2": self.frobenius,
                "l1_norm_manhattan": self.l1,
                "l_inf_norm_chebyshev": self.l_inf,
                "cosine_similarity": self.cosine_sim,
                "dot_product": self.dot_prod,
            },
            "svd_analysis": svd_dict,
            "algorithmic_metrics": {
                "mse": self.mse,
                "psnr_db": self.psnr_db,
                "histogram_intersection": self.hist_intersection,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the report.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate a human-readable summary string.

        Returns:
            Formatted multi-line string with all key metrics.
        """
        lines = [
            "=" * 60,
            "  PicPicComparison Report",
            "=" * 60,
            f"  Image A:     {self.path_a}",
            f"  Image B:     {self.path_b}",
            f"  Dimensions:  {self.dimensions[0]} x {self.dimensions[1]}",
            "-" * 60,
            "  Linear Algebra Metrics:",
            f"    Frobenius (L2):          {self.frobenius:.6f}",
            f"    L1 (Manhattan):          {self.l1:.6f}",
            f"    L-infinity (Chebyshev):  {self.l_inf:.6f}",
            f"    Cosine Similarity:       {self.cosine_sim:.6f}",
            f"    Dot Product:             {self.dot_prod:.6f}",
            "-" * 60,
            "  SVD Analysis:",
            f"    Energy A:                {self.svd.energy_a:.6f}",
            f"    Energy B:                {self.svd.energy_b:.6f}",
            f"    Energy Ratio (A/B):      {self.svd.energy_ratio:.6f}",
            f"    Cosine Distance (SVs):   {self.svd.cosine_distance_sv:.6f}",
            f"    Top-K Cosine Sim:        {self.svd.top_k_singular_cosine:.6f}",
            "-" * 60,
            "  Algorithmic Metrics:",
            f"    MSE:                     {self.mse:.6f}",
            f"    PSNR (dB):               {self.psnr_db:.2f}",
            f"    Histogram Intersection:  {self.hist_intersection:.6f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class ImageComparator:
    """Main orchestrator class that ties all modules together.

    Provides a high-level API for comparing two images using both
    linear algebra and algorithmic techniques.

    Usage:
        comparator = ImageComparator()
        report = comparator.compare("image_a.png", "image_b.png")
        print(report.summary())
        comparator.save_visualizations(report, output_dir="output/")
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        svd_top_k: int = 50,
        histogram_bins: int = 256,
    ) -> None:
        """Initialize the comparator.

        Args:
            target_size: Optional (width, height) to resize both images to.
                        If None, image B is resized to match image A.
            svd_top_k: Number of top singular values for structural comparison.
            histogram_bins: Number of bins for histogram computation.
        """
        self.target_size = target_size
        self.svd_top_k = svd_top_k
        self.histogram_bins = histogram_bins

    def compare(
        self,
        path_a: Union[str, Path],
        path_b: Union[str, Path],
    ) -> ComparisonReport:
        """Run the full comparison pipeline on two images.

        Loads, preprocesses, and computes all metrics for the image pair.

        Args:
            path_a: Path to the first image.
            path_b: Path to the second image.

        Returns:
            ComparisonReport containing all computed metrics.

        Raises:
            ImageLoadError: If either image cannot be loaded.
            DimensionMismatchError: If images can't be matched in size.
        """
        # Step 1: Load and preprocess
        pair = load_and_prepare_images(path_a, path_b, self.target_size)

        # Step 2: Linear algebra metrics
        frob = frobenius_norm(pair.image_a, pair.image_b)
        l1 = l1_norm(pair.image_a, pair.image_b)
        l_inf = l_inf_norm(pair.image_a, pair.image_b)
        cos_sim = cosine_similarity(pair.vector_a, pair.vector_b)
        dp = dot_product(pair.vector_a, pair.vector_b)
        svd_result = svd_energy_comparison(
            pair.image_a, pair.image_b, top_k=self.svd_top_k
        )

        # Step 3: Algorithmic metrics
        mse_val = mse(pair.image_a, pair.image_b)
        psnr_val = psnr(pair.image_a, pair.image_b)
        hist_int = histogram_intersection(
            pair.image_a, pair.image_b, bins=self.histogram_bins
        )

        return ComparisonReport(
            path_a=pair.path_a,
            path_b=pair.path_b,
            dimensions=(pair.height, pair.width),
            frobenius=frob,
            l1=l1,
            l_inf=l_inf,
            cosine_sim=cos_sim,
            dot_prod=dp,
            mse=mse_val,
            psnr_db=psnr_val,
            hist_intersection=hist_int,
            svd=svd_result,
        )

    def compare_images(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        path_a: str = "<array_a>",
        path_b: str = "<array_b>",
    ) -> ComparisonReport:
        """Compare two pre-loaded image arrays directly.

        Useful when images are already loaded as numpy arrays.
        Images must have matching dimensions.

        Args:
            img_a: First image matrix (H x W), float64 in [0, 1].
            img_b: Second image matrix (H x W), float64 in [0, 1].
            path_a: Label for image A (for reporting).
            path_b: Label for image B (for reporting).

        Returns:
            ComparisonReport containing all computed metrics.

        Raises:
            ValueError: If images have different shapes.
        """
        if img_a.shape != img_b.shape:
            raise ValueError(
                f"Image dimensions must match: A={img_a.shape}, B={img_b.shape}"
            )

        # Ensure float64
        img_a = img_a.astype(np.float64)
        img_b = img_b.astype(np.float64)

        h, w = img_a.shape

        # Linear algebra metrics
        frob = frobenius_norm(img_a, img_b)
        l1_val = l1_norm(img_a, img_b)
        l_inf_val = l_inf_norm(img_a, img_b)
        cos_sim = cosine_similarity(img_a.flatten(), img_b.flatten())
        dp = dot_product(img_a.flatten(), img_b.flatten())
        svd_result = svd_energy_comparison(img_a, img_b, top_k=self.svd_top_k)

        # Algorithmic metrics
        mse_val = mse(img_a, img_b)
        psnr_val = psnr(img_a, img_b)
        hist_int = histogram_intersection(img_a, img_b, bins=self.histogram_bins)

        return ComparisonReport(
            path_a=path_a,
            path_b=path_b,
            dimensions=(h, w),
            frobenius=frob,
            l1=l1_val,
            l_inf=l_inf_val,
            cosine_sim=cos_sim,
            dot_prod=dp,
            mse=mse_val,
            psnr_db=psnr_val,
            hist_intersection=hist_int,
            svd=svd_result,
        )

    def save_visualizations(
        self,
        report: ComparisonReport,
        output_dir: Union[str, Path] = "output",
    ) -> List[str]:
        """Generate and save all visualization plots for a comparison report.

        Creates individual plots and a combined dashboard, saving them to disk.

        Args:
            report: A ComparisonReport from compare() or compare_images().
            output_dir: Directory to save the plots.

        Returns:
            List of file paths where plots were saved.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load images again for visualization
        pair = load_and_prepare_images(
            report.path_a, report.path_b, self.target_size
        )

        diff = difference_image(pair.image_a, pair.image_b)
        saved_paths: List[str] = []

        # Individual plots
        p1 = str(output_dir / "original_images.png")
        plot_original_images(
            pair.image_a, pair.image_b,
            title_a=f"Image A: {Path(report.path_a).name}",
            title_b=f"Image B: {Path(report.path_b).name}",
            save_path=p1,
        )
        saved_paths.append(p1)

        p2 = str(output_dir / "difference_heatmap.png")
        plot_difference_heatmap(diff, save_path=p2)
        saved_paths.append(p2)

        p3 = str(output_dir / "svd_spectrum.png")
        plot_svd_spectrum(
            report.svd.singular_values_a,
            report.svd.singular_values_b,
            top_k=self.svd_top_k,
            save_path=p3,
        )
        saved_paths.append(p3)

        # Histogram comparison
        hist_result = histogram_intersection_detailed(
            pair.image_a, pair.image_b, bins=self.histogram_bins
        )
        p4 = str(output_dir / "histogram_comparison.png")
        plot_histogram_comparison(
            hist_result.hist_a, hist_result.hist_b,
            hist_result.bin_edges, hist_result.intersection,
            save_path=p4,
        )
        saved_paths.append(p4)

        # Full dashboard
        p5 = str(output_dir / "full_dashboard.png")
        plot_full_comparison(
            pair.image_a, pair.image_b, diff,
            report.svd.singular_values_a, report.svd.singular_values_b,
            save_path=p5,
        )
        saved_paths.append(p5)

        return saved_paths

    def save_report(
        self,
        report: ComparisonReport,
        output_path: Union[str, Path] = "output/report.json",
    ) -> str:
        """Save the comparison report to a JSON file.

        Args:
            report: A ComparisonReport from compare() or compare_images().
            output_path: Path for the JSON output file.

        Returns:
            The path where the report was saved.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.to_json(indent=2))

        return str(output_path)
