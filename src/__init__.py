"""
LinAlg-ImageCompare: Advanced Picture-to-Picture Comparison

A computational imaging library that quantifies image similarity using
rigorous Linear Algebra and Algorithmic techniques.
"""

from src.loader import (
    load_and_prepare_images,
    auto_foreground_mask,
    bbox_from_mask,
    crop_to_foreground,
)
from src.linalg_metrics import (
    frobenius_norm,
    l1_norm,
    l_inf_norm,
    cosine_similarity,
    dot_product,
    svd_energy_comparison,
    zscore_normalize,
    apply_mask_flat,
    cosine_masked,
    frobenius_norm_masked,
    patch_cosine_similarity,
    pca_projection_similarity,
    svd_directional_similarity,
)
from src.algo_metrics import (
    mse,
    psnr,
    histogram_intersection,
    difference_image,
    sobel_edge_magnitude,
    normalized_cross_correlation,
)
from src.visualizer import (
    plot_original_images,
    plot_difference_heatmap,
    plot_svd_spectrum,
    plot_histogram_comparison,
    plot_full_comparison,
)
from src.comparator import ImageComparator, ComparisonReport

__all__ = [
    # Loader
    "load_and_prepare_images",
    "auto_foreground_mask",
    "bbox_from_mask",
    "crop_to_foreground",
    # Linear algebra metrics
    "frobenius_norm",
    "l1_norm",
    "l_inf_norm",
    "cosine_similarity",
    "dot_product",
    "svd_energy_comparison",
    "zscore_normalize",
    "apply_mask_flat",
    "cosine_masked",
    "frobenius_norm_masked",
    "patch_cosine_similarity",
    "pca_projection_similarity",
    "svd_directional_similarity",
    # Algorithmic metrics
    "mse",
    "psnr",
    "histogram_intersection",
    "difference_image",
    "sobel_edge_magnitude",
    "normalized_cross_correlation",
    # Visualizer
    "plot_original_images",
    "plot_difference_heatmap",
    "plot_svd_spectrum",
    "plot_histogram_comparison",
    "plot_full_comparison",
    # Orchestrator
    "ImageComparator",
    "ComparisonReport",
]
