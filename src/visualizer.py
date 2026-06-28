"""
Visualization module for image comparison results.

Creates matplotlib figures showing original images, difference heatmaps,
SVD singular value spectrums, and histogram comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_original_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    title_a: str = "Image A",
    title_b: str = "Image B",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot the two original images side by side.

    Args:
        img_a: First image matrix (H x W), float64 in [0, 1].
        img_b: Second image matrix (H x W), float64 in [0, 1].
        title_a: Title for the first image subplot.
        title_b: Title for the second image subplot.
        save_path: Optional path to save the figure. If None, not saved.
        figsize: Figure size as (width, height) in inches.

    Returns:
        The matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(img_a, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(title_a, fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_b, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(title_b, fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Original Image Pair", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_difference_heatmap(
    diff: np.ndarray,
    title: str = "Pixel Difference Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "hot",
) -> plt.Figure:
    """Plot a heatmap of the absolute pixel difference.

    Args:
        diff: Absolute difference matrix (H x W), float64 >= 0.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size as (width, height) in inches.
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(diff, cmap=cmap, vmin=0.0, vmax=max(diff.max(), 1e-10))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Difference", fontsize=10)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_svd_spectrum(
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    top_k: Optional[int] = None,
    title: str = "SVD Singular Value Spectrum",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot the singular value spectrums of both images.

    Shows the energy distribution across singular values, which represents
    the structural feature importance of each image.

    Args:
        sigma_a: Singular values of image A (sorted descending).
        sigma_b: Singular values of image B (sorted descending).
        top_k: If set, only show the top-k singular values. Default shows all.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size as (width, height) in inches.

    Returns:
        The matplotlib Figure object.
    """
    if top_k is not None:
        sigma_a = sigma_a[:top_k]
        sigma_b = sigma_b[:top_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Line plot of singular values
    indices_a = np.arange(1, len(sigma_a) + 1)
    indices_b = np.arange(1, len(sigma_b) + 1)

    ax1.plot(indices_a, sigma_a, "b-o", markersize=2, label="Image A", alpha=0.8)
    ax1.plot(indices_b, sigma_b, "r-s", markersize=2, label="Image B", alpha=0.8)
    ax1.set_xlabel("Index", fontsize=10)
    ax1.set_ylabel("Singular Value (Sigma)", fontsize=10)
    ax1.set_title("Singular Values", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Cumulative energy plot
    energy_a = np.cumsum(sigma_a ** 2)
    energy_b = np.cumsum(sigma_b ** 2)

    # Normalize to [0, 1]
    if energy_a[-1] > 0:
        energy_a_norm = energy_a / energy_a[-1]
    else:
        energy_a_norm = energy_a

    if energy_b[-1] > 0:
        energy_b_norm = energy_b / energy_b[-1]
    else:
        energy_b_norm = energy_b

    ax2.plot(indices_a, energy_a_norm, "b-", linewidth=2, label="Image A", alpha=0.8)
    ax2.plot(indices_b, energy_b_norm, "r-", linewidth=2, label="Image B", alpha=0.8)
    ax2.set_xlabel("Number of Singular Values", fontsize=10)
    ax2.set_ylabel("Cumulative Energy (Normalized)", fontsize=10)
    ax2.set_title("Cumulative Energy Distribution", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_histogram_comparison(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    bin_edges: np.ndarray,
    intersection: float,
    title: str = "Histogram Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot the intensity histograms of both images overlaid.

    Args:
        hist_a: Normalized histogram of image A.
        hist_b: Normalized histogram of image B.
        bin_edges: Bin edges for the histograms.
        intersection: Histogram intersection score for the title.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size as (width, height) in inches.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    ax.fill_between(
        bin_centers, hist_a, alpha=0.3, color="blue", label="Image A"
    )
    ax.fill_between(
        bin_centers, hist_b, alpha=0.3, color="red", label="Image B"
    )
    ax.plot(bin_centers, hist_a, color="blue", linewidth=1.0, alpha=0.8)
    ax.plot(bin_centers, hist_b, color="red", linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Intensity (0-255)", fontsize=10)
    ax.set_ylabel("Normalized Frequency", fontsize=10)
    ax.set_title(
        f"{title}\nIntersection = {intersection:.4f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_full_comparison(
    img_a: np.ndarray,
    img_b: np.ndarray,
    diff: np.ndarray,
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """Create a comprehensive comparison dashboard.

    Generates a 2x3 grid showing:
    - Top-left: Image A
    - Top-center: Image B
    - Top-right: Difference heatmap
    - Bottom-left: SVD spectrum (line plot)
    - Bottom-center: SVD cumulative energy
    - Bottom-right: Histogram comparison

    Args:
        img_a: First image matrix.
        img_b: Second image matrix.
        diff: Absolute difference matrix.
        sigma_a: Singular values of image A.
        sigma_b: Singular values of image B.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        The matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Top row: images and heatmap
    axes[0, 0].imshow(img_a, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Image A", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_b, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Image B", fontweight="bold")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(diff, cmap="hot", vmin=0, vmax=max(diff.max(), 1e-10))
    axes[0, 2].set_title("Difference Heatmap", fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Bottom row: SVD analysis
    k = min(100, len(sigma_a), len(sigma_b))
    indices = np.arange(1, k + 1)

    axes[1, 0].plot(indices, sigma_a[:k], "b-", alpha=0.8, label="Image A")
    axes[1, 0].plot(indices, sigma_b[:k], "r-", alpha=0.8, label="Image B")
    axes[1, 0].set_title("SVD Singular Values", fontweight="bold")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Sigma")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative energy
    energy_a = np.cumsum(sigma_a[:k] ** 2)
    energy_b = np.cumsum(sigma_b[:k] ** 2)
    if energy_a[-1] > 0:
        energy_a /= energy_a[-1]
    if energy_b[-1] > 0:
        energy_b /= energy_b[-1]

    axes[1, 1].plot(indices, energy_a, "b-", linewidth=2, label="Image A")
    axes[1, 1].plot(indices, energy_b, "r-", linewidth=2, label="Image B")
    axes[1, 1].set_title("Cumulative Energy", fontweight="bold")
    axes[1, 1].set_xlabel("Singular Values Used")
    axes[1, 1].set_ylabel("Normalized Energy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.05)

    # Difference histogram
    from src.algo_metrics import _compute_histogram, histogram_intersection

    hist_a, bin_edges = _compute_histogram(img_a)
    hist_b, _ = _compute_histogram(img_b)
    hi = histogram_intersection(img_a, img_b)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    axes[1, 2].fill_between(bin_centers, hist_a, alpha=0.3, color="blue", label="A")
    axes[1, 2].fill_between(bin_centers, hist_b, alpha=0.3, color="red", label="B")
    axes[1, 2].plot(bin_centers, hist_a, color="blue", linewidth=1)
    axes[1, 2].plot(bin_centers, hist_b, color="red", linewidth=1)
    axes[1, 2].set_title(
        f"Histograms (Intersection={hi:.4f})", fontweight="bold"
    )
    axes[1, 2].set_xlabel("Intensity")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("PicPicComparison Dashboard", fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
