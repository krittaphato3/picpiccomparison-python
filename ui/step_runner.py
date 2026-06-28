"""
Advanced mode step-through pipeline engine.

Renders an interactive step-by-step walkthrough of the image comparison
pipeline with LaTeX formulas and intermediate visualizations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


@dataclass
class PipelineStep:
    """A single step in the advanced analysis pipeline.

    Attributes:
        number: Step number (1-based).
        title: Human-readable step title.
        formula_latex: LaTeX formula string for this step.
        description: Short description of what this step does.
    """

    number: int
    title: str
    formula_latex: str
    description: str


# ---------------------------------------------------------------------------
# Define the 8 pipeline steps
# ---------------------------------------------------------------------------

PIPELINE_STEPS: List[PipelineStep] = [
    PipelineStep(
        number=1,
        title="Load & Preprocess",
        formula_latex=r"""
        A_{gray} = 0.299R + 0.587G + 0.114B, \quad
        A_{norm} = \frac{A_{gray}}{255} \in [0, 1]
        """,
        description=(
            "Load both images, convert to grayscale using the standard "
            "luminance formula, and normalize pixel values to [0, 1]."
        ),
    ),
    PipelineStep(
        number=2,
        title="Resize & Align",
        formula_latex=r"""
        A_{target} = \text{cv2.resize}(A, (W_{target}, H_{target}))
        """,
        description=(
            "Resize image B to match image A's dimensions (or both to a "
            "target size) using area interpolation for downscaling."
        ),
    ),
    PipelineStep(
        number=3,
        title="Flatten to Vectors",
        formula_latex=r"""
        A \in \mathbb{R}^{m \times n} \longrightarrow \mathbf{a} \in \mathbb{R}^N, \quad N = m \times n
        """,
        description=(
            "Reshape the 2D image matrix into a 1D vector in N-dimensional "
            "pixel space for vector operations."
        ),
    ),
    PipelineStep(
        number=4,
        title="Norm-Based Distances",
        formula_latex=r"""
        \|A - B\|_F = \sqrt{\sum_{i,j} |a_{ij} - b_{ij}|^2}, \quad
        \|A - B\|_1 = \sum_{i,j} |a_{ij} - b_{ij}|, \quad
        \|A - B\|_\infty = \max_{i,j} |a_{ij} - b_{ij}|
        """,
        description=(
            "Compute pixel-wise divergence using Frobenius (L2), Manhattan (L1), "
            "and Chebyshev (L-infinity) norms."
        ),
    ),
    PipelineStep(
        number=5,
        title="Vector Space Geometry",
        formula_latex=r"""
        \cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
        """,
        description=(
            "Measure the angle between image vectors in N-dimensional space. "
            "Cosine similarity captures structural similarity regardless of "
            "brightness/contrast scaling."
        ),
    ),
    PipelineStep(
        number=6,
        title="SVD Analysis",
        formula_latex=r"""
        A = U \Sigma V^T, \quad
        \text{Energy}_A = \sum_i \sigma_i^2
        """,
        description=(
            "Decompose both images via Singular Value Decomposition. "
            "Compare the distribution of singular values (energy spectra) "
            "to assess structural feature similarity."
        ),
    ),
    PipelineStep(
        number=7,
        title="Algorithmic Metrics",
        formula_latex=r"""
        \text{MSE} = \frac{1}{mn}\sum_{i,j}(a_{ij}-b_{ij})^2, \quad
        \text{PSNR} = 10\log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right), \quad
        \text{HI} = \sum_k \min(h_A^k, h_B^k)
        """,
        description=(
            "Compute standard algorithmic metrics: Mean Squared Error, "
            "Peak Signal-to-Noise Ratio, and Histogram Intersection."
        ),
    ),
    PipelineStep(
        number=8,
        title="Summary Report",
        formula_latex=r"""
        \text{Report} = \{\text{All Metrics}\}
        """,
        description=(
            "Compile all computed metrics into a comprehensive comparison "
            "report available for export."
        ),
    ),
]


def _render_step_1(img_a: np.ndarray, img_b: np.ndarray) -> Dict[str, Any]:
    """Render Step 1: Load & Preprocess."""
    import streamlit as st

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_a, caption=f"Image A — shape: {img_a.shape}, dtype: {img_a.dtype}", clamp=True)
        st.caption(f"Min: {img_a.min():.4f} | Max: {img_a.max():.4f} | Mean: {img_a.mean():.4f}")
    with col2:
        st.image(img_b, caption=f"Image B — shape: {img_b.shape}, dtype: {img_b.dtype}", clamp=True)
        st.caption(f"Min: {img_b.min():.4f} | Max: {img_b.max():.4f} | Mean: {img_b.mean():.4f}")

    return {"height": img_a.shape[0], "width": img_a.shape[1]}


def _render_step_2(
    img_a: np.ndarray, img_b: np.ndarray, pair: Any
) -> Dict[str, Any]:
    """Render Step 2: Resize & Align."""
    import streamlit as st

    st.info(
        f"Image A: {pair.height}×{pair.width} | "
        f"Image B resized to match."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_a, caption=f"Resized A ({pair.height}×{pair.width})", clamp=True)
    with col2:
        st.image(img_b, caption=f"Resized B ({pair.height}×{pair.width})", clamp=True)

    return {"target_shape": (pair.height, pair.width)}


def _render_step_3(pair: Any) -> Dict[str, Any]:
    """Render Step 3: Flatten to Vectors."""
    import streamlit as st

    vec_a = pair.vector_a
    vec_b = pair.vector_b

    st.code(
        f"vector_a.shape = {vec_a.shape}\n"
        f"vector_b.shape = {vec_b.shape}\n"
        f"Total elements N = {vec_a.shape[0]:,}",
        language="python",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.caption("vector_a[:20]:")
        st.code(f"{vec_a[:20]}", language="python")
    with col2:
        st.caption("vector_b[:20]:")
        st.code(f"{vec_b[:20]}", language="python")

    return {"N": vec_a.shape[0]}


def _render_step_4(img_a: np.ndarray, img_b: np.ndarray) -> Dict[str, Any]:
    """Render Step 4: Norm-Based Distances."""
    import streamlit as st
    from src.linalg_metrics import frobenius_norm, l1_norm, l_inf_norm
    from src.algo_metrics import difference_image
    import matplotlib.pyplot as plt

    frob = frobenius_norm(img_a, img_b)
    l1 = l1_norm(img_a, img_b)
    linf = l_inf_norm(img_a, img_b)

    col1, col2, col3 = st.columns(3)
    col1.metric("Frobenius (L2)", f"{frob:.6f}")
    col2.metric("L1 (Manhattan)", f"{l1:.6f}")
    col3.metric("L∞ (Chebyshev)", f"{linf:.6f}")

    # Difference heatmap
    diff = difference_image(img_a, img_b)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im0 = axes[0].imshow(img_a, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Image A")
    axes[0].axis("off")

    im1 = axes[1].imshow(diff, cmap="hot", vmin=0, vmax=max(diff.max(), 1e-10))
    axes[1].set_title("Difference Heatmap")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Pixel-Level Differences", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return {"frobenius": frob, "l1": l1, "l_inf": linf}


def _render_step_5(pair: Any) -> Dict[str, Any]:
    """Render Step 5: Vector Space Geometry."""
    import streamlit as st
    from src.linalg_metrics import cosine_similarity
    import matplotlib.pyplot as plt

    cos_sim = cosine_similarity(pair.vector_a, pair.vector_b)

    st.metric("Cosine Similarity", f"{cos_sim:.6f}")

    # Conceptual angle visualization
    fig, ax = plt.subplots(figsize=(6, 4))

    # Draw conceptual vectors in 2D
    angle = np.arccos(np.clip(cos_sim, -1, 1))
    ax.annotate(
        "", xy=(1, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax.annotate(
        "", xy=(np.cos(angle), np.sin(angle)), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    # Draw arc
    theta = np.linspace(0, angle, 50)
    ax.plot(np.cos(theta) * 0.3, np.sin(theta) * 0.3, "k--", alpha=0.5)
    ax.text(0.35, angle / 2 * 0.3, f"θ = {np.degrees(angle):.1f}°", fontsize=10)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Cosine Similarity = {cos_sim:.4f}\n"
        f"θ = {np.degrees(angle):.1f}°",
        fontweight="bold",
    )
    ax.legend(["Vector a", "Vector b"], loc="upper right")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"dot(a, b) = {np.dot(pair.vector_a, pair.vector_b):.4f} | "
        f"||a||₂ = {np.linalg.norm(pair.vector_a):.4f} | "
        f"||b||₂ = {np.linalg.norm(pair.vector_b):.4f}"
    )

    return {"cosine_similarity": cos_sim}


def _render_step_6(
    img_a: np.ndarray, img_b: np.ndarray, svd_result: Any
) -> Dict[str, Any]:
    """Render Step 6: SVD Analysis."""
    import streamlit as st
    import matplotlib.pyplot as plt

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Energy A", f"{svd_result.energy_a:.2f}")
    col2.metric("Energy B", f"{svd_result.energy_b:.2f}")
    col3.metric("Energy Ratio", f"{svd_result.energy_ratio:.4f}")
    col4.metric("SV Cosine Dist", f"{svd_result.cosine_distance_sv:.6f}")

    # SVD spectrum plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    k = min(50, len(svd_result.singular_values_a), len(svd_result.singular_values_b))
    indices = np.arange(1, k + 1)

    ax1.plot(indices, svd_result.singular_values_a[:k], "b-o", markersize=2, label="A")
    ax1.plot(indices, svd_result.singular_values_b[:k], "r-s", markersize=2, label="B")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Singular Value")
    ax1.set_title("Singular Value Spectrum", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative energy
    energy_a = np.cumsum(svd_result.singular_values_a[:k] ** 2)
    energy_b = np.cumsum(svd_result.singular_values_b[:k] ** 2)
    if energy_a[-1] > 0:
        energy_a /= energy_a[-1]
    if energy_b[-1] > 0:
        energy_b /= energy_b[-1]

    ax2.plot(indices, energy_a, "b-", linewidth=2, label="A")
    ax2.plot(indices, energy_b, "r-", linewidth=2, label="B")
    ax2.set_xlabel("Singular Values Used")
    ax2.set_ylabel("Cumulative Energy")
    ax2.set_title("Energy Distribution", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return {
        "energy_a": svd_result.energy_a,
        "energy_b": svd_result.energy_b,
        "energy_ratio": svd_result.energy_ratio,
    }


def _render_step_7(img_a: np.ndarray, img_b: np.ndarray) -> Dict[str, Any]:
    """Render Step 7: Algorithmic Metrics."""
    import streamlit as st
    from src.algo_metrics import mse, psnr, histogram_intersection, histogram_intersection_detailed
    import matplotlib.pyplot as plt

    mse_val = mse(img_a, img_b)
    psnr_val = psnr(img_a, img_b)
    hi_val = histogram_intersection(img_a, img_b)

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse_val:.6f}")
    col2.metric("PSNR", f"{psnr_val:.2f} dB")
    col3.metric("Histogram Intersection", f"{hi_val:.6f}")

    # Histogram comparison
    hist_result = histogram_intersection_detailed(img_a, img_b)
    bin_centers = (hist_result.bin_edges[:-1] + hist_result.bin_edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(bin_centers, hist_result.hist_a, alpha=0.3, color="blue", label="Image A")
    ax.fill_between(bin_centers, hist_result.hist_b, alpha=0.3, color="red", label="Image B")
    ax.plot(bin_centers, hist_result.hist_a, color="blue", linewidth=1)
    ax.plot(bin_centers, hist_result.hist_b, color="red", linewidth=1)
    ax.set_xlabel("Intensity (0-255)")
    ax.set_ylabel("Normalized Frequency")
    ax.set_title(
        f"Intensity Histograms (Intersection = {hi_val:.4f})",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return {"mse": mse_val, "psnr_db": psnr_val, "hist_intersection": hi_val}


def _render_step_8(report: Any) -> Dict[str, Any]:
    """Render Step 8: Summary Report."""
    import streamlit as st

    st.json(report.to_dict(), expanded=True)

    return {"report_complete": True}


def run_advanced_pipeline(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pair: Any,
    report: Any,
) -> None:
    """Run the advanced step-through pipeline.

    Renders 8 expandable steps, each with LaTeX formulas and
    intermediate visualizations.

    Args:
        img_a: Preprocessed image A (H x W), float64.
        img_b: Preprocessed image B (H x W), float64.
        pair: ImagePair dataclass from the loader.
        report: ComparisonReport from the comparator.
    """
    import streamlit as st

    st.subheader("🔬 Advanced Analysis Pipeline")
    st.caption("Click each step to expand and view detailed calculations.")

    step_functions = [
        lambda: _render_step_1(img_a, img_b),
        lambda: _render_step_2(img_a, img_b, pair),
        lambda: _render_step_3(pair),
        lambda: _render_step_4(img_a, img_b),
        lambda: _render_step_5(pair),
        lambda: _render_step_6(img_a, img_b, report.svd),
        lambda: _render_step_7(img_a, img_b),
        lambda: _render_step_8(report),
    ]

    for step, render_fn in zip(PIPELINE_STEPS, step_functions):
        with st.expander(
            f"**Step {step.number}: {step.title}**",
            expanded=False,
        ):
            st.markdown(f"_{step.description}_")
            st.latex(step.formula_latex)

            with st.spinner(f"Computing Step {step.number}..."):
                start = time.time()
                result = render_fn()
                elapsed = time.time() - start

            st.caption(f"⏱ Computed in {elapsed*1000:.1f} ms")
