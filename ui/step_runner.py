"""Advanced analysis pipeline — 8 steps with LaTeX formulas and visualizations.

Each step computes metrics and renders matplotlib charts inside Streamlit
expanders. Expanders default to collapsed (except step 1) for progressive
disclosure. All helpers are inlined to avoid Streamlit module-cache issues.
"""

import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ── Dark-theme matplotlib helpers ───────────────────────────────────
_DARK_BG = "#111118"
_DARK_FG = "#e8e8ec"
_DARK_GRID = "#2a2a35"

def _dark_fig(w=10, h=5):
    fig = plt.figure(figsize=(w, h), facecolor=_DARK_BG)
    return fig

def _dark_ax(ax, title=None):
    ax.set_facecolor(_DARK_BG)
    ax.tick_params(colors=_DARK_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_DARK_GRID)
    if title:
        ax.set_title(title, color=_DARK_FG, fontsize=11, fontweight=600, pad=10)


# ── Step definitions ───────────────────────────────────────────────
STEPS = [
    {
        "title": "Load & Preprocess",
        "latex": r"I_{\text{gray}} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B, \quad I_{\text{norm}} = \frac{I}{255}",
        "desc": "Convert to grayscale and normalize to [0, 1].",
    },
    {
        "title": "Resize & Align",
        "latex": r"I_{\text{resized}} = \text{cv2.resize}(I, (T, T), \text{interpolation}=\text{INTER\_AREA})",
        "desc": "Resize both images to the target dimension for fair comparison.",
    },
    {
        "title": "Flatten to Vectors",
        "latex": r"a = \text{vec}(A) \in \mathbb{R}^{n}, \quad n = T \times T",
        "desc": "Reshape 2D matrices into 1D vectors for linear algebra operations.",
    },
    {
        "title": "Norm-Based Distances",
        "latex": r"\|A - B\|_F = \sqrt{\sum_{i,j} (a_{ij} - b_{ij})^2}, \quad L_1 = \sum |a_{ij} - b_{ij}|, \quad L_\infty = \max |a_{ij} - b_{ij}|",
        "desc": "Compute Frobenius, L1, and L∞ norms of the difference matrix.",
    },
    {
        "title": "Vector Space Geometry",
        "latex": r"\cos(\theta) = \frac{a \cdot b}{\|a\| \cdot \|b\|}",
        "desc": "Measure the angle between image vectors in high-dimensional space.",
    },
    {
        "title": "SVD Analysis",
        "latex": r"A = U \Sigma V^T, \quad \text{Energy} = \sum_i \sigma_i^2, \quad \text{Ratio} = \frac{\sum_{i=1}^{K} \sigma_i^2}{\sum \sigma_i^2}",
        "desc": "Decompose images via SVD and compare singular value spectra.",
    },
    {
        "title": "Algorithmic Metrics",
        "latex": r"\text{MSE} = \frac{1}{n}\sum(a_i - b_i)^2, \quad \text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right), \quad \text{HI} = \sum \min(h_A, h_B)",
        "desc": "Compute pixel-level error metrics and histogram similarity.",
    },
    {
        "title": "Summary Report",
        "latex": r"\text{All metrics aggregated into a unified comparison report}",
        "desc": "Aggregate all metrics into a unified comparison summary.",
    },
]


def _render_step_header(step_num: int, title: str, desc: str) -> None:
    """Render a step header with number badge, title, and description."""
    st.html(f"""
    <div style="display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.5rem">
        <span class="step-number">{step_num}</span>
        <div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
    </div>
    """)


# ── Step renderers ──────────────────────────────────────────────────

def _render_step_1(img_a, img_b):
    """Load & Preprocess — show grayscale images."""
    _render_step_header(1, STEPS[0]["title"], STEPS[0]["desc"])
    st.latex(STEPS[0]["latex"])

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_a, caption=f"Image A — shape={img_a.shape}, dtype={img_a.dtype}", use_container_width=True)
    with col2:
        st.image(img_b, caption=f"Image B — shape={img_b.shape}, dtype={img_b.dtype}", use_container_width=True)

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"Range: [{img_a.min():.3f}, {img_a.max():.3f}]")
    with col_info2:
        st.info(f"Range: [{img_b.min():.3f}, {img_b.max():.3f}]")


def _render_step_3(img_a, img_b):
    """Flatten to Vectors — show vector stats."""
    _render_step_header(3, STEPS[2]["title"], STEPS[2]["desc"])
    st.latex(STEPS[2]["latex"])

    vec_a = img_a.flatten()
    vec_b = img_b.flatten()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vector A length", f"{len(vec_a):,}")
    with col2:
        st.metric("Vector B length", f"{len(vec_b):,}")
    with col3:
        st.metric("‖a‖₂", f"{np.linalg.norm(vec_a):.2f}")
    with col4:
        st.metric("‖b‖₂", f"{np.linalg.norm(vec_b):.2f}")

    fig = _dark_fig(8, 3)
    ax = fig.add_subplot(111)
    _dark_ax(ax, "Vector Components (first 500)")
    n = min(500, len(vec_a))
    ax.plot(vec_a[:n], color="#666", linewidth=0.8, alpha=0.8, label="Vector A")
    ax.plot(vec_b[:n], color="#aaa", linewidth=0.8, alpha=0.8, label="Vector B")
    legend = ax.legend(framealpha=0.3, facecolor=_DARK_BG, edgecolor=_DARK_GRID, labelcolor=_DARK_FG)
    st.pyplot(fig)
    plt.close(fig)


def _render_step_4(img_a, img_b):
    """Norm-Based Distances — compute and visualize norms."""
    _render_step_header(4, STEPS[3]["title"], STEPS[3]["desc"])
    st.latex(STEPS[3]["latex"])

    from src.linalg_metrics import frobenius_norm, l1_norm, l_inf_norm

    diff = img_a - img_b
    frob = frobenius_norm(img_a, img_b)
    l1 = l1_norm(img_a, img_b)
    linf = l_inf_norm(img_a, img_b)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Frobenius Norm", f"{frob:.4f}")
    with col2:
        st.metric("L1 Norm", f"{l1:.2f}")
    with col3:
        st.metric("L∞ Norm", f"{linf:.6f}")

    fig = _dark_fig(8, 4)
    ax = fig.add_subplot(111)
    _dark_ax(ax, "Difference Distribution")
    flat_diff = np.abs(diff).flatten()
    flat_diff = flat_diff[flat_diff > 0] if np.any(flat_diff > 0) else flat_diff
    ax.hist(flat_diff, bins=50, color="#555", edgecolor="#333", alpha=0.8)
    ax.set_xlabel("|a_ij − b_ij|", color=_DARK_FG, fontsize=9)
    ax.set_ylabel("Count", color=_DARK_FG, fontsize=9)
    st.pyplot(fig)
    plt.close(fig)


def _render_step_5(img_a, img_b):
    """Vector Space Geometry — cosine similarity."""
    _render_step_header(5, STEPS[4]["title"], STEPS[4]["desc"])
    st.latex(STEPS[4]["latex"])

    from src.linalg_metrics import cosine_similarity

    cos_sim = cosine_similarity(img_a.flatten(), img_b.flatten())
    angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cosine Similarity", f"{cos_sim:.6f}")
    with col2:
        st.metric("Angle (degrees)", f"{angle:.4f}°")

    # Visualize as unit vectors on a circle (2D projection)
    fig = _dark_fig(6, 6)
    ax = fig.add_subplot(111)
    _dark_ax(ax, "Vector Direction (2D Projection)")
    theta_a = 0
    theta_b = angle
    ax.annotate("", xy=(np.cos(np.radians(theta_b)), np.sin(np.radians(theta_b))),
                xytext=(np.cos(np.radians(theta_a)), np.sin(np.radians(theta_a))),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5))
    circle = plt.Circle((0, 0), 1, fill=False, color=_DARK_GRID, linewidth=1)
    ax.add_patch(circle)
    ax.plot(1, 0, "o", color="#666", markersize=8, label="Vector A")
    ax.plot(np.cos(np.radians(theta_b)), np.sin(np.radians(theta_b)), "o", color="#aaa", markersize=8, label="Vector B")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.legend(framealpha=0.3, facecolor=_DARK_BG, edgecolor=_DARK_GRID, labelcolor=_DARK_FG)
    st.pyplot(fig)
    plt.close(fig)


def _render_step_6(img_a, img_b):
    """SVD Analysis — singular values and energy."""
    _render_step_header(6, STEPS[5]["title"], STEPS[5]["desc"])
    st.latex(STEPS[5]["latex"])

    from src.linalg_metrics import svd_energy_comparison

    result = svd_energy_comparison(img_a, img_b)
    sv = result.singular_values

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Energy Ratio", f"{result.energy_ratio:.6f}")
    with col2:
        st.metric("Cosine Distance (SV)", f"{result.cosine_distance_sv:.6f}")
    with col3:
        st.metric("Top-K Cosine", f"{result.top_k_singular_cosine:.6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=_DARK_BG)

    _dark_ax(ax1, "Singular Values")
    ax1.plot(range(1, len(sv) + 1), sv, color=_DARK_FG, linewidth=1.5, marker="o", markersize=3)

    _dark_ax(ax2, "Cumulative Energy")
    cum = np.cumsum(sv) / (np.sum(sv) + 1e-12)
    ax2.fill_between(range(1, len(cum) + 1), cum, alpha=0.3, color="#666")
    ax2.plot(range(1, len(cum) + 1), cum, color=_DARK_FG, linewidth=1.5)
    ax2.set_ylabel("Energy", color=_DARK_FG, fontsize=9)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_step_7(img_a, img_b):
    """Algorithmic Metrics — MSE, PSNR, histogram."""
    _render_step_header(7, STEPS[6]["title"], STEPS[6]["desc"])
    st.latex(STEPS[6]["latex"])

    from src.algo_metrics import mse, psnr, histogram_intersection, histogram_intersection_detailed

    mse_val = mse(img_a, img_b)
    psnr_val = psnr(img_a, img_b)
    hist_int = histogram_intersection(img_a, img_b)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE", f"{mse_val:.8f}")
    with col2:
        st.metric("PSNR", f"{psnr_val:.2f} dB" if psnr_val != float("inf") else "∞ dB")
    with col3:
        st.metric("Histogram Intersection", f"{hist_int:.6f}")

    hist_result = histogram_intersection_detailed(img_a, img_b, bins=64)

    fig = _dark_fig(8, 4)
    ax = fig.add_subplot(111)
    _dark_ax(ax, "Histogram Comparison")
    edges = hist_result.edges
    centers = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centers, hist_result.histogram_a, alpha=0.4, color="#666", label="Image A")
    ax.fill_between(centers, hist_result.histogram_b, alpha=0.4, color="#aaa", label="Image B")
    ax.set_xlabel("Intensity", color=_DARK_FG, fontsize=9)
    ax.set_ylabel("Density", color=_DARK_FG, fontsize=9)
    legend = ax.legend(framealpha=0.3, facecolor=_DARK_BG, edgecolor=_DARK_GRID, labelcolor=_DARK_FG)
    st.pyplot(fig)
    plt.close(fig)


def _render_step_8(report):
    """Summary Report — all metrics in a table."""
    _render_step_header(8, STEPS[7]["title"], STEPS[7]["desc"])
    st.latex(STEPS[7]["latex"])

    import pandas as pd

    rows = [
        ("Frobenius Norm", f"{report.frobenius_norm:.6f}"),
        ("Cosine Similarity", f"{report.cosine_similarity:.6f}"),
        ("MSE", f"{report.mse:.8f}"),
        ("PSNR", f"{report.psnr:.2f} dB" if report.psnr != float("inf") else "∞ dB"),
        ("L1 Norm", f"{report.l1_norm:.4f}"),
        ("L∞ Norm", f"{report.l_inf_norm:.6f}"),
        ("Histogram Intersection", f"{report.histogram_intersection:.6f}"),
        ("SVD Energy Ratio", f"{report.svd_energy_ratio:.6f}"),
        ("Cosine Distance (SV)", f"{report.cosine_distance_sv:.6f}"),
        ("SVD Top-K Cosine", f"{report.top_k_singular_cosine:.6f}"),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Full JSON Report", expanded=False):
        st.json(report.to_dict())


# ── Pipeline runner ─────────────────────────────────────────────────

def run_advanced_pipeline(img_a, img_b, pair, report):
    """Execute the 8-step advanced analysis pipeline.

    Expanders default to collapsed except step 1 for progressive disclosure.
    Each step is wrapped in try/except for graceful error handling.
    """
    steps = [
        ("1. Load & Preprocess",       lambda: _render_step_1(img_a, img_b)),
        # Steps 1 & 2 merged: resize info is shown as caption in step 1
        ("2. Resize & Align",          lambda: _render_step_2_info(img_a, img_b)),
        ("3. Flatten to Vectors",      lambda: _render_step_3(img_a, img_b)),
        ("4. Norm-Based Distances",    lambda: _render_step_4(img_a, img_b)),
        ("5. Vector Space Geometry",   lambda: _render_step_5(img_a, img_b)),
        ("6. SVD Analysis",            lambda: _render_step_6(img_a, img_b)),
        ("7. Algorithmic Metrics",     lambda: _render_step_7(img_a, img_b)),
        ("8. Summary Report",          lambda: _render_step_8(report)),
    ]

    for i, (label, renderer) in enumerate(steps):
        with st.expander(label, expanded=(i == 0)):
            try:
                renderer()
            except Exception as e:
                st.error(f"Step failed: {e}")
                with st.expander("Show details"):
                    st.code(traceback.format_exc())


def _render_step_2_info(img_a, img_b):
    """Step 2 — Resize & Align (info display, no redundant image render)."""
    _render_step_header(2, STEPS[1]["title"], STEPS[1]["desc"])
    st.latex(STEPS[1]["latex"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Image A Size", f"{img_a.shape[1]}×{img_a.shape[0]}")
    with col2:
        st.metric("Image B Size", f"{img_b.shape[1]}×{img_b.shape[0]}")
    with col3:
        st.metric("Channels", f"{img_a.shape[2] if len(img_a.shape) > 2 else 1}")
    with col4:
        st.metric("Pixel Count", f"{img_a.shape[0] * img_a.shape[1]:,}")

    st.info("Both images are resized to the same dimensions using INTER_AREA interpolation for accurate downsampling.")
