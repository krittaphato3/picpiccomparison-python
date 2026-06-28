"""Page 3 — Export Results.

Download JSON reports, individual plot images, a ZIP bundle, or view the raw metrics table.
Figures are generated once and cached in session state for performance.
All helper functions inlined to avoid Streamlit module-cache issues.
"""

import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ui.styles import inject_custom_css

st.set_page_config(page_title="Export Results", page_icon="📥", layout="wide")
inject_custom_css()

# ── Guard ───────────────────────────────────────────────────────────
if st.session_state.report is None:
    st.warning("No comparison results yet. Run a comparison from the Home page first.")
    st.html("""
    <div class="nav-links">
        <a class="nav-link" href="/" target="_self">← Back to Home</a>
    </div>
    """)
    st.stop()

report = st.session_state.report
img_a = st.session_state.img_a
img_b = st.session_state.img_b
pair = st.session_state.pair

# ── Dark-theme matplotlib helpers ───────────────────────────────────
_DARK_BG = "#111118"
_DARK_FG = "#e8e8ec"
_DARK_GRID = "#2a2a35"


# ── Header ──────────────────────────────────────────────────────────
st.html("""
<div class="page-breadcrumb">
    <a href="/" target="_self">Home</a>
    <span class="sep">/</span>
    <span>Export Results</span>
</div>
<div class="page-header">
    <h1>Export Results</h1>
    <p>Download reports, visualizations, and raw data</p>
</div>
""")


# ── Generate all figures ONCE, cache in session state ───────────────
if "_export_figs" not in st.session_state:
    with st.spinner("Generating visualizations..."):
        from src.visualizer import (
            plot_original_images,
            plot_difference_heatmap,
            plot_svd_spectrum,
            plot_histogram_comparison,
            plot_full_comparison,
        )
        from src.algo_metrics import histogram_intersection_detailed, difference_image
        from src.linalg_metrics import svd_energy_comparison

        hist_result = histogram_intersection_detailed(img_a, img_b, bins=64)
        svd_result = svd_energy_comparison(img_a, img_b)
        diff = difference_image(img_a, img_b)

        st.session_state._export_figs = {
            "originals": plot_original_images(img_a, img_b),
            "heatmap": plot_difference_heatmap(diff),
            "svd": plot_svd_spectrum(svd_result.singular_values_a, svd_result.singular_values_b),
            "histogram": plot_histogram_comparison(hist_result.hist_a, hist_result.hist_b, hist_result.bin_edges, hist_result.intersection),
            "full": plot_full_comparison(img_a, img_b, diff, svd_result.singular_values_a, svd_result.singular_values_b),
            "hist_result": hist_result,
            "svd_result": svd_result,
        }

_figs = st.session_state._export_figs


def _fig_to_png(fig: plt.Figure) -> bytes:
    """Convert a matplotlib figure to PNG bytes with dark background."""
    import copy
    fig_copy = copy.deepcopy(fig)
    buf = io.BytesIO()
    fig_copy.savefig(buf, format="png", dpi=150, facecolor=_DARK_BG, bbox_inches="tight")
    plt.close(fig_copy)
    buf.seek(0)
    return buf.getvalue()


# ── Tier 1: Quick Export ──────────────────────────────────────────
st.markdown("#### ⚡ Quick Export")

col_json, col_dash = st.columns(2)
with col_json:
    st.caption("Structured data for programmatic use")
    json_data = report.to_json()
    st.download_button(
        label="📄 Download JSON Report",
        data=json_data,
        file_name="comparison_report.json",
        mime="application/json",
        use_container_width=True,
    )
    st.caption(f"Size: {len(json_data.encode()) / 1024:.1f} KB")

with col_dash:
    st.caption("All visualizations in a single image")
    full_png = _fig_to_png(_figs["full"])
    st.download_button(
        label="🖼️ Download Dashboard PNG",
        data=full_png,
        file_name="full_dashboard.png",
        mime="image/png",
        use_container_width=True,
    )
    st.caption(f"Size: {len(full_png) / 1024:.0f} KB")

# JSON preview
with st.expander("Preview JSON Report", expanded=False):
    st.json(json.loads(json_data))


# ── Tier 2: Individual Assets ─────────────────────────────────────
st.markdown("#### 📊 Individual Visualizations")

plot_items = [
    ("Original Images", "originals", "original_images.png", "Side-by-side grayscale originals"),
    ("Difference Heatmap", "heatmap", "difference_heatmap.png", "Absolute pixel difference map"),
    ("SVD Spectrum", "svd", "svd_spectrum.png", "Singular values and cumulative energy"),
    ("Histogram Comparison", "histogram", "histogram_comparison.png", "Pixel intensity distributions"),
]

col1, col2 = st.columns(2)
for i, (name, fig_key, filename, desc) in enumerate(plot_items):
    col = col1 if i % 2 == 0 else col2
    with col:
        st.caption(desc)
        png_data = _fig_to_png(_figs[fig_key])
        st.image(png_data, caption=name, use_container_width=True)
        st.download_button(
            label=f"📥 {name}",
            data=png_data,
            file_name=filename,
            mime="image/png",
            use_container_width=True,
        )


# ── Tier 3: Complete Archive ───────────────────────────────────────
st.markdown("#### 📦 Complete Archive")

st.caption("All plots + JSON report in a single ZIP file")

zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("report.json", report.to_json())
    for fig_key, fname in [
        ("originals", "original_images.png"),
        ("heatmap", "difference_heatmap.png"),
        ("svd", "svd_spectrum.png"),
        ("histogram", "histogram_comparison.png"),
        ("full", "full_dashboard.png"),
    ]:
        png_data = _fig_to_png(_figs[fig_key])
        zf.writestr(fname, png_data)

zip_buf.seek(0)
zip_data = zip_buf.getvalue()

col_zip, col_size = st.columns([1, 1])
with col_zip:
    st.download_button(
        label="📦 Download ZIP Bundle",
        data=zip_data,
        file_name="picpiccomparison_export.zip",
        mime="application/zip",
        use_container_width=True,
    )
with col_size:
    st.caption(f"Size: {len(zip_data) / 1024:.0f} KB (5 plots + JSON)")


# ── Raw metrics table ──────────────────────────────────────────────
st.markdown("#### 📋 Raw Metrics")

metric_rows = [
    ("Frobenius Norm", f"{report.frobenius:.6f}", "Matrix distance measure — lower means more similar"),
    ("Cosine Similarity", f"{report.cosine_sim:.6f}", "Vector angle similarity — closer to 1 is better"),
    ("MSE", f"{report.mse:.8f}", "Mean Squared Error — lower means less pixel difference"),
    ("PSNR", f"{report.psnr_db:.2f} dB" if report.psnr_db != float("inf") else "∞ dB", "Peak Signal-to-Noise Ratio — higher is better"),
    ("L1 Norm", f"{report.l1:.4f}", "Sum of absolute differences"),
    ("L∞ Norm", f"{report.l_inf:.6f}", "Maximum absolute pixel difference"),
    ("Histogram Intersection", f"{report.hist_intersection:.6f}", "Intensity distribution overlap — closer to 1 is better"),
    ("SVD Energy Ratio", f"{report.svd.energy_ratio:.6f}", "Ratio of top-K singular value energy"),
    ("Cosine Distance (SV)", f"{report.svd.cosine_distance_sv:.6f}", "1 - cosine(svd_a, svd_b) — lower means more similar"),
    ("SVD Top-K Cosine", f"{report.svd.top_k_singular_cosine:.6f}", "Cosine of top-K singular vectors — closer to 1 is better"),
]

df = pd.DataFrame(metric_rows, columns=["Metric", "Value", "Description"])
st.dataframe(df, use_container_width=True, hide_index=True)


# ── Navigation ──────────────────────────────────────────────────────
st.html("""
<div class="nav-links">
    <a class="nav-link" href="/" target="_self">← Home</a>
    <a class="nav-link" href="/Basic_Comparison" target="_self">📊 Basic Comparison →</a>
    <a class="nav-link" href="/Advanced_Analysis" target="_self">🔬 Advanced Analysis →</a>
</div>
""")
