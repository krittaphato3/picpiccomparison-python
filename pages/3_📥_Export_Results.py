"""Page 3 — Export Results.

Download JSON reports, individual plot images, a ZIP bundle, or view the raw metrics table.
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
    st.stop()

report = st.session_state.report
img_a = st.session_state.img_a
img_b = st.session_state.img_b
pair = st.session_state.pair

# ── Header ──────────────────────────────────────────────────────────
st.html("""
<div class="page-header">
    <h1>Export Results</h1>
    <p>Download reports, visualizations, and raw data</p>
</div>
""")


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


# ── Generate all figures ───────────────────────────────────────────
from src.visualizer import (
    plot_original_images,
    plot_difference_heatmap,
    plot_svd_spectrum,
    plot_histogram_comparison,
    plot_full_comparison,
)
from src.algo_metrics import histogram_intersection_detailed
from src.linalg_metrics import svd_energy_comparison

hist_result = histogram_intersection_detailed(img_a, img_b, bins=64)
svd_result = svd_energy_comparison(img_a, img_b)

fig_originals = plot_original_images(img_a, img_b)
fig_heatmap = plot_difference_heatmap(img_a, img_b)
fig_svd = plot_svd_spectrum(svd_result)
fig_histogram = plot_histogram_comparison(hist_result)
fig_full = plot_full_comparison(img_a, img_b, svd_result, hist_result)


def _fig_to_png(fig: plt.Figure) -> bytes:
    """Convert a matplotlib figure to PNG bytes with dark background."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=_DARK_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ── JSON Report ────────────────────────────────────────────────────
st.markdown("#### JSON Report")

col_json, col_preview = st.columns([1, 1])
with col_json:
    json_data = report.to_json()
    st.download_button(
        label="📄 Download JSON",
        data=json_data,
        file_name="comparison_report.json",
        mime="application/json",
        use_container_width=True,
    )
    json_size_kb = len(json_data.encode()) / 1024
    st.caption(f"Size: {json_size_kb:.1f} KB")

with col_preview:
    with st.expander("Preview JSON", expanded=False):
        st.json(json.loads(json_data))


# ── Individual plots ───────────────────────────────────────────────
st.markdown("#### Visualizations")

plot_items = [
    ("Original Images", fig_originals, "original_images.png"),
    ("Difference Heatmap", fig_heatmap, "difference_heatmap.png"),
    ("SVD Spectrum", fig_svd, "svd_spectrum.png"),
    ("Histogram Comparison", fig_histogram, "histogram_comparison.png"),
]

col1, col2 = st.columns(2)
for i, (name, fig, filename) in enumerate(plot_items):
    col = col1 if i % 2 == 0 else col2
    with col:
        png_data = _fig_to_png(fig)
        st.image(png_data, caption=name, use_container_width=True)
        st.download_button(
            label=f"📥 {name}",
            data=png_data,
            file_name=filename,
            mime="image/png",
            use_container_width=True,
        )


# ── Full dashboard ─────────────────────────────────────────────────
st.markdown("#### Full Dashboard")

full_png = _fig_to_png(fig_full)
st.image(full_png, caption="Full Dashboard", use_container_width=True)

col_dl, col_info = st.columns([1, 1])
with col_dl:
    st.download_button(
        label="🖼️ Download Dashboard PNG",
        data=full_png,
        file_name="full_dashboard.png",
        mime="image/png",
        use_container_width=True,
    )
with col_info:
    st.caption(f"PNG size: {len(full_png) / 1024:.0f} KB")


# ── ZIP bundle ─────────────────────────────────────────────────────
st.markdown("#### ZIP Bundle")

# Regenerate figures for the bundle (since we closed them above)
fig_orig2 = plot_original_images(img_a, img_b)
fig_heat2 = plot_difference_heatmap(img_a, img_b)
fig_svd2 = plot_svd_spectrum(svd_result)
fig_hist2 = plot_histogram_comparison(hist_result)
fig_full2 = plot_full_comparison(img_a, img_b, svd_result, hist_result)

zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("report.json", report.to_json())
    for name, fig, fname in [
        ("original_images", fig_orig2, "original_images.png"),
        ("difference_heatmap", fig_heat2, "difference_heatmap.png"),
        ("svd_spectrum", fig_svd2, "svd_spectrum.png"),
        ("histogram_comparison", fig_hist2, "histogram_comparison.png"),
        ("full_dashboard", fig_full2, "full_dashboard.png"),
    ]:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=_DARK_BG, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        zf.writestr(fname, buf.getvalue())

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
    st.caption(f"ZIP size: {len(zip_data) / 1024:.0f} KB (5 plots + JSON)")


# ── Raw metrics table ──────────────────────────────────────────────
st.markdown("#### Raw Metrics")

metric_rows = [
    ("Frobenius Norm", f"{report.frobenius_norm:.6f}", "Matrix distance measure"),
    ("Cosine Similarity", f"{report.cosine_similarity:.6f}", "Vector angle similarity [0,1]"),
    ("MSE", f"{report.mse:.8f}", "Mean Squared Error"),
    ("PSNR", f"{report.psnr:.2f} dB" if report.psnr != float("inf") else "∞ dB", "Peak Signal-to-Noise Ratio"),
    ("L1 Norm", f"{report.l1_norm:.4f}", "Sum of absolute differences"),
    ("L∞ Norm", f"{report.l_inf_norm:.6f}", "Maximum absolute difference"),
    ("Histogram Intersection", f"{report.histogram_intersection:.6f}", "Normalized overlap [0,1]"),
    ("SVD Energy Ratio", f"{report.svd_energy_ratio:.6f}", "Ratio of top-K singular energy"),
    ("Cosine Distance (SV)", f"{report.cosine_distance_sv:.6f}", "1 - cosine(svd_a, svd_b)"),
    ("SVD Top-K Cosine", f"{report.top_k_singular_cosine:.6f}", "Cosine of top-K singular vectors"),
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
