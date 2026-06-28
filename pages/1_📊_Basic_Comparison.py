"""Page 1 — Basic Comparison Results.

Displays summary interpretation, metric cards with color coding,
side-by-side images, difference heatmap, SVD spectrum, histogram, and downloads.
All helper functions inlined to avoid Streamlit module-cache issues.
"""

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ui.styles import inject_custom_css

st.set_page_config(page_title="Basic Comparison", page_icon="📊", layout="wide")
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


# ── Header ──────────────────────────────────────────────────────────
st.html("""
<div class="page-breadcrumb">
    <a href="/" target="_self">Home</a>
    <span class="sep">/</span>
    <span>Basic Comparison</span>
</div>
<div class="page-header">
    <h1>Basic Comparison</h1>
    <p>Side-by-side image metrics and visualizations</p>
</div>
""")


# ── Summary banner ──────────────────────────────────────────────────
cosine = report.cosine_similarity
if cosine >= 0.95:
    banner_cls, icon = "good", "✅"
    text = f"These images are <strong>very similar</strong> — Cosine Similarity: <strong>{cosine:.1%}</strong>"
elif cosine >= 0.70:
    banner_cls, icon = "warn", "⚠️"
    text = f"These images are <strong>moderately similar</strong> — Cosine Similarity: <strong>{cosine:.1%}</strong>"
else:
    banner_cls, icon = "bad", "❌"
    text = f"These images are <strong>significantly different</strong> — Cosine Similarity: <strong>{cosine:.1%}</strong>"

st.html(f"""
<div class="summary-banner {banner_cls}">
    <span class="summary-banner-icon">{icon}</span>
    <span class="summary-banner-text">{text}</span>
</div>
""")


# ── Side-by-side images ────────────────────────────────────────────
st.markdown("#### Original Images")
col1, col2 = st.columns(2)
with col1:
    st.image(img_a, caption="Image A", use_container_width=True)
with col2:
    st.image(img_b, caption="Image B", use_container_width=True)


# ── Metric cards with color coding ──────────────────────────────────
st.markdown("#### All Metrics")

def _metric_cls(name, value_str):
    """Return metric-good/warn/bad based on metric name and value."""
    try:
        val = float(value_str.replace(" dB", "").replace("∞", "999"))
    except ValueError:
        return ""
    if name == "Cosine Similarity":
        return "metric-good" if val >= 0.95 else ("metric-warn" if val >= 0.70 else "metric-bad")
    if name == "MSE":
        return "metric-good" if val < 0.001 else ("metric-warn" if val < 0.01 else "metric-bad")
    if name == "Frobenius Norm":
        return "metric-good" if val < 1 else ("metric-warn" if val < 10 else "metric-bad")
    if name == "L∞ Norm":
        return "metric-good" if val < 0.01 else ("metric-warn" if val < 0.1 else "metric-bad")
    return ""

metrics = [
    ("Frobenius Norm", f"{report.frobenius_norm:.4f}", "Matrix distance — lower means more similar"),
    ("Cosine Similarity", f"{report.cosine_similarity:.6f}", "Vector angle — closer to 1 means more similar"),
    ("MSE", f"{report.mse:.8f}", "Mean Squared Error — lower means less pixel difference"),
    ("PSNR", f"{report.psnr:.2f} dB" if report.psnr != float("inf") else "∞ dB", "Peak Signal-to-Noise — higher is better"),
    ("L1 Norm", f"{report.l1_norm:.2f}", "Sum of absolute pixel differences"),
    ("L∞ Norm", f"{report.l_inf_norm:.6f}", "Maximum single-pixel difference"),
    ("Hist. Intersection", f"{report.histogram_intersection:.6f}", "Intensity distribution overlap — closer to 1 is better"),
    ("SVD Energy Ratio", f"{report.svd_energy_ratio:.6f}", "Energy captured by top-K singular values"),
]

grid_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin-bottom:2rem">'
for name, value, desc in metrics:
    cls = _metric_cls(name, value)
    grid_html += f'''
    <div class="metric-card {cls}">
        <div class="metric-name">{name}</div>
        <div class="metric-value">{value}</div>
    </div>'''
grid_html += "</div>"
st.html(grid_html)


# ── Difference heatmap ─────────────────────────────────────────────
st.markdown("#### Difference Heatmap")
from src.algo_metrics import difference_image

diff = difference_image(img_a, img_b)
fig_heat = _dark_fig(8, 4)
ax_heat = fig_heat.add_subplot(111)
_dark_ax(ax_heat, "Absolute Difference")
im = ax_heat.imshow(diff, cmap="magma", aspect="auto")
fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
st.pyplot(fig_heat)
plt.close(fig_heat)


# ── SVD spectrum ───────────────────────────────────────────────────
st.markdown("#### SVD Spectrum")
from src.linalg_metrics import svd_energy_comparison

svd_result = svd_energy_comparison(img_a, img_b)
sv = svd_result.singular_values

fig_svd = _dark_fig(8, 4)
ax_svd = fig_svd.add_subplot(111)
_dark_ax(ax_svd, "Singular Values & Cumulative Energy")

ax_svd.plot(range(1, len(sv) + 1), sv, color=_DARK_FG, linewidth=1.5, marker="o", markersize=3)
ax_svd.set_xlabel("Index", color=_DARK_FG, fontsize=9)
ax_svd.set_ylabel("Singular Value", color=_DARK_FG, fontsize=9)

ax2 = ax_svd.twinx()
cum_energy = np.cumsum(sv) / (np.sum(sv) + 1e-12)
ax2.plot(range(1, len(cum_energy) + 1), cum_energy, color="#888", linewidth=1.2, linestyle="--")
ax2.set_ylabel("Cumulative Energy", color="#888", fontsize=9)
ax2.tick_params(colors="#888", labelsize=9)
ax2.spines["right"].set_color(_DARK_GRID)
ax2.spines["left"].set_color(_DARK_GRID)

st.pyplot(fig_svd)
plt.close(fig_svd)


# ── Histogram comparison ───────────────────────────────────────────
st.markdown("#### Histogram Comparison")
from src.algo_metrics import histogram_intersection_detailed

hist_result = histogram_intersection_detailed(img_a, img_b, bins=64)

fig_hist = _dark_fig(8, 4)
ax_hist = fig_hist.add_subplot(111)
_dark_ax(ax_hist, "Pixel Intensity Histograms")

edges = hist_result.edges
centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]
ax_hist.fill_between(centers, hist_result.histogram_a, alpha=0.4, color="#666", label="Image A")
ax_hist.fill_between(centers, hist_result.histogram_b, alpha=0.4, color="#aaa", label="Image B")
ax_hist.set_xlabel("Intensity", color=_DARK_FG, fontsize=9)
ax_hist.set_ylabel("Density", color=_DARK_FG, fontsize=9)
legend = ax_hist.legend(framealpha=0.3, facecolor=_DARK_BG, edgecolor=_DARK_GRID, labelcolor=_DARK_FG)

st.pyplot(fig_hist)
plt.close(fig_hist)


# ── Downloads ───────────────────────────────────────────────────────
st.html('<div style="height:1rem"></div>')
st.markdown("#### Export")

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        label="📄 JSON Report",
        data=report.to_json(),
        file_name="comparison_report.json",
        mime="application/json",
        use_container_width=True,
    )
with col_dl2:
    from src.visualizer import plot_full_comparison
    fig_full = plot_full_comparison(img_a, img_b, svd_result, hist_result)
    buf = io.BytesIO()
    fig_full.savefig(buf, format="png", dpi=150, facecolor=_DARK_BG, bbox_inches="tight")
    plt.close(fig_full)
    buf.seek(0)
    st.download_button(
        label="🖼️ Full Dashboard PNG",
        data=buf.getvalue(),
        file_name="full_dashboard.png",
        mime="image/png",
        use_container_width=True,
    )


# ── Navigation ──────────────────────────────────────────────────────
st.html("""
<div class="nav-links">
    <a class="nav-link" href="/" target="_self">← Home</a>
    <a class="nav-link" href="/Advanced_Analysis" target="_self">🔬 Advanced Analysis →</a>
    <a class="nav-link" href="/Export_Results" target="_self">📥 Export Results →</a>
</div>
""")
