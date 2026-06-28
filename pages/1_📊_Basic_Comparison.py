#!/usr/bin/env python3
"""PicPicComparison — Basic Comparison (Dark, Fullscreen)"""

from __future__ import annotations
import io, sys
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.styles import inject_custom_css
from src.algo_metrics import difference_image, histogram_intersection_detailed
inject_custom_css()

def _kpi(metrics):
    cols = st.columns(len(metrics))
    for col, (label, (value, gradient)) in zip(cols, metrics.items()):
        with col:
            st.html(
                f'<div class="kpi-card" style="background:{gradient};">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div></div>'
            )

def _dark_fig(w=10,h=5):
    fig = plt.figure(figsize=(w,h)); fig.patch.set_facecolor("#0b0b1e"); return fig

def _dark_ax(ax):
    ax.set_facecolor("#0b0b1e"); ax.tick_params(colors="#6666aa")
    ax.title.set_color("#e0e0f0")
    for s in ax.spines.values(): s.set_color("#1a1a44")

# Guard
if st.session_state.report is None:
    st.warning("⚠️ No results yet. Upload images on the Home page.")
    st.page_link("app.py", label="🏠 Home", icon="🏠"); st.stop()

r = st.session_state.report
img_a, img_b = st.session_state.img_a, st.session_state.img_b

# Header
st.markdown(f'<div class="page-header"><h1>📊 Basic Comparison</h1><p>{r.path_a} vs {r.path_b} — {r.dimensions[0]}×{r.dimensions[1]}px</p></div>', unsafe_allow_html=True)

# KPI
_kpi({
    "Cosine Similarity": (f"{r.cosine_sim*100:.1f}%", "linear-gradient(135deg,#7c6aef,#a855f7)"),
    "MSE": (f"{r.mse:.6f}", "linear-gradient(135deg,#ec4899,#f43f5e)"),
    "PSNR": (f"{r.psnr_db:.1f} dB" if r.psnr_db!=float("inf") else "∞ dB", "linear-gradient(135deg,#06b6d4,#3b82f6)"),
    "Frobenius (L2)": (f"{r.frobenius:.4f}", "linear-gradient(135deg,#22c55e,#10b981)"),
})

# Metrics grid
st.markdown("### 📐 All Metrics")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Frobenius (L2)", f"{r.frobenius:.6f}")
c2.metric("Cosine Similarity", f"{r.cosine_sim:.6f}")
c3.metric("MSE", f"{r.mse:.8f}")
c4.metric("PSNR", f"{r.psnr_db:.2f} dB" if r.psnr_db!=float("inf") else "∞ dB")
c5,c6,c7,c8 = st.columns(4)
c5.metric("L1 (Manhattan)", f"{r.l1:.4f}")
c6.metric("L∞ (Chebyshev)", f"{r.l_inf:.6f}")
c7.metric("Hist. Intersection", f"{r.hist_intersection:.6f}")
c8.metric("SVD Energy Ratio", f"{r.svd.energy_ratio:.4f}")

# Images
st.markdown("---"); st.markdown("### 🖼️ Original Images")
ca,cb = st.columns(2, gap="medium")
with ca:
    st.markdown(f'<div style="text-align:center;font-weight:600;color:#7c6aef;margin-bottom:8px;">📎 {r.path_a}</div>', unsafe_allow_html=True)
    st.image(img_a, use_container_width=True, clamp=True)
with cb:
    st.markdown(f'<div style="text-align:center;font-weight:600;color:#a855f7;margin-bottom:8px;">📎 {r.path_b}</div>', unsafe_allow_html=True)
    st.image(img_b, use_container_width=True, clamp=True)

# Heatmap + SVD
st.markdown("---")
ch,cs = st.columns(2, gap="medium")
with ch:
    st.markdown("#### 🔥 Difference Heatmap")
    diff = difference_image(img_a, img_b)
    fig = _dark_fig(10,5); ax = fig.add_subplot(111); _dark_ax(ax)
    im = ax.imshow(diff, cmap="magma", vmin=0, vmax=max(diff.max(),1e-10))
    ax.set_title("Pixel Differences", fontweight="bold", fontsize=13); ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="#6666aa"); plt.setp(cb.ax.yaxis.get_ticklabels(), color="#6666aa")
    st.pyplot(fig); plt.close(fig)
    st.caption(f"Max: {diff.max():.4f} | Mean: {diff.mean():.6f} | Non-zero: {np.count_nonzero(diff):,}/{diff.size:,}")
with cs:
    st.markdown("#### 📈 SVD Spectrum")
    from src.visualizer import plot_svd_spectrum
    fig2 = plot_svd_spectrum(r.svd.singular_values_a, r.svd.singular_values_b, top_k=50)
    fig2.patch.set_facecolor("#0b0b1e")
    for a in fig2.get_axes(): _dark_ax(a)
    st.pyplot(fig2); plt.close(fig2)
    st.caption(f"Energy A: {r.svd.energy_a:.2f} | Energy B: {r.svd.energy_b:.2f}")

# Histogram
st.markdown("---"); st.markdown("#### 📊 Histogram Comparison")
hr = histogram_intersection_detailed(img_a, img_b)
bc = (hr.bin_edges[:-1]+hr.bin_edges[1:])/2.0
fig3,ax3 = plt.subplots(figsize=(12,4)); _dark_ax(ax3)
ax3.fill_between(bc, hr.hist_a, alpha=0.3, color="#7c6aef", label="Image A")
ax3.fill_between(bc, hr.hist_b, alpha=0.3, color="#a855f7", label="Image B")
ax3.plot(bc, hr.hist_a, color="#7c6aef", linewidth=1); ax3.plot(bc, hr.hist_b, color="#a855f7", linewidth=1)
ax3.set_xlabel("Intensity (0-255)", color="#6666aa"); ax3.set_ylabel("Frequency", color="#6666aa")
ax3.set_title(f"Intersection = {hr.intersection:.4f}", fontweight="bold")
ax3.legend(facecolor="#111130", edgecolor="#1a1a44", labelcolor="#e0e0f0")
fig3.patch.set_facecolor("#0b0b1e"); st.pyplot(fig3); plt.close(fig3)

# Download
st.markdown("---"); st.markdown("### 📥 Download")
cd1,cd2 = st.columns(2)
with cd1:
    st.download_button("📄 Download Report (JSON)", r.to_json(indent=2), "comparison_report.json", "application/json", use_container_width=True)
with cd2:
    from src.visualizer import plot_full_comparison
    figd = plot_full_comparison(img_a, img_b, diff, r.svd.singular_values_a, r.svd.singular_values_b)
    figd.patch.set_facecolor("#0b0b1e")
    for a in figd.get_axes(): _dark_ax(a)
    buf = io.BytesIO(); figd.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0b0b1e"); plt.close(figd)
    st.download_button("🖼️ Download Dashboard (PNG)", buf.getvalue(), "dashboard.png", "image/png", use_container_width=True)

st.markdown("")
cn1,cn2 = st.columns(2)
with cn1: st.page_link("pages/2_🔬_Advanced_Analysis.py", label="🔬 Advanced", icon="🔬")
with cn2: st.page_link("pages/3_📥_Export_Results.py", label="📥 Export", icon="📥")
