#!/usr/bin/env python3
"""PicPicComparison — Export Results (Dark, Fullscreen)"""

from __future__ import annotations
import io, json, sys, zipfile
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.styles import inject_custom_css
from src.algo_metrics import difference_image, histogram_intersection_detailed
from src.visualizer import (plot_original_images, plot_difference_heatmap, plot_svd_spectrum,
                            plot_histogram_comparison, plot_full_comparison)
inject_custom_css()

if st.session_state.report is None:
    st.warning("⚠️ No results yet. Upload images on the Home page.")
    st.page_link("app.py", label="🏠 Home", icon="🏠"); st.stop()

r = st.session_state.report
img_a, img_b = st.session_state.img_a, st.session_state.img_b
diff = difference_image(img_a, img_b)

st.markdown(f'<div class="page-header"><h1>📥 Export Results</h1><p>Download all outputs from {r.path_a} vs {r.path_b}</p></div>', unsafe_allow_html=True)

# Generate figures
figs = {}
figs["original_images.png"] = plot_original_images(img_a, img_b, title_a=r.path_a, title_b=r.path_b)
figs["difference_heatmap.png"] = plot_difference_heatmap(diff)
figs["svd_spectrum.png"] = plot_svd_spectrum(r.svd.singular_values_a, r.svd.singular_values_b, top_k=50)
hr = histogram_intersection_detailed(img_a, img_b)
figs["histogram_comparison.png"] = plot_histogram_comparison(hr.hist_a, hr.hist_b, hr.bin_edges, hr.intersection)
figs["full_dashboard.png"] = plot_full_comparison(img_a, img_b, diff, r.svd.singular_values_a, r.svd.singular_values_b)

# Style all dark
for f in figs.values():
    f.patch.set_facecolor("#0b0b1e")
    for a in f.get_axes():
        a.set_facecolor("#0b0b1e"); a.tick_params(colors="#6666aa")
        a.title.set_color("#e0e0f0")
        for s in a.spines.values(): s.set_color("#1a1a44")

# JSON
st.markdown("### 📄 JSON Report")
rj = r.to_json(indent=2)
cd1,cd2 = st.columns([2,1])
with cd1: st.download_button("⬇️ Download Report (JSON)", rj, "comparison_report.json", "application/json", use_container_width=True, type="primary")
with cd2: st.metric("Size", f"{len(rj):,} bytes")
with st.expander("👁️ Preview JSON", expanded=False): st.json(rj)

# Plots
st.markdown("---"); st.markdown("### 🖼️ Download Plots")
cols = st.columns(3)
for i,(fname,fig) in enumerate(figs.items()):
    with cols[i%3]:
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0b0b1e"); plt.close(fig)
        st.download_button(f"📥 {fname}", buf.getvalue(), fname, "image/png", use_container_width=True)

# ZIP
st.markdown("---"); st.markdown("### 📦 Download All")
zbuf = io.BytesIO()
with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("comparison_report.json", rj)
    for fname,fig in figs.items():
        ib = io.BytesIO(); fig.savefig(ib, format="png", dpi=150, bbox_inches="tight", facecolor="#0b0b1e"); plt.close(fig)
        zf.writestr(fname, ib.getvalue())
cd3,cd4 = st.columns([2,1])
with cd3: st.download_button("📦 Download All (ZIP)", zbuf.getvalue(), "picpiccomparison_results.zip", "application/zip", use_container_width=True, type="primary")
with cd4: st.metric("ZIP Size", f"{len(zbuf.getvalue()):,} bytes")

# Table
st.markdown("---"); st.markdown("### 📋 Raw Metrics")
df = pd.DataFrame({"Metric": ["Frobenius (L2)","L1 (Manhattan)","L∞ (Chebyshev)","Cosine Similarity","Dot Product","MSE","PSNR (dB)","Hist. Intersection","SVD Energy A","SVD Energy B","SVD Energy Ratio","SVD Cosine Distance","SVD Top-K Cosine"],
    "Value": [f"{r.frobenius:.6f}",f"{r.l1:.6f}",f"{r.l_inf:.6f}",f"{r.cosine_sim:.6f}",f"{r.dot_prod:.4f}",f"{r.mse:.8f}",
              f"{r.psnr_db:.2f}" if r.psnr_db!=float("inf") else "∞",f"{r.hist_intersection:.6f}",
              f"{r.svd.energy_a:.4f}",f"{r.svd.energy_b:.4f}",f"{r.svd.energy_ratio:.4f}",f"{r.svd.cosine_distance_sv:.6f}",f"{r.svd.top_k_singular_cosine:.6f}"],
    "Meaning": ["Euclidean distance","Sum of absolute diffs","Max absolute difference","Structural similarity",
                "Vector dot product","Mean squared error","Signal-to-noise ratio","Intensity overlap",
                "Sum of σ² for A","Sum of σ² for B","Energy A / Energy B","1 - cos(σ_A, σ_B)","Top-K feature similarity"]})
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("")
cn1,cn2 = st.columns(2)
with cn1: st.page_link("pages/1_📊_Basic_Comparison.py", label="📊 Basic", icon="📊")
with cn2: st.page_link("pages/2_🔬_Advanced_Analysis.py", label="🔬 Advanced", icon="🔬")
