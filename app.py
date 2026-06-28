#!/usr/bin/env python3
"""PicPicComparison — Home Page (Dark, Fullscreen)"""

from __future__ import annotations
import sys, tempfile, os
from pathlib import Path
import numpy as np, streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.styles import inject_custom_css

# --- Helpers (inlined to avoid module-cache issues) ---
def _kpi(metrics):
    cols = st.columns(len(metrics))
    for col, (label, (value, gradient)) in zip(cols, metrics.items()):
        with col:
            st.html(
                f'<div class="kpi-card" style="background:{gradient};">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div></div>'
            )

def _features():
    feats = [("📐","Linear Algebra","Frobenius, L1, Cosine, SVD"),
             ("🧮","Algorithmic","MSE, PSNR, Histogram"),
             ("🔬","Advanced","Patch-Cosine, PCA, NCC"),
             ("📊","Visuals","Heatmaps, SVD, Histograms")]
    cols = st.columns(4)
    for col, (icon, title, sub) in zip(cols, feats):
        with col:
            st.html(
                f'<div class="feat-card">'
                f'<div class="feat-icon">{icon}</div>'
                f'<div class="feat-title">{title}</div>'
                f'<div class="feat-sub">{sub}</div>'
                f'</div>'
            )

# --- Config ---
st.set_page_config(page_title="PicPicComparison", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
inject_custom_css()

for k,v in {"advanced_mode":False,"img_a":None,"img_b":None,"pair":None,"report":None}.items():
    if k not in st.session_state: st.session_state[k] = v

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    advanced = st.toggle("🔬 Advanced Mode", value=st.session_state.advanced_mode, help="Step-by-step analysis.")
    st.session_state.advanced_mode = advanced
    st.markdown("---"); st.markdown("### 🎛️ Parameters")
    resize_size = st.slider("Resize to (px)", 64, 1024, 256, 32)
    svd_top_k = st.slider("SVD Top-K", 5, 100, 50, 5)
    hist_bins = st.select_slider("Histogram Bins", [64,128,256,512], 256)
    st.markdown("---")
    st.markdown(f'<div class="mode-badge">Mode: {"🔬 Advanced" if advanced else "📊 Basic"}</div>', unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="page-header"><h1>🔍 PicPicComparison</h1><p>Advanced Picture-to-Picture Comparison using Linear Algebra & Algorithmic Metrics</p></div>', unsafe_allow_html=True)
_features()

# --- Upload ---
st.markdown("### 📤 Upload Images")
col_a, col_b = st.columns(2, gap="large")
with col_a:
    file_a = st.file_uploader("Image A", type=["png","jpg","jpeg","bmp","webp","tif","tiff"], key="upload_a")
    if file_a:
        file_a.seek(0); st.image(Image.open(file_a).convert("L"), caption=f"📎 {file_a.name}", use_container_width=True)
with col_b:
    file_b = st.file_uploader("Image B", type=["png","jpg","jpeg","bmp","webp","tif","tiff"], key="upload_b")
    if file_b:
        file_b.seek(0); st.image(Image.open(file_b).convert("L"), caption=f"📎 {file_b.name}", use_container_width=True)

# --- Compare ---
st.markdown("")
c1,c2,c3 = st.columns([1,2,1])
with c2:
    if st.button("🚀 Compare Images", disabled=not(file_a and file_b), use_container_width=True, type="primary"):
        progress = st.progress(0, "🔄 Loading...")
        try:
            tmp = tempfile.mkdtemp(prefix="picpic_")
            pa,pb = os.path.join(tmp,"a.png"), os.path.join(tmp,"b.png")
            file_a.seek(0); open(pa,"wb").write(file_a.read())
            file_b.seek(0); open(pb,"wb").write(file_b.read())
            progress.progress(20, "🔄 Preprocessing...")
            from src.loader import load_image, resize_to_match, ImagePair, flatten_image
            ia,ib = load_image(pa), load_image(pb)
            ia,ib = resize_to_match(ia,ib,target_size=(resize_size,resize_size))
            pair = ImagePair(ia,ib,flatten_image(ia),flatten_image(ib),ia.shape[0],ia.shape[1],file_a.name,file_b.name)
            progress.progress(50, "🔄 Computing...")
            from src.comparator import ImageComparator
            report = ImageComparator(target_size=(resize_size,resize_size),svd_top_k=int(svd_top_k),histogram_bins=int(hist_bins)).compare_images(ia,ib)
            progress.progress(80, "🔄 Finalizing...")
            st.session_state.update(img_a=ia,img_b=ib,pair=pair,report=report)
            progress.progress(100, "✅ Done!"); st.success("✅ Comparison complete!")
        except Exception as e:
            st.error(f"❌ {e}"); progress.empty()

# --- Summary ---
if st.session_state.report:
    st.markdown("---"); st.markdown("### 📊 Quick Summary")
    r = st.session_state.report
    _kpi({
        "Cosine Similarity": (f"{r.cosine_sim*100:.1f}%", "linear-gradient(135deg,#7c6aef,#a855f7)"),
        "MSE": (f"{r.mse:.6f}", "linear-gradient(135deg,#ec4899,#f43f5e)"),
        "PSNR": (f"{r.psnr_db:.1f} dB" if r.psnr_db!=float("inf") else "∞ dB", "linear-gradient(135deg,#06b6d4,#3b82f6)"),
        "Frobenius": (f"{r.frobenius:.4f}", "linear-gradient(135deg,#22c55e,#10b981)"),
    })
    c1,c2,c3 = st.columns(3)
    with c1: st.page_link("pages/1_📊_Basic_Comparison.py", label="📊 Basic Comparison", icon="📊")
    with c2: st.page_link("pages/2_🔬_Advanced_Analysis.py", label="🔬 Advanced Analysis", icon="🔬")
    with c3: st.page_link("pages/3_📥_Export_Results.py", label="📥 Export Results", icon="📥")
