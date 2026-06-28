"""PicPicComparison — Home page.

Upload two images, configure comparison settings, and run the analysis.
All helper functions are inlined here (not in ui/components) to avoid
Streamlit module-cache issues.
"""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from ui.styles import inject_custom_css

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="PicPicComparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# ── Session state defaults ──────────────────────────────────────────
for key, default in [
    ("img_a", None),
    ("img_b", None),
    ("pair", None),
    ("report", None),
    ("advanced_mode", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.html('<div style="margin-bottom:0.5rem"><span class="mode-badge"><span class="mode-badge-dot"></span> PicPicComparison v1.0</span></div>')

    st.markdown("### Settings")

    st.session_state.advanced_mode = st.toggle(
        "Advanced Mode",
        value=st.session_state.advanced_mode,
        help="Enable the full 8-step analysis pipeline with LaTeX formulas",
    )

    target_size = st.slider(
        "Resize to",
        min_value=64,
        max_value=1024,
        value=256,
        step=32,
        help="Target dimension for comparison — larger values give more detail but take longer",
    )

    svd_top_k = st.slider(
        "SVD Top-K",
        min_value=5,
        max_value=100,
        value=50,
        step=5,
        help="Number of top singular values to keep for SVD energy analysis",
    )

    histogram_bins = st.select_slider(
        "Histogram Bins",
        options=[32, 64, 128, 256],
        value=64,
        help="Number of bins for pixel intensity histograms",
    )

    # New comparison button in sidebar
    if st.session_state.report is not None:
        st.divider()
        if st.button("🔄  New Comparison", use_container_width=True):
            for k in ["img_a", "img_b", "pair", "report"]:
                st.session_state[k] = None
            st.rerun()

    st.divider()
    st.markdown(
        '<p style="color:#5a5a65;font-size:0.72rem;margin:0">Compare images using linear algebra & algorithmic metrics.</p>',
        unsafe_allow_html=True,
    )


# ── Header ──────────────────────────────────────────────────────────
st.html("""
<div class="page-header">
    <h1>PicPicComparison</h1>
    <p>Compare two images using linear algebra and algorithmic metrics</p>
</div>
""")


# ── Feature cards (always visible — even before upload) ─────────────
st.html("""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem">
    <div class="feature-card">
        <span class="feature-icon">📐</span>
        <div class="feature-title">Linear Algebra</div>
        <div class="feature-desc">Frobenius, L1, L∞, Cosine, SVD Energy</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">⚙️</span>
        <div class="feature-title">Algorithmic</div>
        <div class="feature-desc">MSE, PSNR, Histogram Intersection</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🔬</span>
        <div class="feature-title">Advanced Pipeline</div>
        <div class="feature-desc">Sobel, NCC, PCA, Patch Cosine, SVD Directional</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <div class="feature-title">Visualizations</div>
        <div class="feature-desc">Heatmaps, SVD Spectrum, Histograms, Dashboard</div>
    </div>
</div>
""")


# ── Empty state (when no comparison has been run) ───────────────────
if st.session_state.report is None:
    st.html("""
    <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <div class="empty-state-title">Upload two images to compare</div>
        <div class="empty-state-desc">
            Drop two images below to measure their similarity using 10+ metrics
            across linear algebra, signal processing, and algorithmic analysis.
            Results include side-by-side visualizations and an exportable report.
        </div>
    </div>
    """)


# ── Upload section ──────────────────────────────────────────────────
col_a, col_gap, col_b = st.columns([5, 1, 5])

with col_a:
    file_a = st.file_uploader(
        "Image A",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="upload_a",
        help="Upload the first image — PNG, JPG, BMP, TIFF, or WebP",
    )
    if file_a:
        file_a.seek(0)
        img_a_preview = Image.open(file_a)
        st.image(img_a_preview, caption="Image A (preview)", use_container_width=True)
        st.caption("Comparison uses grayscale internally — preview shows original colors")

with col_b:
    file_b = st.file_uploader(
        "Image B",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="upload_b",
        help="Upload the second image — PNG, JPG, BMP, TIFF, or WebP",
    )
    if file_b:
        file_b.seek(0)
        img_b_preview = Image.open(file_b)
        st.image(img_b_preview, caption="Image B (preview)", use_container_width=True)
        st.caption("Comparison uses grayscale internally — preview shows original colors")

with col_gap:
    st.html('<div style="display:flex;align-items:center;justify-content:center;height:100%;padding-top:3rem"><span style="color:#5a5a65;font-size:1.5rem;font-weight:600">vs</span></div>')


# ── Compare button ──────────────────────────────────────────────────
st.html('<div style="height:0.75rem"></div>')

compare_clicked = st.button(
    "Compare Images",
    type="primary",
    use_container_width=True,
    disabled=not (file_a and file_b),
)

if compare_clicked:
    with st.spinner("Analyzing images — this may take a moment..."):
        try:
            # Save uploads to temp files
            tmp_dir = Path(tempfile.mkdtemp())
            path_a = tmp_dir / "img_a.png"
            path_b = tmp_dir / "img_b.png"

            file_a.seek(0)
            path_a.write_bytes(file_a.read())
            file_b.seek(0)
            path_b.write_bytes(file_b.read())

            from src.loader import load_and_prepare_images
            from src.comparator import ImageComparator

            ts = (target_size, target_size)

            pair = load_and_prepare_images(
                str(path_a), str(path_b), target_size=ts
            )

            comparator = ImageComparator(
                target_size=ts,
                svd_top_k=svd_top_k,
                histogram_bins=histogram_bins,
            )
            report = comparator.compare_images(pair.image_a, pair.image_b)

            st.session_state.img_a = pair.image_a
            st.session_state.img_b = pair.image_b
            st.session_state.pair = pair
            st.session_state.report = report
        except Exception as e:
            st.error(f"Comparison failed: {e}")
            st.stop()

    st.toast("Comparison complete!", icon="✅")


# ── Results preview ─────────────────────────────────────────────────
if st.session_state.report is not None:
    report = st.session_state.report
    img_a = st.session_state.img_a
    img_b = st.session_state.img_b

    st.html('<div style="height:1rem"></div>')

    # ── Summary banner with plain-language interpretation ──
    cosine = report.cosine_sim
    if cosine >= 0.95:
        banner_cls = "good"
        icon = "✅"
        text = f"These images are <strong>very similar</strong> — Cosine Similarity: <strong>{cosine:.1%}</strong>"
    elif cosine >= 0.70:
        banner_cls = "warn"
        icon = "⚠️"
        text = f"These images are <strong>moderately similar</strong> — Cosine Similarity: <strong>{cosine:.1%}</strong>"
    else:
        banner_cls = "bad"
        icon = "❌"
        text = f"These images are <strong>significantly different</strong> — Cosine Similarity: <strong>{cosine:.1%}</strong>"

    st.html(f"""
    <div class="summary-banner {banner_cls}">
        <span class="summary-banner-icon">{icon}</span>
        <span class="summary-banner-text">{text}</span>
    </div>
    """)

    # ── Side-by-side thumbnails ──
    st.markdown("#### Comparison Preview")
    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        st.image(img_a, caption="Image A", use_container_width=True)
    with prev_col2:
        st.image(img_b, caption="Image B", use_container_width=True)

    # ── KPI strip with color coding ──
    kpi_data = [
        ("Cosine Similarity", f"{cosine:.4f}", "good" if cosine >= 0.95 else ("warn" if cosine >= 0.70 else "bad")),
        ("MSE", f"{report.mse:.6f}", "good" if report.mse < 0.001 else ("warn" if report.mse < 0.01 else "bad")),
        ("PSNR", f"{report.psnr_db:.2f} dB" if report.psnr_db != float("inf") else "∞ dB", "good"),
        ("Frobenius", f"{report.frobenius:.2f}", "good" if report.frobenius < 1 else ("warn" if report.frobenius < 10 else "bad")),
    ]

    kpi_html = '<div class="kpi-strip">'
    for label, value, cls in kpi_data:
        kpi_html += f'''
        <div class="kpi-card metric-{cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>'''
    kpi_html += "</div>"
    st.html(kpi_html)

    # ── Navigation links ──
    st.html("""
    <div class="nav-links">
        <a class="nav-link" href="/Basic_Comparison" target="_self">📊 Basic Comparison</a>
        <a class="nav-link" href="/Advanced_Analysis" target="_self">🔬 Advanced Analysis</a>
        <a class="nav-link" href="/Export_Results" target="_self">📥 Export Results</a>
    </div>
    """)
