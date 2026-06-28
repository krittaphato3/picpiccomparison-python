#!/usr/bin/env python3
"""PicPicComparison — Advanced Analysis (Dark, Fullscreen)"""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.styles import inject_custom_css
from ui.step_runner import run_advanced_pipeline
inject_custom_css()

if st.session_state.report is None:
    st.warning("⚠️ No results yet. Upload images on the Home page.")
    st.page_link("app.py", label="🏠 Home", icon="🏠"); st.stop()

r = st.session_state.report

st.markdown(f'<div class="page-header"><h1>🔬 Advanced Analysis</h1><p>Step-by-step pipeline for {r.path_a} vs {r.path_b}</p></div>', unsafe_allow_html=True)
st.markdown('<div class="info-banner"><div style="font-size:20px;">💡</div><div><strong>How it works:</strong> Click each step to expand and view the LaTeX formula, intermediate computation, and visualization.</div></div>', unsafe_allow_html=True)

run_advanced_pipeline(st.session_state.img_a, st.session_state.img_b, st.session_state.pair, r)

st.markdown("")
cn1,cn2 = st.columns(2)
with cn1: st.page_link("pages/1_📊_Basic_Comparison.py", label="📊 Basic", icon="📊")
with cn2: st.page_link("pages/3_📥_Export_Results.py", label="📥 Export", icon="📥")
