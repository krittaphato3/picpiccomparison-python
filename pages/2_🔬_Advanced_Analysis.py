"""Page 2 — Advanced Analysis.

Runs the 8-step pipeline with LaTeX formulas and intermediate visualizations.
All helper functions inlined to avoid Streamlit module-cache issues.
"""

import streamlit as st

from ui.styles import inject_custom_css

st.set_page_config(page_title="Advanced Analysis", page_icon="🔬", layout="wide")
inject_custom_css()

# ── Guard ───────────────────────────────────────────────────────────
if st.session_state.report is None:
    st.warning("No comparison results yet. Run a comparison from the Home page first.")
    st.stop()

# ── Header ──────────────────────────────────────────────────────────
st.html("""
<div class="page-header">
    <h1>Advanced Analysis</h1>
    <p>8-step pipeline with detailed metrics and visualizations</p>
</div>
""")

# ── Run pipeline ────────────────────────────────────────────────────
from ui.step_runner import run_advanced_pipeline

run_advanced_pipeline(
    img_a=st.session_state.img_a,
    img_b=st.session_state.img_b,
    pair=st.session_state.pair,
    report=st.session_state.report,
)

# ── Navigation ──────────────────────────────────────────────────────
st.html("""
<div class="nav-links">
    <a class="nav-link" href="/" target="_self">← Home</a>
    <a class="nav-link" href="/Basic_Comparison" target="_self">📊 Basic Comparison →</a>
    <a class="nav-link" href="/Export_Results" target="_self">📥 Export Results →</a>
</div>
""")
