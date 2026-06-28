"""
PicPicComparison — Dark Mode CSS Only.
No helper functions here — avoids Streamlit module-cache issues.
"""

from __future__ import annotations


def inject_custom_css() -> None:
    """Inject full-width dark-mode CSS."""
    import streamlit as st

    st.html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* === FULLSCREEN === */
        .block-container {
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 1.5rem !important;
        }
        .stApp { font-family: 'Inter', sans-serif; background: #0b0b1e !important; }

        /* === HEADER === */
        .page-header {
            background: linear-gradient(135deg, rgba(124,106,239,0.18) 0%, rgba(168,85,247,0.12) 40%, rgba(6,182,212,0.06) 100%);
            border: 1px solid rgba(124,106,239,0.15);
            border-radius: 16px;
            padding: 32px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .page-header::before {
            content: ''; position: absolute; top: -60px; right: -40px;
            width: 240px; height: 240px;
            background: radial-gradient(circle, rgba(124,106,239,0.1) 0%, transparent 70%);
            border-radius: 50%;
        }
        .page-header h1 {
            color: #fff !important; font-size: 1.9rem !important; font-weight: 800 !important;
            margin: 0 !important; letter-spacing: -0.02em;
            background: linear-gradient(135deg, #fff 0%, #c4b5fd 60%, #67e8f9 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .page-header p { color: rgba(224,224,240,0.55) !important; margin: 6px 0 0 0 !important; font-size: 0.95rem !important; }

        /* === SIDEBAR === */
        section[data-testid="stSidebar"] { background: #08081a !important; border-right: 1px solid rgba(255,255,255,0.04) !important; }
        section[data-testid="stSidebar"] .stMarkdown h2 { color: #e0e0f0 !important; font-weight: 700 !important; }
        section[data-testid="stSidebar"] .stMarkdown h3 { color: #8888aa !important; font-weight: 600 !important; }

        /* === METRIC CARDS === */
        div[data-testid="stMetric"] {
            background: rgba(17,17,48,0.7); border: 1px solid rgba(124,106,239,0.1);
            border-radius: 12px; padding: 18px 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.25); transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(124,106,239,0.3);
            box-shadow: 0 4px 20px rgba(0,0,0,0.35), 0 0 15px rgba(124,106,239,0.06);
            transform: translateY(-1px);
        }
        div[data-testid="stMetric"] label { font-size: 11px !important; font-weight: 600 !important; color: #7777aa !important; text-transform: uppercase; letter-spacing: 0.05em; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 800 !important; color: #fff !important; }

        /* === BUTTONS === */
        .stButton > button { border-radius: 10px !important; font-weight: 600 !important; transition: all 0.25s ease !important; }
        .stButton > button:hover { transform: translateY(-1px) !important; }
        .stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #7c6aef, #a855f7) !important; color: #fff !important;
            border: none !important; box-shadow: 0 4px 16px rgba(124,106,239,0.25) !important;
        }
        .stButton > button[kind="primary"]:hover { box-shadow: 0 6px 24px rgba(124,106,239,0.4) !important; }
        .stButton > button:disabled { opacity: 0.3 !important; transform: none !important; }

        /* === FILE UPLOADER === */
        div[data-testid="stFileUploader"] { border: 2px dashed rgba(124,106,239,0.2) !important; border-radius: 12px !important; background: rgba(124,106,239,0.02) !important; }
        div[data-testid="stFileUploader"]:hover { border-color: rgba(124,106,239,0.45) !important; }

        /* === EXPANDER === */
        div[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 10px !important; background: rgba(17,17,48,0.5) !important; }
        div[data-testid="stExpander"]:hover { border-color: rgba(124,106,239,0.15) !important; }

        /* === IMAGES === */
        .stImage { border-radius: 10px !important; overflow: hidden; }

        /* === DIVIDER === */
        hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.04) !important; }

        /* === SCROLLBAR === */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(124,106,239,0.2); border-radius: 3px; }

        /* === KPI STRIP === */
        .kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }
        .kpi-card {
            border-radius: 14px; padding: 20px 18px; text-align: center; color: #fff;
            position: relative; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.3); transition: all 0.3s ease;
        }
        .kpi-card::after { content: ''; position: absolute; inset: 0; background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 50%); pointer-events: none; }
        .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.4); }
        .kpi-card .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.8; margin-bottom: 6px; }
        .kpi-card .kpi-value { font-size: 28px; font-weight: 900; letter-spacing: -0.02em; }

        /* === FEATURE CARDS === */
        .feat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
        .feat-card { background: rgba(17,17,48,0.5); border: 1px solid rgba(255,255,255,0.04); border-radius: 10px; padding: 16px 12px; text-align: center; transition: all 0.3s ease; }
        .feat-card:hover { border-color: rgba(124,106,239,0.2); background: rgba(17,17,48,0.7); }
        .feat-icon { font-size: 26px; margin-bottom: 6px; }
        .feat-title { font-weight: 700; font-size: 13px; color: #e0e0f0; }
        .feat-sub { font-size: 11px; color: #6666aa; margin-top: 2px; }

        /* === MODE BADGE === */
        .mode-badge { text-align: center; padding: 10px; background: rgba(124,106,239,0.08); border: 1px solid rgba(124,106,239,0.12); border-radius: 10px; font-weight: 600; font-size: 0.85rem; color: #e0e0f0; }

        /* === INFO BANNER === */
        .info-banner { background: rgba(124,106,239,0.05); border: 1px solid rgba(124,106,239,0.1); border-radius: 10px; padding: 14px 18px; display: flex; align-items: center; gap: 10px; margin-bottom: 20px; color: #9999bb; font-size: 0.9rem; }

        /* === DOWNLOAD === */
        div[data-testid="stDownloadButton"] > button { background: rgba(34,197,94,0.1) !important; color: #22c55e !important; border: 1px solid rgba(34,197,94,0.2) !important; border-radius: 10px !important; }
        div[data-testid="stDownloadButton"] > button:hover { background: rgba(34,197,94,0.18) !important; }

        /* === SLIDER === */
        .stSlider [data-baseweb="thumb"] { background: #7c6aef !important; box-shadow: 0 0 8px rgba(124,106,239,0.4) !important; }

        /* === CODE === */
        div[data-testid="stCode"] { border-radius: 8px !important; border: 1px solid rgba(255,255,255,0.05) !important; }

        /* === ANIMATION === */
        @keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        .stApp { animation: fadeUp 0.4s ease-out; }
        </style>
        """
    )
