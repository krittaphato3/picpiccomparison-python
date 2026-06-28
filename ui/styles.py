"""Custom CSS injection for the PicPicComparison dark-mode UI.

Black & gray palette with smooth motion, glass morphism, and premium feel.
Only exports ``inject_custom_css()`` — all helper functions live in the page
files to avoid Streamlit module-cache issues.
"""


def inject_custom_css() -> None:
    """Inject the full custom CSS block into the Streamlit page."""
    css = """
<style>
/* ================================================================
   0.  RESET & GLOBALS
   ================================================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    /* Backgrounds */
    --bg-base:      #0a0a0f;
    --bg-surface:   #111118;
    --bg-card:      #16161e;
    --bg-elevated:  #1c1c26;
    --bg-hover:     #22222e;
    /* Borders */
    --border:       #2a2a35;
    --border-light: #333340;
    /* Text */
    --text-primary:   #e8e8ec;
    --text-secondary: #8a8a95;
    --text-muted:     #5a5a65;
    --white:          #ffffff;
    /* Metric colors */
    --metric-good:  #4ade80;
    --metric-warn:  #fbbf24;
    --metric-bad:   #f87171;
    /* Radii */
    --radius-sm:    8px;
    --radius-md:    12px;
    --radius-lg:    16px;
    --radius-xl:    20px;
    /* Timing */
    --transition-fast: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-med:  0.35s cubic-bezier(0.4, 0, 0.2, 1);
    /* Shadows */
    --shadow-sm:  0 1px 3px rgba(0,0,0,0.4);
    --shadow-md:  0 4px 16px rgba(0,0,0,0.5);
    --shadow-lg:  0 8px 32px rgba(0,0,0,0.6);
    --shadow-xl:  0 16px 48px rgba(0,0,0,0.7);
}

html { scroll-behavior: smooth; }
* { box-sizing: border-box; }

/* ── Scrollbar (WebKit + Firefox) ─────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; transition: background var(--transition-fast); }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
* { scrollbar-width: thin; scrollbar-color: var(--border) var(--bg-base); }

/* ── Hide Streamlit chrome ────────────────────────────────────── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Full viewport ────────────────────────────────────────────── */
.block-container {
    max-width: 100% !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-base);
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── Focus-visible (WCAG 2.4.7) ──────────────────────────────── */
:focus-visible {
    outline: 2px solid var(--white);
    outline-offset: 2px;
    border-radius: 4px;
}
button:focus-visible,
.stButton > button:focus-visible {
    outline: 2px solid var(--white);
    outline-offset: 2px;
}

/* ── Streamlit headings ───────────────────────────────────────── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: var(--white) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

/* ── Reduced motion ───────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}


/* ================================================================
   1.  PAGE HEADER
   ================================================================ */
.page-header {
    position: relative;
    padding: 2.5rem 2rem 2rem;
    margin: -2rem -2rem 2rem;
    border-radius: 0 0 var(--radius-xl) var(--radius-xl);
    background: linear-gradient(135deg, #111118 0%, #16161e 100%);
    border-bottom: 1px solid var(--border);
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
    pointer-events: none;
}
.page-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--white);
    margin: 0 0 0.25rem;
    letter-spacing: -0.02em;
}
.page-header p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin: 0;
    font-weight: 400;
}

/* ================================================================
   1b.  PAGE BREADCRUMB
   ================================================================ */
.page-breadcrumb {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.75rem;
    color: var(--text-muted);
}
.page-breadcrumb a {
    color: var(--text-secondary);
    text-decoration: none;
    transition: color var(--transition-fast);
}
.page-breadcrumb a:hover { color: var(--white); }
.page-breadcrumb .sep { color: var(--text-muted); }


/* ================================================================
   2.  SIDEBAR
   ================================================================ */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--white) !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    opacity: 0.5;
}


/* ================================================================
   3.  CARDS — Glass Morphism (staggered entrance)
   ================================================================ */
@keyframes cardEnter {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    transition: transform var(--transition-med), box-shadow var(--transition-med), border-color var(--transition-med);
    position: relative;
    overflow: hidden;
    animation: cardEnter 0.5s ease-out backwards;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 50%, transparent 100%);
}
.glass-card:hover {
    border-color: var(--border-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}


/* ================================================================
   4.  KPI STRIP — Gradient cards with staggered entrance
   ================================================================ */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1rem;
    text-align: center;
    transition: transform var(--transition-med), box-shadow var(--transition-med), border-color var(--transition-med);
    position: relative;
    overflow: hidden;
    animation: cardEnter 0.5s ease-out backwards;
}
.kpi-card:nth-child(1) { animation-delay: 0.05s; }
.kpi-card:nth-child(2) { animation-delay: 0.10s; }
.kpi-card:nth-child(3) { animation-delay: 0.15s; }
.kpi-card:nth-child(4) { animation-delay: 0.20s; }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #444, #666);
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
    border-color: var(--border-light);
}
.kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--white);
    letter-spacing: -0.02em;
    line-height: 1.2;
}


/* ================================================================
   5.  FEATURE CARDS — with staggered entrance
   ================================================================ */
.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem 1.25rem;
    text-align: center;
    transition: transform var(--transition-med), box-shadow var(--transition-med), border-color var(--transition-med);
    cursor: default;
    position: relative;
    overflow: hidden;
    animation: cardEnter 0.5s ease-out backwards;
}
.feature-card:nth-child(1) { animation-delay: 0.10s; }
.feature-card:nth-child(2) { animation-delay: 0.18s; }
.feature-card:nth-child(3) { animation-delay: 0.26s; }
.feature-card:nth-child(4) { animation-delay: 0.34s; }
.feature-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--border-light), transparent);
    opacity: 0;
    transition: opacity var(--transition-med);
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: var(--border-light);
    box-shadow: var(--shadow-lg);
}
.feature-card:hover::after {
    opacity: 1;
}
.feature-icon {
    font-size: 1.75rem;
    margin-bottom: 0.75rem;
    display: block;
}
.feature-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--white);
    margin-bottom: 0.3rem;
}
.feature-desc {
    font-size: 0.72rem;
    color: var(--text-secondary);
    line-height: 1.5;
}


/* ================================================================
   6.  METRIC CARDS — with left accent + staggered entrance
   ================================================================ */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1rem;
    transition: transform var(--transition-med), box-shadow var(--transition-med), border-color var(--transition-med);
    position: relative;
    overflow: hidden;
    animation: cardEnter 0.5s ease-out backwards;
}
.metric-card:nth-child(1) { animation-delay: 0.05s; }
.metric-card:nth-child(2) { animation-delay: 0.08s; }
.metric-card:nth-child(3) { animation-delay: 0.11s; }
.metric-card:nth-child(4) { animation-delay: 0.14s; }
.metric-card:nth-child(5) { animation-delay: 0.17s; }
.metric-card:nth-child(6) { animation-delay: 0.20s; }
.metric-card:nth-child(7) { animation-delay: 0.23s; }
.metric-card:nth-child(8) { animation-delay: 0.26s; }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px;
    height: 100%;
    background: linear-gradient(180deg, #555, #333);
    border-radius: 3px 0 0 3px;
}
.metric-card:hover {
    border-color: var(--border-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.metric-name {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--white);
    letter-spacing: -0.01em;
}

/* ── Metric color coding ──────────────────────────────────────── */
.metric-good .metric-value { color: var(--metric-good); }
.metric-warn .metric-value { color: var(--metric-warn); }
.metric-bad  .metric-value { color: var(--metric-bad); }
.metric-good::before { background: linear-gradient(180deg, var(--metric-good), #22543d) !important; }
.metric-warn::before { background: linear-gradient(180deg, var(--metric-warn), #744210) !important; }
.metric-bad::before  { background: linear-gradient(180deg, var(--metric-bad),  #742a2a) !important; }


/* ================================================================
   6b.  ST.metric OVERRIDE — match dark theme
   ================================================================ */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
}
[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--white) !important;
    font-weight: 700 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] > div {
    color: var(--text-secondary) !important;
}


/* ================================================================
   7.  MODE BADGE
   ================================================================ */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-weight: 500;
    letter-spacing: 0.04em;
    transition: border-color var(--transition-fast), background var(--transition-fast);
}
.mode-badge:hover {
    border-color: var(--border-light);
    background: var(--bg-hover);
}
.mode-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--metric-good);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}


/* ================================================================
   8.  BUTTONS
   ================================================================ */
/* Primary — solid white on dark */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--white) !important;
    color: var(--bg-base) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: transform var(--transition-fast), box-shadow var(--transition-fast), background var(--transition-fast) !important;
    letter-spacing: 0.01em;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #e0e0e0 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255,255,255,0.1);
}
.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="stBaseButton-primary"]:active {
    transform: translateY(0) scale(0.98);
    background: #ccc !important;
}

/* Secondary — ghost / outlined */
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]),
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: transform var(--transition-fast), color var(--transition-fast), border-color var(--transition-fast), background var(--transition-fast) !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    color: var(--white) !important;
    border-color: var(--border-light) !important;
    background: var(--bg-elevated) !important;
    transform: translateY(-1px);
}
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):active,
.stButton > button[data-testid="stBaseButton-secondary"]:active {
    transform: translateY(0) scale(0.98);
}

/* Disabled buttons */
.stButton > button:disabled,
.stButton > button[disabled] {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    transition: transform var(--transition-fast), border-color var(--transition-fast), background var(--transition-fast) !important;
}
.stDownloadButton > button:hover {
    border-color: var(--border-light) !important;
    background: var(--bg-hover) !important;
    transform: translateY(-1px);
}
.stDownloadButton > button:active {
    transform: translateY(0) scale(0.98);
}


/* ================================================================
   9.  FILE UPLOADER
   ================================================================ */
.stFileUploader {
    border: none !important;
}
.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color var(--transition-fast), background var(--transition-fast) !important;
}
.stFileUploader > div:hover {
    border-color: var(--border-light) !important;
    background: var(--bg-elevated) !important;
}
.stFileUploader label {
    color: var(--text-secondary) !important;
}


/* ================================================================
   10.  EXPANDERS
   ================================================================ */
details[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color var(--transition-fast), box-shadow var(--transition-fast) !important;
    margin-bottom: 0.75rem !important;
}
details[data-testid="stExpander"]:hover {
    border-color: var(--border-light) !important;
}
details[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.75rem 1rem !important;
}
details[data-testid="stExpander"][open] {
    box-shadow: var(--shadow-sm) !important;
}


/* ================================================================
   11.  IMAGES
   ================================================================ */
.stImage > img {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    transition: border-color var(--transition-med) !important;
}
.stImage > img:hover {
    border-color: var(--border-light) !important;
}


/* ================================================================
   12.  SLIDER
   ================================================================ */
.stSlider > div > div > div {
    background-color: var(--border) !important;
}
.stSlider > div > div > div > div {
    background-color: var(--white) !important;
}


/* ================================================================
   13.  TABS
   ================================================================ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.82rem;
    transition: color var(--transition-fast), background var(--transition-fast);
    padding: 0.5rem 1rem;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary);
    background: var(--bg-elevated);
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--white) !important;
    border-color: var(--border-light) !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}


/* ================================================================
   14.  DATAFRAME / TABLE
   ================================================================ */
.stDataFrame {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
}


/* ================================================================
   15.  ALERTS
   ================================================================ */
.stAlert {
    border-radius: var(--radius-sm) !important;
    border-left-width: 3px !important;
    font-size: 0.82rem !important;
}


/* ================================================================
   16.  SPINNER
   ================================================================ */
.stSpinner > div {
    border-top-color: var(--text-secondary) !important;
}


/* ================================================================
   17.  PROGRESS BAR
   ================================================================ */
.stProgress > div > div {
    background-color: var(--border) !important;
}
.stProgress > div > div > div {
    background-color: var(--white) !important;
}


/* ================================================================
   18.  COLUMNS — Subtle dividers
   ================================================================ */
[data-testid="stHorizontalBlock"] > div:not(:last-child) {
    border-right: 1px solid var(--border);
    padding-right: 1rem;
}


/* ================================================================
   19.  NAVIGATION LINKS
   ================================================================ */
.nav-links {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}
.nav-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.55rem 1.1rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 500;
    transition: transform var(--transition-fast), color var(--transition-fast), border-color var(--transition-fast), background var(--transition-fast);
    cursor: pointer;
}
.nav-link:hover {
    color: var(--white);
    border-color: var(--border-light);
    background: var(--bg-elevated);
    transform: translateY(-1px);
}
.nav-link:active {
    transform: translateY(0) scale(0.98);
}


/* ================================================================
   20.  STEP PIPELINE
   ================================================================ */
.step-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: border-color var(--transition-med), box-shadow var(--transition-med);
    position: relative;
}
.step-card:hover {
    border-color: var(--border-light);
    box-shadow: var(--shadow-md);
}
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.step-number.active {
    background: var(--white);
    color: var(--bg-base);
    border-color: var(--white);
}
.step-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--white);
    margin-bottom: 0.25rem;
}
.step-desc {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── Pipeline progress bar ────────────────────────────────────── */
.pipeline-progress {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    padding: 0.75rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
}
.pipeline-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    transition: background var(--transition-fast), border-color var(--transition-fast);
    flex-shrink: 0;
}
.pipeline-dot.done {
    background: var(--metric-good);
    border-color: var(--metric-good);
}
.pipeline-dot.active {
    background: var(--white);
    border-color: var(--white);
    box-shadow: 0 0 8px rgba(255,255,255,0.3);
}
.pipeline-line {
    flex: 1;
    height: 2px;
    background: var(--border);
    border-radius: 1px;
}
.pipeline-line.done {
    background: var(--metric-good);
}


/* ================================================================
   21.  EMPTY STATE
   ================================================================ */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    margin: 1rem 0;
    background: var(--bg-card);
    border: 1px dashed var(--border);
    border-radius: var(--radius-lg);
    transition: border-color var(--transition-fast);
}
.empty-state:hover {
    border-color: var(--border-light);
}
.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    opacity: 0.6;
}
.empty-state-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--white);
    margin-bottom: 0.4rem;
}
.empty-state-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}


/* ================================================================
   22.  SUMMARY BANNER (post-comparison)
   ================================================================ */
.summary-banner {
    padding: 1rem 1.5rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    animation: cardEnter 0.5s ease-out;
}
.summary-banner.good {
    background: rgba(74, 222, 128, 0.06);
    border-color: rgba(74, 222, 128, 0.2);
    color: var(--metric-good);
}
.summary-banner.warn {
    background: rgba(251, 191, 36, 0.06);
    border-color: rgba(251, 191, 36, 0.2);
    color: var(--metric-warn);
}
.summary-banner.bad {
    background: rgba(248, 113, 113, 0.06);
    border-color: rgba(248, 113, 113, 0.2);
    color: var(--metric-bad);
}
.summary-banner-icon { font-size: 1.25rem; }
.summary-banner-text { color: var(--text-primary); font-weight: 400; }
.summary-banner-text strong { font-weight: 700; }


/* ================================================================
   23.  SKELETON LOADING
   ================================================================ */
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.skeleton {
    background: linear-gradient(90deg,
        var(--bg-card) 25%,
        var(--bg-elevated) 50%,
        var(--bg-card) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: var(--radius-sm);
}


/* ================================================================
   24.  FORMULA BOXES
   ================================================================ */
.formula-box {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    text-align: center;
    font-size: 0.9rem;
    color: var(--text-primary);
    transition: border-color var(--transition-fast);
}
.formula-box:hover {
    border-color: var(--border-light);
}


/* ================================================================
   25.  NEW COMPARISON BUTTON
   ================================================================ */
.stButton > button.new-comparison {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 1rem !important;
    transition: color var(--transition-fast), border-color var(--transition-fast) !important;
}
.stButton > button.new-comparison:hover {
    color: var(--text-secondary) !important;
    border-color: var(--border-light) !important;
}


/* ================================================================
   26.  CHECKBOX / TOGGLE / RADIO / SELECT
   ================================================================ */
.stCheckbox > label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}
.stCheckbox > label:hover {
    color: var(--text-primary) !important;
}
.stRadio > label,
.stSelectbox > label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}


/* ================================================================
   27.  CAPTION / SMALL TEXT
   ================================================================ */
.stCaption, .caption {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
}


/* ================================================================
   28.  DIVIDER
   ================================================================ */
hr {
    border-color: var(--border) !important;
    opacity: 0.4;
}


/* ================================================================
   29.  CODE BLOCKS
   ================================================================ */
code {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 0.15rem 0.4rem !important;
    font-size: 0.78rem !important;
}


/* ================================================================
   30.  LATEX
   ================================================================ */
.stLatex {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.75rem !important;
}
</style>
"""
    import streamlit as st
    st.html(css)
