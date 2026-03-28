import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Socio-Technical Health Monitor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── ROOT THEME ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-card:       #111827;
    --bg-card-hover: #161f32;
    --bg-sidebar:    #0d1220;
    --border:        #1e2d45;
    --border-accent: #2a3f5f;
    --text-primary:  #e8edf5;
    --text-secondary:#8a9bb5;
    --text-muted:    #4a5a72;
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --green:         #10b981;
    --amber:         #f59e0b;
    --red:           #ef4444;
    --red-soft:      #fca5a5;
    --green-soft:    #6ee7b7;
    --font-main:     'IBM Plex Sans', sans-serif;
    --font-mono:     'IBM Plex Mono', monospace;
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: var(--font-main);
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 1rem 2.5rem !important;
    max-width: 1400px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}
[data-testid="stSidebar"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background-color: #0a0e1a !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] .stSlider > div > div {
    background: var(--border-accent) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:first-child > div:first-child {
    background: var(--accent-blue) !important;
}

/* ── SECTION DIVIDERS IN SIDEBAR ── */
.sidebar-section {
    margin: 1.2rem 0 0.4rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--accent-cyan);
    text-transform: uppercase;
}

/* ── CARDS ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: var(--border-accent); }

/* ── HEADER BANNER ── */
.header-banner {
    background: linear-gradient(135deg, #0d1526 0%, #111f38 50%, #0a1628 100%);
    border: 1px solid var(--border-accent);
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), transparent);
}
.header-title {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.05em;
}
.header-subtitle {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
    font-weight: 300;
}
.header-badge {
    background: #0a1628;
    border: 1px solid var(--border-accent);
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--accent-cyan);
    letter-spacing: 0.08em;
    text-align: right;
    line-height: 1.8;
}

/* ── RISK GAUGE ── */
.gauge-container {
    text-align: center;
    padding: 1rem 0;
}
.risk-value {
    font-family: var(--font-mono);
    font-size: 3.8rem;
    font-weight: 600;
    line-height: 1;
    margin: 0.5rem 0;
}
.risk-label {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.risk-bar-track {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    margin: 0.8rem 0.5rem;
    overflow: hidden;
    position: relative;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.status-pill {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    font-weight: 600;
    text-transform: uppercase;
    margin: 0.5rem 0;
}
.action-box {
    background: #0a0e1a;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-top: 1rem;
    font-size: 0.82rem;
    line-height: 1.6;
    color: var(--text-secondary);
    border-left: 3px solid;
    text-align: left;
}

/* ── FEATURE TILES ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin: 0.5rem 0;
}
.feature-tile {
    background: #0a0e1a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
}
.feature-tile-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.feature-tile-value {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ── SECTION HEADERS ── */
.section-header {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: var(--accent-cyan);
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── METRICS ROW ── */
.metrics-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 0.5rem;
}
.metric-chip {
    background: #0a0e1a;
    border: 1px solid var(--border-accent);
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    flex: 1;
    text-align: center;
}
.metric-chip-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    text-transform: uppercase;
}
.metric-chip-value {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-cyan);
    margin-top: 0.1rem;
}

/* ── FOOTER ── */
.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-left {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    line-height: 1.8;
}
.footer-right {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    text-align: right;
    line-height: 1.8;
}

/* ── STREAMLIT WIDGET OVERRIDES ── */
.stSelectbox > div > div {
    background-color: #0a0e1a !important;
    border-color: var(--border-accent) !important;
    color: var(--text-primary) !important;
}
.stNumberInput > div > div > input {
    background-color: #0a0e1a !important;
    border-color: var(--border-accent) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}
div[data-baseweb="select"] {
    background-color: #0a0e1a !important;
}
.stMarkdown p { color: var(--text-secondary); }

/* ── SHAP FRAME ── */
iframe {
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/isa3_xgboost_honest.pkl')
    explainer = joblib.load('models/isa3_shap_explainer_honest.pkl')
    return model, explainer

model, explainer = load_assets()

# Removed has_enough_emails for the 6 Core Signals approach
features = [
    'email_count_per_ticket', 'subject_length',
    'avg_sentiment', 'sentiment_variance', 'sentiment_trend',
    'priority_numeric'
]

PRIORITY_MAP = {1: "Trivial", 2: "Minor", 3: "Major", 4: "Critical", 5: "Blocker"}


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; font-weight:600; color:#e8edf5; letter-spacing:0.05em; padding-bottom:0.8rem; border-bottom:1px solid #1e2d45; margin-bottom:1rem;">
    ◈ INPUT PARAMETERS
</div>
<div style="font-size:0.75rem; color:#8a9bb5; margin-bottom:1.2rem; line-height:1.6;">
    Adjust the socio-technical signals extracted from JIRA and developer mailing list data.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">01 · Communication Volume</div>', unsafe_allow_html=True)
    email_count = st.number_input("Human Email Count", min_value=1, max_value=100, value=4,
                                   help="Number of human-authored emails referencing this ticket")
    subject_length = st.number_input("Subject Length (words)", min_value=1, max_value=30, value=8,
                                      help="Average word count of email subject lines for this ticket")

    st.markdown('<div class="sidebar-section">02 · Sentiment Signals</div>', unsafe_allow_html=True)
    avg_sent = st.slider("Average Sentiment", -1.0, 1.0, 0.2, 0.01,
                          help="Mean VADER score across all emails (-1 = max stress, +1 = max positive)")
    sent_var = st.slider("Sentiment Variance", 0.0, 1.0, 0.3, 0.01,
                          help="Spread of sentiment scores. Low variance on a stalled ticket = flat, abandoned communication")
    sent_trend = st.slider("Sentiment Trend", -1.0, 1.0, -0.1, 0.01,
                            help="3-month rolling direction of sentiment. Negative = declining morale")

    st.markdown('<div class="sidebar-section">03 · Task Metadata</div>', unsafe_allow_html=True)
    priority = st.selectbox("Ticket Priority", [1, 2, 3, 4, 5], index=2,
                             format_func=lambda x: f"{x} — {PRIORITY_MAP[x]}")

    st.markdown(f"""
<div style="margin-top:1.5rem; padding:0.8rem; background:#0a0e1a; border:1px solid #1e2d45; border-radius:6px; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#4a5a72; line-height:2;">
    priority_numeric → <span style="color:#06b6d4">{priority}</span><br>
    features active → <span style="color:#06b6d4">6 / 6</span>
</div>
""", unsafe_allow_html=True)


# ── PREDICTION ──────────────────────────────────────────────────────────────
# Updated input array to 6 features
input_array = np.array([[email_count, subject_length, avg_sent, sent_var, sent_trend, priority]])
input_df = pd.DataFrame(input_array, columns=features)
prob_stalled = model.predict_proba(input_df)[0][1]
pct = prob_stalled * 100

if pct >= 70:
    risk_level = "HIGH RISK"
    risk_color = "#ef4444"
    risk_bg    = "rgba(239,68,68,0.08)"
    bar_grad   = "linear-gradient(90deg, #ef4444, #dc2626)"
    action_msg = "Immediate managerial intervention recommended. Communication patterns indicate task abandonment or sustained developer frustration. Review thread history and reassign if unassigned."
    action_border = "#ef4444"
    pill_style = f"background:rgba(239,68,68,0.15); color:#fca5a5; border:1px solid rgba(239,68,68,0.4);"
elif pct >= 40:
    risk_level = "ELEVATED"
    risk_color = "#f59e0b"
    risk_bg    = "rgba(245,158,11,0.08)"
    bar_grad   = "linear-gradient(90deg, #f59e0b, #d97706)"
    action_msg = "Monitor closely. Declining sentiment trend or low email engagement may indicate early-stage stalling. Consider a brief check-in with the assigned developer."
    action_border = "#f59e0b"
    pill_style = f"background:rgba(245,158,11,0.15); color:#fde68a; border:1px solid rgba(245,158,11,0.4);"
else:
    risk_level = "HEALTHY"
    risk_color = "#10b981"
    risk_bg    = "rgba(16,185,129,0.08)"
    bar_grad   = "linear-gradient(90deg, #10b981, #059669)"
    action_msg = "Socio-technical indicators are within normal range. Active communication and stable sentiment suggest the task is progressing. No intervention required."
    action_border = "#10b981"
    pill_style = f"background:rgba(16,185,129,0.15); color:#6ee7b7; border:1px solid rgba(16,185,129,0.4);"


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-banner">
    <div>
        <div class="header-title">◈ SOCIO-TECHNICAL HEALTH MONITOR</div>
        <div class="header-subtitle">
            Predicting task stalling risk from pure developer communication and sentiment signals
        </div>
    </div>
    <div class="header-badge">
        MODEL · XGBoost (ISA III)<br>
        FEATURES · 6 Pure Signals<br>
        DATASET · Apache Hadoop 2023–2024
    </div>
</div>
""", unsafe_allow_html=True)


# ── MAIN LAYOUT ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.7], gap="large")

# LEFT COLUMN — Risk Assessment
with col_left:
    st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="card" style="background:{risk_bg}; border-color:{risk_color}40;">
    <div class="gauge-container">
        <div class="risk-label" style="color:{risk_color};">Stall Probability</div>
        <div class="risk-value" style="color:{risk_color};">{pct:.1f}<span style="font-size:1.4rem; color:{risk_color}80;">%</span></div>
        <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{pct}%; background:{bar_grad};"></div>
        </div>
        <div class="status-pill" style="{pill_style}">{risk_level}</div>
    </div>
    <div class="action-box" style="border-left-color:{action_border};">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:{risk_color}; display:block; margin-bottom:0.3rem;">Recommended Action</span>
        {action_msg}
    </div>
</div>
""", unsafe_allow_html=True)

    # Active Feature Snapshot
    st.markdown('<div class="section-header" style="margin-top:1.2rem;">Active Input Snapshot</div>', unsafe_allow_html=True)

    sentiment_label = "Positive" if avg_sent > 0.1 else ("Negative" if avg_sent < -0.1 else "Neutral")
    trend_label = "↑ Improving" if sent_trend > 0.05 else ("↓ Declining" if sent_trend < -0.05 else "→ Flat")

    st.markdown(f"""
<div class="feature-grid">
    <div class="feature-tile">
        <div class="feature-tile-label">Emails</div>
        <div class="feature-tile-value">{email_count}</div>
    </div>
    <div class="feature-tile">
        <div class="feature-tile-label">Subj. Words</div>
        <div class="feature-tile-value">{subject_length}</div>
    </div>
    <div class="feature-tile">
        <div class="feature-tile-label">Priority</div>
        <div class="feature-tile-value">{PRIORITY_MAP[priority]}</div>
    </div>
    <div class="feature-tile">
        <div class="feature-tile-label">Avg Sentiment</div>
        <div class="feature-tile-value" style="font-size:0.85rem; padding-top:0.1rem;">
            {avg_sent:+.2f} <span style="font-size:0.65rem; color:#8a9bb5;">({sentiment_label})</span>
        </div>
    </div>
    <div class="feature-tile">
        <div class="feature-tile-label">Variance</div>
        <div class="feature-tile-value">{sent_var:.2f}</div>
    </div>
    <div class="feature-tile">
        <div class="feature-tile-label">Trend</div>
        <div class="feature-tile-value" style="font-size:0.8rem;">{trend_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Model performance chips with updated ISA III metrics
    st.markdown('<div class="section-header" style="margin-top:1.2rem;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="metrics-row">
    <div class="metric-chip">
        <div class="metric-chip-label">Recall</div>
        <div class="metric-chip-value">0.89</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Precision</div>
        <div class="metric-chip-value">0.79</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">PR-AUC</div>
        <div class="metric-chip-value">0.93</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Threshold</div>
        <div class="metric-chip-value">0.50</div>
    </div>
</div>
""", unsafe_allow_html=True)


# RIGHT COLUMN — SHAP Explainability
with col_right:
    st.markdown('<div class="section-header">Decision Explainability — SHAP Force Plot</div>', unsafe_allow_html=True)

    st.markdown("""
<div style="font-size:0.78rem; color:#8a9bb5; margin-bottom:1rem; line-height:1.6;">
    The force plot below shows which communication signals are pushing the stall risk score
    higher <span style="color:#ef4444;">▶ red</span> or lower
    <span style="color:#3b82f6;">◀ blue</span> for this specific ticket configuration.
    Each bar width represents the magnitude of that feature's contribution.
</div>
""", unsafe_allow_html=True)

    shap_vals = explainer.shap_values(input_df)

    def st_shap(plot, height=180):
            shap_html = f"""
            <html>
            <head>
                <style>
                    body {{ background: #111827 !important; margin: 0; padding: 8px; }}
                    /* Force all SHAP SVG text and lines to be visible in dark mode */
                    text {{ fill: #e8edf5 !important; font-family: 'IBM Plex Mono', monospace !important; }}
                    path {{ stroke: #4a5a72 !important; }}
                    line {{ stroke: #4a5a72 !important; }}
                    * {{ color: #e8edf5 !important; }}
                </style>
                {shap.getjs()}
            </head>
            <body>{plot.html()}</body>
            </html>
            """
            components.html(shap_html, height=height)
    st_shap(shap.force_plot(explainer.expected_value, shap_vals[0, :], input_df.iloc[0, :]), height=200)

    # SHAP feature contribution table
    st.markdown('<div class="section-header" style="margin-top:1.4rem;">Feature Contribution Breakdown</div>', unsafe_allow_html=True)

    shap_series = pd.Series(shap_vals[0], index=features).sort_values(key=abs, ascending=False)

    rows_html = ""
    for feat, val in shap_series.items():
        direction = "▲" if val > 0 else "▼"
        color = "#fca5a5" if val > 0 else "#6ee7b7"
        bar_width = min(abs(val) / (abs(shap_series).max() + 1e-9) * 100, 100)
        bar_color = "#ef444440" if val > 0 else "#10b98140"
        feat_val = input_df[feat].values[0]
        rows_html += f"""
<div style="display:flex; align-items:center; padding:0.5rem 0; border-bottom:1px solid #1e2d45; gap:0.8rem;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8a9bb5; width:180px; flex-shrink:0;">{feat}</div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#4a5a72; width:55px; flex-shrink:0; text-align:right;">{feat_val:.2f}</div>
    <div style="flex:1; background:#0a0e1a; border-radius:3px; height:6px; overflow:hidden;">
        <div style="width:{bar_width}%; height:100%; background:{bar_color}; border-radius:3px;"></div>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:{color}; width:70px; text-align:right; flex-shrink:0;">
        {direction} {abs(val):.3f}
    </div>
</div>
"""

    final_table_html = f"""
<div class="card" style="padding:0.8rem 1.2rem;">
    <div style="display:flex; gap:0.8rem; padding-bottom:0.5rem; border-bottom:1px solid #1e2d45; font-family:'IBM Plex Mono',monospace; font-size:0.62rem; letter-spacing:0.1em; color:#4a5a72; text-transform:uppercase;">
        <div style="width:180px;">Feature</div>
        <div style="width:55px; text-align:right;">Value</div>
        <div style="flex:1;">Impact</div>
        <div style="width:70px; text-align:right;">SHAP Δ</div>
    </div>
    {rows_html}
    <div style="padding-top:0.6rem; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#4a5a72;">
        ▲ Increases stall probability &nbsp;·&nbsp; ▼ Decreases stall probability
    </div>
</div>
"""
    st.markdown(final_table_html.replace('\n', ''), unsafe_allow_html=True)


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <div class="footer-left">
        Socio-Technical Health Monitor · ISA III · MScIDS Sem VI<br>
        Goa Business School, Goa University · Guide: Dr. Swapnil Fadte<br>
        Rudresh Achari · Unnat Umarye · Sarvadhnya Patil · Samuel Bhandari · Harsh Palyekar
    </div>
    <div class="footer-right">
        Model: XGBoost (Honest Features) · 6 Pure Communication Signals<br>
        Training: 1,000 Human Records · Apache Hadoop 2023–2024<br>
        Threshold: 0.50 · Validated by PR-AUC Curve
    </div>
</div>
""", unsafe_allow_html=True)