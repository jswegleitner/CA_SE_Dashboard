# -*- coding: utf-8 -*-
"""Theme: palette, Plotly template, and CSS for the SE Dashboard.

Single source of truth for visual constants so charts feel cohesive.
Call ``register_plotly_template()`` once at app start to activate it.
"""
import plotly.graph_objects as go
import plotly.io as pio

# ---------- Palette ----------
PRIMARY = "#1F4E79"      # deep slate blue — engineering blue (main bars/lines)
ACCENT = "#D97706"       # amber — comparison/secondary
SUCCESS = "#047857"      # deep emerald — active counts
DANGER = "#B91C1C"       # deep red — expirations / events
PURPLE = "#6D28D9"       # plum — avg-age line
NEUTRAL = "#475569"      # slate gray
MUTED = "#94A3B8"        # lighter slate
INK = "#1E293B"          # primary text
SOFT_BG = "#F8FAFC"      # card background
BORDER = "#E2E8F0"       # card / axis lines
GRID = "#EEF2F6"         # plot gridlines

# Categorical sequence for multi-category charts
CATEGORICAL_SEQUENCE = [
    PRIMARY, ACCENT, SUCCESS, DANGER, PURPLE, "#0891B2", NEUTRAL, "#DB2777",
]

# Sequential colorscales matched to palette
BLUE_SCALE = [
    [0.0, "#F0F7FE"],
    [0.25, "#BFDBFE"],
    [0.5, "#60A5FA"],
    [0.75, "#2563EB"],
    [1.0, "#1E3A8A"],
]
RED_SCALE = [
    [0.0, "#FEF2F2"],
    [0.25, "#FECACA"],
    [0.5, "#F87171"],
    [0.75, "#DC2626"],
    [1.0, "#7F1D1D"],
]

FONT_FAMILY = 'Inter, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'


def register_plotly_template():
    """Register and activate the unified 'se_dashboard' Plotly template."""
    template = go.layout.Template()
    template.layout = dict(
        font=dict(family=FONT_FAMILY, color=INK, size=13),
        title=dict(
            font=dict(size=15, color=INK, family=FONT_FAMILY),
            x=0,
            xanchor='left',
            pad=dict(l=8, t=8, b=8),
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        colorway=CATEGORICAL_SEQUENCE,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=BORDER,
            linewidth=1,
            ticks='outside',
            tickcolor=BORDER,
            tickfont=dict(color=NEUTRAL, size=12),
            title=dict(font=dict(color=NEUTRAL, size=12)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks='outside',
            tickcolor=BORDER,
            tickfont=dict(color=NEUTRAL, size=12),
            title=dict(font=dict(color=NEUTRAL, size=12)),
        ),
        legend=dict(
            font=dict(color=INK, size=12),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor=BORDER,
            borderwidth=1,
        ),
        margin=dict(l=56, r=24, t=56, b=48),
        hoverlabel=dict(
            font=dict(family=FONT_FAMILY, size=12, color='white'),
            bgcolor=INK,
            bordercolor=INK,
            align='left',
        ),
        geo=dict(
            bgcolor='white',
            lakecolor='white',
            landcolor='#F8FAFC',
            subunitcolor=BORDER,
        ),
    )
    pio.templates['se_dashboard'] = template
    pio.templates.default = 'se_dashboard'


# ---------- Streamlit CSS ----------
DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---------- Typography ---------- */
html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

h1 {
    font-weight: 700;
    color: #1E293B;
    letter-spacing: -0.02em;
    margin-bottom: 0.15rem;
}
h2, h3, h4 {
    font-weight: 600;
    color: #1E293B;
    letter-spacing: -0.01em;
}

.dashboard-subtitle {
    color: #64748B;
    font-size: 0.95rem;
    margin: 0 0 1.25rem 0;
}

/* ---------- Container spacing ---------- */
.block-container {
    padding-top: 2.25rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
}

/* Smooth out the default <hr/> if any slip through */
hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.5rem 0;
}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #1F4E79;
    transform: translateY(-1px);
}
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #1E293B !important;
    font-size: 1.85rem !important;
    font-weight: 600;
    line-height: 1.1;
}

/* ---------- Filter chip bar ---------- */
.filter-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin: 0.25rem 0 1.5rem 0;
    padding: 12px 14px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}
.filter-chip {
    display: inline-flex;
    align-items: center;
    background: #EFF6FF;
    color: #1E40AF;
    border: 1px solid #BFDBFE;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    line-height: 1.4;
}
.filter-chip-key {
    color: #64748B;
    margin-right: 6px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.7rem;
}
.filter-chip-empty {
    color: #64748B;
    font-size: 0.82rem;
    font-style: italic;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 1rem;
    color: #1E293B;
    margin-top: 0.5rem;
}
section[data-testid="stSidebar"] [data-testid="stHeader"] {
    color: #1E293B;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    border-radius: 8px 8px 0 0;
    color: #64748B;
    font-weight: 500;
    font-size: 0.95rem;
}
.stTabs [aria-selected="true"] {
    color: #1F4E79 !important;
    background-color: #F0F7FE;
    font-weight: 600;
}

/* ---------- Buttons ---------- */
.stButton button {
    border-radius: 8px;
    border: 1px solid #CBD5E1;
    color: #1E293B;
    font-weight: 500;
    background-color: white;
    transition: all 0.15s ease;
}
.stButton button:hover {
    border-color: #1F4E79;
    color: #1F4E79;
    background-color: #F0F7FE;
}

/* ---------- Expanders ---------- */
.streamlit-expanderHeader {
    font-weight: 500;
    color: #475569;
}

/* ---------- Subheaders inside tabs ---------- */
.stTabs h2, .stTabs h3 {
    margin-top: 1rem;
}
</style>
"""
