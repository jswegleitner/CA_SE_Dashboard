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

# Fixed status -> color map so a status keeps its color no matter which other
# statuses survive the current filters (color follows the entity, not its rank).
STATUS_COLORS = {
    'Active': SUCCESS,
    'Cancelled': NEUTRAL,
    'Delinquent': ACCENT,
    'Retired': PRIMARY,
    'Deceased': PURPLE,
    'Revoked': DANGER,
    'Voluntary Surrender Of License': "#DB2777",
}
_STATUS_FALLBACK = ["#0891B2", MUTED, INK]


def status_color(status) -> str:
    """Stable color for a license status; unknown statuses get a deterministic fallback."""
    key = str(status).strip()
    if key in STATUS_COLORS:
        return STATUS_COLORS[key]
    for name, color in STATUS_COLORS.items():
        if name.lower() == key.lower():
            return color
    return _STATUS_FALLBACK[sum(map(ord, key)) % len(_STATUS_FALLBACK)]


# Fixed category -> color map for licensure-history events (History tab lanes and
# cards, Time-tab event lines). Checked with the dataviz palette validator in this
# lane order: every adjacent pair clears the color-vision-deficiency and
# normal-vision separation floors, which the darker PRIMARY/SUCCESS pair did not.
HISTORY_CATEGORY_COLORS = {
    'Licensure': "#2563EB",
    'Exam': ACCENT,
    'Building code': "#0891B2",
    'Legislation': PURPLE,
    'Earthquake or disaster': DANGER,
}


def history_category_color(category) -> str:
    return HISTORY_CATEGORY_COLORS.get(str(category).strip(), NEUTRAL)


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
            # Thin guide line from the hovered point down to the axis.
            showspikes=True,
            spikemode='across',
            spikesnap='data',
            spikethickness=1,
            spikedash='dot',
            spikecolor='#94A3B8',
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
        # Light, compact tooltips that read as part of the page rather than a
        # dark slab dropped on top of the data. namelength=-1 stops trace
        # names being truncated to 15 chars with an ellipsis.
        hoverlabel=dict(
            font=dict(family=FONT_FAMILY, size=12, color=INK),
            bgcolor='rgba(255,255,255,0.96)',
            bordercolor=BORDER,
            align='left',
            namelength=-1,
        ),
        hoverdistance=30,
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
/* Wrap long labels/values/deltas instead of truncating them with an ellipsis
   when four cards share a narrow row. */
[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
[data-testid="stMetricDelta"] {
    height: auto !important;
}
[data-testid="stMetricValue"] {
    font-size: clamp(1.25rem, 2.2vw, 1.85rem) !important;
}
/* Cards in one row share the tallest card's height. */
[data-testid="stColumn"] > div:has(> div > [data-testid="stMetric"]),
[data-testid="stColumn"] [data-testid="stMetric"] {
    height: 100%;
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
section[data-testid="stSidebar"] h1 {
    font-size: 1.75rem;
    color: #1E293B;
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 1rem;
    color: #1E293B;
    margin-top: 0.5rem;
}
section[data-testid="stSidebar"] [data-testid="stHeader"] {
    color: #1E293B;
}


/* Filter section labels — roughly 2× the default sidebar body text */
.sidebar-filter-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1E293B;
    margin: 0.6rem 0 0.15rem 0;
    line-height: 1.3;
}

/* Expander headers in sidebar */
section[data-testid="stSidebar"] .streamlit-expanderHeader,
section[data-testid="stSidebar"] summary {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
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
    font-size: 1.6rem !important;
}
/* Streamlit renders the visible label inside a child div — target it directly */
.stTabs [data-baseweb="tab"] > div,
.stTabs [data-baseweb="tab"] p,
button[data-baseweb="tab"] > div,
button[data-baseweb="tab"] p {
    font-size: 1.6rem !important;
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

/* ---------- History tab ---------- */
.hist-era {
    margin: 1.9rem 0 0.75rem 0;
}
.hist-era h3 {
    margin: 0 0 0.2rem 0;
    padding: 0;
    font-size: 1.25rem;
}
.hist-era-blurb {
    color: #64748B;
    font-size: 0.95rem;
    margin: 0;
    max-width: 80ch;
}
.hist-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #94A3B8;
    border-radius: 10px;
    padding: 14px 18px 12px 18px;
    margin: 0 0 12px 0;
}
.hist-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    align-items: center;
    margin-bottom: 4px;
}
.hist-date {
    font-size: 0.8rem;
    font-weight: 600;
    color: #475569;
    font-variant-numeric: tabular-nums;
}
.hist-cat {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.hist-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}
.hist-title {
    font-size: 1.08rem;
    font-weight: 600;
    color: #1E293B;
    line-height: 1.35;
    margin: 0 0 6px 0;
}
.hist-card p.hist-text {
    font-size: 0.93rem;
    color: #334155;
    line-height: 1.55;
    margin: 0 0 8px 0;
    max-width: 82ch;
}
.hist-label {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748B;
    margin-right: 8px;
}
.hist-card p.hist-data {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px 12px;
}
.hist-card p.hist-src {
    font-size: 0.78rem;
    color: #64748B;
    line-height: 1.5;
    margin: 4px 0 0 0;
}
.hist-src a {
    color: #1F4E79;
    text-decoration: none;
    border-bottom: 1px solid #BFDBFE;
}
.hist-src a:hover {
    border-bottom-color: #1F4E79;
}

/* ---------- Mobile ---------- */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] > div,
    .stTabs [data-baseweb="tab"] p,
    button[data-baseweb="tab"] > div,
    button[data-baseweb="tab"] p {
        font-size: 1rem !important;
    }
}
</style>
"""
