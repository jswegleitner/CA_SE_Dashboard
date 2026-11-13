# -*- coding: utf-8 -*-
"""
Structural Engineers Dashboard - Streamlit Version
Loads pre-processed CSV data and creates interactive dashboard

Created on Fri Sep 12 19:59:58 2025
@author: jwegleitner

Usage:
    streamlit run SE_Dashboard_v7.py
    streamlit run SE_Dashboard_v7.py --server.headless true
"""
from pathlib import Path
from datetime import datetime
import subprocess
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import html
from typing import Optional

from dashboard_lib.timeline import (
    load_timeline_events,
    format_event_description,
    build_event_description_from_fields,
    render_event_descriptions,
    _sanitize_hover,
    _wrap_words,
    _wrap_field,
)
from dashboard_lib.periods import (
    _assign_period,
    _build_full_index,
    _count_by_period,
    to_bucket_label,
)
from dashboard_lib.geo import (
    US_STATE_TO_ABBREV,
    CA_COUNTY_FIPS,
    to_state_code,
    to_county_fips,
)
from dashboard_lib.stats import (
    PROVISIONAL_DEPARTURE_YEARS,
    SENIOR_TENURE_YEARS,
    TENURE_BIN_EDGES,
    issued_per_year,
    departed_per_year,
    recent_vs_baseline,
    trend_streak,
    active_on,
    years_licensed,
    tenure_histogram,
    active_counts_by_year,
    retention_by_decade,
    license_survival,
    survival_at,
    median_survival,
)
from dashboard_lib.history import (
    HISTORY_CATEGORIES,
    ERAS,
    history_path,
    read_history_csv,
    format_event_date,
    plot_position,
    era_index,
    era_label,
    lane_offsets,
    fractional_year,
    license_number_milestones,
    early_record_coverage,
    event_card_html,
)
from dashboard_lib.exam_results import (
    EXAM_ERAS,
    STRANDS,
    STRAND_LABELS,
    exam_results_path,
    read_exam_results_csv,
    era_bounds,
    sittings as exam_sittings,
    licenses_after_sittings,
    era_rate_summary,
)
from dashboard_lib.theme import (
    register_plotly_template,
    DASHBOARD_CSS,
    HISTORY_CATEGORY_COLORS,
    history_category_color,
    EXAM_STRAND_SYMBOLS,
    exam_strand_color,
    GRID,
    INK as INK_TEXT,
    PRIMARY,
    ACCENT,
    SUCCESS,
    DANGER,
    PURPLE,
    NEUTRAL,
    BORDER,
    BLUE_SCALE,
    RED_SCALE,
    CATEGORICAL_SEQUENCE,
    MUTED,
    status_color,
)

# Activate the unified Plotly template once at module load
register_plotly_template()

# ---------- Visual constants ----------
US_MAP_ZMAX = 300
EVENT_TYPE_PALETTE = CATEGORICAL_SEQUENCE
MAIN_BAR_COLOR = PRIMARY
COMPARISON_BAR_COLOR = ACCENT
EXPIRATION_BAR_COLOR = DANGER
ACTIVE_LINE_COLOR = SUCCESS
STATE_BAR_COLOR = PRIMARY
EVENT_MARKER_COLOR = DANGER
NET_LINE_COLOR = ACCENT

# When a categorical axis has more than this many items, hide outside text labels
# on bars to reduce visual noise.
BAR_LABEL_MAX_CATEGORIES = 25

# Non-state US jurisdictions shown in the "US Territories" bubble on the US map
_US_TERRITORIES = {'PR', 'GU', 'VI', 'AS', 'MP', 'AP', 'AA', 'AE'}


# Plotly config shared by every chart.
# - scrollZoom False: the maps (geo traces) otherwise grab the mouse wheel, so
#   scrolling the page stalls whenever the pointer passes over one.
# - Modebar only on hover, without the select/lasso tools nobody uses here.
CHART_CONFIG = {
    'scrollZoom': False,
    'displaylogo': False,
    'displayModeBar': 'hover',
    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
}


def _show_chart(fig):
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


# ---------- Data loading ----------
def _file_hash(src):
    """Return file modification time as a cache-busting key, or None for URLs."""
    p = Path(src)
    if p.exists():
        return p.stat().st_mtime
    return None


@st.cache_data(ttl=3600)
def _read_csv_cached(src, file_mtime=None):
    # `file_mtime` must NOT start with an underscore: Streamlit leaves
    # underscore-prefixed arguments out of the cache key, so the old
    # `_file_mtime` never actually busted the cache. Exceptions propagate
    # (and are not cached), so a transient failure isn't pinned for an hour.
    return pd.read_csv(src)


def load_cleaned_data(src, _file_mtime=None):
    """
    Load pre-cleaned CSV data with caching for better performance.
    `src` can be a local path or a URL.
    `_file_mtime` busts the cache when the file changes on disk.
    """
    try:
        return _read_csv_cached(src, file_mtime=_file_mtime)
    except Exception as e:
        st.error(f"Error loading data from {src}: {str(e)}")
        return None


def parse_dates(df):
    """
    Centralize parsing of date-like columns so visuals and filters can rely on
    consistent datetime dtypes. Returns the dataframe (modified in place).
    """
    if df is None:
        return df

    def _parse(series):
        # The pipeline publishes ISO dates; parsing with an explicit format avoids
        # the slow per-element dateutil fallback. Re-parse without it only if the
        # column turns out not to be ISO (e.g., a hand-made uploaded CSV).
        parsed = pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')
        if parsed.isna().all() and series.notna().any():
            parsed = pd.to_datetime(series, errors='coerce')
        return parsed

    try:
        if 'Original Issue Date' in df.columns:
            df['Original Issue Date'] = _parse(df['Original Issue Date'])
        if 'Expiration Date' in df.columns:
            df['Expiration Date'] = _parse(df['Expiration Date'])
    except Exception:
        # Don't raise; invalid date parsing should not break the app.
        pass

    return df


def get_data_last_updated(src) -> Optional[str]:
    """Return a human-readable date for when the CSV was last refreshed.

    Prefers the file's last git commit date because Streamlit Cloud re-clones
    the repo on every deploy, which resets filesystem mtimes to the deploy
    time. Falls back to mtime when git isn't available (e.g., the source is
    a URL, the file isn't tracked, or git isn't on PATH).
    """
    if not src:
        return None
    try:
        src_path = Path(src)
        if not src_path.exists():
            return None

        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%aI", "--", src_path.name],
                cwd=str(src_path.parent),
                capture_output=True,
                text=True,
                timeout=5,
            )
            iso = result.stdout.strip()
            if result.returncode == 0 and iso:
                return datetime.fromisoformat(iso).strftime("%B %d, %Y")
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

        return datetime.fromtimestamp(src_path.stat().st_mtime).strftime("%B %d, %Y")
    except (OSError, ValueError):
        return None


def resolve_data_source(default_local_filename: str):
    """
    Resolves the data source path based on the environment.
    If running on Streamlit Cloud, use secrets.
    If running locally, use the local CSV file in the same folder as the script.
    """
    try:
        if "DATA_PATH" in st.secrets:
            return st.secrets["DATA_PATH"]
        elif "DATA_URL" in st.secrets:
            return st.secrets["DATA_URL"]
    except Exception:
        pass

    possible_files = [
        default_local_filename,
        default_local_filename + ".csv",
        "structural_engineers_cleaned.csv",
        "ProfEngrsLandSurvyrsGeologist_Data00.csv",
        "ProfEngrsLandSurvyrsGeologist_Data00_structural_engineers_cleaned.csv",
    ]

    current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    search_dirs = [current_dir / 'data', current_dir]

    for d in search_dirs:
        for filename in possible_files:
            local_csv_path = d / filename
            if local_csv_path.exists():
                return str(local_csv_path)

    for d in search_dirs:
        for csv_path in d.glob("*.csv"):
            if csv_path.name.lower() in ['timeline_events.csv', 'license history table.csv',
                                         'licensure_history.csv', 'timeline_events_extra.csv']:
                continue
            try:
                sample_df = pd.read_csv(csv_path, nrows=1)
                if 'License Type' in sample_df.columns or 'License Number' in sample_df.columns:
                    return str(csv_path)
            except Exception:
                continue

    return None


# ---------- URL query-param helpers ----------

def _init_filters_from_url(df):
    """On first page load, populate widget session state from URL query params.

    This lets users share a URL like ``?states=CA,NY&start_year=2010&end_year=2024``
    and have the dashboard open with those filters pre-applied.
    """
    params = st.query_params
    if not params:
        return

    if 'states' in params:
        all_states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
        if all_states and not any(f"state_filter__{s}" in st.session_state for s in all_states):
            url_states = [s.strip() for s in params['states'].split(',') if s.strip()]
            valid = set(all_states)
            selected = {s for s in url_states if s in valid}
            for s in all_states:
                st.session_state[f"state_filter__{s}"] = s in selected

    if 'statuses' in params:
        all_statuses = sorted(df['License Status'].dropna().unique().tolist()) if 'License Status' in df.columns else []
        if all_statuses and not any(f"status_filter__{s}" in st.session_state for s in all_statuses):
            url_statuses = [s.strip() for s in params['statuses'].split(',') if s.strip()]
            valid = set(all_statuses)
            selected = {s for s in url_statuses if s in valid}
            for s in all_statuses:
                st.session_state[f"status_filter__{s}"] = s in selected

    if 'start_year' in params and 'date_range_slider' not in st.session_state:
        try:
            sy = int(params['start_year'])
            ey = int(params.get('end_year', str(sy)))
            # Clamp to the data's year range — an out-of-bounds session-state value
            # makes st.slider raise (StreamlitValueBelowMinError) and break the page.
            if 'Original Issue Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Original Issue Date']):
                min_year = int(df['Original Issue Date'].dt.year.min())
                max_year = int(df['Original Issue Date'].dt.year.max())
                sy = min(max(sy, min_year), max_year)
                ey = min(max(ey, min_year), max_year)
                if sy > ey:
                    sy, ey = ey, sy
                st.session_state['date_range_slider'] = (sy, ey)
        except (ValueError, TypeError):
            pass

    if 'compare' in params and 'comparison_state' not in st.session_state:
        valid_states = set(df['State'].dropna().unique().tolist()) if 'State' in df.columns else set()
        if params['compare'] in valid_states:
            st.session_state['comparison_state'] = params['compare']


def _build_share_params(filters: dict) -> dict:
    """Build a dict of query-param key/values from the current filter state."""
    params = {}
    if filters.get('states'):
        params['states'] = ','.join(filters['states'])
    if filters.get('statuses'):
        params['statuses'] = ','.join(filters['statuses'])
    if filters.get('start_year') is not None:
        params['start_year'] = str(filters['start_year'])
    if filters.get('end_year') is not None:
        params['end_year'] = str(filters['end_year'])
    if filters.get('comparison_state') and filters['comparison_state'] != 'None':
        params['compare'] = filters['comparison_state']
    return params


# ---------- Filter chip summary ----------

def _render_filter_chips(filters: dict, df: pd.DataFrame, container=None):
    """Render a compact chip bar showing the currently-applied filters.

    Only shows chips that differ from the unfiltered default — keeps the bar
    quiet when nothing is filtered. Pass ``container=st.sidebar`` to render
    inside the sidebar instead of the main area.
    """
    if container is None:
        container = st
    all_states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    all_statuses = sorted(df['License Status'].dropna().unique().tolist()) if 'License Status' in df.columns else []

    if 'Original Issue Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Original Issue Date']):
        full_min = int(df['Original Issue Date'].dt.year.min())
        full_max = int(df['Original Issue Date'].dt.year.max())
    else:
        full_min, full_max = None, None

    chips = []

    # None = not filtered; [] = everything unticked (shows zero rows).
    selected_states = filters.get('states')
    if selected_states is not None and set(selected_states) != set(all_states):
        preview = ", ".join(selected_states[:3]) or "None"
        if len(selected_states) > 3:
            preview += f" +{len(selected_states) - 3} more"
        chips.append(("States", preview))

    selected_statuses = filters.get('statuses')
    if selected_statuses is not None and set(selected_statuses) != set(all_statuses):
        chips.append(("Status", ", ".join(selected_statuses) or "None"))

    sy, ey = filters.get('start_year'), filters.get('end_year')
    if sy is not None and ey is not None and (sy != full_min or ey != full_max):
        chips.append(("Years", f"{sy}–{ey}"))

    cmp_state = filters.get('comparison_state')
    if cmp_state and cmp_state != 'None':
        chips.append(("Compare", cmp_state))

    # Must not be named `html` — that shadows the stdlib module for this whole
    # function scope and makes the html.escape() call below raise
    # UnboundLocalError whenever at least one chip is present.
    if not chips:
        chips_html = '<div class="filter-chips"><span class="filter-chip-empty">No filters applied — viewing all records</span></div>'
    else:
        items = "".join(
            f'<span class="filter-chip"><span class="filter-chip-key">{key}</span>{html.escape(str(val))}</span>'
            for key, val in chips
        )
        chips_html = f'<div class="filter-chips">{items}</div>'

    container.markdown(chips_html, unsafe_allow_html=True)


# ---------- Filters & visuals (updated with Plotly) ----------

def _render_checkbox_filter(label: str, options: list, ss_prefix: str) -> list:
    """Excel-style checkbox filter inside a collapsible expander."""
    select_all_key = f"{ss_prefix}__all"

    for opt in options:
        opt_key = f"{ss_prefix}__{opt}"
        if opt_key not in st.session_state:
            st.session_state[opt_key] = True

    all_selected = all(st.session_state.get(f"{ss_prefix}__{opt}", True) for opt in options)
    n_selected = sum(1 for opt in options if st.session_state.get(f"{ss_prefix}__{opt}", True))
    st.session_state[select_all_key] = all_selected

    if all_selected:
        expander_label = f"{label}: All"
    elif n_selected == 0:
        expander_label = f"{label}: None"
    else:
        expander_label = f"{label}: {n_selected} of {len(options)}"

    def on_select_all():
        new_val = st.session_state[select_all_key]
        for opt in options:
            st.session_state[f"{ss_prefix}__{opt}"] = new_val

    with st.sidebar.expander(expander_label, expanded=False):
        st.checkbox("(Select All)", key=select_all_key, on_change=on_select_all)
        selected = []
        for opt in options:
            opt_key = f"{ss_prefix}__{opt}"
            if st.checkbox(opt, key=opt_key):
                selected.append(opt)

    return selected


def create_filters(df):
    filters = {}

    st.sidebar.title("Filters")

    events_df_present = load_timeline_events() is not None

    st.sidebar.subheader("Date Range")
    if 'Original Issue Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Original Issue Date']):
        min_year = int(df['Original Issue Date'].dt.year.min())
        max_year = int(df['Original Issue Date'].dt.year.max())
    else:
        min_year, max_year = 1900, pd.Timestamp.now().year

    filters['start_year'], filters['end_year'] = st.sidebar.slider(
        "Original Issue Date Range (Years)",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        key='date_range_slider',
    )

    filters['show_events'] = st.sidebar.checkbox(
        "Show timeline events",
        value=events_df_present,
        help="Overlay key licensure, exam, code and earthquake dates on the Time-tab charts. "
             "The History tab explains each one.",
        key='show_events_toggle',
    )

    st.sidebar.markdown("---")

    states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    st.sidebar.markdown('<p class="sidebar-filter-title">License Holder\'s State of Residence</p>', unsafe_allow_html=True)
    filters['states'] = _render_checkbox_filter("State", states, 'state_filter')

    filters['comparison_state'] = st.sidebar.selectbox(
        "Compare with State (overlay on bar chart):",
        options=['None'] + states,
        index=0,
        key='comparison_state',
        help="Select a single state to overlay its data on the time series bar chart",
    )

    statuses = sorted(df['License Status'].dropna().unique().tolist()) if 'License Status' in df.columns else []
    st.sidebar.markdown('<p class="sidebar-filter-title">Status of License</p>', unsafe_allow_html=True)
    filters['statuses'] = _render_checkbox_filter("License Status", statuses, 'status_filter')

    if not events_df_present:
        st.sidebar.caption("Add `data/licensure_history.csv` (rows with `on_chart` = 1) to overlay history events on the charts.")
    return filters


def apply_filters(df, filters):
    if df is None or df.empty:
        return df

    filtered_df = df.copy()

    # None means "don't filter on this". A list, even an empty one, is an
    # explicit selection: unticking every box must show zero rows, not all.
    if 'State' in filtered_df.columns and filters.get('states') is not None:
        filtered_df = filtered_df[filtered_df['State'].isin(filters['states'])]

    if 'License Status' in filtered_df.columns and filters.get('statuses') is not None:
        filtered_df = filtered_df[filtered_df['License Status'].isin(filters['statuses'])]

    if 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        mask = (
            (filtered_df['Original Issue Date'].dt.year >= filters['start_year']) &
            (filtered_df['Original Issue Date'].dt.year <= filters['end_year'])
        )
        filtered_df = filtered_df[mask]

    return filtered_df


_BUCKET_SINGULAR = {
    "Yearly": "Year",
    "Half-Yearly": "Half-Year",
    "Quarterly": "Quarter",
    "Monthly": "Month",
}


def _render_count_bubble(count: int, total: int, label: str):
    """Render a circle card summarising a sub-count of license holders."""
    pct = (count / total * 100) if total > 0 else 0.0
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:100%;min-height:260px;gap:10px;">
            <div style="
                width:140px;height:140px;border-radius:50%;
                background:rgba(31,78,121,0.10);
                border:2.5px solid #1F4E79;
                display:flex;flex-direction:column;
                align-items:center;justify-content:center;text-align:center;
                padding:12px;
            ">
                <span style="font-size:1.55rem;font-weight:700;color:#1F4E79;
                             line-height:1.1;">{count:,}</span>
                <span style="font-size:0.68rem;color:#64748B;margin-top:3px;">licenses</span>
            </div>
            <div style="font-size:0.82rem;font-weight:600;color:#475569;
                        text-align:center;line-height:1.45;">
                {label}<br>
                <span style="font-weight:400;color:#94A3B8;">({pct:.1f}% of total)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_visualizations(filtered_df, show_events: bool, start_year: Optional[int] = None, end_year: Optional[int] = None, comparison_state: str = None, full_df=None, status_agnostic_df=None, all_df=None):
    if filtered_df is None or filtered_df.empty:
        st.warning("No data to visualize with current filters.")
        return

    has_multi_state = 'State' in filtered_df.columns and filtered_df['State'].nunique() > 1
    has_ca_county = (
        'State' in filtered_df.columns
        and 'County' in filtered_df.columns
        and not filtered_df[filtered_df['State'] == 'CA'].empty
    )
    has_geo = has_multi_state or has_ca_county
    has_status = 'License Status' in filtered_df.columns and filtered_df['License Status'].nunique() > 1

    history_events, history_error = load_history()
    has_history = history_error is not None or (history_events is not None and not history_events.empty)

    tab_labels = ["Time"]
    if has_geo:
        tab_labels.append("Geography")
    if has_status:
        tab_labels.append("Status")
    if has_history:
        tab_labels.append("History")

    st.markdown(
        '<h3 style="text-decoration:underline;margin-bottom:0.25rem;">Category of Visuals</h3>',
        unsafe_allow_html=True,
    )
    tabs = st.tabs(tab_labels)

    # ---------- Time tab ----------
    with tabs[0]:
        col1, _ = st.columns([1, 3])
        with col1:
            bucket_size = st.selectbox("Time Grouping:", ["Yearly", "Half-Yearly", "Quarterly", "Monthly"])
            lock_y = st.checkbox(
                "Lock Y-axis when zooming",
                value=True,
                help="When checked, zooming/panning will only affect the X axis; Y axis will remain fixed.",
            )

        st.subheader(f"Licenses Issued by {_BUCKET_SINGULAR.get(bucket_size, bucket_size)}")
        events_df = load_timeline_events()
        plot_time_series(filtered_df, bucket_size, lock_y, events_df=events_df, show_events=show_events, start_year=start_year, end_year=end_year, comparison_state=comparison_state, full_df=full_df)
        active_df = status_agnostic_df if status_agnostic_df is not None else filtered_df
        plot_issued_vs_departed(active_df, start_year=start_year, end_year=end_year)
        plot_active_licenses_per_year(active_df, start_year=start_year, end_year=end_year)
        plot_active_per_capita(active_df, start_year=start_year, end_year=end_year)
        plot_avg_license_age_per_year(active_df, start_year=start_year, end_year=end_year)
        plot_years_licensed(active_df)
        plot_retention_by_decade(active_df)
        plot_license_survival(active_df)
        st.subheader("Licenses Issued Over Time")
        plot_time_series_line(filtered_df, lock_y, events_df=events_df, show_events=show_events, start_year=start_year, end_year=end_year)

    # ---------- Geography tab ----------
    if has_geo:
        with tabs[tab_labels.index("Geography")]:
            if has_ca_county:
                ca_data = filtered_df[filtered_df['State'] == 'CA']
                out_of_state_count = (
                    int((filtered_df['State'] != 'CA').sum())
                    if 'State' in filtered_df.columns else 0
                )
                st.subheader("Licenses by California County")
                county_view = st.radio(
                    "Show",
                    [PER_CAPITA_LABEL, TOTAL_LABEL],
                    horizontal=True,
                    key="county_map_view",
                    label_visibility="collapsed",
                )
                per_capita = county_view == PER_CAPITA_LABEL
                if out_of_state_count > 0:
                    col_map, col_bubble = st.columns([4, 1])
                    with col_map:
                        plot_ca_county_map(ca_data, per_capita=per_capita)
                    with col_bubble:
                        _render_count_bubble(out_of_state_count, len(filtered_df), "Out of State")
                else:
                    plot_ca_county_map(ca_data, per_capita=per_capita)
            if has_multi_state:
                territory_count = (
                    int(
                        filtered_df['State'].apply(
                            lambda s: pd.notna(s) and str(s).strip().upper() in _US_TERRITORIES
                        ).sum()
                    )
                    if 'State' in filtered_df.columns else 0
                )
                st.subheader("Licenses by State")
                col_map, col_bubble = st.columns([4, 1])
                with col_map:
                    plot_us_map(filtered_df)
                with col_bubble:
                    _render_count_bubble(territory_count, len(filtered_df), "US Territories")
                plot_state_counts(filtered_df)

    # ---------- Status tab ----------
    if has_status:
        with tabs[tab_labels.index("Status")]:
            plot_status_pie(filtered_df)

    # ---------- History tab ----------
    # Reference material: reads the whole dataset, not the sidebar-filtered one.
    if has_history:
        with tabs[tab_labels.index("History")]:
            render_history_tab(all_df if all_df is not None else filtered_df, history_events, history_error)


def _period_start(period) -> Optional[pd.Timestamp]:
    """Start timestamp of a bucket label (pandas Period or 'YYYY-H1'/'YYYY-H2' string)."""
    if hasattr(period, 'start_time'):
        return period.start_time
    text = str(period)
    if '-H' in text:
        year, half = text.split('-H')
        return pd.Timestamp(year=int(year), month=1 if half == '1' else 7, day=1)
    return None


def _drop_future_periods(counts: pd.Series) -> pd.Series:
    """Remove buckets that start after today (the year-range reindex pads out to Dec 31)."""
    now = pd.Timestamp.now()
    starts = [_period_start(p) for p in counts.index]
    return counts[[st_ is None or st_ <= now for st_ in starts]]


def _tint(hex_color: str, alpha: float) -> str:
    """rgba() version of a #RRGGBB color at the given opacity."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


PARTIAL_FILL_ALPHA = 0.35


def _partial_bar_marker(color: str, is_partial) -> dict:
    """Bar marker that shows partial/provisional bars as a light fill with a solid outline."""
    return dict(
        color=[_tint(color, PARTIAL_FILL_ALPHA) if p else color for p in is_partial],
        line=dict(color=color, width=[1.5 if p else 0 for p in is_partial]),
    )


def _bar_labels(values, is_partial) -> list:
    return [f"{int(v)} (YTD)" if p else f"{int(v)}" for v, p in zip(values, is_partial)]


def _streak_text(streak: int) -> str:
    if streak == 0:
        return "Flat"
    word = "rising" if streak > 0 else "falling"
    n = abs(streak)
    return f"{n} yr{'s' if n > 1 else ''} {word}"


def _anchor_year(end_year: Optional[int]) -> int:
    """Last complete year, capped at the slider's end year."""
    last_complete = pd.Timestamp.now().year - 1
    return min(int(end_year), last_complete) if end_year is not None else last_complete


def _recent_stat_item(series: pd.Series, anchor: int, start_year: Optional[int], noun: str,
                      recent: int = 5, delta_color: str = 'normal') -> Optional[dict]:
    """Stat card: average of the last `recent` complete years vs. up to 10 years before them.

    The baseline window is shortened so it never reaches before the slider's start year
    (years outside the filter would otherwise count as zeros).
    """
    first_recent = anchor - recent + 1
    if start_year is not None and first_recent < int(start_year):
        return None
    base_len = 10 if start_year is None else max(0, min(10, first_recent - int(start_year)))
    recent_avg, base_avg, pct = recent_vs_baseline(series, anchor, recent=recent, baseline=max(base_len, 1))
    delta = None
    if base_len and pct is not None:
        delta = f"{pct:+.0f}% vs. prior {base_len} yrs"
    help_text = f"Average {noun.lower()} per year over the last {recent} complete years"
    if base_len:
        help_text += f", compared with {base_avg:.0f}/yr over the {base_len} years before that"
    return dict(
        label=f"{noun}, {first_recent}–{anchor}",
        value=f"{recent_avg:.0f}/yr",
        delta=delta,
        delta_color=delta_color,
        help=help_text + ".",
    )


def _issuance_stat_items(filtered_df, start_year=None, end_year=None) -> list:
    """Recent-vs-baseline stat cards for the issuance chart (always computed per year)."""
    per_year = issued_per_year(filtered_df)
    if per_year.empty:
        return []
    anchor = _anchor_year(end_year)
    lo = int(start_year) if start_year is not None else int(per_year.index.min())
    if anchor <= lo:
        return []
    full = per_year.reindex(range(lo, anchor + 1), fill_value=0)

    items = []
    recent = _recent_stat_item(full, anchor, start_year, "Issued")
    if recent:
        items.append(recent)
    last, prev = int(full.loc[anchor]), int(full.loc[anchor - 1])
    items.append(dict(
        label=f"Issued in {anchor}",
        value=f"{last:,}",
        delta=_fmt_delta(last - prev, f"vs. {anchor - 1}"),
    ))
    items.append(dict(
        label="Year-over-Year Trend",
        value=_streak_text(trend_streak(full)),
        help=f"Consecutive years in which annual issuance moved in the same direction, ending {anchor}.",
    ))
    items.append(dict(
        label="Peak Year",
        value=f"{int(full.idxmax())} ({int(full.max()):,})",
        help="Year with the most licenses issued in the selected range.",
    ))
    return items


def plot_time_series(filtered_df, bucket_size: str, lock_y: bool = True, events_df: Optional[pd.DataFrame] = None, show_events: bool = False, start_year: Optional[int] = None, end_year: Optional[int] = None, comparison_state: str = None, full_df=None):
    if 'Original Issue Date' not in filtered_df.columns or not pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        st.info("Original Issue Date column not found or not datetime.")
        return

    license_counts = _count_by_period(filtered_df, bucket_size, start_year, end_year)
    license_counts = _drop_future_periods(license_counts)

    if license_counts.empty:
        st.info("No time-series data for the selected filters.")
        return

    show_labels = len(license_counts) <= BAR_LABEL_MAX_CATEGORIES
    current_label = to_bucket_label(pd.Timestamp.now(), bucket_size)
    is_partial = [str(p) == current_label for p in license_counts.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(p) for p in license_counts.index],
        y=license_counts.values,
        name='Main Data',
        marker=_partial_bar_marker(MAIN_BAR_COLOR, is_partial),
        text=_bar_labels(license_counts.values, is_partial) if show_labels else None,
        textposition='outside' if show_labels else 'none',
        customdata=['Partial period (to date)' if p else '' for p in is_partial],
        hovertemplate='<b>Period:</b> %{x}<br><b>Licenses:</b> %{y}<br>%{customdata}<extra></extra>',
    ))

    if comparison_state and comparison_state != 'None' and full_df is not None:
        comp_df = full_df[full_df['State'] == comparison_state].copy()
        if start_year is not None and end_year is not None:
            comp_df = comp_df[
                (comp_df['Original Issue Date'].dt.year >= start_year) &
                (comp_df['Original Issue Date'].dt.year <= end_year)
            ]

        comp_counts = _count_by_period(comp_df, bucket_size, start_year, end_year)
        comp_counts = comp_counts.reindex(license_counts.index, fill_value=0)

        fig.add_trace(go.Bar(
            x=[str(p) for p in comp_counts.index],
            y=comp_counts.values,
            name=f'{comparison_state} (Comparison)',
            marker=_partial_bar_marker(COMPARISON_BAR_COLOR, is_partial),
            text=_bar_labels(comp_counts.values, is_partial) if show_labels else None,
            textposition='outside' if show_labels else 'none',
            opacity=0.75,
            hovertemplate=f'<b>Period:</b> %{{x}}<br><b>{comparison_state} Licenses:</b> %{{y}}<extra></extra>',
        ))

    fig.update_layout(
        xaxis_title="Period",
        yaxis_title="Number of Licenses",
        showlegend=True if (comparison_state and comparison_state != 'None') else False,
        height=460,
        hovermode='closest',
        dragmode='zoom',
        barmode='group',
        bargap=0.18,
    )

    num_periods = len(license_counts)
    if num_periods > 20:
        fig.update_xaxes(tickangle=90)
    elif num_periods > 10:
        fig.update_xaxes(tickangle=45)

    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=lock_y)

    categories = [str(p) for p in license_counts.index]
    fig.update_xaxes(categoryorder='array', categoryarray=categories)

    if show_events and (events_df is not None) and (not events_df.empty):
        ev = events_df.copy()
        if start_year is not None and end_year is not None:
            ev = ev[(ev['start_date'].dt.year >= start_year) & (ev['start_date'].dt.year <= end_year)]
        ev['bucket'] = ev['start_date'].apply(lambda dt: to_bucket_label(dt, bucket_size))
        ev = ev.dropna(subset=['bucket'])
        ev = ev[ev['bucket'].isin(categories)]
        if not ev.empty:
            y_max = max(license_counts.values) if len(license_counts) else 0
            pad = max(1, y_max * 0.06)
            fig.update_yaxes(range=[-pad * 2.5, y_max * 1.05])

            # Two events can land in one bucket (the 1933 earthquake and the Field
            # Act, say). Stacked markers would hide all but one from the hover, so
            # draw one marker per bucket and list every event in its tooltip.
            ev = ev.sort_values('start_date')
            ev['__desc'] = render_event_descriptions(ev)
            ev['__when'] = _event_date_text(ev)
            hover_by_bucket, year_by_bucket = {}, {}
            for bucket, grp in ev.groupby('bucket', sort=False):
                hover_by_bucket[bucket] = '<br><br>'.join(
                    f"<b>{_sanitize_hover(r['label'])}</b><br>"
                    f"{_sanitize_hover(r['__when'])} · {_sanitize_hover(r['type'])}<br>"
                    f"{r['__desc']}"
                    for _, r in grp.iterrows()
                )
                year_by_bucket[bucket] = str(grp['start_date'].iloc[0].year)
            buckets = list(hover_by_bucket)

            fig.add_trace(go.Scatter(
                x=buckets,
                y=[-pad * 0.8] * len(buckets),
                mode='markers+text',
                name='Events',
                marker=dict(symbol='triangle-up', size=10, color=EVENT_MARKER_COLOR),
                text=[year_by_bucket[b] for b in buckets],
                textposition='bottom center',
                textfont=dict(color=EVENT_MARKER_COLOR),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=[hover_by_bucket[b] for b in buckets],
            ))

    _show_chart(fig)

    if any(is_partial):
        st.caption(
            f"The lightly shaded {current_label} bar is a partial period (to date) and will keep "
            "growing until the period ends, so it isn't directly comparable with the full bars."
        )

    items = _issuance_stat_items(filtered_df, start_year, end_year)
    if items:
        _render_stat_row(items)


def _event_date_text(ev: pd.DataFrame) -> pd.Series:
    """Display date per event: the history file's own wording when it has one
    ('1931', 'April 1988'), else the ISO date (legacy event files)."""
    iso = ev['start_date'].dt.strftime('%Y-%m-%d')
    if 'date_text' not in ev.columns:
        return iso
    given = ev['date_text'].fillna('').astype(str)
    return given.where(given != '', iso)


def _detect_issue_date_anomalies(df_line: pd.DataFrame, license_col: str, window: int = 51, min_diff_years: float = 5.0) -> pd.DataFrame:
    """Flag rows whose Original Issue Date is far later than their License Number suggests.

    License numbers are assigned sequentially, so the issue year of each license should
    track its neighbors in license-number order. When DCA backfills a reciprocity/reissue,
    it stamps the *reactivation* date into Original Issue Date — those rows show up as
    large deviations from the rolling median of nearby license numbers' issue years.
    """
    df = df_line.copy()
    df['__lic_num'] = pd.to_numeric(df[license_col], errors='coerce')
    df = df.dropna(subset=['__lic_num']).sort_values('__lic_num').reset_index(drop=True)
    if len(df) < 10:
        return df.iloc[0:0]
    df['__issue_year'] = df['Original Issue Date'].dt.year
    df['__expected'] = df['__issue_year'].rolling(window=window, center=True, min_periods=10).median()
    df['__diff'] = df['__issue_year'] - df['__expected']
    return df[df['__diff'] >= min_diff_years].copy()


def plot_time_series_line(filtered_df, lock_y: bool = True, events_df: Optional[pd.DataFrame] = None, show_events: bool = False, start_year: Optional[int] = None, end_year: Optional[int] = None):
    """
    Line chart of licenses over time. X axis is the Original Issue Date and Y is
    either the license number (when available) or a cumulative index.
    """
    if 'Original Issue Date' not in filtered_df.columns:
        st.info("Original Issue Date column not found.")
        return

    df_line = filtered_df.copy()
    df_line['Original Issue Date'] = pd.to_datetime(df_line['Original Issue Date'], errors='coerce')
    df_line = df_line.dropna(subset=['Original Issue Date']).sort_values('Original Issue Date')

    if df_line.empty:
        st.info("No time-series data for the selected filters.")
        return

    license_col = None
    for col in df_line.columns:
        if col.lower().replace('_', ' ').strip() in (
            'license number', 'license no', 'license no.', 'licensenumber'
        ):
            license_col = col
            break
    if license_col:
        yvals = pd.to_numeric(df_line[license_col], errors='coerce')
        if yvals.isna().all():
            st.warning(f"Detected license column `{license_col}`, but values are not numeric. Plotting cumulative index instead.")
            df_line = df_line.reset_index(drop=True)
            yvals = df_line.index + 1
            y_label = 'Cumulative Index'
        else:
            y_label = license_col
    else:
        st.warning("No license-number-like column found. Plotting cumulative index instead.")
        df_line = df_line.reset_index(drop=True)
        yvals = df_line.index + 1
        y_label = 'Cumulative Index'

    fig = go.Figure()
    # Scattergl (WebGL): ~7k points zoom and pan much more smoothly than SVG.
    fig.add_trace(go.Scattergl(
        x=df_line['Original Issue Date'],
        y=yvals,
        mode='lines+markers',
        name='Licenses',
        line=dict(color=MAIN_BAR_COLOR, width=2),
        marker=dict(size=4, color=MAIN_BAR_COLOR),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y}<extra></extra>',
    ))

    anomalies = _detect_issue_date_anomalies(df_line, license_col) if license_col else None
    if anomalies is not None and not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['Original Issue Date'],
            y=anomalies['__lic_num'],
            mode='markers',
            name='Likely reissue / reciprocity',
            marker=dict(
                size=11,
                color=ACCENT,
                symbol='diamond',
                line=dict(color='white', width=1.5),
            ),
            customdata=anomalies[['__expected', '__diff']].values,
            hovertemplate=(
                '<b>License #%{y}</b><br>'
                'Issue Date: %{x|%Y-%m-%d}<br>'
                'Expected era (from neighbors): ~%{customdata[0]:.0f}<br>'
                'Off by ~%{customdata[1]:.0f} years<br>'
                '<i>Likely a reciprocity or reissue entry — DCA appears to have '
                'stamped the reactivation date into the Original Issue Date field.</i>'
                '<extra></extra>'
            ),
            showlegend=True,
        ))

    fig.update_layout(
        xaxis_title="Original Issue Date",
        yaxis_title=y_label,
        height=400,
        hovermode='closest',
        dragmode='zoom',
    )

    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=lock_y)

    try:
        x_min = pd.to_datetime(f"{start_year}-01-01") if start_year is not None else df_line['Original Issue Date'].min()
        x_max = pd.to_datetime(f"{end_year}-12-31") if end_year is not None else df_line['Original Issue Date'].max()
        if pd.notna(x_min) and pd.notna(x_max):
            fig.update_xaxes(range=[x_min, x_max])
    except Exception:
        pass

    if show_events and (events_df is not None) and (not events_df.empty):
        type_styles = {}
        for i, t in enumerate(events_df['type'].astype(str).unique()):
            fallback = EVENT_TYPE_PALETTE[i % len(EVENT_TYPE_PALETTE)]
            type_styles[t] = dict(color=HISTORY_CATEGORY_COLORS.get(t, fallback), dash='dot')

        y_max = float(pd.Series(yvals).max()) if len(yvals) else 0.0
        y_min = float(pd.Series(yvals).min()) if len(yvals) else 0.0
        head = max(1.0, y_max * 0.08)
        fig.update_yaxes(range=[y_min - head * 0.5, y_max + head * 2.0])

        ev = events_df.copy()
        ev = ev.dropna(subset=['start_date'])
        if start_year is not None and end_year is not None:
            ev = ev[(ev['start_date'].dt.year >= start_year) & (ev['start_date'].dt.year <= end_year)]
        if not ev.empty:
            ev_desc = render_event_descriptions(ev)

            fig.add_trace(go.Scatter(
                x=ev['start_date'],
                y=[y_min - head * 0.3] * len(ev),
                mode='markers',
                name='Events',
                marker=dict(size=12, color='rgba(0,0,0,0)'),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    '%{customdata[3]} · %{customdata[1]}<br>'
                    '%{customdata[2]}<extra></extra>'
                ),
                customdata=pd.concat([
                    ev['label'].fillna(''),
                    ev['type'].astype(str),
                    ev_desc,
                    _event_date_text(ev),
                ], axis=1).values,
                showlegend=False,
            ))

            for _, r in ev.iterrows():
                t = str(r.get('type', 'Event'))
                color = type_styles.get(t, dict(color='gray'))['color']
                dash = type_styles.get(t, dict(dash='dot'))['dash']
                xdt = r['start_date']
                fig.add_vline(x=xdt, line=dict(color=color, dash=dash, width=1))
                try:
                    year_txt = str(pd.to_datetime(xdt).year)
                except Exception:
                    year_txt = ''
                if year_txt:
                    fig.add_annotation(x=xdt, y=y_min - head * 0.3, text=year_txt,
                                       showarrow=False, font=dict(color=EVENT_MARKER_COLOR, size=10))

    if anomalies is not None and not anomalies.empty:
        st.caption(
            f"Amber diamonds flag {len(anomalies)} license(s) whose Original Issue Date "
            "appears far later than their License Number would suggest — most likely "
            "reciprocity reinstatements or DCA database backfills where the reactivation "
            "date overwrote the original issuance. Hover a marker for details."
        )

    _show_chart(fig)


def plot_state_counts(filtered_df):
    st.subheader("Licenses by State")
    state_counts = filtered_df['State'].value_counts()
    show_labels = len(state_counts) <= 15

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=state_counts.index,
            y=state_counts.values,
            name='Licenses by State',
            marker_color=STATE_BAR_COLOR,
            text=state_counts.values if show_labels else None,
            textposition='outside' if show_labels else 'none',
            hovertemplate='<b>State:</b> %{x}<br><b>Licenses:</b> %{y}<extra></extra>',
        ))

        fig.update_layout(
            xaxis_title="State",
            yaxis_title="Number of Licenses",
            showlegend=False,
            height=460,
            xaxis_tickangle=45,
            bargap=0.18,
        )

        _show_chart(fig)

    with col2:
        st.markdown("**Top 5 States**")
        for i, (state, count) in enumerate(state_counts.head().items(), 1):
            percentage = (count / len(filtered_df)) * 100
            st.write(f"{i}. **{state}**: {count} ({percentage:.1f}%)")


def plot_status_pie(filtered_df):
    st.subheader("License Status Distribution")
    st.caption(
        "Definitions for each license status are available on the "
        "[BPELSG license lookup page](https://www.bpelsg.ca.gov/consumers/lic_lookup.shtml)."
    )
    c1, c2 = st.columns([1, 1])
    status_counts = filtered_df['License Status'].value_counts()
    total = int(status_counts.sum())

    with c1:
        # Suppress in-slice labels on tiny slivers — they'd render outside the slice
        # and overflow the column. Their info still appears in the right-column
        # breakdown and on hover.
        text_template = [
            f"{pct:.0%}" if pct >= 0.04 else ""
            for pct in (status_counts.values / total)
        ]

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.5,
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>',
            text=text_template,
            textinfo='text',
            textposition='inside',
            insidetextorientation='horizontal',
            textfont=dict(color='white', size=13),
            marker=dict(
                colors=[status_color(s) for s in status_counts.index],
                line=dict(color='white', width=2),
            ),
            sort=False,
        ))

        fig.update_layout(
            height=420,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(
                text=f"<b>{total:,}</b><br><span style='font-size:11px;color:#64748B'>licenses</span>",
                x=0.5, y=0.5, font=dict(size=20, color='#1E293B'),
                showarrow=False,
            )],
        )

        _show_chart(fig)

    with c2:
        st.markdown("**Status Breakdown**")
        for status, count in status_counts.items():
            percentage = (count / total) * 100
            color = status_color(status)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0;'>"
                f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
                f"background:{color};flex-shrink:0;'></span>"
                f"<span><b>{html.escape(str(status))}</b>: {count:,} ({percentage:.1f}%)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def plot_us_map(filtered_df):
    """Plot a choropleth map of the United States showing total licenses per state."""
    if 'State' not in filtered_df.columns:
        st.info("No `State` column available for US map.")
        return

    state_counts = filtered_df['State'].value_counts()

    all_state_codes = list(US_STATE_TO_ABBREV.values())
    codes = all_state_codes.copy()
    values = [0] * len(all_state_codes)
    hover_text = [f"{code}: 0" for code in all_state_codes]

    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    unmapped = []

    for state, count in state_counts.items():
        code = to_state_code(state)
        if code and code in code_to_idx:
            idx = code_to_idx[code]
            values[idx] = int(count)
            hover_text[idx] = f"{state}: {count}"
        elif code:
            codes.append(code)
            values.append(int(count))
            hover_text.append(f"{state}: {count}")
        else:
            unmapped.append(state)

    if not codes:
        st.info("No mappable US state values found for the map.")
        return

    fig = go.Figure(data=go.Choropleth(
        locations=codes,
        z=values,
        locationmode='USA-states',
        colorscale=BLUE_SCALE,
        zmin=0,
        zmax=US_MAP_ZMAX,
        text=hover_text,
        marker_line_color=BORDER,
        marker_line_width=0.5,
        colorbar=dict(title='Licenses', ticks='outside', outlinewidth=0, thickness=14),
    ))

    fig.update_layout(
        geo=dict(scope='usa', bgcolor='white', lakecolor='white'),
        height=560,
        margin=dict(l=0, r=0, t=20, b=0),
    )

    _show_chart(fig)

    if unmapped:
        st.caption(f"Unmapped state values (not shown on map): {', '.join(unmapped)}")


COUNTY_POPULATION_FILE = Path(__file__).resolve().parent / 'data' / 'ca_county_population.csv'
COUNTY_POPULATION_SOURCE = "U.S. Census Bureau, Vintage 2024 county population estimates"
PER_CAPITA_LABEL = "Per 100k residents"
TOTAL_LABEL = "Total licenses"


@st.cache_data(ttl=86400)
def _read_county_population(file_mtime=None) -> pd.Series:
    pop = pd.read_csv(COUNTY_POPULATION_FILE, dtype={'FIPS': str})
    return pop.set_index('FIPS')['Population']


def load_county_population(_file_mtime=None) -> Optional[pd.Series]:
    """Population by 5-digit county FIPS code, or None if the reference file is missing."""
    try:
        return _read_county_population(file_mtime=_file_mtime)
    except Exception:
        return None


CA_COUNTIES_GEOJSON_FILE = Path(__file__).resolve().parent / 'data' / 'ca_counties.geojson'
_US_COUNTIES_GEOJSON_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'


@st.cache_data
def _read_ca_counties_geojson(file_mtime=None) -> dict:
    import json
    with open(CA_COUNTIES_GEOJSON_FILE, encoding='utf-8') as f:
        return json.load(f)


def load_ca_counties_geojson():
    """California county shapes (58 features, ~70 KB) shipped in data/.

    Falls back to Plotly's full US-counties URL if the local file is missing,
    which makes the browser fetch ~3 MB from GitHub on every view.
    """
    try:
        return _read_ca_counties_geojson(file_mtime=_file_hash(CA_COUNTIES_GEOJSON_FILE))
    except Exception:
        return _US_COUNTIES_GEOJSON_URL


def plot_ca_county_map(filtered_df, per_capita: bool = False):
    """Plot a choropleth map of California counties: total licenses, or licenses per 100k residents."""
    if 'County' not in filtered_df.columns:
        st.info("No `County` column available for California county map.")
        return

    population = load_county_population(_file_mtime=_file_hash(COUNTY_POPULATION_FILE)) if per_capita else None
    if per_capita and population is None:
        st.caption("County population data not found; showing total licenses instead.")
        per_capita = False

    county_counts = filtered_df['County'].value_counts()
    counts_by_fips = {}
    names_by_fips = {code: name for name, code in CA_COUNTY_FIPS.items()}
    unmapped = []
    for county, count in county_counts.items():
        code = to_county_fips(county)
        if code:
            counts_by_fips[code] = counts_by_fips.get(code, 0) + int(count)
            names_by_fips.setdefault(code, str(county))
        else:
            unmapped.append(county)

    all_fips = list(CA_COUNTY_FIPS.values())
    fips = all_fips + [c for c in counts_by_fips if c not in all_fips]
    counts = [counts_by_fips.get(code, 0) for code in fips]

    if per_capita:
        pops = [population.get(code) for code in fips]
        values = [(c / p * 100_000) if p else None for c, p in zip(counts, pops)]
        hover_text = [
            f"<b>{names_by_fips.get(code, code)}</b><br>{v:.1f} per 100k residents<br>"
            f"{c:,} licenses · pop. {p:,.0f}" if v is not None
            else f"<b>{names_by_fips.get(code, code)}</b><br>{c:,} licenses"
            for code, c, p, v in zip(fips, counts, pops, values)
        ]
        colorbar_title = 'Per 100k'
    else:
        values = counts
        hover_text = [f"<b>{names_by_fips.get(code, code)}</b><br>{c:,} licenses" for code, c in zip(fips, counts)]
        colorbar_title = 'Licenses'

    finite = [v for v in values if v is not None]
    if not finite:
        st.info("No mappable California county values found for the map.")
        return

    fig = go.Figure(data=go.Choropleth(
        locations=fips,
        z=values,
        locationmode='geojson-id',
        geojson=load_ca_counties_geojson(),
        colorscale=RED_SCALE,
        zmin=0,
        zmax=max(finite) or 1,
        text=hover_text,
        hoverinfo='text',
        marker_line_color=BORDER,
        marker_line_width=0.5,
        colorbar=dict(title=colorbar_title, ticks='outside', outlinewidth=0, thickness=14),
    ))

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor='white',
    )

    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=20, b=0),
    )

    _show_chart(fig)

    if per_capita:
        st.caption(
            "Licenses per 100,000 residents, so large counties don't dominate simply because "
            f"more people live there. Population: {COUNTY_POPULATION_SOURCE}. Counts follow the "
            "sidebar filters — select only Active in License Status for today's coverage."
        )
    if unmapped:
        st.caption(f"Unmapped county values (not shown on map): {', '.join(map(str, unmapped))}")


# ---------- Issued vs. departed (inflow / outflow / net) ----------

def plot_issued_vs_departed(filtered_df, start_year=None, end_year=None):
    """Combined flow chart: issued bars up, departures down, net change as a line."""
    st.subheader("Licenses Issued vs. Leaving the Rolls by Year")
    st.caption(
        "Bars above zero are licenses issued; bars below zero are licenses that left the rolls "
        "(status no longer Active — Cancelled, Retired, Deceased, Delinquent, Revoked, Voluntary "
        "Surrender — counted by the year their Expiration Date falls in). The line is the net "
        "change. Ignores the License Status sidebar filter so both sides stay comparable. The "
        "current year is excluded while its renewal cycle is in progress."
    )

    has_issue = 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date'])
    has_exp = 'Expiration Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Expiration Date'])
    if not has_issue or not has_exp or 'License Status' not in filtered_df.columns:
        st.info("Original Issue Date, Expiration Date, and License Status are needed for this chart.")
        return

    anchor = _anchor_year(end_year)
    lo = int(start_year) if start_year is not None else int(filtered_df['Original Issue Date'].dt.year.min())
    if lo > anchor:
        st.info("No complete years in the selected range.")
        return

    years = pd.Index(range(lo, anchor + 1))
    issued = issued_per_year(filtered_df).reindex(years, fill_value=0)
    left = departed_per_year(filtered_df).reindex(years, fill_value=0)
    net = issued - left

    provisional_from = pd.Timestamp.now().year - PROVISIONAL_DEPARTURE_YEARS
    is_prov = [y >= provisional_from for y in years]
    show_labels = len(years) <= BAR_LABEL_MAX_CATEGORIES

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=issued.values,
        name='Issued', marker_color=MAIN_BAR_COLOR,
        text=issued.values if show_labels else None,
        textposition='outside' if show_labels else 'none',
        hovertemplate='Issued: %{y:,}<extra></extra>',
    ))
    # Settled and provisional departures are separate traces so the legend explains the shading.
    prov_mask = pd.Series(is_prov, index=years)
    for name, mask, provisional in [
        ('Left the rolls', ~prov_mask, False),
        ('Left the rolls (provisional)', prov_mask, True),
    ]:
        sub = left[mask.values]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub.index, y=-sub.values,
            name=name,
            marker=_partial_bar_marker(EXPIRATION_BAR_COLOR, [provisional] * len(sub)),
            text=sub.values if show_labels else None,
            textposition='outside' if show_labels else 'none',
            customdata=sub.values,
            hovertemplate=f'{name}: %{{customdata:,}}<extra></extra>',
        ))
    fig.add_trace(go.Scatter(
        x=years, y=net.values,
        name='Net change', mode='lines+markers',
        line=dict(color=NET_LINE_COLOR, width=2),
        marker=dict(size=4, color=NET_LINE_COLOR),
        hovertemplate='Net: %{y:+,}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Licenses (issued ↑ / departed ↓)",
        height=480,
        hovermode='x unified',
        barmode='relative',
        bargap=0.18,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    fig.update_yaxes(zeroline=True, zerolinecolor=MUTED, zerolinewidth=1.5)

    _show_chart(fig)

    if any(is_prov):
        st.caption(
            f"**Departures for {provisional_from}–{anchor} are provisional** (lightly shaded bars). Some "
            "licenses counted as leaving in the most recent years may still renew or be "
            "reclassified, so these bars — and the net line — can change in later updates."
        )

    items = []
    for series, noun, color in [(issued, "Issued", 'normal'), (left, "Departed", 'inverse')]:
        item = _recent_stat_item(series, anchor, start_year, noun, delta_color=color)
        if item:
            items.append(item)
    # Net can sit near zero or flip sign, so compare it in licenses/yr rather than percent.
    first_recent = anchor - 4
    if first_recent >= lo:
        recent_net = float(net.loc[first_recent:anchor].mean())
        base = net.loc[max(lo, first_recent - 10):first_recent - 1]
        items.append(dict(
            label=f"Net Change, {first_recent}–{anchor}",
            value=f"{recent_net:+.0f}/yr",
            delta=_fmt_delta(recent_net - float(base.mean()), f"vs. prior {len(base)} yrs", fmt="{:+,.0f}/yr") if len(base) else None,
            help="Average net change per year over the last 5 complete years, compared with "
                 "the years before that.",
        ))
    items.append(dict(
        label=f"Cumulative Net, {lo}–{anchor}",
        value=f"{int(net.sum()):+,}",
        help="Total issued minus total departed over the selected years.",
    ))
    _render_stat_row(items)


# ---------- Active licenses per year ----------

def plot_active_licenses_per_year(filtered_df, start_year=None, end_year=None):
    """Area chart showing how many licenses were active at the end of each year."""
    st.subheader("Active Licenses per Year")
    st.caption(
        "Counts licenses that were active during each year (issued on or before Dec 31, "
        "expiration on or after Jan 1). Ignores the License Status sidebar filter so the "
        "historical view stays consistent.  \n"
        "Active licenses is not an accurate measure of active practitioners. A licenses holder "
        "only has to pay the license renewal fee every 2 years to maintain an active license."
    )

    has_issue = 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date'])
    has_exp = 'Expiration Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Expiration Date'])
    if not has_issue or not has_exp:
        st.info("Both Original Issue Date and Expiration Date are needed for this chart.")
        return

    df = filtered_df.dropna(subset=['Original Issue Date', 'Expiration Date']).copy()
    if df.empty:
        st.info("No records with both issue and expiration dates.")
        return

    min_year = int(df['Original Issue Date'].dt.year.min()) if start_year is None else int(start_year)
    max_year = int(pd.Timestamp.now().year) if end_year is None else min(int(end_year), pd.Timestamp.now().year)

    years = list(range(min_year, max_year + 1))
    active_counts = []
    for y in years:
        yr_start = pd.Timestamp(f"{y}-01-01")
        yr_end = pd.Timestamp(f"{y}-12-31")
        count = int(((df['Original Issue Date'] <= yr_end) & (df['Expiration Date'] >= yr_start)).sum())
        active_counts.append(count)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=active_counts,
        mode='lines+markers', fill='tozeroy',
        name='Active Licenses',
        line=dict(color=ACTIVE_LINE_COLOR, width=2.5),
        marker=dict(size=5, color=ACTIVE_LINE_COLOR),
        fillcolor='rgba(4, 120, 87, 0.12)',
        hovertemplate='<b>%{x}</b><br>Active: %{y:,}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Active Licenses",
        height=440,
        hovermode='x',
    )

    _show_chart(fig)

    by_year = dict(zip(years, active_counts))
    latest_year = years[-1]
    peak_year = max(by_year, key=by_year.get)
    items = [dict(
        label=f"Active in {latest_year}",
        value=f"{by_year[latest_year]:,}",
        delta=_fmt_delta(by_year[latest_year] - by_year[latest_year - 1], f"vs. {latest_year - 1}")
        if latest_year - 1 in by_year else None,
    )]
    for span in (10, 25):
        then = latest_year - span
        if then in by_year and by_year[then]:
            pct = (by_year[latest_year] - by_year[then]) / by_year[then] * 100
            items.append(dict(
                label=f"Change Since {then}",
                value=f"{pct:+.0f}%",
                help=f"{by_year[then]:,} licenses were active in {then}.",
            ))
    items.append(dict(
        label="Peak Year",
        value=f"{peak_year} ({by_year[peak_year]:,})",
        help="Year with the most licenses in force in the selected range.",
    ))
    _render_stat_row(items)


# ---------- Active licenses per capita ----------

CA_POPULATION_FILE = Path(__file__).resolve().parent / 'data' / 'ca_population_annual.csv'
CA_POPULATION_SOURCE = "U.S. Census Bureau via FRED (series CAPOP)"


@st.cache_data(ttl=86400)
def _read_ca_population(file_mtime=None) -> pd.Series:
    pop = pd.read_csv(CA_POPULATION_FILE)
    return pop.set_index('Year')['Population']


def load_ca_population() -> Optional[pd.Series]:
    """California resident population by year, or None if the file is missing."""
    try:
        return _read_ca_population(file_mtime=_file_hash(CA_POPULATION_FILE))
    except Exception:
        return None


def plot_active_per_capita(filtered_df, start_year=None, end_year=None):
    """Active California-resident licenses per 100k Californians, by year."""
    if 'State' not in filtered_df.columns:
        return
    ca = filtered_df[filtered_df['State'] == 'CA']
    population = load_ca_population()
    if ca.empty or population is None:
        return
    if not (pd.api.types.is_datetime64_any_dtype(ca['Original Issue Date'])
            and pd.api.types.is_datetime64_any_dtype(ca['Expiration Date'])):
        return

    st.subheader("Active California Licenses per 100,000 Californians")
    st.caption(
        "Licenses held by California residents and active during each year, divided by the "
        f"state's population ({CA_POPULATION_SOURCE}). Shows whether the supply of structural "
        "engineers is keeping pace with the population. Same active-license rule and the same "
        "status-filter exception as the chart above. Population figures stop at the latest "
        "published year."
    )

    first = int(ca['Original Issue Date'].dt.year.min()) if start_year is None else int(start_year)
    last = int(pd.Timestamp.now().year) if end_year is None else min(int(end_year), pd.Timestamp.now().year)
    years = [y for y in range(first, last + 1) if y in population.index]
    if not years:
        st.info("No population data for the selected years.")
        return

    active = active_counts_by_year(ca, years)
    rate = active / population.reindex(years).astype(float) * 100_000

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=rate.round(2).tolist(),
        mode='lines+markers',
        name='Per 100k residents',
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(size=5, color=PRIMARY),
        customdata=list(zip(active.tolist(), population.reindex(years).tolist())),
        hovertemplate=(
            '<b>%{x}</b><br>%{y:.1f} per 100k residents<br>'
            'Active CA licenses: %{customdata[0]:,}<br>Population: %{customdata[1]:,.0f}<extra></extra>'
        ),
    ))
    fig.update_layout(xaxis_title="Year", yaxis_title="Active licenses per 100k", height=420,
                      hovermode='x', yaxis=dict(rangemode='tozero'))
    _show_chart(fig)

    latest = years[-1]
    items = [dict(label=f"Per 100k in {latest}", value=f"{rate[latest]:.1f}",
                  help=f"{int(active[latest]):,} active CA licenses, population "
                       f"{population[latest]:,.0f}.")]
    for span in (10, 25):
        then = latest - span
        if then in rate.index and rate[then]:
            items.append(dict(label=f"Change Since {then}",
                              value=f"{(rate[latest] - rate[then]) / rate[then] * 100:+.0f}%",
                              help=f"{rate[then]:.1f} per 100k in {then}."))
    peak = int(rate.idxmax())
    items.append(dict(label="Peak Year", value=f"{peak} ({rate[peak]:.1f})"))
    _render_stat_row(items)


# ---------- Cohort retention & survival ----------

def plot_retention_by_decade(filtered_df):
    """Share of each issuance decade's licenses that are still Active today."""
    if 'License Status' not in filtered_df.columns or not pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        return
    table = retention_by_decade(filtered_df)
    if table.empty:
        return

    st.subheader("Share of Each Decade's Licenses Still Active Today")
    st.caption(
        "For licenses first issued in each decade: the percentage with status Active today. "
        "Follows the State and date-range filters but ignores the License Status filter. "
        "DCA's public data may no longer list some long-gone licenses, which would make the "
        "oldest decades look better than they really were."
    )

    fig = go.Figure(go.Bar(
        x=table['label'], y=table['pct_active'].round(1),
        marker_color=SUCCESS,
        text=[f"{p:.0f}%" for p in table['pct_active']],
        textposition='outside',
        customdata=table[['active', 'issued']].values,
        hovertemplate='<b>%{x}</b><br>%{y:.1f}% still active<br>'
                      '%{customdata[0]:,} of %{customdata[1]:,} licenses<extra></extra>',
    ))
    fig.update_layout(xaxis_title="Decade first issued", yaxis_title="% still Active",
                      yaxis=dict(range=[0, 110], ticksuffix='%'), height=420, showlegend=False)
    _show_chart(fig)

    items = []
    for _, row in table[table['decade'].isin([1980, 1990, 2000])].iterrows():
        items.append(dict(label=f"{row['label']} Still Active", value=f"{row['pct_active']:.0f}%",
                          help=f"{int(row['active']):,} of {int(row['issued']):,} licenses."))
    if items:
        _render_stat_row(items)


def plot_license_survival(filtered_df):
    """Kaplan-Meier curve: chance a license is still on the rolls N years after issue."""
    if 'License Status' not in filtered_df.columns or not pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        return
    curve = license_survival(filtered_df, pd.Timestamp.now())
    if len(curve) < 2:
        return

    st.subheader("How Long Licenses Stay on the Rolls")
    st.caption(
        "Estimated chance that a license is still on the rolls N years after it was first "
        "issued (a Kaplan-Meier survival curve). A license leaves the rolls when its status is "
        "anything but Active, as of its last expiration date. Licenses that are still Active "
        "count only for the years they have existed so far, so recent licenses don't skew the "
        "curve. The same caveat about long-gone licenses applies, so the far right of the "
        "curve is the least certain part."
    )

    fig = go.Figure(go.Scatter(
        x=curve.index, y=(curve * 100).round(2),
        mode='lines', line=dict(color=PRIMARY, width=2.5, shape='hv'),
        fill='tozeroy', fillcolor='rgba(31, 78, 121, 0.10)',
        name='Still on the rolls',
        hovertemplate='%{x:.1f} years after issue<br>%{y:.1f}% still on the rolls<extra></extra>',
    ))
    fig.add_hline(y=50, line=dict(color=MUTED, dash='dot', width=1))
    fig.update_layout(xaxis_title="Years since first issued", yaxis_title="% still on the rolls",
                      yaxis=dict(range=[0, 102], ticksuffix='%'), height=420, showlegend=False)
    _show_chart(fig)

    items = []
    med = median_survival(curve)
    if med is not None:
        items.append(dict(label="Median Time on the Rolls", value=f"{med:.0f} yrs",
                          help="Half of licenses are expected to leave the rolls by this point."))
    for t in (20, 30, 40):
        s = survival_at(curve, t)
        if s is not None:
            items.append(dict(label=f"Still On After {t} Years", value=f"{s:.0%}"))
    if items:
        _render_stat_row(items)


# ---------- Average license age per year ----------

def plot_avg_license_age_per_year(filtered_df, start_year=None, end_year=None):
    """Line chart of the average age of active licenses at the end of each year."""
    st.subheader("Average Age of Active Licenses by Year")
    st.caption(
        "Average issue-to-year-end age of licenses active during each year. Ignores the "
        "License Status sidebar filter so the historical view stays consistent."
    )

    has_issue = 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date'])
    has_exp = 'Expiration Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Expiration Date'])
    if not has_issue or not has_exp:
        st.info("Both Original Issue Date and Expiration Date are needed for this chart.")
        return

    df = filtered_df.dropna(subset=['Original Issue Date', 'Expiration Date']).copy()
    if df.empty:
        st.info("No records with both issue and expiration dates.")
        return

    min_year = int(df['Original Issue Date'].dt.year.min()) if start_year is None else int(start_year)
    max_year = int(pd.Timestamp.now().year) if end_year is None else min(int(end_year), pd.Timestamp.now().year)

    years = list(range(min_year, max_year + 1))
    avg_ages = []
    active_counts = []
    for y in years:
        yr_start = pd.Timestamp(f"{y}-01-01")
        yr_end = pd.Timestamp(f"{y}-12-31")
        active = df[(df['Original Issue Date'] <= yr_end) & (df['Expiration Date'] >= yr_start)]
        if active.empty:
            avg_ages.append(None)
            active_counts.append(0)
        else:
            ages_years = (yr_end - active['Original Issue Date']).dt.days / 365.25
            avg_ages.append(float(ages_years.mean()))
            active_counts.append(int(len(active)))

    if not any(a is not None for a in avg_ages):
        st.info("No active licenses in the selected year range.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=avg_ages,
        mode='lines+markers',
        name='Average License Age',
        line=dict(color=PURPLE, width=2.5),
        marker=dict(size=5, color=PURPLE),
        customdata=active_counts,
        hovertemplate='<b>%{x}</b><br>Avg Age: %{y:.1f} yrs<br>Active Licenses: %{customdata:,}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Age (years)",
        height=440,
        hovermode='x',
    )

    _show_chart(fig)

    valid = {y: a for y, a in zip(years, avg_ages) if a is not None}
    if valid:
        latest_year = max(valid)
        latest = valid[latest_year]
        oldest_year = max(valid, key=valid.get)
        items = [dict(
            label=f"Average Age in {latest_year}",
            value=f"{latest:.1f} yrs",
            delta="Highest on record" if oldest_year == latest_year else None,
            delta_color='off',
        )]
        for span in (10, 25):
            then = latest_year - span
            if then in valid:
                items.append(dict(
                    label=f"Change Since {then}",
                    value=f"{latest - valid[then]:+.1f} yrs",
                    help=f"Average age was {valid[then]:.1f} years in {then}.",
                ))
        if oldest_year != latest_year:
            items.append(dict(
                label="Oldest Year",
                value=f"{oldest_year} ({valid[oldest_year]:.1f} yrs)",
                help="Year with the highest average license age in the selected range.",
            ))
        _render_stat_row(items)


# ---------- Years licensed (tenure) of today's active licensees ----------

def plot_years_licensed(filtered_df):
    """Histogram of how long today's Active licensees have held their license."""
    st.subheader("How Long Today's Active Licensees Have Been Licensed")
    st.caption(
        f"Years since Original Issue Date for every license with status Active today, in "
        f"5-year groups. Groups of {SENIOR_TENURE_YEARS}+ years (amber) are the licensees most "
        "likely to retire over the next decade. This is time since licensure, not age. Follows "
        "the State and date-range filters but always shows Active licenses, regardless of the "
        "License Status filter. A few reinstated licenses carry their reinstatement date as the "
        "issue date (see the last chart on this tab), so they appear newer than they are."
    )

    has_issue = 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date'])
    if not has_issue or 'License Status' not in filtered_df.columns:
        st.info("Original Issue Date and License Status are needed for this chart.")
        return

    tenure = years_licensed(filtered_df, pd.Timestamp.now())
    if tenure.empty:
        st.info("No Active licenses in the current selection.")
        return

    hist = tenure_histogram(tenure)
    edges = TENURE_BIN_EDGES + [None]
    senior = [lo >= SENIOR_TENURE_YEARS for lo in edges[:-1]]
    total = int(hist.sum())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(hist.index), y=hist.values,
        marker_color=[ACCENT if s_ else PRIMARY for s_ in senior],
        text=hist.values,
        textposition='outside',
        customdata=[f"{v / total:.1%}" for v in hist.values],
        hovertemplate='<b>%{x} years</b><br>Licensees: %{y:,}<br>Share: %{customdata}<extra></extra>',
    ))
    fig.update_layout(
        xaxis_title="Years licensed",
        yaxis_title="Active licensees",
        height=440,
        showlegend=False,
        bargap=0.12,
    )
    _show_chart(fig)

    n_senior = int((tenure >= SENIOR_TENURE_YEARS).sum())
    n_new = int((tenure < 10).sum())
    _render_stat_row([
        dict(label=f"Licensed {SENIOR_TENURE_YEARS}+ Years", value=f"{n_senior:,} ({n_senior / total:.0%})"),
        dict(label="Licensed Under 10 Years", value=f"{n_new:,} ({n_new / total:.0%})"),
        dict(label="Median Years Licensed", value=f"{tenure.median():.1f} yrs"),
    ])


def _get_current_filters_from_session(df) -> dict:
    """Read the current filter state from session state without rendering any widgets.

    Called before create_filters() so sidebar metrics can be computed on the
    same run that the user changed a filter, rather than one run behind.
    Defaults to all-included when a key is absent (first load).
    """
    states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    statuses = sorted(df['License Status'].dropna().unique().tolist()) if 'License Status' in df.columns else []

    selected_states = [s for s in states if st.session_state.get(f"state_filter__{s}", True)]
    selected_statuses = [s for s in statuses if st.session_state.get(f"status_filter__{s}", True)]

    date_range = st.session_state.get('date_range_slider')
    if date_range:
        start_year, end_year = int(date_range[0]), int(date_range[1])
    elif 'Original Issue Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Original Issue Date']):
        start_year = int(df['Original Issue Date'].dt.year.min())
        end_year = int(df['Original Issue Date'].dt.year.max())
    else:
        start_year, end_year = 1900, pd.Timestamp.now().year

    return {
        'states': selected_states if states else None,
        'statuses': selected_statuses if statuses else None,
        'start_year': start_year,
        'end_year': end_year,
        'comparison_state': st.session_state.get('comparison_state', 'None'),
        'show_events': st.session_state.get('show_events_toggle', True),
    }


def _display_sidebar_metrics(filtered_df):
    """Inline label: value rows for the sidebar Summary Data section."""
    if filtered_df is None or filtered_df.empty:
        st.sidebar.caption("No data loaded.")
        return

    records = f"{len(filtered_df):,}"
    states = str(filtered_df['State'].nunique() if 'State' in filtered_df.columns else 0)

    if 'License Status' in filtered_df.columns:
        active = f"{int((filtered_df['License Status'].str.strip().str.lower() == 'active').sum()):,}"
    else:
        active = "N/A"

    if 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        yr_min = int(filtered_df['Original Issue Date'].dt.year.min())
        yr_max = int(filtered_df['Original Issue Date'].dt.year.max())
        years = f"{yr_min}–{yr_max}"
    else:
        years = "N/A"

    rows = [("Records", records), ("States*", states), ("Active Licenses", active), ("Year Range", years)]
    cells = "".join(
        f'<tr>'
        f'<td style="padding:3px 0;color:#64748B;font-size:0.88rem;">{lbl}</td>'
        f'<td style="padding:3px 0;text-align:right;font-weight:600;color:#1E293B;font-size:0.88rem;">{val}</td>'
        f'</tr>'
        for lbl, val in rows
    )
    # Not named `html`: that would shadow the stdlib module for this scope and
    # break any html.escape() added here later (see _render_filter_chips).
    table_html = (
        f'<table style="width:100%;border-collapse:collapse;">{cells}</table>'
        f'<p style="font-size:0.65rem;color:#94A3B8;margin-top:5px;">'
        f'*Includes U.S. territories, Armed Forces jurisdictions, and international license holders.</p>'
    )
    st.sidebar.markdown(table_html, unsafe_allow_html=True)


def _fmt_delta(diff: Optional[float], suffix: str, fmt: str = "{:+,.0f}") -> Optional[str]:
    """Format a st.metric delta string; Streamlit colors it by the leading sign."""
    if diff is None:
        return None
    if round(diff) == 0:
        return f"0 {suffix}"
    return f"{fmt.format(diff)} {suffix}"


def _render_stat_row(items):
    """Render a row of st.metric cards.

    `items` is a list of dicts with keys label, value, and optional delta, help,
    delta_color. Shown directly under a chart (not in an expander) so the
    takeaway is visible without a click.
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                item['label'],
                item['value'],
                delta=item.get('delta'),
                delta_color=item.get('delta_color', 'normal'),
                help=item.get('help'),
            )


def _display_headline_stats(df):
    """Four headline metrics under the title — the story before the first scroll.

    Always computed from the full, unfiltered dataset so the numbers match the
    public record regardless of sidebar filters. Each number carries a
    comparison so the reader can tell whether it is good, bad, or normal.
    """
    has_status = 'License Status' in df.columns
    has_issue = 'Original Issue Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Original Issue Date'])
    has_exp = 'Expiration Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Expiration Date'])

    now = pd.Timestamp.now().normalize()
    year_ago = now - pd.DateOffset(years=1)
    last_complete_year = now.year - 1

    items = []

    # 1. Active licenses, vs. a year ago
    n_active = int((df['License Status'].str.strip().str.lower() == 'active').sum()) if has_status else None
    active_delta = None
    if has_issue and has_exp:
        active_delta = _fmt_delta(active_on(df, now) - active_on(df, year_ago), "vs. a year ago")
    items.append(dict(
        label="Active SE Licenses",
        value=f"{n_active:,}" if n_active is not None else "N/A",
        delta=active_delta,
        help="Licenses with status Active today. The change compares licenses in force today "
             "with licenses in force one year ago, based on issue and expiration dates.",
    ))

    # 2. Issued in the last 12 months, vs. the 10-year average
    if has_issue:
        issued_12m = int(((df['Original Issue Date'] > year_ago) & (df['Original Issue Date'] <= now)).sum())
        per_year = issued_per_year(df)
        avg_10 = float(per_year.reindex(range(last_complete_year - 9, last_complete_year + 1), fill_value=0).mean())
        items.append(dict(
            label="Issued in Last 12 Months",
            value=f"{issued_12m:,}",
            delta=_fmt_delta(issued_12m - avg_10, f"vs. 10-yr avg ({avg_10:.0f}/yr)"),
            help=f"New SE licenses issued between {year_ago:%b %d, %Y} and today, compared with "
                 f"the average per year over {last_complete_year - 9}–{last_complete_year}.",
        ))

    # 3. Net change in the last complete year, vs. the 10-year average
    if has_status and has_issue and has_exp:
        issued = issued_per_year(df)
        departed = departed_per_year(df)
        years = range(last_complete_year - 10, last_complete_year + 1)
        net = issued.reindex(years, fill_value=0) - departed.reindex(years, fill_value=0)
        net_ly = int(net.loc[last_complete_year])
        avg_prior = float(net.loc[last_complete_year - 10:last_complete_year - 1].mean())
        items.append(dict(
            label=f"Net Change in {last_complete_year}",
            value=f"{net_ly:+,}",
            delta=_fmt_delta(net_ly - avg_prior, f"vs. 10-yr avg ({avg_prior:+.0f}/yr)"),
            help="Licenses issued minus licenses leaving the rolls in the last complete year. "
                 "Departures for recent years are provisional and may shrink as lapsed "
                 "licenses renew. See the Issued vs. Leaving chart on the Time tab.",
        ))

    # 4. Share of active licensees licensed 30+ years
    if has_status and has_issue:
        tenure = years_licensed(df, now)
        if len(tenure):
            senior = int((tenure >= SENIOR_TENURE_YEARS).sum())
            items.append(dict(
                label=f"Active, Licensed {SENIOR_TENURE_YEARS}+ Years",
                value=f"{senior / len(tenure):.0%}",
                help=f"{senior:,} of {len(tenure):,} active licensees received their SE license "
                     f"{SENIOR_TENURE_YEARS} or more years ago — the group most likely to retire "
                     "in the coming decade. See the Years Licensed chart on the Time tab.",
            ))

    _render_stat_row(items)


# ---------- App ----------
# ---------- History tab ----------
# Direct labels on neighbouring events alternate above/below their marker when
# the events are closer together than this many years.
HISTORY_LABEL_GAP_YEARS = 14
HISTORY_FILTER_KEY = 'history_category_filter'
_MS_PER_YEAR = 365.25 * 24 * 3600 * 1000

HISTORY_INTRO = """
In California, Structural Engineer is not a separate license. It is an authority to use a
title, granted to someone who already holds a California Civil Engineer license, has at least
three further years of structural work in responsible charge, and has passed a separate
structural examination. State law reserves two kinds of buildings for structural engineers:
public schools and hospitals. A civil engineer may design the structure of anything else.

That arrangement dates from 1931 and has changed little since. What has changed is everything
around it: the exam, the building code, and the earthquakes that tested both. This page sets
those changes against the number of licenses issued each year.
"""

HISTORY_VOICES = """
Recollections from two engineers who held the license, taken from the Earthquake Engineering
Research Institute's oral history series.

- **Henry Degenkolb** began practice in 1936 and was interviewed between 1984 and 1986. He sat
  sixteen hours of open-book civil exam and sixteen more of structural. When California was
  small, he recalled, references from engineers who knew the candidate counted for as much as
  the exam. By the 1980s he had declined to vouch for people from his own office who were
  licensed anyway.
- Degenkolb argued that an engineer "shouldn't really get your structural license until you've
  chased an earthquake." Reading damage reports, he said, was no substitute for seeing it.
- On enforcement he was blunt. In November 1985 he described California's license as reputedly
  the toughest in the country, but said the Board rarely disciplined engineers whose designs
  had failed, and recounted a case against one firm that dragged on for years before settling.
- **Nicholas Forell** came to San Francisco from firms where, around 1950, engineers saw no
  need for a license at all. He earned his structural license in 1958 and only then joined the
  structural engineers association, having wrongly assumed that membership required it.
"""

HISTORY_ABOUT = """
Compiled in November 2026 from the Board's sunset review reports, legislative background
papers, the national examining council's published history, SEAOC and SEAONC histories, state
statutes, Robert Olson's study of the Field Act, STRUCTURE magazine, and two EERI oral
histories. Each entry lists its sources. Where a statement rests on inference from the license
numbers rather than on a document, the entry says so.

The events are kept in `data/licensure_history.csv`. Corrections and additions are welcome.
"""


@st.cache_data
def _load_history_cached(file_mtime=None):
    """(events, error message). Keyed on the CSV's mtime so hand edits show up on rerun."""
    try:
        return read_history_csv(), None
    except ValueError as exc:
        return None, str(exc)


def load_history():
    path = history_path()
    return _load_history_cached(file_mtime=path.stat().st_mtime if path.exists() else None)


def _hover_lines(text: str, width: int = 62) -> str:
    """Wrap prose for a Plotly tooltip, which does not wrap on its own."""
    lines = _wrap_words(str(text).split(), width=width, first_prefix='', subsequent_prefix='')
    return '<br>'.join(_sanitize_hover(line) for line in lines)


def _history_label_positions(xs, labels, offsets) -> list:
    """Text position per marker.

    A marker nudged off its lane's centerline has a neighbour on the other side,
    so its label goes on the outside (below a lowered marker, above a raised one).
    Centered markers alternate above and below when labelled neighbours are closer
    than HISTORY_LABEL_GAP_YEARS.
    """
    positions = []
    last_year, last_pos = None, 'bottom center'
    for x, label, offset in zip(xs, labels, offsets):
        if not label:
            positions.append('top center')
            continue
        year = fractional_year(x)
        if offset < 0:
            pos = 'bottom center'
        elif offset > 0:
            pos = 'top center'
        else:
            crowded = last_year is not None and (year - last_year) < HISTORY_LABEL_GAP_YEARS
            pos = ('bottom center' if last_pos == 'top center' else 'top center') if crowded else 'top center'
        positions.append(pos)
        last_year, last_pos = year, pos
    return positions


def plot_history_timeline(events: pd.DataFrame, df: Optional[pd.DataFrame]):
    """Event lanes above licenses issued per year, on one shared time axis.

    Two stacked panels rather than one chart with two y-scales: the lanes carry
    identity (what kind of event) and the bars carry magnitude (licenses issued).
    Each lane is named on its axis, so the lane colors are a second cue, not the
    only one.
    """
    from plotly.subplots import make_subplots

    lanes = [c for c in HISTORY_CATEGORIES if (events['category'] == c).any()]
    if not lanes:
        st.info("Pick at least one kind of event to draw the timeline.")
        return

    has_counts = (
        df is not None and 'Original Issue Date' in df.columns
        and pd.api.types.is_datetime64_any_dtype(df['Original Issue Date'])
    )
    now = pd.Timestamp.now()

    if has_counts:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                            row_heights=[0.6, 0.4])
    else:
        fig = make_subplots(rows=1, cols=1)

    lane_y = {c: len(lanes) - 1 - i for i, c in enumerate(lanes)}  # first category on top
    guide_xs = []
    for category in lanes:
        grp = events[events['category'] == category].sort_values('date')
        xs = [plot_position(d, prec) for d, prec in zip(grp['date'], grp['precision'])]
        offsets = lane_offsets([fractional_year(x) for x in xs])
        labels = grp['chart_label'].tolist()
        guide_xs += [x for x, label in zip(xs, labels) if label]
        fig.add_trace(go.Scatter(
            x=xs,
            y=[lane_y[category] + o for o in offsets],
            mode='markers+text',
            name=category,
            # White ring keeps neighbouring markers readable where they touch.
            marker=dict(size=11, color=history_category_color(category), line=dict(color='white', width=2)),
            text=labels,
            textposition=_history_label_positions(xs, labels, offsets),
            textfont=dict(size=11, color=NEUTRAL),
            cliponaxis=False,
            customdata=[
                [_sanitize_hover(title), format_event_date(d, prec), category, _hover_lines(note or summary)]
                for title, d, prec, note, summary in zip(
                    grp['title'], grp['date'], grp['precision'], grp['chart_note'], grp['summary'])
            ],
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                '%{customdata[1]} · %{customdata[2]}<br>'
                '%{customdata[3]}<extra></extra>'
            ),
            showlegend=False,
        ), row=1, col=1)

    fig.update_yaxes(
        tickmode='array',
        tickvals=[lane_y[c] for c in lanes],
        ticktext=lanes,
        range=[-0.7, len(lanes) - 0.3],
        ticks='',
        tickfont=dict(size=12, color=INK_TEXT),
        fixedrange=True,
        row=1, col=1,
    )

    if has_counts:
        counts = issued_per_year(df)
        counts = counts[counts.index <= now.year]
        years = [int(y) for y in counts.index]
        is_partial = [y == now.year for y in years]
        fig.add_trace(go.Bar(
            x=[pd.Timestamp(year=y, month=7, day=1) for y in years],
            y=counts.values,
            width=[0.82 * _MS_PER_YEAR] * len(years),
            name='Licenses issued',
            marker=_partial_bar_marker(MAIN_BAR_COLOR, is_partial),
            customdata=[[y, ' (year to date)' if part else ''] for y, part in zip(years, is_partial)],
            hovertemplate='<b>%{customdata[0]}</b><br>%{y} licenses issued%{customdata[1]}<extra></extra>',
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text='Licenses issued', fixedrange=True, row=2, col=1)

    # Faint guides from each labelled event down through the bars, so an event can
    # be read against the licensing level of its year.
    for x in guide_xs:
        fig.add_shape(type='line', x0=x, x1=x, xref='x', yref='paper', y0=0, y1=1,
                      line=dict(color=BORDER, width=1), layer='below')

    first_year = min(int(events['year'].min()), 1931) - 3
    fig.update_xaxes(
        range=[pd.Timestamp(year=first_year, month=1, day=1), pd.Timestamp(year=now.year + 3, month=1, day=1)],
        tick0='1900-01-01', dtick='M120', tickformat='%Y',
    )
    fig.update_layout(
        height=580 if has_counts else 340,
        hovermode='closest',
        dragmode='zoom',
        bargap=0,
        margin=dict(l=24, r=24, t=28, b=40),
    )
    _show_chart(fig)


def _history_stat_items(df: Optional[pd.DataFrame]) -> list:
    """Stat cards read off the license numbers: how fast the title has been granted."""
    milestones = license_number_milestones(df, step=1000) if df is not None else []
    if len(milestones) < 3:
        return []
    by_number = dict(milestones)
    first_number, first_date = milestones[0]
    top_number = int(pd.to_numeric(df['License Number'], errors='coerce').max())
    thousands = sorted(n for n in by_number if n % 1000 == 0)

    items = [
        dict(label="First title issued", value=f"{first_date.strftime('%b')} {first_date.day}, {first_date.year}",
             help=f"Issue date of license No. {first_number}, the lowest number in the data."),
        dict(label="Titles numbered to date", value=f"{top_number:,}",
             help="Highest license number in the data. Numbers are issued in sequence, so this is "
                  "close to the count of structural titles ever granted, including those long lapsed."),
    ]
    if 1000 in by_number:
        years = (by_number[1000] - first_date).days / 365.25
        items.append(dict(
            label="First 1,000 titles took", value=f"{years:.0f} years",
            help=f"No. {first_number} ({first_date.year}) to No. 1,000 ({by_number[1000].year}).",
        ))
    if len(thousands) >= 2:
        prev_n, last_n = thousands[-2], thousands[-1]
        years = (by_number[last_n] - by_number[prev_n]).days / 365.25
        items.append(dict(
            label="Latest 1,000 titles took", value=f"{years:.0f} years",
            help=f"No. {prev_n:,} ({by_number[prev_n].year}) to No. {last_n:,} ({by_number[last_n].year}).",
        ))
    return items


def _history_data_notes(df: Optional[pd.DataFrame]) -> str:
    """Markdown bullets on how to read the license counts against the history."""
    notes = []
    coverage = early_record_coverage(df, before_year=1940) if df is not None else None
    if coverage:
        present, numbered = coverage
        notes.append(
            f"- **The earliest years are undercounted.** Only {present:,} of roughly {numbered:,} licenses "
            "numbered before 1940 appear in the public data, because the state's feed drops licenses "
            "that lapsed long ago. License numbers are issued in sequence, so the numbering is the "
            "better guide to how many titles were granted in the 1930s and 1940s. From the 1950s on "
            "the records are nearly complete."
        )
    notes.append(
        "- **Licenses arrive in batches, and the batch month has moved.** For most of the license's "
        "history, nearly all of a year's structural licenses were issued within a few days of exam "
        "results. The main batch came in December through 1978 and in January to March from 1980, "
        "with second batches late in 1988, 1990 and 1991 as the exam calendar shifted. That is why "
        "1979, 1989 and 1992 look almost empty on the Time tab. The batch fell just outside the "
        "calendar year; licensing did not stop."
    )
    notes.append(
        "- **Two drops are real, and both follow an exam change.** Only 26 licenses were issued in "
        "February 1985, the first results after grading on a curve ended. New licenses then fell by "
        "about half in 2024 and 2025, the first two years of the computer-based exam. The exam "
        "section below covers both."
    )
    notes.append(
        "- **Some issue dates are not original.** A number of licenses carry an issue date years "
        "later than their license number implies, usually the date of a reinstatement or a "
        "reciprocity grant. The last chart on the Time tab flags them."
    )
    return "\n".join(notes)


# ---------- History tab: exam results ----------
EXAM_NATIONAL_KEY = 'history_exam_national'
EXAM_BAR_ALPHA = 0.30          # license bars sit behind the passer markers
EXAM_STRAND_NUDGE_DAYS = 28    # side-by-side markers for two pieces at one sitting
EXAM_BAR_WIDTH_DAYS = 150

EXAM_INTRO = """
Every structural title in this data rests on a structural exam, and since 2003 that exam has
changed shape four times. A pass rate therefore measures something different in each era: the
whole exam, one of two separate exams, one half of a 16-hour exam, or one of four
computer-based sections. The chart follows the two parts of the exam that run through every
format, design for gravity loads (vertical forces) and design for earthquakes and wind
(lateral forces), and sets the number of California candidates who passed at each sitting
against the licenses issued before the next one.
"""

EXAM_ABOUT = """
California results are the Board's own figures for all California candidates at each sitting
(it has never split first-time from repeat takers, or buildings from bridges). National results
are NCEES first-time takers unless the hover says otherwise. Most come from the Board's archived
statistics pages, its bulletins and meeting packets, NCEES's *Licensure Exchange* and *Squared*
reports, and, for twenty national sittings from 2012 to 2023, one private compilation (William
S. Godfrey's) that matches every NCEES figure it could be checked against.

The full record, including repeat-taker rates, fiscal-year totals, the 2001-2002 national
results and every conflict between sources, is in `research/se_exam_pass_rates.csv` and the
report `reports/CA SE exam pass rates.md`. The rows drawn here are kept in
`data/se_exam_results.csv`.
"""


@st.cache_data
def _load_exam_results_cached(file_mtime=None):
    try:
        return read_exam_results_csv(), None
    except ValueError as exc:
        return None, str(exc)


def load_exam_results():
    path = exam_results_path()
    return _load_exam_results_cached(file_mtime=path.stat().st_mtime if path.exists() else None)


def _sitting_label(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.strftime('%B')} {ts.year}"


def _exam_hover_rows(grp: pd.DataFrame) -> list:
    evidence_text = {
        'primary': 'Published figure',
        'derived': 'Worked out from published figures',
        'secondary': 'Third-party compilation, not confirmed by NCEES',
    }
    rows = []
    for r in grp.to_dict('records'):
        if pd.notna(r['passed']):
            counts = f"{int(r['passed'])} of {int(r['tested'])} passed" if pd.notna(r['tested']) else f"{int(r['passed'])} passed"
        elif pd.notna(r['passed_est']):
            counts = f"about {int(r['passed_est'])} of {int(r['tested'])} passed"
        elif pd.notna(r['tested']):
            counts = f"{int(r['tested'])} tested"
        else:
            counts = 'No counts published'
        who = 'California, all takers' if r['scope'] == 'California' else (
            'National, first-time takers' if r['population'] == 'first_time' else 'National, all takers')
        rows.append([
            _sanitize_hover(r['exam']),
            _sitting_label(r['date']),
            who,
            counts,
            evidence_text.get(r['evidence'], ''),
            ('<br>' + _hover_lines(r['note'])) if r['note'] else '',
        ])
    return rows


_EXAM_HOVER = (
    '<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>'
    '<b>%{y:.0f}%</b> passed · %{customdata[3]}<br>'
    '<i>%{customdata[4]}</i>%{customdata[5]}<extra></extra>'
)


def plot_exam_results(results: pd.DataFrame, licenses_df: Optional[pd.DataFrame], show_national: bool):
    """Pass rates at each sitting above passers and the licenses that followed.

    Two stacked panels on one time axis rather than one chart with two scales:
    rates are percentages, passers and licenses are head counts. Lines break at
    each format change because a rate on one side of the break is not the same
    measure as a rate on the other. Each strand keeps its color and marker shape
    across formats; hollow markers are worked out from other published figures.
    """
    from plotly.subplots import make_subplots

    now = pd.Timestamp.now().normalize()
    x_lo = pd.Timestamp('2002-07-01')
    x_hi = now + pd.DateOffset(months=9)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.55, 0.45])

    # Era bands across both panels, with the format named at the top of each.
    for i, (key, _start, _end, label, desc) in enumerate(EXAM_ERAS):
        lo, hi = era_bounds(key, today=x_hi)
        lo, hi = max(lo, x_lo), min(hi, x_hi)
        if lo >= hi:
            continue
        if i % 2 == 1:
            # exclude_empty_subplots=False: the bands go on before any trace exists.
            fig.add_vrect(x0=lo, x1=hi, fillcolor=_tint(PRIMARY, 0.06), layer='below', line_width=0,
                          row='all', col=1, exclude_empty_subplots=False)
        if i > 0:
            fig.add_vline(x=lo, line=dict(color=MUTED, width=1, dash='dot'), layer='below',
                          row='all', col=1, exclude_empty_subplots=False)
        fig.add_annotation(
            x=lo + (hi - lo) / 2, y=1.0, xref='x', yref='y domain', yanchor='bottom',
            text=f"<b>{html.escape(label)}</b>", showarrow=False, font=dict(size=11, color=NEUTRAL),
            hovertext=html.escape(desc),
        )

    ca = results[results['scope'] == 'California']
    ca_sittings = set(ca.loc[ca['pass_rate_pct'].notna(), 'date'])
    national = results[results['scope'] == 'National']
    # National-only sittings (for now, just the newest) are always drawn so the
    # chart does not end on the 2025 low when a later result exists.
    national_only = national[~national['date'].isin(ca_sittings) & (national['date'] > max(ca_sittings))]

    nudge = {'vertical': -EXAM_STRAND_NUDGE_DAYS, 'lateral': EXAM_STRAND_NUDGE_DAYS, 'whole': -EXAM_STRAND_NUDGE_DAYS}

    def _markers(grp, color, strand, hollow_mask, size=9):
        return dict(
            size=size,
            symbol=[EXAM_STRAND_SYMBOLS[strand] + ('-open' if h else '') for h in hollow_mask],
            color=color,
            line=dict(color=[color if h else 'white' for h in hollow_mask], width=[2 if h else 1.5 for h in hollow_mask]),
        )

    # ---- Panel 1: pass rates ----
    shown_in_legend = set()
    for strand in STRANDS:
        color = exam_strand_color(strand)
        for era_key in [e[0] for e in EXAM_ERAS]:
            grp = ca[(ca['strand'] == strand) & (ca['era'] == era_key)].sort_values('date')
            if grp['pass_rate_pct'].notna().sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=grp['date'], y=grp['pass_rate_pct'],
                mode='lines+markers', connectgaps=False,
                name=STRAND_LABELS[strand], legendgroup=strand,
                showlegend=strand not in shown_in_legend,
                line=dict(color=color, width=2),
                marker=_markers(grp, color, strand, (grp['evidence'] == 'derived').tolist()),
                customdata=_exam_hover_rows(grp), hovertemplate=_EXAM_HOVER,
            ), row=1, col=1)
            shown_in_legend.add(strand)

        nat = national[national['strand'] == strand]
        if not show_national:
            nat = nat[nat.index.isin(national_only.index)]
        for era_key in [e[0] for e in EXAM_ERAS]:
            grp = nat[nat['era'] == era_key].sort_values('date')
            if grp['pass_rate_pct'].notna().sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=grp['date'], y=grp['pass_rate_pct'],
                mode='lines+markers', connectgaps=False,
                name=f"National · {STRAND_LABELS[strand].lower()}", legendgroup=f"nat-{strand}",
                showlegend=show_national and f"nat-{strand}" not in shown_in_legend,
                line=dict(color=color, width=1.5, dash='dot'),
                marker=_markers(grp, color, strand, [True] * len(grp), size=8),
                opacity=0.85,
                customdata=_exam_hover_rows(grp), hovertemplate=_EXAM_HOVER,
            ), row=1, col=1)
            shown_in_legend.add(f"nat-{strand}")

    if not show_national and not national_only.empty:
        top = national_only.sort_values('pass_rate_pct', ascending=False).iloc[0]
        fig.add_annotation(
            x=top['date'], y=top['pass_rate_pct'], xref='x', yref='y', ax=-58, ay=-18,
            text=f"National, {top['date'].strftime('%b %Y')}", showarrow=True, arrowhead=0,
            arrowcolor=MUTED, font=dict(size=11, color=NEUTRAL),
        )

    fig.update_yaxes(title_text='Passed at the sitting', ticksuffix='%', range=[0, 72],
                     dtick=20, fixedrange=True, row=1, col=1)

    # ---- Panel 2: licenses after each sitting, and California passers ----
    windows = licenses_after_sittings(licenses_df, exam_sittings(results), today=now)
    if not windows.empty:
        fig.add_trace(go.Bar(
            x=windows['date'], y=windows['licenses'],
            width=[EXAM_BAR_WIDTH_DAYS * 86400000] * len(windows),
            name='Licenses issued before the next sitting',
            marker=dict(
                color=[_tint(PRIMARY, 0.12 if p else EXAM_BAR_ALPHA) for p in windows['partial']],
                line=dict(color=PRIMARY, width=[1.5 if p else 0 for p in windows['partial']]),
            ),
            customdata=[
                [f"{s:%b %Y}", f"{e:%b %Y}", ' (so far)' if p else '']
                for s, e, p in zip(windows['window_start'], windows['window_end'], windows['partial'])
            ],
            hovertemplate='<b>%{y} licenses issued%{customdata[2]}</b><br>'
                          '%{customdata[0]} to %{customdata[1]}<extra></extra>',
        ), row=2, col=1)

    for strand in STRANDS:
        color = exam_strand_color(strand)
        grp = ca[(ca['strand'] == strand) & ca['passed_est'].notna()].sort_values('date')
        if grp.empty:
            continue
        hollow = grp['passed'].isna().tolist()
        fig.add_trace(go.Scatter(
            x=grp['date'] + pd.to_timedelta(nudge[strand], unit='D'), y=grp['passed_est'],
            mode='markers', name=STRAND_LABELS[strand], legendgroup=strand, showlegend=False,
            marker=_markers(grp, color, strand, hollow),
            customdata=[
                [_sanitize_hover(r['exam']), _sitting_label(r['date']),
                 ('about ' if pd.isna(r['passed']) else '') + f"{int(r['passed_est'])}",
                 f"{int(r['tested'])}" if pd.notna(r['tested']) else '?']
                for r in grp.to_dict('records')
            ],
            hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>'
                          '%{customdata[2]} of %{customdata[3]} California candidates passed<extra></extra>',
        ), row=2, col=1)

    fig.update_yaxes(title_text='People', rangemode='tozero', fixedrange=True, row=2, col=1)
    fig.update_xaxes(range=[x_lo, x_hi], dtick='M24', tickformat='%Y')
    fig.update_layout(
        height=680,
        hovermode='closest',
        barmode='overlay',
        margin=dict(l=24, r=24, t=36, b=90),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='left', x=0),
    )
    _show_chart(fig)


def _exam_stat_items(results: pd.DataFrame) -> list:
    """One card per exam format: the median California pass rate per sitting."""
    summary = era_rate_summary(results, scope='California')
    what = {
        'state': 'the whole exam',
        'split': 'either of the two separate exams (Structural II or SE III)',
        'sixteen': 'one of the two components',
        'cbt': 'one buildings depth section',
    }
    items = []
    for r in summary.to_dict('records'):
        span = f"{r['first'].year}" if r['first'].year == r['last'].year else f"{r['first'].year}–{r['last'].year}"
        if r['n'] == 1:
            row = results[(results['scope'] == 'California') & (results['era'] == r['era'])
                          & results['pass_rate_pct'].notna()].iloc[0]
            help_text = (f"{_sitting_label(row['date'])}: {int(row['passed'])} of {int(row['tested'])} candidates "
                         "passed. The only California-written structural exam with a published result.")
        else:
            help_text = (f"Middle value of {r['n']} California results, {r['low']:.0f}% to {r['high']:.0f}%. "
                         f"Each is the share passing {what.get(r['era'], 'one piece of the exam')} at one sitting, "
                         "so eras are not strictly comparable.")
        items.append(dict(label=f"{r['label']}, {span}", value=f"{r['median']:.0f}%", help=help_text))
    return items


def _exam_notes(results: pd.DataFrame, licenses_df: Optional[pd.DataFrame]) -> str:
    """Markdown bullets tying the exam results to the license counts.

    Figures that come from the data are computed here so the text cannot drift
    from the chart.
    """
    now = pd.Timestamp.now().normalize()
    ca = results[results['scope'] == 'California']
    windows = licenses_after_sittings(licenses_df, exam_sittings(results), today=now).set_index('date')
    notes = [
        "- **Pass rates are not one series.** A 2003 rate is for the whole exam, a 2004–2011 rate for "
        "one of two separate exams, a 2011–2023 rate for one of two components and a rate from 2024 for "
        "one of four sections, and a candidate had to clear every piece. Colors and shapes follow the "
        "part of the exam rather than the format: blue circles for vertical forces, amber diamonds for "
        "lateral forces and seismic design (California's own SE III exam, then the Lateral component "
        "and section), green squares for a whole exam or the general national Structural II. Lines "
        "break where the format changed.",
    ]

    # The last SE III sitting (October 2011) shared its window with the first
    # 16-hour completions, so only the years when SE III was the last step count.
    se3 = ca[ca['exam'].str.contains('SE III') & ca['passed'].notna() & (ca['date'] < pd.Timestamp('2011-04-01'))]
    se3 = se3[se3['date'].isin(windows.index)]
    if not se3.empty:
        diffs = (se3.set_index('date')['passed'] - windows.loc[se3['date'], 'licenses']).abs()
        best = se3.sort_values('passed').iloc[-1]
        notes.append(
            f"- **The head counts line up better than the rates.** From {se3['date'].min().year} to "
            f"{se3['date'].max().year} the number passing California's SE III each October came within "
            f"{int(diffs.max())} of the licenses issued before the next October exam. "
            f"{int(best['passed'])} passed in {_sitting_label(best['date'])}, the most in the record, and "
            f"{int(windows.loc[best['date'], 'licenses'])} licenses followed. On the 16-hour exam a candidate "
            "needed both components, often passed at different sittings, so each batch tracks the two "
            "component counts loosely rather than either one."
        )

    notes.append(
        "- **Before 2003 there is only a recollection.** No California structural pass rate survives "
        "from before October 2003, when 25% passed the last state-written exam. Henry Degenkolb recalled "
        "the share passing falling from over 30% to about 10% when grading on a curve ended in the "
        "mid-1980s. If he meant the structural exam, that would explain why only 26 licenses were "
        "issued in February 1985."
    )

    cbt = ca[(ca['era'] == 'cbt') & ca['pass_rate_pct'].notna()]
    if not cbt.empty:
        est = cbt['passed_est'].dropna()
        cbt_windows = windows[(windows.index.isin(cbt['date'])) & ~windows['partial']]
        nat_cbt = results[(results['scope'] == 'National') & (results['era'] == 'cbt')
                          & (results['date'] < pd.Timestamp('2026-01-01'))]['pass_rate_pct'].dropna()
        text = (
            "- **2024–2025: the depth sections were the bottleneck.** California's buildings depth "
            f"sections passed {cbt['pass_rate_pct'].min():.0f}% to {cbt['pass_rate_pct'].max():.0f}% of exams, "
            f"about {int(est.min())} to {int(est.max())} people per section per sitting"
        )
        if not nat_cbt.empty:
            text += f", against {nat_cbt.min():.0f}% to {nat_cbt.max():.0f}% nationally"
        text += ". "
        if not cbt_windows.empty:
            text += (
                f"Licenses issued between sittings ran at {int(cbt_windows['licenses'].min())} to "
                f"{int(cbt_windows['licenses'].max())}, well above that, because candidates who had "
                "passed one paper component by October 2023 needed only the other half, and some "
                "licenses probably went to engineers already licensed in other states. "
            )
        text += ("Nationally the breadth sections passed 40% to 49% of first-time takers throughout, "
                 "so the drop came from the depth format.")
        notes.append(text)

    nat_latest = results[(results['scope'] == 'National') & (results['date'] >= pd.Timestamp('2026-01-01'))]
    if not nat_latest.empty:
        rates = nat_latest.set_index('strand')['pass_rate_pct']
        last = nat_latest['date'].max()
        text = (
            "- **2026: rates recover.** NCEES added an hour to the depth sections, and its audit "
            f"committee listed a standard-setting study on the {_sitting_label(last)} exams. "
            "National first-time rates on the "
            f"buildings depth sections rose to {rates.get('vertical', float('nan')):.0f}% vertical and "
            f"{rates.get('lateral', float('nan')):.0f}% lateral. No source says how much of the change "
            "came from the extra time and how much from a new passing standard. The Board's California "
            "figures for January to June 2026, which combine breadth and depth, were 46% vertical and "
            "45% lateral."
        )
        if licenses_df is not None and 'Original Issue Date' in licenses_df.columns:
            issued = pd.to_datetime(licenses_df['Original Issue Date'], errors='coerce')
            ytd = int(((issued.dt.year == now.year) & (issued <= now)).sum())
            prior = int((issued.dt.year == now.year - 1).sum())
            text += f" {ytd} licenses have been issued so far in {now.year}, against {prior} in all of {now.year - 1}."
        notes.append(text)

    notes.append(
        "- **What the record cannot show.** Neither the Board nor NCEES publishes how many candidates "
        "finished the whole exam, so no figure here is a rate of becoming a structural engineer. The "
        "Board never published Fall 2017 results. The 2024–2025 California section results were read "
        "from charts in Board meeting packets and are worth checking against the originals. Hollow "
        "markers are worked out from other published figures."
    )
    return "\n".join(notes)


def _render_exam_section(results: Optional[pd.DataFrame], error: Optional[str], all_df: Optional[pd.DataFrame]):
    if error:
        st.warning(f"The exam results file could not be read: {error}")
        return
    if results is None or results.empty:
        return

    st.subheader("The exam as the gate: pass rates since 2003")
    st.markdown(EXAM_INTRO)
    show_national = st.toggle(
        "Compare with national first-time takers",
        value=False,
        key=EXAM_NATIONAL_KEY,
        help="Adds NCEES national results as dotted lines. They cover first-time takers and California "
             "covers everyone, so the two are not the same measure.",
    )
    plot_exam_results(results, all_df, show_national)
    st.caption(
        "Top: the share of California candidates passing each piece of the exam at each sitting. "
        "Bottom: how many passed (markers) and how many structural licenses were issued between that "
        "sitting and the next (bars; the last bar is still filling). Shaded bands mark the four exam "
        "formats. Hover any point for the exam, the counts and where the figure comes from."
    )
    stat_items = _exam_stat_items(results)
    if stat_items:
        _render_stat_row(stat_items)

    st.markdown("**Reading the exam results against the license counts**")
    st.markdown(_exam_notes(results, all_df))

    with st.expander("Exam results data and sources"):
        table = results.copy()
        table['Sitting'] = table['date'].map(_sitting_label)
        table['Who'] = [
            'California, all' if s == 'California' else ('National, first-time' if p == 'first_time' else 'National, all')
            for s, p in zip(table['scope'], table['population'])
        ]
        table = table.rename(columns={
            'exam': 'Exam', 'tested': 'Tested', 'passed': 'Passed', 'pass_rate_pct': 'Pass rate (%)',
            'evidence': 'Evidence', 'note': 'Note', 'source': 'Source',
        })
        st.dataframe(
            table[['Sitting', 'Exam', 'Who', 'Tested', 'Passed', 'Pass rate (%)', 'Evidence', 'Note', 'Source']],
            hide_index=True, use_container_width=True,
        )
        st.markdown(EXAM_ABOUT)


def render_history_tab(all_df: Optional[pd.DataFrame], events: Optional[pd.DataFrame], error: Optional[str]):
    """History tab: what the license is, a timeline against licenses issued, and
    a sourced entry for every event. Uses the whole dataset, not the sidebar filters."""
    if error:
        st.warning(f"The licensure history file could not be read: {error}")
        return
    if events is None or events.empty:
        st.info("No licensure history events found in `data/licensure_history.csv`.")
        return

    st.subheader("How the Structural Engineer license came to be")
    st.markdown(HISTORY_INTRO)

    stat_items = _history_stat_items(all_df)
    if stat_items:
        _render_stat_row(stat_items)

    st.subheader("Timeline: events and licenses issued")
    present = [c for c in HISTORY_CATEGORIES if (events['category'] == c).any()]
    selected = st.pills(
        "Kinds of event",
        present,
        selection_mode="multi",
        default=present,
        key=HISTORY_FILTER_KEY,
        help="Filters both the timeline and the entries below it.",
    )
    shown = events[events['category'].isin(selected or [])]

    plot_history_timeline(shown, all_df)
    st.caption(
        "Each dot is an event, in a lane for its kind. Hover a dot for a one-line summary; the full "
        "entries are below. Bars count licenses by Original Issue Date across the whole dataset and "
        f"ignore the sidebar filters. The lightly shaded {pd.Timestamp.now().year} bar is year to date."
    )

    st.subheader("Reading the license counts against this history")
    st.markdown(_history_data_notes(all_df))

    exam_results, exam_error = load_exam_results()
    _render_exam_section(exam_results, exam_error, all_df)

    st.subheader("Chronology")
    if shown.empty:
        st.info("Pick at least one kind of event above to list the entries.")
    for i, (_start, _end, _title, blurb) in enumerate(ERAS):
        era_events = shown[shown['year'].map(era_index) == i]
        if era_events.empty:
            continue
        cards = ''.join(
            event_card_html(row, history_category_color(row['category']))
            for row in era_events.to_dict('records')
        )
        st.markdown(
            f'<div class="hist-era"><h3>{html.escape(era_label(i))}</h3>'
            f'<p class="hist-era-blurb">{html.escape(blurb)}</p></div>{cards}',
            unsafe_allow_html=True,
        )

    st.subheader("From the oral histories")
    st.markdown(HISTORY_VOICES)

    with st.expander("About this history"):
        st.markdown(HISTORY_ABOUT)


def main():
    st.set_page_config(
        page_title="CA SE License Statistics",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Apply unified visual theme
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    default_csv = "ProfEngrsLandSurvyrsGeologist_Data00.xls_structural_engineers_cleaned"
    src = resolve_data_source(default_csv)

    data_updated_str = get_data_last_updated(src)

    st.title("🏗️ Statistics for the California Structural Engineer License")
    subtitle_html = (
        '<p class="dashboard-subtitle">'
        'Interactive view of California structural engineer license data from the DCA public record.<br>'
        'Select a tab below to toggle between different visual representation categories of the data.<br>'
        'Scroll down on each page to see all visuals of the chosen category.<br>'
        'See sidebar to the left to filter the data by time, state of residence, and license status.'
        '</p>'
    )
    if data_updated_str:
        subtitle_html += f'<p class="dashboard-subtitle">Data last updated: {data_updated_str}</p>'
    st.markdown(subtitle_html, unsafe_allow_html=True)

    df = None
    if src:
        with st.spinner("Loading cleaned data..."):
            df = load_cleaned_data(src, _file_mtime=_file_hash(src))
            df = parse_dates(df)

    if df is None:
        st.sidebar.warning("No CSV file found automatically.")
        st.sidebar.markdown("**Please upload your CSV file:**")
        uploaded = st.sidebar.file_uploader("Upload a cleaned CSV", type=["csv"])
        if uploaded is not None:
            with st.spinner("Loading uploaded data..."):
                df = load_cleaned_data(uploaded)
                df = parse_dates(df)

    if df is None or df.empty:
        st.error("Unable to load data from the CSV file.")
        st.info("""
**Expected columns (case sensitive):**
`State`, `License Status`, `Original Issue Date`, `Expiration Date`

**To get started:**
1. Place your CSV file in the same directory as this script
2. Or use the file uploader in the sidebar
3. Make sure your CSV has the required columns listed above
        """)
        return

    _display_headline_stats(df)

    with st.expander("About the data"):
        st.markdown(
            """
- **Source.** California Department of Consumer Affairs (DCA) public licensee roster for the
  Board for Professional Engineers, Land Surveyors, and Geologists (BPELSG), filtered to
  Structural Engineer licenses. Individual statuses can be verified on the
  [BPELSG license lookup](https://www.bpelsg.ca.gov/consumers/lic_lookup.shtml).
- **Refresh cadence.** The public feed is downloaded and reconciled roughly monthly; the
  "data last updated" date above reflects the latest refresh.
- **How drop-offs are handled.** Licenses that disappear from the public feed are marked
  *Cancelled* provisionally, then re-checked against the DCA license-detail pages to recover
  their true status (Retired, Deceased, Delinquent, etc.).
- **Known quirks.** For some reinstated or reciprocity licenses, DCA appears to stamp the
  *reactivation* date into the Original Issue Date field; the last chart on the Time tab
  flags these automatically.
- **Caveats.** An *active license* is not the same as an active practitioner — renewal only
  requires a fee every two years. Only each license's most recent expiration date is
  published, so brief mid-career lapses aren't visible. Recent attrition counts can shrink
  over time as delinquent licenses renew.
"""
        )

    # Dev-only column inspector — enable with ?debug=1 in the URL
    if st.query_params.get('debug') == '1':
        with st.expander("Data Column Information (debug)"):
            st.write("**Available columns:**")
            for col in df.columns:
                st.write(f"- `{col}` ({df[col].dtype})")
            st.write(f"\n**Data shape:** {df.shape[0]} rows x {df.shape[1]} columns")

    _init_filters_from_url(df)

    # Read session state before widgets render so sidebar metrics are current
    preview_filters = _get_current_filters_from_session(df)
    preview_df = apply_filters(df, preview_filters)

    # Sidebar: Summary Data
    st.sidebar.title("Summary Data")
    _display_sidebar_metrics(preview_df)

    st.sidebar.markdown("---")

    # Sidebar: active filter chips
    _render_filter_chips(preview_filters, df, container=st.sidebar)

    # Sidebar: Filters (title + widgets)
    filters = create_filters(df)

    st.sidebar.markdown("---")
    if st.sidebar.button("Share current filters"):
        st.query_params.from_dict(_build_share_params(filters))
        st.sidebar.success("URL updated — copy it from your browser address bar.")

    # Base for the comparison-state overlay: every filter except State, so the
    # overlay honours License Status (it used to include all statuses).
    full_df = apply_filters(df, {**filters, 'states': None})
    filtered_df = apply_filters(df, filters)
    status_agnostic_df = apply_filters(df, {**filters, 'statuses': None})

    if filtered_df is None or filtered_df.empty:
        if filters.get('states') == [] or filters.get('statuses') == []:
            st.warning(
                "Every option is unticked in a sidebar filter, so no records are shown. "
                "Tick **(Select All)** in the State or License Status filter to bring them back."
            )
        else:
            st.warning("No records match the selected filters. Try adjusting your filter criteria.")
        return

    create_visualizations(
        filtered_df,
        filters.get('show_events', False),
        filters.get('start_year'),
        filters.get('end_year'),
        filters.get('comparison_state'),
        full_df,
        status_agnostic_df,
        all_df=df,
    )


if __name__ == "__main__":
    main()
