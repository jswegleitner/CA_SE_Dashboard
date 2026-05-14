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
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re
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
from dashboard_lib.theme import (
    register_plotly_template,
    DASHBOARD_CSS,
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

# When a categorical axis has more than this many items, hide outside text labels
# on bars to reduce visual noise.
BAR_LABEL_MAX_CATEGORIES = 25


# ---------- Data loading ----------
def _file_hash(src):
    """Return file modification time as a cache-busting key, or None for URLs."""
    p = Path(src)
    if p.exists():
        return p.stat().st_mtime
    return None


@st.cache_data(ttl=3600)
def load_cleaned_data(src, _file_mtime=None):
    """
    Load pre-cleaned CSV data with caching for better performance.
    `src` can be a local path or a URL.
    `_file_mtime` is used only to bust the cache when the file changes on disk.
    """
    try:
        df = pd.read_csv(src)
        return df
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

    try:
        if 'Original Issue Date' in df.columns:
            df['Original Issue Date'] = pd.to_datetime(df['Original Issue Date'], errors='coerce')
        if 'Expiration Date' in df.columns:
            df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors='coerce')
    except Exception:
        # Don't raise; invalid date parsing should not break the app.
        pass

    return df


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
            if csv_path.name.lower() in ['timeline_events.csv', 'license history table.csv']:
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

    if 'states' in params and 'state_filter' not in st.session_state:
        url_states = [s.strip() for s in params['states'].split(',') if s.strip()]
        valid = set(df['State'].dropna().unique()) if 'State' in df.columns else set()
        filtered = [s for s in url_states if s in valid]
        if filtered:
            st.session_state['state_filter'] = filtered

    if 'statuses' in params and 'status_filter' not in st.session_state:
        url_statuses = [s.strip() for s in params['statuses'].split(',') if s.strip()]
        valid = set(df['License Status'].dropna().unique()) if 'License Status' in df.columns else set()
        filtered = [s for s in url_statuses if s in valid]
        if filtered:
            st.session_state['status_filter'] = filtered

    if 'start_year' in params and 'date_range_slider' not in st.session_state:
        try:
            sy = int(params['start_year'])
            ey = int(params.get('end_year', str(sy)))
            st.session_state['date_range_slider'] = (sy, ey)
        except (ValueError, TypeError):
            pass

    if 'compare' in params and 'comparison_state' not in st.session_state:
        st.session_state['comparison_state'] = params['compare']

    if 'exp' in params and 'expired_filter' not in st.session_state:
        url_exp = [s.strip() for s in params['exp'].split(',') if s.strip()]
        if url_exp:
            st.session_state['expired_filter'] = url_exp


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
    if filters.get('expiration_types'):
        params['exp'] = ','.join(filters['expiration_types'])
    return params


# ---------- Filter chip summary ----------

def _render_filter_chips(filters: dict, df: pd.DataFrame):
    """Render a compact chip bar showing the currently-applied filters.

    Only shows chips that differ from the unfiltered default — keeps the bar
    quiet when nothing is filtered.
    """
    all_states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    all_statuses = sorted(df['License Status'].dropna().unique().tolist()) if 'License Status' in df.columns else []

    if 'Original Issue Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Original Issue Date']):
        full_min = int(df['Original Issue Date'].dt.year.min())
        full_max = int(df['Original Issue Date'].dt.year.max())
    else:
        full_min, full_max = None, None

    chips = []

    selected_states = filters.get('states') or []
    if selected_states and set(selected_states) != set(all_states):
        preview = ", ".join(selected_states[:3])
        if len(selected_states) > 3:
            preview += f" +{len(selected_states) - 3} more"
        chips.append(("States", preview))

    selected_statuses = filters.get('statuses') or []
    if selected_statuses and set(selected_statuses) != set(all_statuses):
        chips.append(("Status", ", ".join(selected_statuses)))

    sy, ey = filters.get('start_year'), filters.get('end_year')
    if sy is not None and ey is not None and (sy != full_min or ey != full_max):
        chips.append(("Years", f"{sy}–{ey}"))

    exp = filters.get('expiration_types') or []
    if exp and set(exp) != {'Active Only', 'Expired Only'}:
        chips.append(("Show", ", ".join(exp).replace(' Only', '')))

    cmp_state = filters.get('comparison_state')
    if cmp_state and cmp_state != 'None':
        chips.append(("Compare", cmp_state))

    if not chips:
        html = '<div class="filter-chips"><span class="filter-chip-empty">No filters applied — viewing all records</span></div>'
    else:
        items = "".join(
            f'<span class="filter-chip"><span class="filter-chip-key">{key}</span>{val}</span>'
            for key, val in chips
        )
        html = f'<div class="filter-chips">{items}</div>'

    st.markdown(html, unsafe_allow_html=True)


# ---------- Filters & visuals (updated with Plotly) ----------
def create_filters(df):
    filters = {}

    st.sidebar.header("Filters")
    events_df_present = load_timeline_events() is not None
    filters['show_events'] = st.sidebar.checkbox(
        "Show timeline events",
        value=events_df_present,
        help="Toggle overlays of exam/code history on charts (if available).",
        key='show_events_toggle',
    )

    st.sidebar.subheader("State")
    states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    filters['states'] = st.sidebar.multiselect(
        "Select States:",
        options=states,
        default=states,
        key='state_filter',
        help="Select one or more states. Leave empty to include all states.",
    )

    filters['comparison_state'] = st.sidebar.selectbox(
        "Compare with State (overlay on bar chart):",
        options=['None'] + states,
        index=0,
        key='comparison_state',
        help="Select a single state to overlay its data on the time series bar chart",
    )

    st.sidebar.subheader("License Status")
    statuses = sorted(df['License Status'].dropna().unique().tolist()) if 'License Status' in df.columns else []
    filters['statuses'] = st.sidebar.multiselect(
        "Select License Statuses:",
        options=statuses,
        default=statuses,
        key='status_filter',
        help="Select one or more license statuses. Leave empty to include all statuses.",
    )

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

    expiration_options = ['Active Only', 'Expired Only']
    filters['expiration_types'] = st.sidebar.multiselect(
        "License Expiration Status:",
        options=expiration_options,
        default=expiration_options,
        key='expired_filter',
        help="Active = not yet expired, Expired = past expiration date. Select both for all licenses.",
    )

    if not events_df_present:
        st.sidebar.caption("Add `timeline_events.csv` with columns: start_date, end_date (optional), label, description, type.")
    return filters


def apply_filters(df, filters):
    if df is None or df.empty:
        return df

    filtered_df = df.copy()

    if 'State' in filtered_df.columns and filters['states']:
        filtered_df = filtered_df[filtered_df['State'].isin(filters['states'])]

    if 'License Status' in filtered_df.columns and filters['statuses']:
        filtered_df = filtered_df[filtered_df['License Status'].isin(filters['statuses'])]

    if 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        mask = (
            (filtered_df['Original Issue Date'].dt.year >= filters['start_year']) &
            (filtered_df['Original Issue Date'].dt.year <= filters['end_year'])
        )
        filtered_df = filtered_df[mask]

    if 'Expiration Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Expiration Date']):
        current_date = pd.Timestamp.now()
        if filters['expiration_types']:
            inc_active = 'Active Only' in filters['expiration_types']
            inc_expired = 'Expired Only' in filters['expiration_types']
            mask = pd.Series(False, index=filtered_df.index)
            if inc_active:
                mask = mask | (filtered_df['Expiration Date'] >= current_date)
            if inc_expired:
                mask = mask | (filtered_df['Expiration Date'] < current_date)
            filtered_df = filtered_df[mask]

    return filtered_df


def create_visualizations(filtered_df, show_events: bool, start_year: Optional[int] = None, end_year: Optional[int] = None, comparison_state: str = None, full_df=None):
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

    tab_labels = ["Activity"]
    if has_geo:
        tab_labels.append("Geography")
    if has_status:
        tab_labels.append("Status")

    tabs = st.tabs(tab_labels)

    # ---------- Activity tab ----------
    with tabs[0]:
        st.subheader("Licenses Issued Over Time")
        col1, _ = st.columns([1, 3])
        with col1:
            bucket_size = st.selectbox("Time Grouping:", ["Yearly", "Half-Yearly", "Quarterly", "Monthly"])
            lock_y = st.checkbox(
                "Lock Y-axis when zooming",
                value=True,
                help="When checked, zooming/panning will only affect the X axis; Y axis will remain fixed.",
            )

        events_df = load_timeline_events()
        plot_time_series(filtered_df, bucket_size, lock_y, events_df=events_df, show_events=show_events, start_year=start_year, end_year=end_year, comparison_state=comparison_state, full_df=full_df)
        plot_time_series_line(filtered_df, lock_y, events_df=events_df, show_events=show_events, start_year=start_year, end_year=end_year)
        plot_yearly_retirements(filtered_df, start_year=start_year, end_year=end_year)
        plot_active_licenses_per_year(filtered_df, start_year=start_year, end_year=end_year)
        plot_avg_license_age_per_year(filtered_df, start_year=start_year, end_year=end_year)

    # ---------- Geography tab ----------
    if has_geo:
        with tabs[tab_labels.index("Geography")]:
            if has_multi_state:
                plot_state_counts(filtered_df)
                plot_us_map(filtered_df)
            if has_ca_county:
                ca_data = filtered_df[filtered_df['State'] == 'CA']
                st.subheader("California Licenses by County")
                plot_ca_county_map(ca_data)

    # ---------- Status tab ----------
    if has_status:
        with tabs[tab_labels.index("Status")]:
            plot_status_pie(filtered_df)


def plot_time_series(filtered_df, bucket_size: str, lock_y: bool = True, events_df: Optional[pd.DataFrame] = None, show_events: bool = False, start_year: Optional[int] = None, end_year: Optional[int] = None, comparison_state: str = None, full_df=None):
    if 'Original Issue Date' not in filtered_df.columns or not pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        st.info("Original Issue Date column not found or not datetime.")
        return

    license_counts = _count_by_period(filtered_df, bucket_size, start_year, end_year)

    if license_counts.empty:
        st.info("No time-series data for the selected filters.")
        return

    show_labels = len(license_counts) <= BAR_LABEL_MAX_CATEGORIES

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(p) for p in license_counts.index],
        y=license_counts.values,
        name='Main Data',
        marker_color=MAIN_BAR_COLOR,
        text=license_counts.values if show_labels else None,
        textposition='outside' if show_labels else 'none',
        hovertemplate='<b>Period:</b> %{x}<br><b>Licenses:</b> %{y}<extra></extra>',
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
            marker_color=COMPARISON_BAR_COLOR,
            text=comp_counts.values if show_labels else None,
            textposition='outside' if show_labels else 'none',
            opacity=0.75,
            hovertemplate=f'<b>Period:</b> %{{x}}<br><b>{comparison_state} Licenses:</b> %{{y}}<extra></extra>',
        ))

    fig.update_layout(
        xaxis_title="Period",
        yaxis_title="Number of Licenses",
        showlegend=True if (comparison_state and comparison_state != 'None') else False,
        height=460,
        hovermode='x unified',
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

            text_year = ev['start_date'].dt.year.astype('Int64').astype(str).fillna('')

            ev_desc = render_event_descriptions(ev)

            fig.add_trace(go.Scatter(
                x=ev['bucket'],
                y=[-pad * 0.8] * len(ev),
                mode='markers+text',
                name='Events',
                marker=dict(symbol='triangle-up', size=10, color=EVENT_MARKER_COLOR),
                text=text_year,
                textposition='bottom center',
                textfont=dict(color=EVENT_MARKER_COLOR),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'Date: %{customdata[1]}<br>'
                    'Type: %{customdata[2]}<br>'
                    '%{customdata[3]}<extra></extra>'
                ),
                customdata=pd.concat([
                    ev['label'].fillna(''),
                    ev['start_date'].dt.strftime('%Y-%m-%d'),
                    ev['type'].astype(str),
                    ev_desc,
                ], axis=1).values,
            ))

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Time Series Statistics"):
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Peak Period", str(license_counts.idxmax()))
        with c2: st.metric("Peak Count", int(license_counts.max()))
        with c3: st.metric("Average per Period", f"{license_counts.mean():.1f}")


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
    fig.add_trace(go.Scatter(
        x=df_line['Original Issue Date'],
        y=yvals,
        mode='lines+markers',
        name='Licenses',
        line=dict(color=MAIN_BAR_COLOR, width=2),
        marker=dict(size=4, color=MAIN_BAR_COLOR),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title="Original Issue Date",
        yaxis_title=y_label,
        height=400,
        hovermode='x unified',
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
            type_styles[t] = dict(color=EVENT_TYPE_PALETTE[i % len(EVENT_TYPE_PALETTE)], dash='dot')

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
                    'Date: %{x|%Y-%m-%d}<br>'
                    'Type: %{customdata[1]}<br>'
                    '%{customdata[2]}<extra></extra>'
                ),
                customdata=pd.concat([
                    ev['label'].fillna(''),
                    ev['type'].astype(str),
                    ev_desc,
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

    st.plotly_chart(fig, use_container_width=True)


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

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 5 States**")
        for i, (state, count) in enumerate(state_counts.head().items(), 1):
            percentage = (count / len(filtered_df)) * 100
            st.write(f"{i}. **{state}**: {count} ({percentage:.1f}%)")


def plot_status_pie(filtered_df):
    st.subheader("License Status Distribution")
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
                colors=CATEGORICAL_SEQUENCE,
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

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Status Breakdown**")
        for i, (status, count) in enumerate(status_counts.items()):
            percentage = (count / total) * 100
            color = CATEGORICAL_SEQUENCE[i % len(CATEGORICAL_SEQUENCE)]
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0;'>"
                f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
                f"background:{color};flex-shrink:0;'></span>"
                f"<span><b>{status}</b>: {count:,} ({percentage:.1f}%)</span>"
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

    st.plotly_chart(fig, use_container_width=True)

    if unmapped:
        st.caption(f"Unmapped state values (not shown on map): {', '.join(unmapped)}")


def plot_ca_county_map(filtered_df):
    """Plot a choropleth map of California showing total licenses per county."""
    if 'County' not in filtered_df.columns:
        st.info("No `County` column available for California county map.")
        return

    county_counts = filtered_df['County'].value_counts()

    all_county_names = list(CA_COUNTY_FIPS.keys())
    all_fips = list(CA_COUNTY_FIPS.values())
    fips = all_fips.copy()
    values = [0] * len(all_fips)
    hover_text = [f"{name}: 0" for name in all_county_names]

    fips_to_idx = {code: idx for idx, code in enumerate(fips)}
    unmapped = []

    for county, count in county_counts.items():
        code = to_county_fips(county)
        if code and code in fips_to_idx:
            idx = fips_to_idx[code]
            values[idx] = int(count)
            hover_text[idx] = f"{county}: {count}"
        elif code:
            fips.append(code)
            values.append(int(count))
            hover_text.append(f"{county}: {count}")
        else:
            unmapped.append(county)

    if not fips:
        st.info("No mappable California county values found for the map.")
        return

    fig = go.Figure(data=go.Choropleth(
        locations=fips,
        z=values,
        locationmode='geojson-id',
        geojson='https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json',
        colorscale=RED_SCALE,
        zmin=0,
        zmax=max(values) if values else 100,
        text=hover_text,
        marker_line_color=BORDER,
        marker_line_width=0.5,
        colorbar=dict(title='Licenses', ticks='outside', outlinewidth=0, thickness=14),
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

    st.plotly_chart(fig, use_container_width=True)

    if unmapped:
        st.caption(f"Unmapped county values (not shown on map): {', '.join(unmapped)}")


# ---------- Yearly retirements (expirations) ----------

def plot_yearly_retirements(filtered_df, start_year=None, end_year=None):
    """Bar chart of license expirations by year."""
    st.subheader("License Expirations (Retirements) by Year")

    if 'Expiration Date' not in filtered_df.columns:
        st.info("No Expiration Date column available.")
        return
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['Expiration Date']):
        st.info("Expiration Date is not in datetime format.")
        return

    df = filtered_df.dropna(subset=['Expiration Date']).copy()
    current_year = pd.Timestamp.now().year
    df['Exp_Year'] = df['Expiration Date'].dt.year.astype(int)

    df = df[df['Exp_Year'] <= current_year]
    if start_year is not None:
        df = df[df['Exp_Year'] >= int(start_year)]
    if end_year is not None:
        df = df[df['Exp_Year'] <= min(int(end_year), current_year)]

    yearly = df.groupby('Exp_Year').size().reset_index(name='Expired')
    yearly = yearly.sort_values('Exp_Year')

    if yearly.empty:
        st.info("No expiration data for the selected range.")
        return

    show_labels = len(yearly) <= BAR_LABEL_MAX_CATEGORIES

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly['Exp_Year'], y=yearly['Expired'],
        name='Licenses Expired', marker_color=EXPIRATION_BAR_COLOR,
        text=yearly['Expired'] if show_labels else None,
        textposition='outside' if show_labels else 'none',
        hovertemplate='<b>Year %{x}</b><br>Expired: %{y}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Licenses Expired",
        height=460,
        showlegend=False,
        hovermode='x unified',
        bargap=0.18,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Expiration Statistics"):
        c1, c2, c3 = st.columns(3)
        peak_idx = yearly['Expired'].idxmax()
        with c1:
            st.metric("Peak Year", int(yearly.loc[peak_idx, 'Exp_Year']))
        with c2:
            st.metric("Peak Expirations", int(yearly['Expired'].max()))
        with c3:
            st.metric("Avg per Year", f"{yearly['Expired'].mean():.1f}")


# ---------- Active licenses per year ----------

def plot_active_licenses_per_year(filtered_df, start_year=None, end_year=None):
    """Area chart showing how many licenses were active at the end of each year."""
    st.subheader("Active Licenses per Year")

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
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Active License Statistics"):
        c1, c2, c3 = st.columns(3)
        peak_idx = active_counts.index(max(active_counts))
        with c1:
            st.metric("Peak Year", years[peak_idx])
        with c2:
            st.metric("Peak Count", f"{max(active_counts):,}")
        with c3:
            st.metric("Current", f"{active_counts[-1]:,}" if active_counts else "N/A")


# ---------- Average license age per year ----------

def plot_avg_license_age_per_year(filtered_df, start_year=None, end_year=None):
    """Line chart of the average age of active licenses at the end of each year."""
    st.subheader("Average Age of Active Licenses by Year")

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
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)

    valid = [(y, a) for y, a in zip(years, avg_ages) if a is not None]
    if valid:
        ages_only = [a for _, a in valid]
        min_idx = ages_only.index(min(ages_only))
        max_idx = ages_only.index(max(ages_only))
        with st.expander("Average Age Statistics"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Youngest Cohort Year", f"{valid[min_idx][0]} ({valid[min_idx][1]:.1f} yrs)")
            with c2:
                st.metric("Oldest Cohort Year", f"{valid[max_idx][0]} ({valid[max_idx][1]:.1f} yrs)")
            with c3:
                st.metric("Most Recent Year", f"{valid[-1][0]} ({valid[-1][1]:.1f} yrs)")


def display_summary_metrics(filtered_df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}" if filtered_df is not None else "0")
    with col2:
        st.metric("States*", filtered_df['State'].nunique() if (filtered_df is not None and 'State' in filtered_df.columns) else 0)
    with col3:
        if filtered_df is not None and 'Expiration Date' in filtered_df.columns:
            current_date = pd.Timestamp.now()
            active_licenses = len(filtered_df[filtered_df['Expiration Date'] >= current_date])
            st.metric("Active Licenses", f"{active_licenses:,}")
        else:
            st.metric("Active Licenses", "0")
    with col4:
        if filtered_df is not None and 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
            date_range = f"{filtered_df['Original Issue Date'].dt.year.min()}–{filtered_df['Original Issue Date'].dt.year.max()}"
            st.metric("Year Range", date_range)
        else:
            st.metric("Year Range", "N/A")

    st.caption("*Includes U.S. territories (DC, GU, PR), Armed Forces jurisdictions (AP), and international license holders. Total count may exceed 50 states.")


# ---------- App ----------
def main():
    st.set_page_config(
        page_title="SE License Dashboard",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply unified visual theme
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    st.title("🏗️ Structural Engineers License Dashboard")
    st.markdown(
        '<p class="dashboard-subtitle">Interactive view of California structural engineer license data from the DCA public record.</p>',
        unsafe_allow_html=True,
    )

    st.sidebar.title("Data Source")

    default_csv = "ProfEngrsLandSurvyrsGeologist_Data00.xls_structural_engineers_cleaned"
    src = resolve_data_source(default_csv)

    df = None
    if src:
        with st.spinner("Loading cleaned data..."):
            df = load_cleaned_data(src, _file_mtime=_file_hash(src))
            df = parse_dates(df)
            if df is not None:
                st.sidebar.success(f"Loaded: {Path(src).name}")
                if 'Dashboard_Updated' in df.columns:
                    latest = df['Dashboard_Updated'].dropna()
                    if not latest.empty:
                        st.sidebar.caption(f"Data processed: {latest.iloc[0]}")

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

    # Dev-only column inspector — enable with ?debug=1 in the URL
    if st.query_params.get('debug') == '1':
        with st.expander("Data Column Information (debug)"):
            st.write("**Available columns:**")
            for col in df.columns:
                st.write(f"- `{col}` ({df[col].dtype})")
            st.write(f"\n**Data shape:** {df.shape[0]} rows x {df.shape[1]} columns")

    _init_filters_from_url(df)

    filters = create_filters(df)

    st.sidebar.markdown("---")
    if st.sidebar.button("Share current filters"):
        st.query_params.from_dict(_build_share_params(filters))
        st.sidebar.success("URL updated — copy it from your browser address bar.")

    full_df = df.copy()
    filtered_df = apply_filters(df, filters)

    # Filter summary chip bar
    _render_filter_chips(filters, df)

    # Metrics
    display_summary_metrics(filtered_df)

    if filtered_df is None or filtered_df.empty:
        st.warning("No records match the selected filters. Try adjusting your filter criteria.")
        return

    create_visualizations(
        filtered_df,
        filters.get('show_events', False),
        filters.get('start_year'),
        filters.get('end_year'),
        filters.get('comparison_state'),
        full_df,
    )


if __name__ == "__main__":
    main()
