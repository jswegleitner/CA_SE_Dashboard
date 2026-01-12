# -*- coding: utf-8 -*-
"""
Structural Engineers Dashboard - Streamlit Version
Loads pre-processed CSV data and creates interactive dashboard

Created on Fri Sep 12 19:59:58 2025
@author: jwegleitner

Use these commands in the terminal to run the app locally:
    Opens browser: & 'C:/Users/jwegleitner/Miniforge3/python.exe' -m streamlit run ./SE_Dashboard_v7.py
    Run headless (no browser): & 'C:/Users/jwegleitner/Miniforge3/python.exe' -m streamlit run ./SE_Dashboard_v7.py --server.headless true


"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re
import textwrap
from typing import Optional

# ---------- Data loading ----------
@st.cache_data
def load_cleaned_data(src):
    """
    Load pre-cleaned CSV data with caching for better performance.
    `src` can be a local path or a URL.
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

    # Fallback for local use: try multiple possible filenames
    possible_files = [
        default_local_filename,
        default_local_filename + ".csv",
        "structural_engineers_cleaned.csv",
        "ProfEngrsLandSurvyrsGeologist_Data00.csv",
        "ProfEngrsLandSurvyrsGeologist_Data00_structural_engineers_cleaned.csv"
    ]

    # Use script directory for more reliable path resolution on Streamlit Cloud
    current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    
    # First try the specific filenames we know
    for filename in possible_files:
        local_csv_path = current_dir / filename
        if local_csv_path.exists():
            return str(local_csv_path)

    # Look for CSV files with required license columns (not timeline or other data)
    csv_files = list(current_dir.glob("*.csv"))
    for csv_path in csv_files:
        # Skip known non-license files
        if csv_path.name.lower() in ['timeline_events.csv', 'license history table.csv']:
            continue
        # Check if it has license data columns
        try:
            sample_df = pd.read_csv(csv_path, nrows=1)
            if 'License Type' in sample_df.columns or 'License Number' in sample_df.columns:
                return str(csv_path)
        except Exception:
            continue

# ---------- Optional timeline events (for chart annotations) ----------
@st.cache_data
def load_timeline_events() -> Optional[pd.DataFrame]:
    """
    Load optional timeline events used to annotate time-series charts.
    Supported locations (in priority order):
    - st.secrets['EVENTS_PATH'] or st.secrets['EVENTS_URL']
    - ./timeline_events.csv
    - ./data/timeline_events.csv

    Expected columns (case-insensitive):
    - start_date (required)
    - end_date (optional)
    - label (short title shown on chart)
    - description (longer hover text)
    - type (category for coloring/filtering)
    """
    # Try secrets first
    try:
        if 'EVENTS_PATH' in st.secrets:
            df = pd.read_csv(st.secrets['EVENTS_PATH'])
        elif 'EVENTS_URL' in st.secrets:
            df = pd.read_csv(st.secrets['EVENTS_URL'])
        else:
            df = None
    except Exception:
        df = None

    # Fallbacks in repo
    if df is None:
        # Use script directory for reliable path resolution
        script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        for candidate in [
            script_dir / 'timeline_events.csv',
            script_dir / 'data' / 'timeline_events.csv',
        ]:
            if candidate.exists():
                try:
                    df = pd.read_csv(candidate)
                    break
                except Exception:
                    pass

    if df is None:
        return None

    # Normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    def col(name):
        # Return actual column name matching a case-insensitive key
        for k, v in cols.items():
            if k == name:
                return v
        return None

    s_col = col('start_date') or col('start') or col('date')
    e_col = col('end_date') or col('end')
    l_col = col('label') or col('title')
    d_col = col('description') or col('details')
    code_col = col('code') or col('codes')
    fmt_col = col('format')
    notes_col = col('notes')
    t_col = col('type') or col('category')

    if not s_col:
        return None

    # Build normalized frame
    out = pd.DataFrame()
    out['start_date'] = pd.to_datetime(df[s_col], errors='coerce')
    if e_col and e_col in df.columns:
        out['end_date'] = pd.to_datetime(df[e_col], errors='coerce')
    else:
        out['end_date'] = pd.NaT
    out['label'] = df[l_col] if l_col in df.columns else ''
    out['type'] = df[t_col] if t_col in df.columns else 'Event'

    # Prefer structured fields if present; else try to parse description
    if any(x in df.columns for x in filter(None, [code_col, fmt_col, notes_col])):
        out['code'] = df[code_col] if code_col in df.columns else ''
        out['format'] = df[fmt_col] if fmt_col in df.columns else ''
        out['notes'] = df[notes_col] if notes_col in df.columns else ''
        # Also provide a combined description for legacy callers
        out['description'] = (
            'Codes: ' + out['code'].fillna('') + '\n' +
            'Format: ' + out['format'].fillna('') + '\n' +
            'Notes: ' + out['notes'].fillna('')
        ).str.strip()
    else:
        # Keep original description, but try splitting into fields for downstream formatting
        desc_series = df[d_col] if d_col in df.columns else pd.Series([''] * len(df))
        out['description'] = desc_series
        # Quick token extraction for code/format/notes within single description text
        def extract_token(s: str, token: str):
            import re as _re
            if not isinstance(s, str):
                return ''
            # Capture text after token label until next token or end
            pattern = _re.compile(rf"{token}\s*(.*?)(?:(?:\n|\r)\s*(Codes:|Format:|Notes:)|$)", _re.IGNORECASE | _re.DOTALL)
            m = pattern.search(s)
            if not m:
                return ''
            return m.group(1).strip()

        out['code'] = desc_series.apply(lambda x: extract_token(x, 'Codes:'))
        out['format'] = desc_series.apply(lambda x: extract_token(x, 'Format:'))
        out['notes'] = desc_series.apply(lambda x: extract_token(x, 'Notes:'))

    # Drop rows without valid start dates
    out = out.dropna(subset=['start_date']).reset_index(drop=True)
    return out

# ---------- Helpers ----------
def _sanitize_hover(text: str) -> str:
    """Escape characters that can interfere with Plotly's hover template parsing."""
    if text is None:
        return ''
    # Escape braces which Plotly might try to interpret
    return str(text).replace('{', '&#123;').replace('}', '&#125;')

def _wrap_words(words, width: int, first_prefix: str = '', subsequent_prefix: str = '  ') -> list:
    """Greedy word wrap returning list of lines with given width constraints."""
    lines = []
    current = first_prefix.rstrip()
    for w in words:
        w = w.strip()
        if not w:
            continue
        if not current:
            current = first_prefix.rstrip()
        tentative = (current + ' ' + w).strip() if current else w
        if len(tentative) > width and current:
            lines.append(current)
            current = f"{subsequent_prefix}{w}".rstrip()
        else:
            current = tentative
    if current:
        lines.append(current)
    return lines

def _wrap_field(label: str, content: str, width: int = 55) -> str:
    """Wrap a single field (label + content) producing <br>-joined lines."""
    if content is None:
        return ''
    txt = str(content).strip()
    if not txt:
        return ''
    words = txt.replace('\r\n', ' ').replace('\n', ' ').split()
    lines = _wrap_words(words, width=width, first_prefix=f"{label} ", subsequent_prefix='  ')
    return '<br>'.join(_sanitize_hover(l) for l in lines)

def _wrap_legacy(raw: str, width: int = 55) -> str:
    """Wrap a legacy combined description string by detecting tokens and wrapping segments."""
    if raw is None:
        return ''
    s = str(raw).strip()
    if not s:
        return ''
    tokens = ['Codes:', 'Format:', 'Breadth:', 'Depth:', 'Notes:', 'Vertical:', 'Lateral:', 'Updated:', 'Components:']
    # Ensure each token starts new line for splitting
    for tok in tokens:
        s = re.sub(rf'(?<!^)\s*{re.escape(tok)}', f'\n{tok}', s)
    segments = []
    for seg in s.split('\n'):
        seg = seg.strip()
        if not seg:
            continue
        # Find label if present
        m = re.match(r'^(\w+:)\s*(.*)$', seg)
        if m:
            label, content = m.group(1), m.group(2)
            segments.append(_wrap_field(label, content, width=width))
        else:
            # Generic wrapping
            words = seg.split()
            lines = _wrap_words(words, width=width, first_prefix='', subsequent_prefix='  ')
            segments.append('<br>'.join(_sanitize_hover(l) for l in lines))
    return '<br>'.join(segments)

def format_event_description(desc: str) -> str:
    """Public interface for wrapping legacy description strings into hover-friendly HTML."""
    return _wrap_legacy(desc, width=55)

def build_event_description_from_fields(code: str, fmt: str, notes: str) -> str:
    parts = []
    if code and str(code).strip():
        parts.append(_wrap_field('Codes:', code, width=55))
    if fmt and str(fmt).strip():
        parts.append(_wrap_field('Format:', fmt, width=55))
    if notes and str(notes).strip():
        parts.append(_wrap_field('Notes:', notes, width=55))
    return '<br>'.join(parts)

# ---------- Filters & visuals (updated with Plotly) ----------
def create_filters(df):
    filters = {}

    st.sidebar.header("🔍 Data Filters")
    # Timeline events toggle at the very top of the Data Filters section
    events_df_present = load_timeline_events() is not None
    filters['show_events'] = st.sidebar.checkbox(
        "Show timeline events",
        value=events_df_present,
        help="Toggle overlays of exam/code history on charts (if available).",
        key='show_events_toggle'
    )

    st.sidebar.subheader("State")
    states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    filters['states'] = st.sidebar.multiselect(
        "Select States:",
        options=states,
        default=states,
        key='state_filter',
        help="Select one or more states. Leave empty to include all states."
    )
    
    # Comparison state overlay
    filters['comparison_state'] = st.sidebar.selectbox(
        "Compare with State (overlay on bar chart):",
        options=['None'] + states,
        index=0,
        key='comparison_state',
        help="Select a single state to overlay its data on the time series bar chart"
    )

    st.sidebar.subheader("License Status")
    statuses = sorted(df['License Status'].dropna().unique().tolist()) if 'License Status' in df.columns else []
    filters['statuses'] = st.sidebar.multiselect(
        "Select License Statuses:",
        options=statuses,
        default=statuses,
        key='status_filter',
        help="Select one or more license statuses. Leave empty to include all statuses."
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
        key='date_range_slider'
    )

    expiration_options = ['Active Only', 'Expired Only']
    filters['expiration_types'] = st.sidebar.multiselect(
        "License Expiration Status:",
        options=expiration_options,
        default=expiration_options,
        key='expired_filter',
        help="Active = not yet expired, Expired = past expiration date. Select both for all licenses."
    )

    # Small hint if no events file present
    if not events_df_present:
        st.sidebar.caption("Add `timeline_events.csv` with columns: start_date, end_date (optional), label, description, type.")
    return filters

def apply_filters(df, filters):
    if df is None or df.empty:
        return df

    filtered_df = df.copy()

    # State filter
    if 'State' in filtered_df.columns and filters['states']:
        filtered_df = filtered_df[filtered_df['State'].isin(filters['states'])]

    # Status filter
    if 'License Status' in filtered_df.columns and filters['statuses']:
        filtered_df = filtered_df[filtered_df['License Status'].isin(filters['statuses'])]

    # Date range filter
    if 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        mask = (
            (filtered_df['Original Issue Date'].dt.year >= filters['start_year']) &
            (filtered_df['Original Issue Date'].dt.year <= filters['end_year'])
        )
        filtered_df = filtered_df[mask]

    # Expiration filter
    if 'Expiration Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Expiration Date']):
        current_date = pd.Timestamp.now()
        if filters['expiration_types']:
            # Only include rows that match any selected expiration category
            inc_active = 'Active Only' in filters['expiration_types']
            inc_expired = 'Expired Only' in filters['expiration_types']
            mask = pd.Series(False, index=filtered_df.index)
            if inc_active:
                mask = mask | (filtered_df['Expiration Date'] >= current_date)
            if inc_expired:
                mask = mask | (filtered_df['Expiration Date'] < current_date)
            filtered_df = filtered_df[mask]

    return filtered_df

def create_visualizations(filtered_df, show_events: bool, start_year: Optional[int] = None, end_year: Optional[int] = None, comparison_state: str = None, full_df = None):
    if filtered_df is None or filtered_df.empty:
        st.warning("No data to visualize with current filters.")
        return
    # High-level visual section: each visual is its own function to make
    # independent editing and testing straightforward.
    st.subheader("📈 Licenses Issued Over Time")

    col1, _ = st.columns([1, 3])
    with col1:
        bucket_size = st.selectbox("Time Grouping:", ["Yearly", "Half-Yearly", "Quarterly", "Monthly"]) 
        lock_y = st.checkbox("Lock Y-axis when zooming", value=True, help="When checked, zooming/panning will only affect the X axis; Y axis will remain fixed.")

    # Load events (if present)
    events_df = load_timeline_events()
    # Time series
    plot_time_series(filtered_df, bucket_size, lock_y, events_df=events_df, show_events=show_events, start_year=start_year, end_year=end_year, comparison_state=comparison_state, full_df=full_df)
    # Line chart variant
    plot_time_series_line(filtered_df, bucket_size, lock_y, events_df=events_df, show_events=show_events, start_year=start_year, end_year=end_year)
    st.markdown("---")

    # By State
    if 'State' in filtered_df.columns and filtered_df['State'].nunique() > 1:
        plot_state_counts(filtered_df)
        # Also render a US choropleth map showing counts by state
        plot_us_map(filtered_df)
        st.markdown("---")
    
    # California county map - show whenever there's CA data
    if 'State' in filtered_df.columns and 'County' in filtered_df.columns:
        ca_data = filtered_df[filtered_df['State'] == 'CA']
        if not ca_data.empty:
            st.subheader("🗺️ California Licenses by County")
            plot_ca_county_map(ca_data)
            st.markdown("---")

    # Status pie chart
    if 'License Status' in filtered_df.columns and filtered_df['License Status'].nunique() > 1:
        plot_status_pie(filtered_df)
        st.markdown("---")


def plot_time_series(filtered_df, bucket_size: str, lock_y: bool = True, events_df: Optional[pd.DataFrame] = None, show_events: bool = False, start_year: Optional[int] = None, end_year: Optional[int] = None, comparison_state: str = None, full_df = None):
    if 'Original Issue Date' not in filtered_df.columns or not pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
        st.info("Original Issue Date column not found or not datetime.")
        return

    df_viz = filtered_df.copy()
    if bucket_size == "Yearly":
        df_viz['Period'] = df_viz['Original Issue Date'].dt.to_period('Y')
        title_period = "Yearly"
    elif bucket_size == "Half-Yearly":
        df_viz['Year'] = df_viz['Original Issue Date'].dt.year
        df_viz['Half'] = ((df_viz['Original Issue Date'].dt.month - 1) // 6) + 1
        df_viz['Period'] = df_viz['Year'].astype(str) + '-H' + df_viz['Half'].astype(str)
        title_period = "Half-Yearly"
    else:
        df_viz['Period'] = df_viz['Original Issue Date'].dt.to_period('Q')
        title_period = "Quarterly"

    if bucket_size == "Monthly":
        # Overwrite Period for monthly grouping if requested
        df_viz['Period'] = df_viz['Original Issue Date'].dt.to_period('M')
        title_period = "Monthly"

    # Build full period range to include empty buckets (e.g., ensure 1980 appears even if zero)
    if bucket_size == "Yearly":
        full_index = None
        if start_year is not None and end_year is not None:
            full_index = pd.period_range(start=f"{int(start_year)}-01-01", end=f"{int(end_year)}-12-31", freq='Y')
        license_counts = df_viz.groupby('Period').size()
        if full_index is not None:
            license_counts = license_counts.reindex(full_index, fill_value=0)
        license_counts = license_counts.sort_index()
    elif bucket_size == "Monthly":
        full_index = None
        if start_year is not None and end_year is not None:
            full_index = pd.period_range(start=f"{int(start_year)}-01-01", end=f"{int(end_year)}-12-31", freq='M')
        license_counts = df_viz.groupby('Period').size()
        if full_index is not None:
            license_counts = license_counts.reindex(full_index, fill_value=0)
        license_counts = license_counts.sort_index()
    elif bucket_size == "Half-Yearly":
        license_counts = df_viz.groupby('Period').size()
        if start_year is not None and end_year is not None:
            labels = []
            for y in range(int(start_year), int(end_year) + 1):
                labels.append(f"{y}-H1")
                labels.append(f"{y}-H2")
            license_counts = license_counts.reindex(labels, fill_value=0)
        license_counts = license_counts.sort_index()
    else:  # Quarterly
        full_index = None
        if start_year is not None and end_year is not None:
            full_index = pd.period_range(start=f"{int(start_year)}-01-01", end=f"{int(end_year)}-12-31", freq='Q')
        license_counts = df_viz.groupby('Period').size()
        if full_index is not None:
            license_counts = license_counts.reindex(full_index, fill_value=0)
        license_counts = license_counts.sort_index()

    if license_counts.empty:
        st.info("No time-series data for the selected filters.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(p) for p in license_counts.index],
        y=license_counts.values,
        name='Main Data',
        marker_color='steelblue',
        text=license_counts.values,
        textposition='outside',
        hovertemplate='<b>Period:</b> %{x}<br><b>Licenses:</b> %{y}<extra></extra>'
    ))
    
    # Add comparison state overlay if selected
    if comparison_state and comparison_state != 'None' and full_df is not None:
        comp_df = full_df[full_df['State'] == comparison_state].copy()
        # Apply same date filters as main data
        if start_year is not None and end_year is not None:
            comp_df = comp_df[
                (comp_df['Original Issue Date'].dt.year >= start_year) &
                (comp_df['Original Issue Date'].dt.year <= end_year)
            ]
        
        # Apply same bucketing logic
        if bucket_size == "Yearly":
            comp_df['Period'] = comp_df['Original Issue Date'].dt.to_period('Y')
        elif bucket_size == "Half-Yearly":
            comp_df['Year'] = comp_df['Original Issue Date'].dt.year
            comp_df['Half'] = ((comp_df['Original Issue Date'].dt.month - 1) // 6) + 1
            comp_df['Period'] = comp_df['Year'].astype(str) + '-H' + comp_df['Half'].astype(str)
        elif bucket_size == "Monthly":
            comp_df['Period'] = comp_df['Original Issue Date'].dt.to_period('M')
        else:  # Quarterly
            comp_df['Period'] = comp_df['Original Issue Date'].dt.to_period('Q')
        
        # Reindex to match main data categories with zeros
        comp_counts = comp_df.groupby('Period').size()
        comp_counts = comp_counts.reindex(license_counts.index, fill_value=0)
        
        fig.add_trace(go.Bar(
            x=[str(p) for p in comp_counts.index],
            y=comp_counts.values,
            name=f'{comparison_state} (Comparison)',
            marker_color='coral',
            text=comp_counts.values,
            textposition='outside',
            opacity=0.7,
            hovertemplate=f'<b>Period:</b> %{{x}}<br><b>{comparison_state} Licenses:</b> %{{y}}<extra></extra>'
        ))

    fig.update_layout(
        title=f"Structural Engineer Licenses Issued Over Time ({title_period})",
        xaxis_title="Period",
        yaxis_title="Number of Licenses",
        showlegend=True if (comparison_state and comparison_state != 'None') else False,
        height=500,
        hovermode='x unified',
        dragmode='zoom',
        hoverlabel=dict(align='left'),
        barmode='group'
    )

    num_periods = len(license_counts)
    if num_periods > 20:
        fig.update_xaxes(tickangle=90)
    elif num_periods > 10:
        fig.update_xaxes(tickangle=45)

    # Ensure X axis is zoomable but optionally lock Y axis so zoom only affects X
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=lock_y)

    # Ensure category axis order stays tied to license data only
    categories = [str(p) for p in license_counts.index]
    fig.update_xaxes(categoryorder='array', categoryarray=categories)

    # Overlay timeline events as markers aligned to the current bucket
    if show_events and (events_df is not None) and (not events_df.empty):
        # Map event start dates to the current bucket label used on x-axis
        def to_bucket_label(dt):
            if pd.isna(dt):
                return None
            if bucket_size == 'Yearly':
                return str(pd.Period(dt, 'Y'))
            if bucket_size == 'Half-Yearly':
                year = dt.year
                half = 1 if dt.month <= 6 else 2
                return f"{year}-H{half}"
            if bucket_size == 'Quarterly':
                return str(pd.Period(dt, 'Q'))
            if bucket_size == 'Monthly':
                return str(pd.Period(dt, 'M'))
            return None

        ev = events_df.copy()
        # Restrict events to selected year range if provided
        if start_year is not None and end_year is not None:
            ev = ev[(ev['start_date'].dt.year >= start_year) & (ev['start_date'].dt.year <= end_year)]
        ev['bucket'] = ev['start_date'].apply(to_bucket_label)
        ev = ev.dropna(subset=['bucket'])
        # Only keep events whose bucket exists in the data categories so axis isn't extended by events
        ev = ev[ev['bucket'].isin(categories)]
        if not ev.empty:
            # Place markers at the bottom of the chart (slightly below 0)
            y_max = max(license_counts.values) if len(license_counts) else 0
            pad = max(1, y_max * 0.06)
            # Extend y-axis to include more negative space for markers and text labels
            fig.update_yaxes(range=[-pad * 2.5, y_max * 1.05])

            # Year-only text in red; full details in hover
            text_year = ev['start_date'].dt.year.astype('Int64').astype(str).fillna('')

            # Prefer structured fields if present
            if all(col in ev.columns for col in ['code', 'format', 'notes']):
                ev_desc = [
                    build_event_description_from_fields(c, f, n)
                    for c, f, n in zip(ev['code'].fillna(''), ev['format'].fillna(''), ev['notes'].fillna(''))
                ]
            else:
                ev_desc = ev['description'].fillna('').map(format_event_description)
            # Ensure Series type for concat
            if not isinstance(ev_desc, pd.Series):
                ev_desc = pd.Series(ev_desc, index=ev.index)

            fig.add_trace(go.Scatter(
                x=ev['bucket'],
                y=[-pad * 0.8] * len(ev),
                mode='markers+text',
                name='Events',
                marker=dict(symbol='triangle-up', size=10, color='crimson'),
                text=text_year,
                textposition='bottom center',
                textfont=dict(color='crimson'),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'  # label
                    'Date: %{customdata[1]}<br>'
                    'Type: %{customdata[2]}<br>'
                    '%{customdata[3]}<extra></extra>'
                ),
                customdata=pd.concat([
                    ev['label'].fillna(''),
                    ev['start_date'].dt.strftime('%Y-%m-%d'),
                    ev['type'].astype(str),
                    ev_desc
                ], axis=1).values
            ))

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Time Series Statistics"):
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Peak Period", str(license_counts.idxmax()))
        with c2: st.metric("Peak Count", int(license_counts.max()))
        with c3: st.metric("Average per Period", f"{license_counts.mean():.1f}")


def plot_time_series_line(filtered_df, bucket_size: str, lock_y: bool = True, events_df: Optional[pd.DataFrame] = None, show_events: bool = False, start_year: Optional[int] = None, end_year: Optional[int] = None):
    """
    Line chart of licenses over time. X axis is a datetime representing the start of the
    selected period (year/half/quarter/month) and Y is the number of licenses issued.
    """
    # New behavior: plot each individual license by its Original Issue Date on the X axis
    # and the cumulative license count on the Y axis so the line monotonically increases.
    if 'Original Issue Date' not in filtered_df.columns:
        st.info("Original Issue Date column not found.")
        return

    df_line = filtered_df.copy()
    df_line['Original Issue Date'] = pd.to_datetime(df_line['Original Issue Date'], errors='coerce')
    df_line = df_line.dropna(subset=['Original Issue Date']).sort_values('Original Issue Date')

    if df_line.empty:
        st.info("No time-series data for the selected filters.")
        return

    # Prefer plotting the actual license number field (if present) on Y vs the date on X.
    # Look for common license-number-like column names.
    def find_license_column(columns):
        patterns = [
            r'license.*num', r'licen[s|c]e.*num', r'lic.*num', r'licenseid', r'license.*id',
            r'licno', r'licence', r'registration.*num', r'regnum', r'license#', r'lic#'
        ]
        for col in columns:
            name = re.sub(r'[^a-z0-9]', '', col.lower())
            for pat in patterns:
                if re.search(pat, name):
                    return col
        # fallback common exact matches
        for col in columns:
            if col.lower() in ('license number', 'license_number', 'licensenumber', 'license no', 'license no.','license'):
                return col
        return None

    license_col = find_license_column(df_line.columns)
    if license_col:
        yvals = pd.to_numeric(df_line[license_col], errors='coerce')
        # If values are non-numeric, warn and fall back to cumulative index
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
        line=dict(color='steelblue'),
        marker=dict(size=4),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Structural Engineer Licenses Over Time (individual licenses)",
        xaxis_title="Original Issue Date",
        yaxis_title=y_label,
        height=420,
        hovermode='x unified',
        dragmode='zoom',
        hoverlabel=dict(align='left')
    )

    # Allow zooming on X; optionally lock Y so it doesn't rescale
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=lock_y)

    # Constrain X-axis strictly to the filtered license dates (ignore events for autorange)
    try:
        x_min = pd.to_datetime(f"{start_year}-01-01") if start_year is not None else df_line['Original Issue Date'].min()
        x_max = pd.to_datetime(f"{end_year}-12-31") if end_year is not None else df_line['Original Issue Date'].max()
        if pd.notna(x_min) and pd.notna(x_max):
            fig.update_xaxes(range=[x_min, x_max])
    except Exception:
        pass

    # Overlay timeline events as vertical lines on the datetime X axis
    if show_events and (events_df is not None) and (not events_df.empty):
        # Create consistent colors by event type
        type_styles = {}
        palette = ['crimson', 'darkorange', 'seagreen', 'mediumpurple', 'teal', 'goldenrod']
        for i, t in enumerate(events_df['type'].astype(str).unique()):
            type_styles[t] = dict(color=palette[i % len(palette)], dash='dot')

        # Reserve headroom for year labels
        y_max = float(pd.Series(yvals).max()) if len(yvals) else 0.0
        y_min = float(pd.Series(yvals).min()) if len(yvals) else 0.0
        head = max(1.0, y_max * 0.08)
        # Extend both top and bottom to accommodate labels
        fig.update_yaxes(range=[y_min - head * 0.5, y_max + head * 2.0])

        # Add vlines and year-only annotations
        ev = events_df.copy()
        ev = ev.dropna(subset=['start_date'])
        # Restrict events to the same visible date range so they don't expand axes
        if start_year is not None and end_year is not None:
            ev = ev[(ev['start_date'].dt.year >= start_year) & (ev['start_date'].dt.year <= end_year)]
        if not ev.empty:
            # Add all invisible markers at a fixed y just above the line to enable hover
            # Prefer structured fields if present
            if all(col in ev.columns for col in ['code', 'format', 'notes']):
                ev_desc = [
                    build_event_description_from_fields(c, f, n)
                    for c, f, n in zip(ev['code'].fillna(''), ev['format'].fillna(''), ev['notes'].fillna(''))
                ]
            else:
                ev_desc = ev['description'].fillna('').map(format_event_description)
            # Ensure Series type for concat
            if not isinstance(ev_desc, pd.Series):
                ev_desc = pd.Series(ev_desc, index=ev.index)

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
                    ev_desc
                ], axis=1).values,
                showlegend=False
            ))

            for _, r in ev.iterrows():
                t = str(r.get('type', 'Event'))
                color = type_styles.get(t, dict(color='gray'))['color']
                dash = type_styles.get(t, dict(dash='dot'))['dash']
                xdt = r['start_date']
                fig.add_vline(x=xdt, line=dict(color=color, dash=dash, width=1))
                # Year-only label in red at the bottom
                try:
                    year_txt = str(pd.to_datetime(xdt).year)
                except Exception:
                    year_txt = ''
                if year_txt:
                    fig.add_annotation(x=xdt, y=y_min - head * 0.3, text=year_txt,
                                       showarrow=False, font=dict(color='crimson', size=10))

    st.plotly_chart(fig, use_container_width=True)


def plot_state_counts(filtered_df):
    st.subheader("🗺️ Licenses by State")
    state_counts = filtered_df['State'].value_counts()

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=state_counts.index,
            y=state_counts.values,
            name='Licenses by State',
            marker_color='lightcoral',
            text=state_counts.values,
            textposition='outside',
            hovertemplate='<b>State:</b> %{x}<br><b>Licenses:</b> %{y}<extra></extra>'
        ))

        fig.update_layout(
            title="Number of Structural Engineer Licenses by State",
            xaxis_title="State",
            yaxis_title="Number of Licenses",
            showlegend=False,
            height=500,
            xaxis_tickangle=45
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 5 States:**")
        for i, (state, count) in enumerate(state_counts.head().items(), 1):
            percentage = (count / len(filtered_df)) * 100
            st.write(f"{i}. **{state}**: {count} ({percentage:.1f}%)")


def plot_status_pie(filtered_df):
    st.subheader("📋 License Status Distribution")
    c1, c2 = st.columns([1, 1])
    status_counts = filtered_df['License Status'].value_counts()

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            textinfo='label+percent',
            textposition='auto',
            marker=dict(line=dict(color='#000000', width=2))
        ))

        fig.update_layout(
            title="License Status Distribution",
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Status Breakdown:**")
        for status, count in status_counts.items():
            percentage = (count / len(filtered_df)) * 100
            st.write(f"**{status}**: {count} ({percentage:.1f}%)")


def plot_us_map(filtered_df):
    """
    Plot a choropleth map of the United States showing total licenses per state.
    The function attempts to map common state names to USPS state codes.
    """
    if 'State' not in filtered_df.columns:
        st.info("No `State` column available for US map.")
        return

    # Mapping of full state names to USPS codes
    us_state_to_abbrev = {
        'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
        'Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID',
        'Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA',
        'Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS',
        'Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
        'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
        'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
        'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
        'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY','District Of Columbia':'DC','District of Columbia':'DC'
    }

    def to_state_code(x: str):
        if pd.isna(x):
            return None
        s = str(x).strip()
        # If already a 2-letter code
        if len(s) == 2 and s.upper() in us_state_to_abbrev.values():
            return s.upper()
        # Try title-case full name
        key = s.title()
        if key in us_state_to_abbrev:
            return us_state_to_abbrev[key]
        # Try common abbreviations passed in mixed case
        if s.upper() in us_state_to_abbrev.values():
            return s.upper()
        return None

    state_counts = filtered_df['State'].value_counts()
    
    # Start with all US states set to 0
    all_state_codes = list(us_state_to_abbrev.values())
    codes = all_state_codes.copy()
    values = [0] * len(all_state_codes)
    hover_text = [f"{code}: 0" for code in all_state_codes]
    
    # Build a mapping from code to index
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    unmapped = []

    # Update with actual counts
    for state, count in state_counts.items():
        code = to_state_code(state)
        if code and code in code_to_idx:
            idx = code_to_idx[code]
            values[idx] = int(count)
            hover_text[idx] = f"{state}: {count}"
        elif code:
            # State code found but not in our list (shouldn't happen)
            codes.append(code)
            values.append(int(count))
            hover_text.append(f"{state}: {count}")
        else:
            unmapped.append(state)

    if not codes:
        st.info("No mappable US state values found for the map.")
        return

    # Cap the color scale to 300 at the high end for better visual distinction
    fig = go.Figure(data=go.Choropleth(
        locations=codes,
        z=values,
        locationmode='USA-states',
        colorscale='Blues',
        zmin=0,
        zmax=300,
        text=hover_text,
        marker_line_color='black',
        colorbar=dict(title='Licenses', ticks='outside')
    ))

    fig.update_layout(
        title_text='🗺️ CA SE licenses by Resident State',
        geo_scope='usa',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    if unmapped:
        st.caption(f"Unmapped state values (not shown on map): {', '.join(unmapped)}")


def plot_ca_county_map(filtered_df):
    """
    Plot a choropleth map of California showing total licenses per county.
    Attempts to map county names to California county FIPS codes.
    """
    if 'County' not in filtered_df.columns:
        st.info("No `County` column available for California county map.")
        return

    # California county name to FIPS code mapping
    ca_county_fips = {
        'Alameda': '06001', 'Alpine': '06003', 'Amador': '06005', 'Butte': '06007',
        'Calaveras': '06009', 'Colusa': '06011', 'Contra Costa': '06013', 'Del Norte': '06015',
        'El Dorado': '06017', 'Fresno': '06019', 'Glenn': '06021', 'Humboldt': '06023',
        'Imperial': '06025', 'Inyo': '06027', 'Kern': '06029', 'Kings': '06031',
        'Lake': '06033', 'Lassen': '06035', 'Los Angeles': '06037', 'Madera': '06039',
        'Marin': '06041', 'Mariposa': '06043', 'Mendocino': '06045', 'Merced': '06047',
        'Modoc': '06049', 'Mono': '06051', 'Monterey': '06053', 'Napa': '06055',
        'Nevada': '06057', 'Orange': '06059', 'Placer': '06061', 'Plumas': '06063',
        'Riverside': '06065', 'Sacramento': '06067', 'San Benito': '06069', 'San Bernardino': '06071',
        'San Diego': '06073', 'San Francisco': '06075', 'San Joaquin': '06077', 'San Luis Obispo': '06079',
        'San Mateo': '06081', 'Santa Barbara': '06083', 'Santa Clara': '06085', 'Santa Cruz': '06087',
        'Shasta': '06089', 'Sierra': '06091', 'Siskiyou': '06093', 'Solano': '06095',
        'Sonoma': '06097', 'Stanislaus': '06099', 'Sutter': '06101', 'Tehama': '06103',
        'Trinity': '06105', 'Tulare': '06107', 'Tuolumne': '06109', 'Ventura': '06111',
        'Yolo': '06113', 'Yuba': '06115'
    }

    def to_county_fips(x: str):
        if pd.isna(x):
            return None
        s = str(x).strip()
        # Try title case
        key = s.title()
        if key in ca_county_fips:
            return ca_county_fips[key]
        # Try removing " County" suffix
        cleaned = re.sub(r'\s+County$', '', key, flags=re.IGNORECASE).title()
        if cleaned in ca_county_fips:
            return ca_county_fips[cleaned]
        return None

    county_counts = filtered_df['County'].value_counts()
    
    # Start with all CA counties set to 0
    all_county_names = list(ca_county_fips.keys())
    all_fips = list(ca_county_fips.values())
    fips = all_fips.copy()
    values = [0] * len(all_fips)
    hover_text = [f"{name}: 0" for name in all_county_names]
    
    # Build a mapping from FIPS to index
    fips_to_idx = {code: idx for idx, code in enumerate(fips)}
    unmapped = []

    # Update with actual counts
    for county, count in county_counts.items():
        code = to_county_fips(county)
        if code and code in fips_to_idx:
            idx = fips_to_idx[code]
            values[idx] = int(count)
            hover_text[idx] = f"{county}: {count}"
        elif code:
            # County code found but not in our list (shouldn't happen)
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
        colorscale='Reds',
        zmin=0,
        zmax=max(values) if values else 100,
        text=hover_text,
        marker_line_color='black',
        colorbar=dict(title='Licenses', ticks='outside')
    ))

    fig.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig.update_layout(
        title_text='🗺️ CA SE Licenses by County',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    if unmapped:
        st.caption(f"Unmapped county values (not shown on map): {', '.join(unmapped)}")


def display_summary_metrics(filtered_df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{len(filtered_df):,}" if filtered_df is not None else "0")
    with col2:
        st.metric("🗺️ States*", filtered_df['State'].nunique() if (filtered_df is not None and 'State' in filtered_df.columns) else 0)
    with col3:
        if filtered_df is not None and 'Expiration Date' in filtered_df.columns:
            current_date = pd.Timestamp.now()
            active_licenses = len(filtered_df[filtered_df['Expiration Date'] >= current_date])
            st.metric("✅ Active Licenses", f"{active_licenses:,}")
        else:
            st.metric("✅ Active Licenses", "0")
    with col4:
        if filtered_df is not None and 'Original Issue Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Original Issue Date']):
            date_range = f"{filtered_df['Original Issue Date'].dt.year.min()}-{filtered_df['Original Issue Date'].dt.year.max()}"
            st.metric("📅 Year Range", date_range)
        else:
            st.metric("📅 Year Range", "N/A")

    # Clarifying note for States metric
    st.caption("*Includes U.S. territories (DC, GU, PR), Armed Forces jurisdictions (AP), and international license holders. Total count may exceed 50 states.")

# ---------- App ----------
def main():
    st.set_page_config(page_title="SE License Dashboard", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")
    st.title("🏗️ Structural Engineers License Dashboard")
    st.markdown("Interactive dashboard for analyzing structural engineer license data")
    st.markdown("---")

    st.sidebar.title("📁 Data Source")

    # Default repo-bundled CSV path (without extension, will be added automatically)
    default_csv = "ProfEngrsLandSurvyrsGeologist_Data00.xls_structural_engineers_cleaned"
    src = resolve_data_source(default_csv)

    df = None
    if src:
        with st.spinner("Loading cleaned data..."):
            df = load_cleaned_data(src)
            # Ensure date columns are parsed consistently for the rest of the app
            df = parse_dates(df)
            if df is not None:
                st.sidebar.success(f"✅ Loaded: {Path(src).name}")
    
    # If no file found automatically, show upload option
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

    st.success(f"✅ Successfully loaded {len(df):,} Structural Engineer records")

    # Show column info for debugging
    with st.expander("📋 Data Column Information"):
        st.write("**Available columns:**")
        for col in df.columns:
            st.write(f"- `{col}` ({df[col].dtype})")
        
        st.write(f"\n**Data shape:** {df.shape[0]} rows ├ù {df.shape[1]} columns")

    # (file information expander removed by user request)

    # Filters
    filters = create_filters(df)
    full_df = df.copy()  # Keep unfiltered copy for comparison overlay
    filtered_df = apply_filters(df, filters)

    # Metrics
    st.subheader("📊 Summary Metrics")
    display_summary_metrics(filtered_df)
    st.markdown("---")

    # Visuals
    if filtered_df is None or filtered_df.empty:
        st.warning("🔍 No records match the selected filters. Try adjusting your filter criteria.")
        return

    create_visualizations(
        filtered_df,
        filters.get('show_events', False),
        filters.get('start_year'),
        filters.get('end_year'),
        filters.get('comparison_state'),
        full_df
    )

    # Table
    st.subheader("📋 Filtered Data Table")
    if st.checkbox("Show detailed data table", value=False):
        display_df = filtered_df.copy()
        if 'Original Issue Date' in display_df.columns:
            display_df['Original Issue Date'] = pd.to_datetime(display_df['Original Issue Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        if 'Expiration Date' in display_df.columns:
            display_df['Expiration Date'] = pd.to_datetime(display_df['Expiration Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        display_df = display_df.sort_values('Original Issue Date', ascending=False, na_position='last')
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    # Downloads
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"structural_engineers_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        summary_data = {
            'Metric': ['Total Records', 'States', 'Active Licenses', 'Date Range', 'Filters Applied'],
            'Value': [
                len(filtered_df),
                filtered_df['State'].nunique() if 'State' in filtered_df.columns else 0,
                len(filtered_df[filtered_df['Expiration Date'] >= pd.Timestamp.now()]) if 'Expiration Date' in filtered_df.columns else 0,
                f"{filtered_df['Original Issue Date'].dt.year.min()}-{filtered_df['Original Issue Date'].dt.year.max()}" if 'Original Issue Date' in filtered_df.columns else "N/A",
                f"States: {len(filters['states']) if filters['states'] else 'All'}, "
                f"Statuses: {len(filters['statuses']) if filters['statuses'] else 'All'}, "
                f"Years: {filters['start_year']}-{filters['end_year']}, "
                f"Expiration: {len(filters['expiration_types']) if filters['expiration_types'] else 'All'}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Summary Report (CSV)",
            data=summary_csv,
            file_name=f"dashboard_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
