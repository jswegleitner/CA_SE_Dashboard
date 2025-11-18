# -*- coding: utf-8 -*-
"""
Structural Engineers Dashboard - Streamlit Version
Loads pre-processed CSV data and creates interactive dashboard

Created on Fri Sep 12 19:59:58 2025
@author: jwegleitner

Use these commands in the terminal to run the app locally:
    Opens browser: & 'C:/Users/jwegleitner/Miniforge3/python.exe' -m streamlit run .\SE_Dashboard_v7.py
    Run headless (no browser): & 'C:/Users/jwegleitner/Miniforge3/python.exe' -m streamlit run .\SE_Dashboard_v7.py --server.headless true


"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re

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
        "ProfEngrsLandSurvyrsGeologist_Data00.csv"
    ]
    
    current_dir = Path.cwd()
    
    for filename in possible_files:
        local_csv_path = current_dir / filename
        if local_csv_path.exists():
            return str(local_csv_path)
    
    # Also check for any CSV files in the current directory
    csv_files = list(current_dir.glob("*.csv"))
    if csv_files:
        st.sidebar.info(f"Found CSV files: {[f.name for f in csv_files]}")
        return str(csv_files[0])  # Use the first CSV file found
    
    return None


# ---------- Filters & visuals (updated with Plotly) ----------
def create_filters(df):
    st.sidebar.header("🔍 Data Filters")
    filters = {}

    st.sidebar.subheader("State")
    states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    filters['states'] = st.sidebar.multiselect(
        "Select States:",
        options=states,
        default=states,
        key='state_filter',
        help="Select one or more states. Leave empty to include all states."
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

def create_visualizations(filtered_df):
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

    # Time series
    plot_time_series(filtered_df, bucket_size, lock_y)
    # Line chart variant
    plot_time_series_line(filtered_df, bucket_size, lock_y)
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


def plot_time_series(filtered_df, bucket_size: str, lock_y: bool = True):
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

    license_counts = df_viz.groupby('Period').size().sort_index()

    if license_counts.empty:
        st.info("No time-series data for the selected filters.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(p) for p in license_counts.index],
        y=license_counts.values,
        name='Licenses Issued',
        marker_color='steelblue',
        text=license_counts.values,
        textposition='outside',
        hovertemplate='<b>Period:</b> %{x}<br><b>Licenses:</b> %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Structural Engineer Licenses Issued Over Time ({title_period})",
        xaxis_title="Period",
        yaxis_title="Number of Licenses",
        showlegend=False,
        height=500,
        hovermode='x unified',
        dragmode='zoom'
    )

    num_periods = len(license_counts)
    if num_periods > 20:
        fig.update_xaxes(tickangle=90)
    elif num_periods > 10:
        fig.update_xaxes(tickangle=45)

    # Ensure X axis is zoomable but optionally lock Y axis so zoom only affects X
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=lock_y)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Time Series Statistics"):
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Peak Period", str(license_counts.idxmax()))
        with c2: st.metric("Peak Count", int(license_counts.max()))
        with c3: st.metric("Average per Period", f"{license_counts.mean():.1f}")


def plot_time_series_line(filtered_df, bucket_size: str, lock_y: bool = True):
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
        dragmode='zoom'
    )

    # Allow zooming on X; optionally lock Y so it doesn't rescale
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=lock_y)

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
        st.metric("🗺️ States", filtered_df['State'].nunique() if (filtered_df is not None and 'State' in filtered_df.columns) else 0)
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
    filtered_df = apply_filters(df, filters)

    # Metrics
    st.subheader("📊 Summary Metrics")
    display_summary_metrics(filtered_df)
    st.markdown("---")

    # Visuals
    if filtered_df is None or filtered_df.empty:
        st.warning("🔍 No records match the selected filters. Try adjusting your filter criteria.")
        return

    create_visualizations(filtered_df)

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
