# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 19:59:58 2025

@author: jwegleitner
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 19:59:58 2025

@author: jwegleitner
"""

# -*- coding: utf-8 -*-
"""
Structural Engineers Dashboard - Streamlit Version
Loads pre-processed CSV data and creates interactive dashboard
"""
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------- Data loading ----------
@st.cache_data
def load_cleaned_data(src):
    """
    Load pre-cleaned CSV data with caching for better performance.
    `src` can be a local path or a URL.
    """
    try:
        df = pd.read_csv(src)
        # Ensure dates are parsed as datetime
        if 'Original Issue Date' in df.columns:
            df['Original Issue Date'] = pd.to_datetime(df['Original Issue Date'], errors="coerce")
        if 'Expiration Date' in df.columns:
            df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading data from {src}: {str(e)}")
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

    st.subheader("📈 Licenses Issued Over Time")

    col1, _ = st.columns([1, 3])
    with col1:
        bucket_size = st.selectbox("Time Grouping:", ["Yearly", "Half-Yearly", "Quarterly"])

    if 'Original Issue Date' not in filtered_df.columns:
        st.info("Original Issue Date column not found.")
        return

    df_viz = filtered_df.copy()
    if bucket_size == "Yearly":
        df_viz['Period'] = df_viz['Original Issue Date'].dt.to_period('Y')
        title_period = "Yearly"
    elif bucket_size == "Half-Yearly":
        # Create proper half-year periods (H1, H2)
        df_viz['Year'] = df_viz['Original Issue Date'].dt.year
        df_viz['Half'] = ((df_viz['Original Issue Date'].dt.month - 1) // 6) + 1
        df_viz['Period'] = df_viz['Year'].astype(str) + '-H' + df_viz['Half'].astype(str)
        title_period = "Half-Yearly"
    else:
        df_viz['Period'] = df_viz['Original Issue Date'].dt.to_period('Q')
        title_period = "Quarterly"

    license_counts = df_viz.groupby('Period').size().sort_index()

    if not license_counts.empty:
        # Create interactive bar chart with Plotly
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
            hovermode='x unified'
        )

        # Adjust x-axis labels for readability
        num_periods = len(license_counts)
        if num_periods > 20:
            fig.update_xaxes(tickangle=90)
        elif num_periods > 10:
            fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Time Series Statistics"):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Peak Period", str(license_counts.idxmax()))
            with c2: st.metric("Peak Count", int(license_counts.max()))
            with c3: st.metric("Average per Period", f"{license_counts.mean():.1f}")

    st.markdown("---")

    # By State
    if 'State' in filtered_df.columns and filtered_df['State'].nunique() > 1:
        st.subheader("🗺️ Licenses by State")
        state_counts = filtered_df['State'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # Create interactive bar chart for states
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
        st.markdown("---")

    # Status pie chart
    if 'License Status' in filtered_df.columns and filtered_df['License Status'].nunique() > 1:
        st.subheader("📋 License Status Distribution")
        c1, c2 = st.columns([1, 1])
        status_counts = filtered_df['License Status'].value_counts()
        
        with c1:
            # Create interactive pie chart
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
        st.markdown("---")

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
        
        st.write(f"\n**Data shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    with st.expander("📄 Data File Information"):
        try:
            # Try to show filesystem info if src is a path
            if isinstance(src, str) and not src.startswith('http'):
                p = Path(src)
                if p.exists():
                    st.write(f"**File path:** `{p}`")
                    st.write(f"**File size:** {p.stat().st_size / 1024:.1f} KB")
                    st.write(f"**Last modified:** {pd.Timestamp(p.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.write(f"**Source:** {src}")
            else:
                st.write(f"**Source:** {src}")
        except Exception:
            st.write(f"**Source:** {src}")

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