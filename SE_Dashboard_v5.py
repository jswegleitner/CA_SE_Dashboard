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

@author: jwegleitner
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

@st.cache_data
def load_cleaned_data(csv_path):
    """
    Load pre-cleaned CSV data with caching for better performance
    
    Parameters:
    csv_path (str): Path to the cleaned CSV file
    
    Returns:
    pandas.DataFrame: Loaded and parsed data
    """
    try:
        # Load CSV file
        df = pd.read_csv(csv_path)
        
        # Convert dates back to datetime (they were saved as strings in CSV)
        df['Original Issue Date'] = pd.to_datetime(df['Original Issue Date'])
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_filters(df):
    """Create multi-select filters for the dashboard"""
    st.sidebar.header("🔍 Data Filters")
    
    filters = {}
    
    # State filter
    st.sidebar.subheader("State")
    states = sorted(df['State'].dropna().unique().tolist())
    filters['states'] = st.sidebar.multiselect(
        "Select States:",
        options=states,
        default=states,  # Default to all states selected
        key='state_filter',
        help="Select one or more states. Leave empty to include all states."
    )
    
    # License Status filter
    st.sidebar.subheader("License Status")
    statuses = sorted(df['License Status'].dropna().unique().tolist())
    filters['statuses'] = st.sidebar.multiselect(
        "Select License Statuses:",
        options=statuses,
        default=statuses,  # Default to all statuses selected
        key='status_filter',
        help="Select one or more license statuses. Leave empty to include all statuses."
    )
    
    # Date range filters
    st.sidebar.subheader("Date Range")
    
    # Original Issue Date range
    min_year = int(df['Original Issue Date'].dt.year.min())
    max_year = int(df['Original Issue Date'].dt.year.max())
    
    filters['start_year'], filters['end_year'] = st.sidebar.slider(
        "Original Issue Date Range (Years)",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        key='date_range_slider'
    )
    
    # Expiration Date filter option
    expiration_options = ['Active Only', 'Expired Only']
    filters['expiration_types'] = st.sidebar.multiselect(
        "License Expiration Status:",
        options=expiration_options,
        default=expiration_options,  # Default to both selected
        key='expired_filter',
        help="Active = not yet expired, Expired = past expiration date. Select both for all licenses."
    )
    
    return filters

def apply_filters(df, filters):
    """Apply selected filters to the dataframe"""
    filtered_df = df.copy()
    
    # Apply state filter
    if filters['states']:  # Only filter if states are selected
        filtered_df = filtered_df[filtered_df['State'].isin(filters['states'])]
    
    # Apply status filter
    if filters['statuses']:  # Only filter if statuses are selected
        filtered_df = filtered_df[filtered_df['License Status'].isin(filters['statuses'])]
    
    # Apply date range filter
    mask = (
        (filtered_df['Original Issue Date'].dt.year >= filters['start_year']) & 
        (filtered_df['Original Issue Date'].dt.year <= filters['end_year'])
    )
    filtered_df = filtered_df[mask]
    
    # Apply expiration filter
    current_date = pd.Timestamp.now()
    if filters['expiration_types']:
        expiration_mask = pd.Series(False, index=filtered_df.index)
        
        if 'Active Only' in filters['expiration_types']:
            expiration_mask |= (filtered_df['Expiration Date'] >= current_date)
        
        if 'Expired Only' in filters['expiration_types']:
            expiration_mask |= (filtered_df['Expiration Date'] < current_date)
        
        filtered_df = filtered_df[expiration_mask]
    
    return filtered_df

def create_visualizations(filtered_df):
    """Create charts and visualizations"""
    
    if filtered_df.empty:
        st.warning("No data to visualize with current filters.")
        return
    
    # Time-based analysis
    st.subheader("📈 Licenses Issued Over Time")
    
    # Bucket size selection
    col1, col2 = st.columns([1, 3])
    with col1:
        bucket_size = st.selectbox("Time Grouping:", ["Yearly", "Half-Yearly", "Quarterly"])
    
    # Create time periods
    df_viz = filtered_df.copy()
    if bucket_size == "Yearly":
        df_viz['Period'] = df_viz['Original Issue Date'].dt.to_period('Y')
        title_period = "Yearly"
    elif bucket_size == "Half-Yearly":
        df_viz['Period'] = df_viz['Original Issue Date'].dt.to_period('6M')
        title_period = "Half-Yearly"
    else:  # Quarterly
        df_viz['Period'] = df_viz['Original Issue Date'].dt.to_period('Q')
        title_period = "Quarterly"
    
    # Count licenses per period
    license_counts = df_viz.groupby('Period').size().sort_index()
    
    # Plot time series
    if not license_counts.empty:
        # Calculate figure width based on number of periods for better readability
        num_periods = len(license_counts)
        fig_width = max(12, min(20, num_periods * 0.6))  # Between 12-20 inches, scale with data
        
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        bars = ax.bar(range(len(license_counts)), license_counts.values, color='steelblue', alpha=0.7)
        
        # Customize the plot
        ax.set_title(f"Structural Engineer Licenses Issued Over Time ({title_period})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Period", fontsize=12)
        ax.set_ylabel("Number of Licenses", fontsize=12)
        
        # Set x-axis labels with improved formatting
        ax.set_xticks(range(len(license_counts)))
        period_labels = [str(period) for period in license_counts.index]
        
        # Adjust label rotation and size based on number of periods
        if num_periods > 30:
            rotation_angle = 90
            fontsize = 7
            # Show every nth label to avoid overcrowding
            step = max(1, num_periods // 20)
            ax.set_xticks(range(0, len(license_counts), step))
            period_labels = [period_labels[i] if i % step == 0 else '' for i in range(len(period_labels))]
        elif num_periods > 15:
            rotation_angle = 90
            fontsize = 8
        elif num_periods > 8:
            rotation_angle = 45
            fontsize = 9
        else:
            rotation_angle = 0
            fontsize = 10
            
        ax.set_xticklabels([period_labels[i] for i in ax.get_xticks().astype(int) if i < len(period_labels)], 
                          rotation=rotation_angle, ha='center' if rotation_angle == 0 else 'right', fontsize=fontsize)
        
        # Add value labels on bars (but only if not too many bars)
        if num_periods <= 25:
            for i, (bar, value) in enumerate(zip(bars, license_counts.values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(1, license_counts.max() * 0.01),
                       str(value), ha='center', va='bottom', fontsize=max(6, 10 - num_periods // 5))
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show summary stats for the chart
        with st.expander("📊 Time Series Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Period", str(license_counts.idxmax()))
            with col2:
                st.metric("Peak Count", int(license_counts.max()))
            with col3:
                st.metric("Average per Period", f"{license_counts.mean():.1f}")
    
    st.markdown("---")
    
    # State analysis (if multiple states)
    if len(filtered_df['State'].unique()) > 1:
        st.subheader("🗺️ Licenses by State")
        state_counts = filtered_df['State'].value_counts()
        
        # Create two columns for chart and stats
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(state_counts.index, state_counts.values, color='lightcoral', alpha=0.7)
            ax.set_title("Number of Structural Engineer Licenses by State", fontsize=14, fontweight='bold')
            ax.set_xlabel("State", fontsize=12)
            ax.set_ylabel("Number of Licenses", fontsize=12)
            
            # Add value labels on bars
            for bar, value in zip(bars, state_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(value), ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Top 5 States:**")
            for i, (state, count) in enumerate(state_counts.head().items(), 1):
                percentage = (count / len(filtered_df)) * 100
                st.write(f"{i}. **{state}**: {count} ({percentage:.1f}%)")
        
        st.markdown("---")
    
    # License status distribution (if multiple statuses)
    if len(filtered_df['License Status'].unique()) > 1:
        st.subheader("📋 License Status Distribution")
        
        col1, col2 = st.columns([1, 1])
        
        status_counts = filtered_df['License Status'].value_counts()
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.Set3(range(len(status_counts)))
            wedges, texts, autotexts = ax.pie(status_counts.values, labels=status_counts.index, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title("License Status Distribution", fontsize=14, fontweight='bold')
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Status Breakdown:**")
            for status, count in status_counts.items():
                percentage = (count / len(filtered_df)) * 100
                st.write(f"**{status}**: {count} ({percentage:.1f}%)")
        
        st.markdown("---")

def display_summary_metrics(filtered_df):
    """Display summary metrics in columns"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Records", f"{len(filtered_df):,}")
    
    with col2:
        st.metric("🗺️ States", filtered_df['State'].nunique())
    
    with col3:
        if not filtered_df.empty:
            current_date = pd.Timestamp.now()
            active_licenses = len(filtered_df[filtered_df['Expiration Date'] >= current_date])
            st.metric("✅ Active Licenses", f"{active_licenses:,}")
        else:
            st.metric("✅ Active Licenses", "0")
    
    with col4:
        if not filtered_df.empty:
            date_range = f"{filtered_df['Original Issue Date'].dt.year.min()}-{filtered_df['Original Issue Date'].dt.year.max()}"
            st.metric("📅 Year Range", date_range)
        else:
            st.metric("📅 Year Range", "N/A")

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="SE License Dashboard", 
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏗️ Structural Engineers License Dashboard")
    st.markdown("Interactive dashboard for analyzing structural engineer license data")
    st.markdown("---")
    
    # File input section
    st.sidebar.title("📁 Data Source")
    
    # Default paths - update these to match your preprocessor output
    default_path = r"C:\Users\jwegleitner\OneDrive - Degenkolb Engineers\Desktop\CA License Dashboard\ProfEngrsLandSurvyrsGeologist_Data00.xls_structural_engineers_cleaned.csv"
    
    use_custom_path = st.sidebar.checkbox("Use custom file path")
    
    if use_custom_path:
        csv_path = st.sidebar.text_input(
            "Enter path to cleaned CSV file:",
            placeholder="C:\\path\\to\\your\\cleaned_data.csv",
            help="This should be the CSV file generated by the data preprocessor script"
        )
    else:
        csv_path = default_path
        st.sidebar.info("Using default cleaned data file")
    
    # Check if file exists
    if not csv_path or not os.path.exists(csv_path):
        st.error("⚠️ CSV file not found!")
        if not use_custom_path:
            st.info("""
            **Next Steps:**
            1. Run the data preprocessor script first to create the cleaned CSV file
            2. Make sure the file path matches the preprocessor output
            3. Or check 'Use custom file path' and specify your CSV location
            """)
        else:
            st.info("Please check the file path and make sure the CSV file exists.")
        return
    
    # Load data
    with st.spinner("Loading cleaned data..."):
        df = load_cleaned_data(csv_path)
    
    if df is None or df.empty:
        st.error("Unable to load data from the CSV file.")
        return
    
    # Success message
    st.success(f"✅ Successfully loaded {len(df):,} Structural Engineer records")
    
    # Show file info
    with st.expander("📄 Data File Information"):
        st.write(f"**File path:** `{csv_path}`")
        st.write(f"**File size:** {os.path.getsize(csv_path) / 1024:.1f} KB")
        st.write(f"**Last modified:** {pd.Timestamp.fromtimestamp(os.path.getmtime(csv_path)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create filters
    filters = create_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Display summary metrics
    st.subheader("📊 Summary Metrics")
    display_summary_metrics(filtered_df)
    st.markdown("---")
    
    # Check if we have data after filtering
    if filtered_df.empty:
        st.warning("🔍 No records match the selected filters. Try adjusting your filter criteria.")
        return
    
    # Create visualizations
    create_visualizations(filtered_df)
    
    # Data table section
    st.subheader("📋 Filtered Data Table")
    
    # Show/hide table option
    show_table = st.checkbox("Show detailed data table", value=False)
    
    if show_table:
        # Format dates for display
        display_df = filtered_df.copy()
        display_df['Original Issue Date'] = display_df['Original Issue Date'].dt.strftime('%Y-%m-%d')
        display_df['Expiration Date'] = display_df['Expiration Date'].dt.strftime('%Y-%m-%d')
        
        # Sort by Original Issue Date (most recent first)
        display_df = display_df.sort_values('Original Issue Date', ascending=False)
        
        # Display with pagination
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            height=400
        )
    
    # Download section
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download filtered data
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"structural_engineers_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download summary report
        summary_data = {
            'Metric': ['Total Records', 'States', 'Active Licenses', 'Date Range', 'Filters Applied'],
            'Value': [
                len(filtered_df),
                filtered_df['State'].nunique(),
                len(filtered_df[filtered_df['Expiration Date'] >= pd.Timestamp.now()]),
                f"{filtered_df['Original Issue Date'].dt.year.min()}-{filtered_df['Original Issue Date'].dt.year.max()}",
                f"States: {len(filters['states']) if filters['states'] else 'All'}, Statuses: {len(filters['statuses']) if filters['statuses'] else 'All'}, Years: {filters['start_year']}-{filters['end_year']}, Expiration: {len(filters['expiration_types']) if filters['expiration_types'] else 'All'}"
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