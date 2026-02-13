# Structural Engineers Dashboard Pipeline

Complete automation pipeline for downloading, processing, and visualizing California structural engineer license data.

## Overview

This pipeline consists of 4 main components:

1. **Data Downloader** (`Offline Processing/License_List_Claude_v3.py`) - Downloads .xls files from the DCA website
2. **Data Processor** (`Offline Processing/SE_Data_Process_V9.py`) - Handles corrupted files and cleans data
3. **Dashboard** (`SE_Dashboard_v7.py`) - Interactive Streamlit dashboard
4. **Pipeline Coordinator** (`Offline Processing/pipeline_coordinator.py`) - Runs all scripts in sequence

## Quick Start

### Option 1: Manual Steps
```bash
# 1. Download data
python "Offline Processing/License_List_Claude_v3.py"

# 2. Process data (handles corrupted .xls files)
python "Offline Processing/SE_Data_Process_V9.py"

# 3. Launch dashboard
streamlit run SE_Dashboard_v7.py
```

### Option 2: Coordinator Script
```bash
python "Offline Processing/pipeline_coordinator.py"
```

## Requirements

### Python Packages
```bash
pip install -r requirements.txt
```

### Additional Requirements
- Chrome browser (for data downloading)
- ChromeDriver (automatically managed by Selenium)

## File Structure

```
CA License Dashboard/
├── SE_Dashboard_v7.py                    # Streamlit dashboard
├── requirements.txt                      # Python dependencies
├── .gitignore
├── README.md                             # This file
├── timeline_events.csv                   # Timeline event annotations
├── License History Table.csv             # Source for timeline events
├── Offline Processing/
│   ├── License_List_Claude_v3.py         # Downloads data from DCA
│   ├── SE_Data_Process_V9.py             # Processes and cleans data
│   └── pipeline_coordinator.py           # Coordinates all scripts
├── tools/
│   └── reprocess_timeline.py             # Regenerates timeline_events.csv
└── .streamlit/
    └── config.toml                       # Streamlit UI configuration
```

## Key Features

### Enhanced Data Processing
- **Handles corrupted .xls files** - Multiple fallback methods for reading problematic Excel files
- **Flexible column matching** - Works with variations in column names
- **Robust date parsing** - Handles various date formats
- **Data validation** - Removes duplicates and invalid records

### Interactive Dashboard
- **Real-time filtering** - By state, license status, date ranges
- **Interactive visualizations** - Built with Plotly for better user experience
- **Comparison overlays** - Compare license counts between states
- **Geographic maps** - US choropleth and California county maps
- **Responsive design** - Works on desktop and mobile

### Automation Features
- **Pipeline coordination** - Run entire pipeline with single command
- **Error handling** - Comprehensive logging and error recovery
- **Skip options** - Reuse existing downloaded files
- **Progress tracking** - Clear status updates at each step

## Timeline Events (Exam/Code History)

The dashboard overlays key timeline events on time-series charts using `timeline_events.csv`.

Required columns:
- `start_date` (YYYY-MM-DD)
- `label` (short title)
- `type` (e.g., `SE Exam`)

Preferred detail columns:
- `Code` — code references (e.g., IBC/ASCE/ACI)
- `Format` — exam format details
- `Notes` — additional context

If your CSV only has a single `description` field (with lines starting `Codes:`, `Format:`, `Notes:`), the app parses it automatically.

Update the timeline file:

```bash
python tools/reprocess_timeline.py
```

Notes:
- The events toggle is under "Data Filters" in the sidebar.
- Date filters constrain both charts and events; events do not extend axes.

## Troubleshooting

### Common Issues

**Corrupted .xls file errors:**
The enhanced data processor now handles this automatically with multiple fallback methods.

**Chrome/ChromeDriver issues:**
```bash
pip install --upgrade selenium
```

**Missing columns:**
The processor now flexibly matches column names. Check the output for column mapping details.

**Streamlit not opening:**
```bash
streamlit run SE_Dashboard_v7.py --server.headless=false
```

### Manual Fallbacks

If the automated download fails:
1. Manually download from: https://www.dca.ca.gov/consumers/public_info/index.shtml
2. Look for "ProfEngrsLandSurvyrsGeologist_7500" folder
3. Download "ProfEngrsLandSurvyrsGeologist_Data00.xls"
4. Run the data processor on the downloaded file

### Debug Mode

For detailed logging, edit the scripts to increase verbosity or check the log files generated in the output directory.

## Configuration

### Paths
Update paths in each script as needed:
- Download directory in `License_List_Claude_v3.py`
- Input/output paths in `SE_Data_Process_V9.py`
- Default CSV path in `SE_Dashboard_v7.py`

### Dashboard Settings
The dashboard looks for CSV files in this order:
1. Streamlit secrets (`DATA_PATH` or `DATA_URL`)
2. Default filename: `ProfEngrsLandSurvyrsGeologist_Data00.xls_structural_engineers_cleaned.csv`
3. Any CSV file in the current directory with license data columns
4. File upload widget

## Data Privacy

This pipeline processes public license data from the California Department of Consumer Affairs. No private or sensitive information is collected or stored beyond what is publicly available.

## Support

For issues or questions:
1. Check the log files generated during pipeline execution
2. Review error messages in the console output
3. Try running each script individually to isolate issues
4. Verify all dependencies are installed correctly

## Version History

- **v1.0**: Initial three separate scripts
- **v2.0**: Enhanced data processor with corruption handling
- **v3.0**: Added pipeline coordinator and automation scripts
- **Current**: Complete integrated pipeline with robust error handling
