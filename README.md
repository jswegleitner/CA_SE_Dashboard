# Structural Engineers Dashboard Pipeline

Complete automation pipeline for downloading, processing, and visualizing California structural engineer license data.

## Overview

This pipeline consists of 4 main components:

1. **Data Downloader** (`License_List_Claude_v3.py`) - Downloads .xls files from the DCA website
2. **Data Processor** (`SE_Data_Process_V1.py`) - Handles corrupted files and cleans data
3. **Dashboard** (`SE_Dashboard_v7.py`) - Interactive Streamlit dashboard
4. **Pipeline Coordinator** (`pipeline_coordinator.py`) - Runs all scripts in sequence

## Quick Start

### Option 1: Automated (Recommended)
1. Double-click `run_pipeline.bat` (Windows) or `./run_pipeline.sh` (Linux/Mac)
2. Follow the prompts
3. Dashboard will open in your browser automatically

### Option 2: Manual Steps
```bash
# 1. Download data
python License_List_Claude_v3.py

# 2. Process data (handles corrupted .xls files)
python SE_Data_Process_V1.py

# 3. Launch dashboard  
streamlit run SE_Dashboard_v7.py
```

### Option 3: Coordinator Script
```bash
python pipeline_coordinator.py
```

## Requirements

### Python Packages
```bash
pip install pandas streamlit plotly selenium xlrd openpyxl
```

### Additional Requirements
- Chrome browser (for data downloading)
- ChromeDriver (automatically managed by Selenium)

## File Structure

```
your_project_folder/
├── License_List_Claude_v3.py          # Downloads data from DCA
├── SE_Data_Process_V2.py              # Processes and cleans data  
├── SE_Dashboard_v7.py                 # Streamlit dashboard
├── pipeline_coordinator.py            # Coordinates all scripts
├── run_pipeline.bat                   # Windows batch runner
├── run_pipeline.sh                    # Linux/Mac shell runner
└── README.md                          # This file
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
- **Export capabilities** - Download filtered data and summary reports
- **Responsive design** - Works on desktop and mobile

### Automation Features  
- **One-click execution** - Run entire pipeline with single command
- **Error handling** - Comprehensive logging and error recovery
- **Skip options** - Reuse existing downloaded files
- **Progress tracking** - Clear status updates at each step

## Timeline Events (Exam/Code History)

The dashboard overlays key timeline events on time‑series charts using `Github Repo/timeline_events.csv`.

Required columns:
- `start_date` (YYYY‑MM‑DD)
- `label` (short title)
- `type` (e.g., `SE Exam`)

Preferred detail columns:
- `Code` — code references (e.g., IBC/ASCE/ACI)
- `Format` — exam format details
- `Notes` — additional context

If your CSV only has a single `description` field (with lines starting `Codes:`, `Format:`, `Notes:`), the app parses it automatically.

Update the timeline file (Windows PowerShell):

```powershell
Push-Location "C:\Users\jwegleitner\OneDrive\Documents\Side Projects\CA License Dashboard\Github Repo"
& 'C:/Users/jwegleitner/Miniforge3/python.exe' tools/reprocess_timeline.py
git add "Github Repo/timeline_events.csv"; git commit -m "Update timeline events"; git push origin main
```

Notes:
- The events toggle is under "🔍 Data Filters" in the sidebar.
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
- Input/output paths in `SE_Data_Process_V1.py` 
- Default CSV path in `SE_Dashboard_v7.py`

### Dashboard Settings
The dashboard looks for CSV files in this order:
1. Streamlit secrets (`DATA_PATH` or `DATA_URL`)
2. Default filename: `ProfEngrsLandSurvyrsGeologist_Data00.xls_structural_engineers_cleaned.csv`
3. Any CSV file in the current directory
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