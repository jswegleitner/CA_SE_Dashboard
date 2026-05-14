# Structural Engineers Dashboard Pipeline

Complete automation pipeline for downloading, processing, reconciling, and visualizing California structural engineer license data.

## Overview

The pipeline has two layers:

**Monthly feed refresh** (fast, fully automated):
1. **Data Downloader** (`Offline Processing/License_List_Claude_v3.py`) — Downloads `.xls` from the DCA public feed
2. **Data Processor** (`Offline Processing/SE_Data_Process_V9.py`) — Filters to SEs, upserts into the master record, reconciles drop-offs to `Cancelled`, and auto-publishes the cleaned CSV to `data/`
3. **Pipeline Coordinator** (`Offline Processing/pipeline_coordinator.py`) — Orchestrates the above; invoked by `run_pipeline.bat`
4. **Dashboard** (`SE_Dashboard_v7.py`) — Interactive Streamlit dashboard; reads `data/...cleaned.csv` locally or from Streamlit secrets in the cloud

**Optional HTML rescan layer** (slow, run when convenient):
5. **License Fetcher** (`Offline Processing/DCA_License_Fetcher.py`) — Selenium scraper for individual license detail pages. `--rescan` mode replaces placeholder values with real DCA data for licenses the reconciler flagged
6. **HTML Parser** (`Offline Processing/DCA_License_HTML_Parser.py`) — Parses scraped HTML and auto-merges into the master record
7. **Merge helper** (`Offline Processing/merge_parsed_licenses.py`) — One-shot merge of an existing parsed-HTML batch file

## Quick Start

### One-click (recommended)
Double-click from File Explorer at the project root:
- `run_pipeline.bat` — full monthly refresh
- `run_dashboard.bat` — launch dashboard only

### Command-line equivalents
```bash
# Monthly feed refresh
python "Offline Processing/pipeline_coordinator.py"

# Dashboard
streamlit run SE_Dashboard_v7.py

# Rescan dropped licenses to replace placeholder Cancelled values with DCA truth
python "Offline Processing/DCA_License_Fetcher.py" --rescan
python "Offline Processing/DCA_License_HTML_Parser.py"

# Merge an existing parsed-HTML batch into master
python "Offline Processing/merge_parsed_licenses.py"
```

## Monthly Workflow

1. Run `run_pipeline.bat`. It downloads the feed, reconciles drop-offs, and updates `data/`. Any licenses that dropped off the public feed are flipped to placeholder status `Cancelled` with `Status_Changed_Date` stamped and added to `Offline Processing/licenses/needs_rescan.txt`.
2. (Optional) Run the rescan to replace placeholders with DCA truth:
   - `python "Offline Processing/DCA_License_Fetcher.py" --rescan` — scrape fresh HTML for queued licenses
   - `python "Offline Processing/DCA_License_HTML_Parser.py"` — parse and auto-merge
3. Commit the updated `data/ProfEngrsLandSurvyrsGeologist_Data00_structural_engineers_cleaned.csv` so Streamlit Cloud picks it up.

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
├── SE_Dashboard_v7.py                    # Streamlit dashboard (orchestrator)
├── run_pipeline.bat                      # One-click monthly refresh
├── run_dashboard.bat                     # One-click dashboard launch
├── dashboard_lib/                        # Extracted helpers
│   ├── timeline.py                       # Timeline event loading + hover formatting
│   ├── periods.py                        # Period bucketing primitives
│   └── geo.py                            # State/county name → code lookups
├── data/                                 # Tracked CSV data (auto-populated by pipeline)
│   ├── ProfEngrsLandSurvyrsGeologist_Data00_structural_engineers_cleaned.csv
│   ├── timeline_events.csv               # Timeline event annotations
│   └── License History Table.csv         # Source for timeline events
├── requirements.txt                      # Python dependencies
├── .gitignore
├── README.md                             # This file
├── CLAUDE.md                             # AI-assistant context / project conventions
├── Offline Processing/                   # (gitignored)
│   ├── License_List_Claude_v3.py         # Feed downloader (Selenium)
│   ├── SE_Data_Process_V9.py             # Processor + reconciler + auto-publish
│   ├── pipeline_coordinator.py           # Orchestrator for the .bat
│   ├── DCA_License_Fetcher.py            # License-detail scraper (undetected-chromedriver)
│   ├── DCA_License_HTML_Parser.py        # HTML parser + auto-merge
│   ├── merge_parsed_licenses.py          # One-shot parsed-HTML merger
│   ├── ProfEngrsLandSurvyrsGeologist_Data00_SE_master_record.xlsx   # Master record
│   └── licenses/                         # Scraper outputs + rescan queue
│       ├── raw/<n>.html                  # Captured detail pages
│       ├── notfound/<n>.html             # No-results pages
│       ├── archive/<n>_<label>_<ts>.html # Pre-rescan HTML snapshots
│       └── needs_rescan.txt              # Rescan queue (managed by reconciler + fetcher)
├── tools/
│   └── reprocess_timeline.py             # Regenerates data/timeline_events.csv
├── tests/                                # pytest suite (run with `pytest`)
└── .streamlit/
    └── config.toml                       # Streamlit UI configuration
```

## Key Features

### Enhanced Data Processing
- **Handles corrupted .xls files** - Multiple fallback methods for reading problematic Excel files
- **Flexible column matching** - Works with variations in column names
- **Robust date parsing** - Handles various date formats
- **Data validation** - Removes duplicates and invalid records

### Drop-off Reconciliation
- Licenses present in the master but absent from a fresh DCA feed are flipped to placeholder status `Cancelled` with `Status_Changed_Date` stamped (today's date).
- Terminal statuses (`Cancelled`, `Deceased`, `Retired`, `Revoked`, `Voluntary Surrender Of License`) are never overwritten.
- DCA's own spelling — `Cancelled`, two Ls — is mirrored.
- The processor takes `--source feed|html`: `feed` enables reconciliation (full DCA snapshot), `html` is a pure upsert (gap-fill batches).

### HTML Rescan Flow
- The reconciler appends every license it flipped to `Offline Processing/licenses/needs_rescan.txt`.
- The fetcher's `--rescan` mode reads that queue, archives any stale HTML to `licenses/archive/`, re-fetches detail pages, and removes successful entries.
- The parser auto-merges results back into the master with `source='html'`. Rescan revealing a non-`Cancelled` status (false alarm — license is still active in DCA's database, just dropped from the public feed) clears `Status_Changed_Date`.

### Auto-publish to `data/`
- Every processor run copies the cleaned CSV from `Offline Processing/` to `data/` under the dashboard-expected filename.
- Streamlit Cloud sees the new data once you commit `data/`.

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
1. Streamlit secrets (`DATA_PATH` or `DATA_URL`) — used on Streamlit Cloud; if `DATA_PATH` is set, update it to point at `data/...` after this layout change.
2. Known filenames inside `data/`, then in the project root (legacy fallback).
3. Any CSV in `data/` or the project root with license data columns.
4. File upload widget.

Timeline events (`timeline_events.csv`) are looked up the same way: `data/` first, then the project root.

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
- **v4.0**: Complete integrated pipeline with robust error handling
- **Current** (2026-05): Drop-off reconciliation + HTML rescan flow
  - Reconciler flips dropped licenses to `Cancelled` with `Status_Changed_Date` stamping
  - New `--source feed|html` flag separates feed snapshots from gap-fill batches
  - HTML fetcher gains `--rescan` mode driven by `licenses/needs_rescan.txt`
  - HTML parser auto-merges into master on completion
  - Processor auto-publishes cleaned CSV to `data/` so the dashboard always sees fresh data
  - Pipeline entry points (`run_pipeline.bat`, `run_dashboard.bat`) live at the project root
  - Pipeline coordinator self-locates; hardcoded feed filename guards against picking the master file as input
  - Emojis stripped from all processing scripts (Windows cp1252 console compatibility)
  - Date columns normalized before row diff to prevent spurious `Last_Updated` churn
