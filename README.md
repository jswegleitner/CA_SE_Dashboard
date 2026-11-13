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
│   ├── timeline.py                       # Chart-overlay events + hover formatting
│   ├── history.py                        # Licensure-history loader and helpers (History tab)
│   ├── periods.py                        # Period bucketing primitives
│   ├── geo.py                            # State/county name → code lookups
│   └── theme.py                          # Unified palette, Plotly template, CSS overrides
├── data/                                 # Tracked CSV data (auto-populated by pipeline)
│   ├── ProfEngrsLandSurvyrsGeologist_Data00_structural_engineers_cleaned.csv
│   └── licensure_history.csv             # Licensure, exam, code, law and earthquake history (hand-maintained)
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
├── research/
│   ├── SOURCES.md                        # Index of sources behind the History tab
│   └── sources/                          # Source PDFs (gitignored: copyrighted material)
├── tests/                                # pytest suite (run with `pytest tests`)
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
- **Tabbed layout** — Time / Geography / Status / History, replacing a single long scroll
- **History tab** — what the Structural Engineer license is and how it got that way: a timeline of licensure, exam, building-code, legislative and earthquake events set against licenses issued per year, with a sourced write-up of each event and notes on what the license counts do and don't show
- **Filter chip summary** — chip bar above the metrics shows only the filters that differ from defaults, with a quiet "No filters applied" state when clean
- **Unified visual theme** — single palette, Plotly template, and CSS injected from `dashboard_lib/theme.py` so every chart, card, tab, and chip reads as one design
- **Real-time filtering** — by state, license status, year range, and comparison state. The historical-cohort charts (Active Licenses, Average Age) bypass the License Status filter so picking "Active" doesn't collapse the past-year view.
- **Comparison overlays** — overlay a second state on the time-series bar chart
- **Geographic maps** — US choropleth and California county choropleth
- **Shareable URLs** — "Share current filters" writes the current filter state into URL query params
- **Debug expander** — `?debug=1` exposes the column-info expander; hidden from public viewers by default

#### Charts (Activity tab)
1. **Licenses Issued Over Time** — bar chart with Yearly/Half-Yearly/Quarterly/Monthly bucketing; supports comparison-state overlay and timeline-event markers
2. **Licenses Over Time (line)** — individual licenses by Original Issue Date with optional event vertical lines. Overlays amber diamond markers on data anomalies (licenses whose Issue Date is far later than their License Number suggests — almost always reciprocity reinstatements or DCA backfills) with an explanatory tooltip and caption.
3. **License Expirations (Retirements) by Year** — bar chart of true retirements per year. Filters to non-Active statuses (Cancelled, Retired, Deceased, Delinquent, Revoked, Voluntary Surrender) and excludes the current year so the in-progress renewal cycle doesn't dominate.
4. **Active Licenses per Year** — area chart counting licenses active at the end of each year (issued ≤ Dec 31, expiration ≥ Jan 1). Ignores the License Status sidebar filter so the historical view stays consistent.
5. **Average Age of Active Licenses by Year** — line chart of mean issue-to-year-end age for the active cohort each year. Same status-filter-bypass as chart 4.

#### Charts (Geography tab)
- **Licenses by State** bar chart + Top 5 callout
- **US choropleth** (Blues scale, capped at zmax for visual range)
- **California Licenses by County** choropleth (Reds scale)

#### Charts (Status tab)
- **License Status Distribution** donut with in-slice percentages, color-coded breakdown legend, and a center "N licenses" total. Caption links to the [BPELSG license lookup page](https://www.bpelsg.ca.gov/consumers/lic_lookup.shtml) where each status is defined.

### Automation Features
- **Pipeline coordination** - Run entire pipeline with single command
- **Error handling** - Comprehensive logging and error recovery
- **Skip options** - Reuse existing downloaded files
- **Progress tracking** - Clear status updates at each step

## Licensure History (History tab and chart overlays)

`data/licensure_history.csv` is the single, hand-maintained record of licensure, exam,
building-code, legislative and earthquake events. The **History** tab shows every row as a
sourced entry and plots them against licenses issued per year. Rows with `on_chart` = 1 are
also overlaid on the Time-tab charts. There is no build step: edit the CSV and rerun.

Columns:
- `date` (YYYY-MM-DD) and `precision` (`day`, `month` or `year`: how exactly the date is known;
  a year-only event is stored as January 1 and displayed as just the year)
- `category`: `Licensure`, `Exam`, `Building code`, `Legislation` or `Earthquake or disaster`
- `title`, `summary` (what happened), `significance` (why it matters)
- `data_note` (optional): what the license data shows around that event
- `codes`, `exam_format` (optional): detail shown on exam entries and in chart tooltips
- `on_chart` (0/1), `chart_note` (one-line tooltip, required when `on_chart` is 1),
  `chart_label` (optional short label drawn on the History timeline)
- `sources`: `Label <https://url>; Label <https://url>; Label with no link`

`tests/test_history.py` checks the file (valid categories and dates, a source on every entry,
well-formed links), so a bad hand edit fails the tests rather than the page.
`research/SOURCES.md` lists the documents behind the entries.

Notes:
- The events toggle is at the top of the "Filters" section in the sidebar.
- Date filters constrain the Time-tab charts and their event overlays; the History tab always
  shows the whole record.

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

History events are read from `data/licensure_history.csv`. A legacy `timeline_events.csv` (in `data/` or the project root) is still honored for the chart overlays if the history file is absent.

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
- **2026-05a**: Drop-off reconciliation + HTML rescan flow
  - Reconciler flips dropped licenses to `Cancelled` with `Status_Changed_Date` stamping
  - New `--source feed|html` flag separates feed snapshots from gap-fill batches
  - HTML fetcher gains `--rescan` mode driven by `licenses/needs_rescan.txt`
  - HTML parser auto-merges into master on completion
  - Processor auto-publishes cleaned CSV to `data/` so the dashboard always sees fresh data
  - Pipeline entry points (`run_pipeline.bat`, `run_dashboard.bat`) at the project root (local-only, gitignored)
  - Pipeline coordinator self-locates; hardcoded feed filename guards against picking the master file as input
  - Emojis stripped from all processing scripts (Windows cp1252 console compatibility)
  - Date columns normalized before row diff to prevent spurious `Last_Updated` churn
- **2026-05a**: Dashboard visual refresh
  - New `dashboard_lib/theme.py` centralizes the palette, Plotly template, and Streamlit CSS
  - Charts reorganized into Activity / Geography / Status tabs
  - Filter chip summary bar shows non-default filters above the metrics
  - Metric cards, sidebar, tabs, buttons styled via injected CSS for a consistent feel
  - Status donut redesigned: in-slice labels, color-coded breakdown legend, center total
  - Conditional bar labels on long time-series and state charts (hidden when category count is high)
  - Average license age per year chart (added 2026-04) is now part of the Activity tab
  - Column-info expander gated behind `?debug=1`; redundant load banner removed
- **Current** (2026-05): Chart-semantics cleanup + anomaly detection
  - **Active Licenses / Average Age charts** now bypass the License Status sidebar filter. Picking "Active" in the sidebar was silently collapsing the historical cohort (every past-year license that has since changed status would drop). `main()` now builds a second `status_agnostic_df` and routes it to these two charts. Both gained captions explaining their per-year logic.
  - **License Expirations (Retirements) by Year** now filters to non-Active statuses (Cancelled, Retired, Deceased, Delinquent, Revoked, Voluntary Surrender) and excludes the current year. The previous chart's giant current-year spike was the in-progress renewal cycle (active licenses whose 2-year cycle ends this year and haven't renewed yet), not real retirements. A caption explains both rules.
  - **License Expiration Status filter removed** from the sidebar — the "Active Only / Expired Only" multiselect duplicated the License Status filter and made the time-window charts misleading. URL share params, filter chips, and the apply-filters branch all cleaned up.
  - **License Status Distribution caption** now links to the BPELSG license lookup page where each status is defined.
  - **Issue-date anomaly detection** on the line chart. `_detect_issue_date_anomalies()` sorts by License Number, computes a rolling 51-row median of issue years, and flags rows whose actual issue year exceeds the local median by ≥ 5 years. `plot_time_series_line` overlays each as an amber diamond marker with a tooltip explaining the discrepancy. These rows are almost always reciprocity reinstatements or DCA database backfills where the agency stamped the *reactivation* date into the Original Issue Date field rather than preserving the true first issuance — the bad dates cluster on specific batch days (`2002-07-26`, `1987-02-27`) and most are `Out Of State` county. Detection is data-driven and refreshes each month with no maintenance.
