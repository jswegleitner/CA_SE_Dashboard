# Development Plan

Backlog from the full codebase review of 2026-07-10 (efficiency / accuracy / security / content).
Items are ordered by priority within each section. Line numbers reference the code as of that date
and will drift — search for the quoted symbols if they do.

## Completed 2026-07-10 (for reference)

- URL query params validated/clamped in `_init_filters_from_url` (out-of-range `?start_year` crashed the page — verified with `StreamlitValueBelowMinError`)
- `html.escape()` on data-derived values in filter chips and status breakdown
- Removed dead `display_summary_metrics` (contradictory "Active" definition) and unused `import re`
- `parse_dates` uses explicit ISO format with dateutil fallback (kills the slow-parse warning)
- `.streamlit/secrets.toml` added to `.gitignore`; `requirements.txt` pinned; `pytest` moved to `requirements-dev.txt`
- New **Net Change in Licenses by Year** diverging chart (Time tab, status-agnostic)
- Headline stats row (Active / Total issued / Net change) + **About the data** expander
- Earthquake events (1933 Long Beach, 1971 San Fernando, 1989 Loma Prieta, 1994 Northridge) added via
  `data/timeline_events_extra.csv`, merged by `tools/reprocess_timeline.py` on every rebuild

---

## Completed 2026-09-12 — pandas 3 breakage + silent-failure triage

The 2026-09 monthly run failed and still reported success. Three separate bugs, all fixed:

- **pandas 3.0 crash in the master upsert** — the local env had drifted to pandas 3.0.3 while
  `requirements.txt` pins 2.2.3. `Speciality Code`, `School`, `Year Graduated` and `Degree` are
  100% empty and round-trip through Excel as all-NaN `float64`; the whole-row assignment in
  `save_or_append_master_excel` raised `TypeError: Invalid value '<ArrowStringArray>...' for
  dtype 'float64'` where pandas 2 silently upcast. Fixed with `_align_text_column_dtypes()`,
  which coerces all-NaN float64 columns to object on both frames before the upsert.
  Version-agnostic — works on pandas 2 and 3.
- **Text cleaning silently skipped under pandas 3** — `clean_and_standardize_data` gated on
  `dtype == 'object'`, but pandas 3 gives text columns dtype `str`, so null-byte / CRLF /
  whitespace stripping ran on *nothing*. Now tests `is_object_dtype or is_string_dtype`.
  (`clean_df_for_excel`'s `select_dtypes` had the same latent issue; now passes `'str'` explicitly.)
- **Failed runs reported as successes** — items 3 and 9 below, now fixed (see them for detail).

Still open from that run: the local env remains on pandas 3.0.3 rather than the pinned 2.2.3.
The code now works on both, but `pip install -r requirements.txt` would remove the drift.
Also note `Offline Processing/ProfEngrsLandSurvyrsGeologist_Data00_structural_engineers_cleaned.csv`
is a stale Jun-2026 leftover from an older naming scheme — it matches the coordinator's
`*structural_engineers_cleaned.csv` glob and is safe to delete.

---

## P1 — Pipeline data integrity (fix before/with the next monthly run)

These can silently corrupt or stale-out the master record.

1. **Upsert overwrites good master data with blanks** — `SE_Data_Process_V9.py` ~548-556.
   When any compared column differs, *all* columns are copied from the incoming row, including
   empty ones. A partial HTML parse (address regex at `DCA_License_HTML_Parser.py` ~185 requires
   exactly a 3-line address block) wipes City/State/Zip/County; a missing expiration date becomes
   the `2099-12-31` placeholder and overwrites the real date.
   *Fix:* on `source='html'` upserts, never overwrite a non-empty master cell with an
   empty/placeholder incoming value (column-wise mask instead of whole-row assignment).

2. **Rows dropped for bad `Original Issue Date` are falsely reconciled to `Cancelled`** —
   `SE_Data_Process_V9.py` ~255-272 (drop) vs ~603-627 (reconciliation). The feed row was present
   but gets treated as a drop-off; if the detail page also lacks the date, the rescan correction
   is discarded by the same `dropna` and the false `Cancelled` never heals.
   *Fix:* track license numbers dropped for bad dates and exclude them from the "missing from
   feed" set; log them for manual review.

3. **Downloader declares success from pre-existing files; month-2 downloads land in `... (1).xls`** —
   `License_List_Claude_v3.py` ~849-908 (`check_for_download` globs `*.xls`/`*.xlsx` without
   diffing against `initial_files`); `main()` always exits 0; coordinator fallback only checks
   that `ProfEngrsLandSurvyrsGeologist_Data00.xls` exists (last month's file satisfies it).
   *Fix:* archive/delete the previous feed before downloading (or download into a clean temp dir),
   verify a *new* file appeared, and `sys.exit(1)` on failure so the coordinator stops.
   **Partly done 2026-09-12:** both `main()`s now return real exit codes and the scripts
   `sys.exit()` on them, and `run_processor` requires the cleaned CSV's mtime to be newer than
   the run start (this is what let the 2026-09 failure masquerade as a success). **Still open:**
   `check_for_download` does not diff against `initial_files`, so the downloader itself can still
   claim success from a pre-existing feed, and nothing archives the previous feed first.

## P2 — Pipeline robustness

4. **Back up the master before rewriting it** — `SE_Data_Process_V9.py` ~679 writes in place;
   `pipeline_coordinator.py` ~154-160 kills the processor at 120 s. A kill mid-save corrupts the
   only accumulating record. *Fix:* copy to `.bak` (or write-temp-then-rename) before `to_excel`;
   consider raising/removing the timeout.
5. **Rescan-queue lost-update** — `DCA_License_Fetcher.py` ~502, 544-554 rewrites
   `needs_rescan.txt` from its startup snapshot; entries appended by a concurrent monthly run are
   erased. *Fix:* re-read + union the file before each rewrite (or use a lock file).
6. **Float-key poisoning of `License Number`** — `SE_Data_Process_V9.py` ~237, ~509. One blank
   cell makes the column float64 → keys become `"1234.0"` → the whole master looks "dropped" and
   gets flipped to Cancelled + duplicated. *Fix:* normalize keys (`str.replace(r'\.0$', '')` or
   nullable `Int64`) before the merge; assert key formats match before reconciling.
7. **Fetcher dequeues on "HTML saved", not "master updated"** — `DCA_License_Fetcher.py` ~523-524;
   a page that fails parsing leaves the placeholder Cancelled with no retry. *Fix:* dequeue after
   successful parse/merge, or re-queue parse failures.
8. **Parser fallback truncates multi-word statuses** — `DCA_License_HTML_Parser.py` ~162-164 keeps
   only the first word ("Voluntary"), which isn't in `TERMINAL_STATUSES`, defeating the
   terminal-status protection. *Fix:* keep the full matched string, title-cased.
9. ~~**Coordinator + downloader `input()` deadlock**~~ — **Done 2026-09-12.**
   `License_List_Claude_v3.py` now takes `--no-prompt` and also skips the prompt whenever stdin
   isn't a TTY; `pipeline_coordinator.run_downloader` passes `--no-prompt`. Separately, the
   coordinator's "Skip download and use this existing feed file?" prompt now treats a blank Enter
   as **yes** (it previously meant "no, re-download", contradicting `_prompt`, which was the
   trigger for the 2026-09 failed run).
10. Low-severity pipeline items: `parse_date` maps unknown month names to January
    (`DCA_License_HTML_Parser.py` ~77-89); `audit_notfound.py` ~134 crashes when no filenames are
    numeric; fetcher checkpoint stamps the last *list element*, not the last processed license,
    after an aborted run; the "automatic cleaned overwrite" step re-reads/rewrites the CSV it just
    wrote with different quoting (`SE_Data_Process_V9.py` ~1007-1060).

## P3 — Dashboard correctness & UX

11. **Deselecting every checkbox shows ALL records** — `apply_filters` skips empty filter lists,
    so "State: None" displays everything. *Fix:* treat an explicit empty selection as zero rows,
    or snap back to select-all with a toast.
12. **Comparison-state overlay ignores the status filter** — built from `full_df` in
    `plot_time_series`, so with "Active only" selected the overlay still includes all statuses.
    *Fix:* apply the status filter to the comparison frame, or state the difference in the
    selectbox help text.
13. **Status colors are positional, not stable** — the donut + breakdown assign
    `CATEGORICAL_SEQUENCE` by index, so a status changes color when filters change which statuses
    appear. *Fix:* a fixed status→color map in `theme.py` (color follows the entity, never its rank).
14. **Retirements chart caption** — note that `Delinquent` is a reinstatable lapse and that
    reconciliation-placeholder `Cancelled` rows are provisional, so recent years may shrink over
    time as rescans/renewals land.

## P4 — Performance & polish

15. **Fetcher: reuse the search form** — `DCA_License_Fetcher.py` ~289-363 reloads the page and
    re-selects both dropdowns (with fixed sleeps) for every license; reuse the form and reload
    only on error. Biggest single time-saver for full runs.
16. **Vendor a CA-only GeoJSON** — `plot_ca_county_map` has the browser fetch Plotly's full
    US-counties file (~2.4 MB) from raw.githubusercontent.com on every view; ship a CA-only file
    in `data/` (also removes the GitHub-availability dependency).
17. **Cache-busting for `load_timeline_events`** — add an mtime key like `load_cleaned_data`'s so
    edits to `timeline_events.csv` show up without an app restart; also avoid caching a `None`
    load failure for the full 1-hour TTL in `load_cleaned_data`.
18. **`go.Scattergl` for the license-number scatter** — snappier zoom/pan on the ~6.7k-point
    "Licenses Issued Over Time" chart.
19. Vectorize `License_Status_Category` / `Expiring_Soon` `.apply` lambdas in the processor;
    skip the second full parse of the master workbook (`nrows=5` still parses the whole sheet).

## P5 — Story & feature backlog

20. **Cohort retention chart** — % of each issuance decade's licenses still active (or median
    license lifespan from issue → terminal status). Pairs with the average-age chart to complete
    the "aging profession" story. Route through `status_agnostic_df`.
21. **Per-capita views** — licenses per 100k residents for the county map (raw counts mostly map
    population) and optionally per-capita active licenses over time (needs a small CA population
    reference CSV).
22. **Survival curve** — Kaplan-Meier-style "probability a license is still active after N years";
    the data (issue date, expiration date, terminal status) supports it directly.
23. More timeline candidates: 1925 Santa Barbara (pre-1931, would need chart range change),
    2013 SFOBB East Span opening, code-adoption years for CBC editions.

## Security (remaining)

24. **Dev container disables XSRF/CORS** (`.devcontainer/devcontainer.json`) — fine as a
    Codespaces workaround; never mirror these flags to production. Revisit whether the forwarded
    preview works with protections on.
25. Periodically re-pin `requirements.txt` after testing upgrades (pins added 2026-07-10).

*Clean bill of health (verified 2026-07-10): no secrets or PII anywhere in git history; license
numbers integer-coerced before filesystem use; all subprocess calls list-form; no unsafe
deserialization; TLS verification intact.*
