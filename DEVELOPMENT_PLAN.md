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

## Completed 2026-11-12 — backlog sweep

Pipeline (`Offline Processing/`, gitignored; pre-edit copies were taken before changing them):

- **P1-1 Upsert blank-overwrite** — `source='html'` upserts now fill empty incoming cells (and the
  `2099-12-31` expiration placeholder) from the master column by column (`_fill_blanks_from_master`)
  before diffing. Brand-new html licenses with no issue date are skipped with a warning.
- **P1-2 Bad-issue-date false Cancels** — `invalid_issue_date_licenses()` records feed rows dropped
  for an unparseable date; reconciliation skips them and writes `licenses/bad_issue_date_review.txt`.
  The html path no longer drops such rows (`clean_and_standardize_data(drop_invalid_issue_dates=False)`),
  so a rescan's status still lands. The 2026-11 feed has 4 of these: 2179, 3340, 3366, 3392.
- **P1-3 Stale-feed success** — the downloader side was already fixed (archives the old feed, waits for a
  new mtime). The coordinator now also rejects a feed whose mtime predates the download run.
- **P2-4 Master backup** — `_backup_master` keeps the last 6 copies in `master_backups/`;
  the master is written temp-file-then-`os.replace`. Coordinator processor timeout raised 2 → 15 min.
- **P2-5 Rescan-queue lost update** — `write_queue(..., resolved=)` re-reads the file and writes
  (file ∪ remaining) − resolved; it also runs in `finally`, so Ctrl+C keeps progress.
- **P2-6 Float-key poisoning** — `normalize_license_key()` on both sides; blank keys dropped. Safety
  stop: a feed reconciliation that would flip more than max(50, 10% of the master) to Cancelled aborts
  without saving (`--allow-mass-dropoff` overrides).
- **P2-7 Dequeue on parse** — `--rescan` dequeues a saved page only if `parse_html` yields a status.
- **P2-8 Truncated status** — the parser's fallback keeps the full status string.
- **P2-10** — unknown month names → None (were January); `audit_notfound.py` no longer crashes on zero
  rows; fetcher checkpoints the last license actually processed; the "automatic cleaned overwrite" step
  skips the CSV it just wrote.
- **P4-19** — status-category lambdas vectorized; save verification uses a read-only row count.
- **Found during testing — pandas 3 stamp bug.** `Status_Changed_Date` blanks stay missing through
  `astype(str)` on pandas 3, and `NaN != ''` is True. Every run since the drift treated all ~4.8k
  non-Cancelled rows as "re-emerged" (rewriting their `Last_Updated`), and **new drop-offs were never
  stamped**. Fixed with `_has_stamp()`; stamp columns are now included in `_align_text_column_dtypes`.
  Dashboard data was not affected (neither column is published).
- Stale Jun-2026 `..._Data00_structural_engineers_cleaned.csv` moved to `Archived License Lists/`.

Dashboard:

- **P3-11** — `apply_filters` treats `None` as "no filter" and `[]` as "zero rows"; chips show "None"
  and the empty page explains how to recover.
- **P3-12** — comparison overlay is built from every filter except State, so it honours License Status.
- **P4-16** — `data/ca_counties.geojson` (58 counties, ~70 KB) replaces the 3 MB GitHub fetch.
- **P4-17** — cache keys fixed. `_file_mtime` was underscore-prefixed, which Streamlit *excludes* from
  the cache key, so file-change busting never worked. Load errors are no longer cached for the TTL.
  Timeline events are keyed on the CSV's mtime.
- **P4-18** — `go.Scattergl` for the license-number scatter.
- **P5-20/21/22** — new Time-tab charts, all status-agnostic: *Active California Licenses per 100,000
  Californians* (`data/ca_population_annual.csv`, FRED `CAPOP`), *Share of Each Decade's Licenses
  Still Active Today*, and a Kaplan-Meier *How Long Licenses Stay on the Rolls*. Pure helpers in
  `dashboard_lib/stats.py`.
- **P5-23 (partial)** — 2013 SFOBB East Span opening added to `timeline_events_extra.csv`.

---

## Completed 2026-11-13 — History tab

- New **History** tab: intro on what the S.E. title authority is, stat row from the license
  numbering, an event-lane timeline over licenses issued per year (`plot_history_timeline`),
  a kind-of-event filter, notes on reading the counts, a sourced chronology grouped by era,
  and recollections from the EERI oral histories.
- `data/licensure_history.csv` (37 events) is now the single source for history events. Rows with
  `on_chart` = 1 feed the Time-tab overlays, so `data/timeline_events.csv`,
  `data/timeline_events_extra.csv`, `data/License History Table.csv` and
  `tools/reprocess_timeline.py` were retired. No build step.
- Corrected in passing: NCEES Structural I / II date from fall 1985 / fall 1987 (the old table said
  1997); the unsupported 1957 "separates CE and SE practice scopes" row and the unverifiable
  per-cycle "exam references updated" rows were dropped.
- Time-tab bar overlay draws one marker per bucket and lists every event in it (two events in one
  year used to hide each other); event lines on the last chart use fixed per-category colors.
- `dashboard_lib/history.py` (pure helpers) + `tests/test_history.py`, including guards on the
  shipped CSV. Sources indexed in `research/SOURCES.md`; PDFs in `research/sources/` are gitignored.
- Open questions are listed at the end of `research/SOURCES.md` (text of Chapter 254, Statutes of
  1931; whether the first ~370 titles were granted without examination).

---

## Still open

- **P2 (verify on next run)** — none of the pipeline changes have run against a live DCA download or
  rescan yet. They were tested against a copy of the 2026-11 feed and master (the dashboard CSV
  it produced had the same content as the published one). Watch the first December run's log.
- **P4-15 Fetcher: reuse the search form** — not done; needs tuning against the live site and its
  Cloudflare behaviour. Still the biggest time-saver for full runs.
- **P5-23** — 1925 Santa Barbara (needs chart-range change); CBC edition adoption years (need sourced dates).
- **Environment** — local pandas is still 3.0.3 versus the 2.2.3 pin. Code is written for both, but the
  new tests have only run on pandas 3.

## Security (remaining)

24. **Dev container disables XSRF/CORS** (`.devcontainer/devcontainer.json`) — fine as a
    Codespaces workaround; never mirror these flags to production. Revisit whether the forwarded
    preview works with protections on.
25. Periodically re-pin `requirements.txt` after testing upgrades (pins added 2026-07-10).

*Clean bill of health (verified 2026-07-10): no secrets or PII anywhere in git history; license
numbers integer-coerced before filesystem use; all subprocess calls list-form; no unsafe
deserialization; TLS verification intact.*
