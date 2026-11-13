# -*- coding: utf-8 -*-
"""Structural exam results: loader and pure helpers for the History tab.

``data/se_exam_results.csv`` holds one row per exam sitting, exam piece and
population, cut down from the full research dataset in
``research/se_exam_pass_rates.csv`` to the series the History tab draws.

The exam changed shape four times, so a pass rate means something different in
each era: the whole exam (2003), one of two separate exams (2004-2011), one
16-hour component (2011-2023), or one depth section (2024 on). Rows therefore
carry a ``strand`` that follows a part of the exam through those changes:

- ``vertical``: gravity design. The Vertical Forces component, then the
  Vertical Depth section.
- ``lateral``: seismic and wind design. California's own SE III exam (which
  kept the state's seismic emphasis), then the Lateral Forces component, then
  the Lateral Depth section.
- ``whole``: a result for the whole exam or for a general exam covering both,
  such as NCEES Structural II.

Everything here is free of Streamlit so it can be unit-tested directly.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd

EXAM_RESULTS_FILENAME = 'se_exam_results.csv'

# (key, first sitting, last sitting or None, short label, what one pass rate measures)
EXAM_ERAS = [
    ('state', '1931-01', '2003-12', 'State exam',
     'California wrote and graded the whole 16-hour exam. One result survives.'),
    ('split', '2004-01', '2011-03', 'Structural II + SE III',
     'Two separate 8-hour exams: the national Structural II and the California SE III.'),
    ('sixteen', '2011-04', '2023-12', '16-hour exam',
     'One national exam in two 8-hour components, each passed separately.'),
    ('cbt', '2024-01', None, 'Computer-based',
     'Four sections passed separately. The depth sections are the hard part.'),
]
ERA_KEYS = [e[0] for e in EXAM_ERAS]

STRANDS = ['vertical', 'lateral', 'whole']
STRAND_LABELS = {
    'vertical': 'Vertical forces',
    'lateral': 'Lateral forces and seismic',
    'whole': 'Whole or general exam',
}
SCOPES = ('California', 'National')
POPULATIONS = ('all', 'first_time')
EVIDENCE = ('primary', 'secondary', 'derived', '')

REQUIRED_COLUMNS = ['sitting', 'era', 'strand', 'exam', 'scope', 'population',
                    'tested', 'passed', 'pass_rate_pct', 'evidence', 'note', 'source']


def exam_results_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'data' / EXAM_RESULTS_FILENAME


def sitting_date(sitting: str) -> pd.Timestamp:
    """'2013-04' -> 15 April 2013, mid-month so markers sit inside their month."""
    ts = pd.to_datetime(f"{sitting}-15", format='%Y-%m-%d')
    return pd.Timestamp(ts)


def read_exam_results_csv(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Read and validate the exam results file. Returns None when it is absent.

    Raises ValueError on a malformed file so a bad hand edit fails a test
    instead of silently dropping points from the chart.
    """
    path = Path(path) if path is not None else exam_results_path()
    if not path.exists():
        return None

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {', '.join(missing)}")
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    def _bad(mask, what):
        if mask.any():
            where = ', '.join(sorted(set(df.loc[mask, 'sitting'])))
            raise ValueError(f"{path.name}: {what} (sittings: {where})")

    _bad(~df['sitting'].str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])'), 'sitting must be YYYY-MM')
    _bad(~df['era'].isin(ERA_KEYS), f"era must be one of {', '.join(ERA_KEYS)}")
    _bad(~df['strand'].isin(STRANDS), f"strand must be one of {', '.join(STRANDS)}")
    _bad(~df['scope'].isin(SCOPES), 'scope must be California or National')
    _bad(~df['population'].isin(POPULATIONS), 'population must be all or first_time')
    _bad(~df['evidence'].isin(EVIDENCE), 'evidence must be primary, secondary or derived')

    for col in ('tested', 'passed', 'pass_rate_pct'):
        raw = df[col]
        df[col] = pd.to_numeric(raw.replace('', None), errors='coerce')
        _bad(raw.ne('') & df[col].isna(), f"{col} is not a number")
    _bad(df['pass_rate_pct'].lt(0) | df['pass_rate_pct'].gt(100), 'pass_rate_pct outside 0-100')
    _bad(df['passed'].gt(df['tested']), 'passed exceeds tested')
    _bad(df['pass_rate_pct'].notna() & df['evidence'].eq(''), 'a result needs an evidence flag')

    df['date'] = df['sitting'].map(sitting_date)
    df['passed_est'] = estimated_passed(df)
    return df.sort_values(['date', 'scope', 'strand'], kind='stable').reset_index(drop=True)


def estimated_passed(df: pd.DataFrame) -> pd.Series:
    """Passed count, or tested x rate when only those two were published.

    The Board's computer-based exam charts give exams administered and a pass
    rate but no count, so the count is rounded from the two.
    """
    est = (df['tested'] * df['pass_rate_pct'] / 100.0).round()
    return df['passed'].where(df['passed'].notna(), est)


def era_of(ts) -> str:
    """Era key for a date, by the era boundaries (not the row's own era)."""
    month = pd.Timestamp(ts).strftime('%Y-%m')
    for key, start, end, _label, _desc in EXAM_ERAS:
        if month >= start and (end is None or month <= end):
            return key
    return EXAM_ERAS[-1][0]


def era_bounds(key: str, today: Optional[pd.Timestamp] = None):
    """(start, end) timestamps of an era, the open era ending today."""
    for k, start, end, _label, _desc in EXAM_ERAS:
        if k == key:
            lo = pd.Timestamp(f"{start}-01")
            hi = (pd.Timestamp(f"{end}-01") + pd.offsets.MonthEnd(0)) if end else pd.Timestamp(today or pd.Timestamp.now())
            return lo, hi
    raise KeyError(key)


def sittings(results: pd.DataFrame, since: str = '2003-01') -> List[pd.Timestamp]:
    """Every exam sitting in the file from ``since`` on, California or national."""
    dates = results.loc[results['sitting'] >= since, 'date']
    return sorted(pd.Timestamp(d) for d in dates.unique())


def licenses_after_sittings(licenses: pd.DataFrame, sitting_dates: List[pd.Timestamp],
                            today: Optional[pd.Timestamp] = None,
                            date_col: str = 'Original Issue Date') -> pd.DataFrame:
    """Licenses issued between each sitting and the next.

    A window runs from the first of the month after a sitting to the end of the
    month of the next sitting, so the license batch that follows each exam's
    results lands in that exam's window. The last window runs to ``today`` and
    is flagged partial.
    """
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    cols = ['date', 'window_start', 'window_end', 'licenses', 'partial']
    if licenses is None or date_col not in licenses.columns or not sitting_dates:
        return pd.DataFrame(columns=cols)
    issued = pd.to_datetime(licenses[date_col], errors='coerce').dropna()

    rows = []
    for i, s in enumerate(sitting_dates):
        start = (pd.Timestamp(s) + pd.offsets.MonthBegin(1)).normalize()
        if i + 1 < len(sitting_dates):
            end = (pd.Timestamp(sitting_dates[i + 1]) + pd.offsets.MonthEnd(0)).normalize()
            partial = end > today
        else:
            end, partial = today, True
        end = min(end, today)
        if start > today:
            continue
        n = int(((issued >= start) & (issued < end + pd.Timedelta(days=1))).sum())
        rows.append(dict(date=pd.Timestamp(s), window_start=start, window_end=end, licenses=n, partial=partial))
    return pd.DataFrame(rows, columns=cols)


def era_rate_summary(results: pd.DataFrame, scope: str = 'California') -> pd.DataFrame:
    """Median, low and high pass rate per era across exam pieces and sittings.

    Rows are grouped by their own ``era`` (the exam they belong to), so the
    last SE III sitting in October 2011 counts with SE III. From 2011 only the
    separately passed pieces count: each component, then each depth section.
    The 2011 results the Board printed for the whole 16-hour exam are left out
    because they measure something else.
    """
    work = results[(results['scope'] == scope) & results['pass_rate_pct'].notna()]
    work = work[~(work['era'].isin(['sixteen', 'cbt']) & (work['strand'] == 'whole'))]
    out = []
    for key, _start, _end, label, _desc in EXAM_ERAS:
        grp = work[work['era'] == key]
        if grp.empty:
            continue
        rates = grp['pass_rate_pct']
        out.append(dict(
            era=key, label=label, n=len(grp),
            median=float(rates.median()), low=float(rates.min()), high=float(rates.max()),
            first=grp['date'].min(), last=grp['date'].max(),
        ))
    return pd.DataFrame(out)
