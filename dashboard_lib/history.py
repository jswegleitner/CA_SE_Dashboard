# -*- coding: utf-8 -*-
"""Licensure history: loader and pure helpers for the History tab.

``data/licensure_history.csv`` is the single hand-maintained record of
licensure, exam, building-code, legislative and earthquake events. The History
tab renders all of it; rows flagged ``on_chart`` also feed the event overlays
on the Time tab (see ``dashboard_lib.timeline``).

Everything here is free of Streamlit so it can be unit-tested directly.
"""
import html
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

HISTORY_FILENAME = 'licensure_history.csv'

# Lane order on the History chart, top to bottom. Colors live in theme.py.
HISTORY_CATEGORIES = [
    'Licensure',
    'Exam',
    'Building code',
    'Legislation',
    'Earthquake or disaster',
]

REQUIRED_COLUMNS = ['date', 'precision', 'category', 'title', 'summary', 'significance', 'sources']
OPTIONAL_COLUMNS = ['data_note', 'codes', 'exam_format', 'on_chart', 'chart_label', 'chart_note']
PRECISIONS = ('day', 'month', 'year')

# (first year, last year or None for open-ended, heading, one-line framing)
ERAS = [
    (None, 1928, "Before the license",
     "Earthquakes and a dam failure, with no engineering license and no statewide seismic rule."),
    (1929, 1933, "A license, a title and a purpose",
     "In five years California licensed civil engineers, created the structural title and "
     "reserved public schools for the people who held it."),
    (1934, 1970, "The profession writes the code",
     "Licensing grew slowly while California's structural engineers built the seismic "
     "provisions the rest of the West adopted."),
    (1971, 1984, "San Fernando and its consequences",
     "Modern buildings failed, hospitals joined schools on the reserved list, and the codes "
     "were rebuilt."),
    (1985, 2003, "New exams, and the code tested twice",
     "Scoring and exam content changed, and two damaging earthquakes turned attention to "
     "existing buildings."),
    (2004, None, "National codes, a national exam",
     "California gave up its own structural exam and its own building code within a few "
     "years of each other."),
]


def history_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'data' / HISTORY_FILENAME


def read_history_csv(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Read and validate the history CSV. Returns None when the file is absent.

    Raises ValueError on a malformed file so a bad hand-edit is caught in tests
    rather than silently dropping events.
    """
    path = Path(path) if path is not None else history_path()
    if not path.exists():
        return None

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {', '.join(missing)}")
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    bad_dates = df[df['date'].isna()]
    if not bad_dates.empty:
        raise ValueError(f"{path.name}: unparseable date for: {', '.join(bad_dates['title'])}")

    df['precision'] = df['precision'].str.lower().replace('', 'day')
    bad_precision = df[~df['precision'].isin(PRECISIONS)]
    if not bad_precision.empty:
        raise ValueError(f"{path.name}: precision must be day, month or year "
                         f"({', '.join(bad_precision['title'])})")

    bad_category = df[~df['category'].isin(HISTORY_CATEGORIES)]
    if not bad_category.empty:
        raise ValueError(f"{path.name}: unknown category for: {', '.join(bad_category['title'])}")

    df['on_chart'] = df['on_chart'].str.lower().isin(['1', 'true', 'yes', 'y'])
    df['year'] = df['date'].dt.year.astype(int)
    df = df.sort_values(['date', 'title'], kind='stable').reset_index(drop=True)
    return df


def format_event_date(ts, precision: str = 'day') -> str:
    """'April 10, 1933', 'May 1933' or '1931' depending on how precisely the date is known."""
    ts = pd.Timestamp(ts)
    if precision == 'year':
        return str(ts.year)
    if precision == 'month':
        return f"{ts.strftime('%B')} {ts.year}"
    return f"{ts.strftime('%B')} {ts.day}, {ts.year}"


def plot_position(ts, precision: str = 'day') -> pd.Timestamp:
    """Where to draw an event on a time axis.

    A year-only event sits mid-year and a month-only event mid-month, so they
    don't all pile up on January 1 or the 1st.
    """
    ts = pd.Timestamp(ts)
    if precision == 'year':
        return pd.Timestamp(year=ts.year, month=7, day=1)
    if precision == 'month':
        return pd.Timestamp(year=ts.year, month=ts.month, day=15)
    return ts


_SOURCE_RE = re.compile(r'^(?P<label>.*?)\s*<(?P<url>https?://[^>\s]+)>\s*$')


def parse_sources(text: str) -> List[Tuple[str, Optional[str]]]:
    """Split 'Label <url>; Label; Label <url>' into (label, url-or-None) pairs."""
    out = []
    for part in str(text or '').split(';'):
        part = part.strip()
        if not part:
            continue
        m = _SOURCE_RE.match(part)
        if m:
            out.append((m.group('label').strip() or m.group('url'), m.group('url')))
        else:
            out.append((part, None))
    return out


def era_index(year: int) -> int:
    """Index into ERAS for a year."""
    for i, (start, end, _title, _blurb) in enumerate(ERAS):
        if (start is None or year >= start) and (end is None or year <= end):
            return i
    return len(ERAS) - 1


def era_label(index: int, today_year: Optional[int] = None) -> str:
    """Heading like 'A license, a title and a purpose (1929 to 1933)'."""
    start, end, title, _ = ERAS[index]
    if start is None:
        span = f"to {end}"
    elif end is None:
        span = f"{start} to today"
    else:
        span = f"{start} to {end}"
    return f"{title} ({span})"


def lane_offsets(years: List[float], min_gap: float = 2.6, step: float = 0.2) -> List[float]:
    """Vertical nudges that keep markers close in time from sitting on top of each other.

    ``years`` are fractional years for the events of one lane, in chronological
    order. An event within ``min_gap`` years of the previous one is nudged to the
    opposite side of the lane's centerline; isolated events stay centered.
    """
    offsets: List[float] = []
    prev_year = None
    prev_offset = 0.0
    for y in years:
        if prev_year is not None and (y - prev_year) < min_gap:
            offset = step if prev_offset <= 0 else -step
            if prev_offset == 0.0 and offsets:
                # First collision: move the earlier marker down so the pair straddles the line.
                offsets[-1] = -step
        else:
            offset = 0.0
        offsets.append(offset)
        prev_year, prev_offset = y, offset
    return offsets


def fractional_year(ts) -> float:
    ts = pd.Timestamp(ts)
    return ts.year + (ts.dayofyear - 1) / 366.0


def license_number_milestones(df: pd.DataFrame, step: int = 1000,
                              number_col: str = 'License Number',
                              date_col: str = 'Original Issue Date',
                              window: int = 15) -> List[Tuple[int, pd.Timestamp]]:
    """Approximate issue date of license No. 1 and of every ``step``-th license.

    License numbers are sequential, so the date a round number was reached is the
    typical issue date of its neighbors. The median over a small window of
    neighboring numbers keeps one re-dated record (see the issue-date anomaly
    notes) from moving a milestone by decades.
    """
    if df is None or df.empty or number_col not in df.columns or date_col not in df.columns:
        return []
    work = pd.DataFrame({
        'n': pd.to_numeric(df[number_col], errors='coerce'),
        'd': pd.to_datetime(df[date_col], errors='coerce'),
    }).dropna()
    if work.empty:
        return []
    work = work.sort_values('n')

    out: List[Tuple[int, pd.Timestamp]] = []
    first = work.iloc[0]
    out.append((int(first['n']), first['d']))
    top = int(work['n'].max())
    for target in range(step, top + 1, step):
        near = work[(work['n'] >= target - window) & (work['n'] <= target + window)]
        if near.empty:
            continue
        # Middle element by date order: resolution-agnostic (pandas 2 stores ns,
        # pandas 3 may store s/us), unlike a median over the integer view.
        ordered = near['d'].sort_values()
        out.append((target, pd.Timestamp(ordered.iloc[len(ordered) // 2])))
    return out


def early_record_coverage(df: pd.DataFrame, before_year: int = 1940,
                          number_col: str = 'License Number',
                          date_col: str = 'Original Issue Date') -> Optional[Tuple[int, int]]:
    """(records present, licenses numbered) for licenses issued before ``before_year``.

    The public feed has lost most of the earliest licenses, so the count of
    surviving records is far below the count the numbering implies.
    """
    if df is None or df.empty or number_col not in df.columns or date_col not in df.columns:
        return None
    work = pd.DataFrame({
        'n': pd.to_numeric(df[number_col], errors='coerce'),
        'd': pd.to_datetime(df[date_col], errors='coerce'),
    }).dropna()
    early = work[work['d'].dt.year < before_year]
    if early.empty:
        return None
    # 0.98 quantile rather than max: a single re-numbered record should not set the total.
    numbered = int(early['n'].quantile(0.98))
    return len(early), max(numbered, len(early))


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def event_card_html(row, color: str) -> str:
    """One event as a self-contained HTML card (single line, safe for st.markdown)."""
    parts = [
        f'<div class="hist-card" style="border-left-color:{_esc(color)};">',
        '<div class="hist-meta">',
        f'<span class="hist-date">{_esc(format_event_date(row["date"], row["precision"]))}</span>',
        f'<span class="hist-cat"><span class="hist-dot" style="background:{_esc(color)};"></span>'
        f'{_esc(row["category"])}</span>',
        '</div>',
        f'<div class="hist-title">{_esc(row["title"])}</div>',
        f'<p class="hist-text">{_esc(row["summary"])}</p>',
        f'<p class="hist-text"><span class="hist-label">Why it matters</span>{_esc(row["significance"])}</p>',
    ]
    if row.get('data_note'):
        parts.append(
            f'<p class="hist-text hist-data"><span class="hist-label">In the license data</span>'
            f'{_esc(row["data_note"])}</p>'
        )
    if row.get('exam_format'):
        parts.append(
            f'<p class="hist-text"><span class="hist-label">Exam format</span>{_esc(row["exam_format"])}</p>'
        )
    if row.get('codes'):
        parts.append(
            f'<p class="hist-text"><span class="hist-label">Codes referenced</span>{_esc(row["codes"])}</p>'
        )
    sources = parse_sources(row.get('sources', ''))
    if sources:
        links = []
        for label, url in sources:
            if url:
                links.append(f'<a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(label)}</a>')
            else:
                links.append(_esc(label))
        parts.append(f'<p class="hist-src">Sources: {" · ".join(links)}</p>')
    parts.append('</div>')
    return ''.join(parts)
