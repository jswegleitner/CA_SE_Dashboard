"""Timeline event loading and hover-description formatting."""
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


@st.cache_data
def load_timeline_events() -> Optional[pd.DataFrame]:
    """
    Load optional timeline events used to annotate time-series charts.
    Supported locations (in priority order):
    - st.secrets['EVENTS_PATH'] or st.secrets['EVENTS_URL']
    - ./data/timeline_events.csv  (preferred)
    - ./timeline_events.csv       (legacy fallback)

    Expected columns (case-insensitive):
    - start_date (required)
    - end_date (optional)
    - label (short title shown on chart)
    - description (longer hover text)
    - type (category for coloring/filtering)
    """
    try:
        if 'EVENTS_PATH' in st.secrets:
            df = pd.read_csv(st.secrets['EVENTS_PATH'])
        elif 'EVENTS_URL' in st.secrets:
            df = pd.read_csv(st.secrets['EVENTS_URL'])
        else:
            df = None
    except Exception:
        df = None

    if df is None:
        project_root = Path(__file__).resolve().parents[1]
        for candidate in [
            project_root / 'data' / 'timeline_events.csv',
            project_root / 'timeline_events.csv',
        ]:
            if candidate.exists():
                try:
                    df = pd.read_csv(candidate)
                    break
                except Exception:
                    pass

    if df is None:
        return None

    cols = {c.lower().strip(): c for c in df.columns}

    def col(name):
        for k, v in cols.items():
            if k == name:
                return v
        return None

    s_col = col('start_date') or col('start') or col('date')
    e_col = col('end_date') or col('end')
    l_col = col('label') or col('title')
    d_col = col('description') or col('details')
    code_col = col('code') or col('codes')
    fmt_col = col('format')
    notes_col = col('notes')
    t_col = col('type') or col('category')

    if not s_col:
        return None

    out = pd.DataFrame()
    out['start_date'] = pd.to_datetime(df[s_col], errors='coerce')
    if e_col and e_col in df.columns:
        out['end_date'] = pd.to_datetime(df[e_col], errors='coerce')
    else:
        out['end_date'] = pd.NaT
    out['label'] = df[l_col] if l_col in df.columns else ''
    out['type'] = df[t_col] if t_col in df.columns else 'Event'

    if any(x in df.columns for x in filter(None, [code_col, fmt_col, notes_col])):
        out['code'] = df[code_col] if code_col in df.columns else ''
        out['format'] = df[fmt_col] if fmt_col in df.columns else ''
        out['notes'] = df[notes_col] if notes_col in df.columns else ''
        out['description'] = (
            'Codes: ' + out['code'].fillna('') + '\n' +
            'Format: ' + out['format'].fillna('') + '\n' +
            'Notes: ' + out['notes'].fillna('')
        ).str.strip()
    else:
        desc_series = df[d_col] if d_col in df.columns else pd.Series([''] * len(df))
        out['description'] = desc_series

        def extract_token(s: str, token: str):
            if not isinstance(s, str):
                return ''
            pattern = re.compile(
                rf"{token}\s*(.*?)(?:(?:\n|\r)\s*(Codes:|Format:|Notes:)|$)",
                re.IGNORECASE | re.DOTALL,
            )
            m = pattern.search(s)
            if not m:
                return ''
            return m.group(1).strip()

        out['code'] = desc_series.apply(lambda x: extract_token(x, 'Codes:'))
        out['format'] = desc_series.apply(lambda x: extract_token(x, 'Format:'))
        out['notes'] = desc_series.apply(lambda x: extract_token(x, 'Notes:'))

    out = out.dropna(subset=['start_date']).reset_index(drop=True)
    return out


def _sanitize_hover(text: str) -> str:
    """Escape characters that can interfere with Plotly's hover template parsing."""
    if text is None:
        return ''
    return str(text).replace('{', '&#123;').replace('}', '&#125;')


def _wrap_words(words, width: int, first_prefix: str = '', subsequent_prefix: str = '  ') -> list:
    """Greedy word wrap returning list of lines with given width constraints."""
    lines = []
    current = first_prefix.rstrip()
    for w in words:
        w = w.strip()
        if not w:
            continue
        if not current:
            current = first_prefix.rstrip()
        tentative = (current + ' ' + w).strip() if current else w
        if len(tentative) > width and current:
            lines.append(current)
            current = f"{subsequent_prefix}{w}".rstrip()
        else:
            current = tentative
    if current:
        lines.append(current)
    return lines


def _wrap_field(label: str, content: str, width: int = 55) -> str:
    """Wrap a single field (label + content) producing <br>-joined lines."""
    if content is None:
        return ''
    txt = str(content).strip()
    if not txt:
        return ''
    words = txt.replace('\r\n', ' ').replace('\n', ' ').split()
    lines = _wrap_words(words, width=width, first_prefix=f"{label} ", subsequent_prefix='  ')
    return '<br>'.join(_sanitize_hover(l) for l in lines)


def _wrap_legacy(raw: str, width: int = 55) -> str:
    """Wrap a legacy combined description string by detecting tokens and wrapping segments."""
    if raw is None:
        return ''
    s = str(raw).strip()
    if not s:
        return ''
    tokens = ['Codes:', 'Format:', 'Breadth:', 'Depth:', 'Notes:', 'Vertical:', 'Lateral:', 'Updated:', 'Components:']
    for tok in tokens:
        s = re.sub(rf'(?<!^)\s*{re.escape(tok)}', f'\n{tok}', s)
    segments = []
    for seg in s.split('\n'):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r'^(\w+:)\s*(.*)$', seg)
        if m:
            label, content = m.group(1), m.group(2)
            segments.append(_wrap_field(label, content, width=width))
        else:
            words = seg.split()
            lines = _wrap_words(words, width=width, first_prefix='', subsequent_prefix='  ')
            segments.append('<br>'.join(_sanitize_hover(l) for l in lines))
    return '<br>'.join(segments)


def format_event_description(desc: str) -> str:
    """Public interface for wrapping legacy description strings into hover-friendly HTML."""
    return _wrap_legacy(desc, width=55)


def build_event_description_from_fields(code: str, fmt: str, notes: str) -> str:
    parts = []
    if code and str(code).strip():
        parts.append(_wrap_field('Codes:', code, width=55))
    if fmt and str(fmt).strip():
        parts.append(_wrap_field('Format:', fmt, width=55))
    if notes and str(notes).strip():
        parts.append(_wrap_field('Notes:', notes, width=55))
    return '<br>'.join(parts)


def render_event_descriptions(ev: pd.DataFrame) -> pd.Series:
    """Build a per-row description series for hover text.

    Prefers structured (code/format/notes) columns when present; otherwise
    falls back to wrapping the legacy ``description`` column.
    """
    if all(col in ev.columns for col in ['code', 'format', 'notes']):
        out = [
            build_event_description_from_fields(c, f, n)
            for c, f, n in zip(
                ev['code'].fillna(''),
                ev['format'].fillna(''),
                ev['notes'].fillna(''),
            )
        ]
        return pd.Series(out, index=ev.index)
    return ev['description'].fillna('').map(format_event_description)
