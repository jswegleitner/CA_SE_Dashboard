import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
SOURCE_TABLE = DATA_DIR / 'License History Table.csv'
# Hand-maintained events (earthquakes etc.) merged in on every rebuild —
# add rows there, not to timeline_events.csv, or they'll be overwritten.
EXTRA_EVENTS = DATA_DIR / 'timeline_events_extra.csv'
DST = DATA_DIR / 'timeline_events.csv'

if not SOURCE_TABLE.exists():
    raise SystemExit(f"Source file not found: {SOURCE_TABLE}")

df = pd.read_csv(SOURCE_TABLE)

# Normalize expected columns
cols = {c.lower().strip(): c for c in df.columns}
get = lambda name: cols.get(name)

s_col = get('year')
l_col = get('event / format update') or get('event') or get('label')
code_col = get('building code reference') or get('code') or get('codes')
fmt_col = get('question format') or get('format')
notes_col = get('notes')

if not s_col or not l_col:
    raise SystemExit('Source CSV missing required columns (Year or Event / Format Update).')

out = pd.DataFrame()
# Convert Year to YYYY-01-01 start_date format
out['start_date'] = pd.to_datetime(df[s_col].astype(str) + '-01-01', errors='coerce').dt.strftime('%Y-%m-%d')
out['end_date'] = ''
out['label'] = df[l_col].fillna('')
out['type'] = 'SE Exam'

code_vals, fmt_vals, notes_vals = [], [], []

# Extract structured fields from the source CSV columns


def cell(idx, col):
    if not col or col not in df.columns:
        return ''
    v = df.at[idx, col]
    return '' if pd.isna(v) else str(v)


for idx in df.index:
    code_vals.append(cell(idx, code_col))
    fmt_vals.append(cell(idx, fmt_col))
    notes_vals.append(cell(idx, notes_col))

out['Code'] = code_vals
out['Format'] = fmt_vals
out['Notes'] = notes_vals

# Merge hand-maintained events (kept in a separate CSV so this rebuild
# doesn't wipe them) and sort chronologically. ISO dates sort as strings.
if EXTRA_EVENTS.exists():
    extra = pd.read_csv(EXTRA_EVENTS).fillna('')
    extra = extra.reindex(columns=out.columns, fill_value='')
    out = pd.concat([out, extra], ignore_index=True)
    print(f"Merged {len(extra)} extra event(s) from {EXTRA_EVENTS.name}")

out = out.sort_values('start_date').reset_index(drop=True)

# Save without the old description column
out.to_csv(DST, index=False)
print(f"Rewrote timeline CSV with Code/Format/Notes: {DST} ({len(out)} events)")
