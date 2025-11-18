import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLE = ROOT / 'License History Table.csv'
DST = ROOT / 'timeline_events.csv'

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
for idx in df.index:
    code_vals.append(str(df.at[idx, code_col]) if code_col and code_col in df.columns else '')
    fmt_vals.append(str(df.at[idx, fmt_col]) if fmt_col and fmt_col in df.columns else '')
    notes_vals.append(str(df.at[idx, notes_col]) if notes_col and notes_col in df.columns else '')

out['Code'] = code_vals
out['Format'] = fmt_vals
out['Notes'] = notes_vals

# Save without the old description column
out.to_csv(DST, index=False)
print(f"Rewrote timeline CSV with Code/Format/Notes: {DST}")
