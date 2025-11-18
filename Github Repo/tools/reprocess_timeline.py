import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'timeline_events.csv'
DST = SRC  # in-place rewrite as requested

if not SRC.exists():
    raise SystemExit(f"Source file not found: {SRC}")

df = pd.read_csv(SRC)

# Normalize expected columns
cols = {c.lower().strip(): c for c in df.columns}
get = lambda name: cols.get(name)

s_col = get('start_date') or get('start') or get('date')
e_col = get('end_date') or get('end')
l_col = get('label') or get('title')
d_col = get('description') or get('details')
t_col = get('type') or get('category')

if not s_col or not l_col or not t_col:
    raise SystemExit('Timeline CSV missing required columns (start_date/label/type).')

out = pd.DataFrame()
out['start_date'] = pd.to_datetime(df[s_col], errors='coerce').dt.strftime('%Y-%m-%d')
out['end_date'] = pd.to_datetime(df[e_col], errors='coerce').dt.strftime('%Y-%m-%d') if e_col in df.columns else ''
out['label'] = df[l_col].fillna('')
out['type'] = df[t_col].fillna('Event')

code_vals, fmt_vals, notes_vals = [], [], []

def extract_block(text: str, token: str) -> str:
    if not isinstance(text, str):
        return ''
    pattern = re.compile(rf"{re.escape(token)}\s*(.*?)(?:(?:\n|\r)\s*(Codes:|Format:|Notes:)|$)", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text)
    if not m:
        return ''
    val = m.group(1).strip()
    # Collapse internal whitespace a bit but keep line breaks
    val = re.sub(r"\s*\n\s*", "\n", val)
    return val

for desc in (df[d_col] if d_col in df.columns else ['']*len(df)):
    codes = extract_block(desc, 'Codes:')
    fmt = extract_block(desc, 'Format:')
    notes = extract_block(desc, 'Notes:')
    code_vals.append(codes)
    fmt_vals.append(fmt)
    notes_vals.append(notes)

out['Code'] = code_vals
out['Format'] = fmt_vals
out['Notes'] = notes_vals

# Save without the old description column
out.to_csv(DST, index=False)
print(f"Rewrote timeline CSV with Code/Format/Notes: {DST}")
