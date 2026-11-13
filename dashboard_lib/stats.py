"""Trend statistics shared by the headline metrics and per-chart stat rows.

Everything here is pure pandas so it can be unit-tested without Streamlit.
"""
from typing import Optional, Tuple

import pandas as pd

# Years of departures still considered provisional: recent lapses can renew,
# so these bars tend to shrink in later monthly updates.
PROVISIONAL_DEPARTURE_YEARS = 2

# Bins for the years-licensed histogram (lower edges, years). The last bin is open-ended.
TENURE_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SENIOR_TENURE_YEARS = 30


def _is_active(status: pd.Series) -> pd.Series:
    return status.fillna('').astype(str).str.strip().str.lower() == 'active'


def issued_per_year(df: pd.DataFrame) -> pd.Series:
    """Count of licenses by Original Issue Date year (missing years not filled)."""
    years = df['Original Issue Date'].dropna().dt.year.astype(int)
    return years.value_counts().sort_index()


def departed_per_year(df: pd.DataFrame) -> pd.Series:
    """Count of non-Active licenses by Expiration Date year (missing years not filled)."""
    leavers = df[~_is_active(df['License Status'])]
    years = leavers['Expiration Date'].dropna().dt.year.astype(int)
    return years.value_counts().sort_index()


def recent_vs_baseline(
    series: pd.Series,
    last_year: int,
    recent: int = 5,
    baseline: int = 10,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Average of the `recent` years ending at `last_year` vs. the `baseline` years before them.

    Years missing from `series` count as zero. Returns (recent_avg, baseline_avg, pct_change);
    pct_change is None when the baseline is zero.
    """
    recent_years = range(last_year - recent + 1, last_year + 1)
    base_years = range(last_year - recent - baseline + 1, last_year - recent + 1)
    full = series.reindex(list(base_years) + list(recent_years), fill_value=0)
    recent_avg = float(full.loc[list(recent_years)].mean())
    base_avg = float(full.loc[list(base_years)].mean())
    pct = (recent_avg - base_avg) / base_avg * 100 if base_avg else None
    return recent_avg, base_avg, pct


def trend_streak(series: pd.Series) -> int:
    """Consecutive year-over-year moves in the same direction ending at the last value.

    Positive = that many straight increases, negative = straight decreases, 0 = flat/none.
    """
    values = series.tolist()
    streak = 0
    for prev, cur in zip(reversed(values[:-1]), reversed(values[1:])):
        step = (cur > prev) - (cur < prev)
        if step == 0 or (streak and (step > 0) != (streak > 0)):
            break
        streak += step
    return streak


def active_on(df: pd.DataFrame, when: pd.Timestamp) -> int:
    """Licenses whose issue..expiration window covers `when` (same rule as the active-per-year chart)."""
    return int(((df['Original Issue Date'] <= when) & (df['Expiration Date'] >= when)).sum())


def years_licensed(df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Years since Original Issue Date for each Active license (NaN dates dropped)."""
    active = df[_is_active(df['License Status'])]
    issued = active['Original Issue Date'].dropna()
    return ((asof - issued).dt.days / 365.25).clip(lower=0)


def tenure_histogram(tenure: pd.Series) -> pd.Series:
    """Bucket years-licensed values into TENURE_BIN_EDGES labels ('0–4', ..., '50+')."""
    edges = TENURE_BIN_EDGES
    labels = [f"{lo}–{hi - 1}" for lo, hi in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
    cut = pd.cut(tenure, bins=edges + [float('inf')], right=False, labels=labels)
    return cut.value_counts().reindex(labels, fill_value=0)


def active_counts_by_year(df: pd.DataFrame, years) -> pd.Series:
    """Licenses active during each year (issued <= Dec 31 and expiring >= Jan 1).

    Same rule as the Active Licenses per Year chart; rows missing either date are ignored.
    """
    d = df.dropna(subset=['Original Issue Date', 'Expiration Date'])
    issue_y = d['Original Issue Date'].dt.year.to_numpy()
    exp_y = d['Expiration Date'].dt.year.to_numpy()
    return pd.Series(
        [int(((issue_y <= y) & (exp_y >= y)).sum()) for y in years],
        index=list(years), dtype=int,
    )


def retention_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Per issuance decade: licenses issued, how many are Active today, and the share.

    Columns: decade (int, e.g. 1980), label ('1980s'), issued, active, pct_active.
    """
    d = df.dropna(subset=['Original Issue Date'])
    if d.empty:
        return pd.DataFrame(columns=['decade', 'label', 'issued', 'active', 'pct_active'])
    decade = (d['Original Issue Date'].dt.year // 10 * 10).astype(int)
    active = _is_active(d['License Status'])
    out = pd.DataFrame({'decade': decade, 'active': active}).groupby('decade').agg(
        issued=('active', 'size'), active=('active', 'sum'))
    out['active'] = out['active'].astype(int)
    out['pct_active'] = out['active'] / out['issued'] * 100
    out = out.reset_index()
    out['label'] = out['decade'].astype(str) + 's'
    return out[['decade', 'label', 'issued', 'active', 'pct_active']]


def kaplan_meier(durations: pd.Series, observed: pd.Series) -> pd.Series:
    """Kaplan-Meier survival estimate.

    durations: time to event or censoring; observed: True where the event happened
    (False = still at risk when last seen). Returns S(t) indexed by each distinct
    duration, starting with S(0) = 1.
    """
    frame = pd.DataFrame({'t': pd.to_numeric(durations), 'e': observed.astype(bool)}).dropna()
    if frame.empty:
        return pd.Series([1.0], index=[0.0])
    grouped = frame.groupby('t')['e'].agg(['sum', 'size']).sort_index()
    at_risk = len(frame) - grouped['size'].cumsum().shift(fill_value=0)
    surv = (1 - grouped['sum'] / at_risk).cumprod()
    return pd.concat([pd.Series([1.0], index=[0.0]), surv[surv.index > 0]])


def license_survival(df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Kaplan-Meier curve of 'still on the rolls N years after issue'.

    A license leaves the rolls when its status is anything but Active, dated by its
    Expiration Date (the same rule as the departures chart). Active licenses are
    censored at `asof`.
    """
    d = df.dropna(subset=['Original Issue Date'])
    left = ~_is_active(d['License Status'])
    exp = d['Expiration Date'].where(d['Expiration Date'].notna(), asof)
    end = exp.where(left, asof).clip(upper=asof)
    years = ((end - d['Original Issue Date']).dt.days / 365.25).clip(lower=0)
    return kaplan_meier(years, left)


def survival_at(curve: pd.Series, t: float) -> Optional[float]:
    """S(t) from a step curve returned by kaplan_meier; None past the last observation."""
    if curve.empty or t > curve.index.max():
        return None
    return float(curve[curve.index <= t].iloc[-1])


def median_survival(curve: pd.Series) -> Optional[float]:
    """First time at which S(t) <= 0.5, or None if the curve never gets there."""
    below = curve[curve <= 0.5]
    return float(below.index[0]) if not below.empty else None
