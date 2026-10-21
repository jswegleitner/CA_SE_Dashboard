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
