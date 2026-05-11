"""Period bucketing primitives for time-series charts."""
from typing import Optional

import pandas as pd


def _assign_period(df, bucket_size: str):
    """Assign a 'Period' column to df based on the selected bucket size. Returns the modified df."""
    if bucket_size == "Yearly":
        df['Period'] = df['Original Issue Date'].dt.to_period('Y')
    elif bucket_size == "Half-Yearly":
        df['Year'] = df['Original Issue Date'].dt.year
        df['Half'] = ((df['Original Issue Date'].dt.month - 1) // 6) + 1
        df['Period'] = df['Year'].astype(str) + '-H' + df['Half'].astype(str)
    elif bucket_size == "Monthly":
        df['Period'] = df['Original Issue Date'].dt.to_period('M')
    else:  # Quarterly
        df['Period'] = df['Original Issue Date'].dt.to_period('Q')
    return df


def _build_full_index(bucket_size: str, start_year: Optional[int], end_year: Optional[int]):
    """Build a full period index to fill empty buckets, or None if years not specified."""
    if start_year is None or end_year is None:
        return None
    start = f"{int(start_year)}-01-01"
    end = f"{int(end_year)}-12-31"
    if bucket_size == "Yearly":
        return pd.period_range(start=start, end=end, freq='Y')
    elif bucket_size == "Monthly":
        return pd.period_range(start=start, end=end, freq='M')
    elif bucket_size == "Half-Yearly":
        labels = []
        for y in range(int(start_year), int(end_year) + 1):
            labels.append(f"{y}-H1")
            labels.append(f"{y}-H2")
        return labels
    else:  # Quarterly
        return pd.period_range(start=start, end=end, freq='Q')


def _count_by_period(df, bucket_size: str, start_year: Optional[int], end_year: Optional[int]):
    """Assign periods, group, reindex to fill gaps, and return sorted counts."""
    df = _assign_period(df.copy(), bucket_size)
    counts = df.groupby('Period').size()
    full_index = _build_full_index(bucket_size, start_year, end_year)
    if full_index is not None:
        counts = counts.reindex(full_index, fill_value=0)
    return counts.sort_index()


def to_bucket_label(dt, bucket_size: str):
    """Map a single datetime to the bucket label used on the x-axis."""
    if pd.isna(dt):
        return None
    if bucket_size == 'Yearly':
        return str(pd.Period(dt, 'Y'))
    if bucket_size == 'Half-Yearly':
        return f"{dt.year}-H{1 if dt.month <= 6 else 2}"
    if bucket_size == 'Quarterly':
        return str(pd.Period(dt, 'Q'))
    if bucket_size == 'Monthly':
        return str(pd.Period(dt, 'M'))
    return None
