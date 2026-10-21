"""Tests for dashboard_lib.stats and the stat-card helpers built on it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from dashboard_lib.stats import (
    recent_vs_baseline,
    trend_streak,
    active_on,
    years_licensed,
    tenure_histogram,
    issued_per_year,
    departed_per_year,
)
from dashboard_lib.theme import status_color, STATUS_COLORS
from SE_Dashboard_v7 import _drop_future_periods, _period_start, _fmt_delta, _streak_text


def _df(rows):
    df = pd.DataFrame(rows, columns=['Original Issue Date', 'Expiration Date', 'License Status'])
    df['Original Issue Date'] = pd.to_datetime(df['Original Issue Date'])
    df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
    return df


class TestRecentVsBaseline:
    def test_basic(self):
        s = pd.Series({2016: 10, 2017: 10, 2018: 20, 2019: 20})
        recent, base, pct = recent_vs_baseline(s, 2019, recent=2, baseline=2)
        assert recent == 20 and base == 10 and pct == pytest.approx(100.0)

    def test_missing_years_count_as_zero(self):
        s = pd.Series({2019: 10})
        recent, base, pct = recent_vs_baseline(s, 2019, recent=2, baseline=2)
        assert recent == 5 and base == 0 and pct is None


class TestTrendStreak:
    @pytest.mark.parametrize('values,expected', [
        ([1, 2, 3], 2),
        ([3, 2, 1], -2),
        ([3, 2, 4], 1),
        ([1, 5, 4, 3], -2),
        ([2, 2], 0),
        ([5], 0),
    ])
    def test_streaks(self, values, expected):
        assert trend_streak(pd.Series(values)) == expected

    def test_streak_text(self):
        assert _streak_text(-3) == "3 yrs falling"
        assert _streak_text(1) == "1 yr rising"
        assert _streak_text(0) == "Flat"


class TestCounts:
    def test_issued_and_departed(self):
        df = _df([
            ('2020-03-01', '2030-01-01', 'Active'),
            ('2020-06-01', '2022-06-30', 'Retired'),
            ('2021-01-01', '2022-01-31', 'Delinquent'),
        ])
        assert issued_per_year(df).to_dict() == {2020: 2, 2021: 1}
        assert departed_per_year(df).to_dict() == {2022: 2}

    def test_active_on(self):
        df = _df([
            ('2020-01-01', '2030-01-01', 'Active'),
            ('2020-01-01', '2021-01-01', 'Cancelled'),
        ])
        assert active_on(df, pd.Timestamp('2020-06-01')) == 2
        assert active_on(df, pd.Timestamp('2022-06-01')) == 1


class TestTenure:
    def test_only_active_counted(self):
        df = _df([
            ('1990-01-01', '2030-01-01', 'Active'),
            ('2020-01-01', '2030-01-01', 'Active'),
            ('1980-01-01', '2000-01-01', 'Retired'),
        ])
        t = years_licensed(df, pd.Timestamp('2025-01-01'))
        assert len(t) == 2
        assert sorted(t.round().tolist()) == [5.0, 35.0]

    def test_histogram_bins(self):
        hist = tenure_histogram(pd.Series([0.5, 4.99, 5.0, 31, 72]))
        assert hist['0–4'] == 2
        assert hist['5–9'] == 1
        assert hist['30–34'] == 1
        assert hist['50+'] == 1
        assert hist.sum() == 5


class TestPeriods:
    def test_drop_future_periods(self):
        now = pd.Timestamp.now()
        idx = pd.period_range(start=f"{now.year}-01", end=f"{now.year + 1}-12", freq='M')
        out = _drop_future_periods(pd.Series(0, index=idx))
        assert out.index.max() == pd.Period(now, 'M')

    def test_half_year_start(self):
        assert _period_start('2026-H2') == pd.Timestamp('2026-07-01')
        assert _period_start('2026-H1') == pd.Timestamp('2026-01-01')


class TestFormatting:
    def test_fmt_delta_signs(self):
        assert _fmt_delta(-62.4, "vs. a year ago") == "-62 vs. a year ago"
        assert _fmt_delta(3, "x") == "+3 x"
        assert _fmt_delta(0.2, "x") == "0 x"
        assert _fmt_delta(None, "x") is None


class TestStatusColors:
    def test_known_status_is_stable(self):
        assert status_color('Active') == STATUS_COLORS['Active']
        assert status_color(' active ') == STATUS_COLORS['Active']

    def test_unknown_status_deterministic(self):
        assert status_color('Suspended') == status_color('Suspended')
