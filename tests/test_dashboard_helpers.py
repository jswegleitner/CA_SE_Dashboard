# -*- coding: utf-8 -*-
"""Unit tests for dashboard helper functions."""

import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

# Import helpers from the dashboard module (avoid triggering Streamlit app startup)
from SE_Dashboard_v7 import (
    _sanitize_hover,
    _wrap_words,
    _wrap_field,
    format_event_description,
    build_event_description_from_fields,
    _assign_period,
    _build_full_index,
    _count_by_period,
    parse_dates,
)


# ---------- _sanitize_hover ----------

class TestSanitizeHover:
    def test_escapes_braces(self):
        assert _sanitize_hover("{hello}") == "&#123;hello&#125;"

    def test_none_returns_empty(self):
        assert _sanitize_hover(None) == ""

    def test_plain_string_unchanged(self):
        assert _sanitize_hover("plain text") == "plain text"


# ---------- _wrap_words ----------

class TestWrapWords:
    def test_basic_wrap(self):
        words = ["a", "very", "long", "sentence", "here"]
        result = _wrap_words(words, width=12)
        assert len(result) > 1  # Should have wrapped

    def test_empty_input(self):
        assert _wrap_words([], width=50) == []

    def test_single_word(self):
        assert _wrap_words(["hello"], width=50) == ["hello"]


# ---------- _wrap_field ----------

class TestWrapField:
    def test_returns_empty_for_none(self):
        assert _wrap_field("Label:", None) == ""

    def test_returns_empty_for_blank(self):
        assert _wrap_field("Label:", "   ") == ""

    def test_basic_field(self):
        result = _wrap_field("Codes:", "IBC 2021, ASCE 7-22")
        assert "Codes:" in result
        assert "IBC" in result


# ---------- format_event_description ----------

class TestFormatEventDescription:
    def test_none_returns_empty(self):
        assert format_event_description(None) == ""

    def test_basic_description(self):
        desc = "Codes: IBC 2021 Format: Computer-based Notes: Updated exam"
        result = format_event_description(desc)
        assert "Codes:" in result
        assert "Format:" in result
        assert "Notes:" in result


# ---------- build_event_description_from_fields ----------

class TestBuildEventDescription:
    def test_all_fields(self):
        result = build_event_description_from_fields("IBC 2021", "CBT", "New format")
        assert "Codes:" in result
        assert "Format:" in result
        assert "Notes:" in result

    def test_empty_fields(self):
        result = build_event_description_from_fields("", "", "")
        assert result == ""

    def test_partial_fields(self):
        result = build_event_description_from_fields("IBC 2021", "", "")
        assert "Codes:" in result
        assert "Format:" not in result


# ---------- _assign_period ----------

class TestAssignPeriod:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'Original Issue Date': pd.to_datetime(['2020-01-15', '2020-07-20', '2021-03-10'])
        })

    def test_yearly(self, sample_df):
        result = _assign_period(sample_df.copy(), "Yearly")
        assert 'Period' in result.columns
        assert str(result['Period'].iloc[0]) == '2020'

    def test_half_yearly(self, sample_df):
        result = _assign_period(sample_df.copy(), "Half-Yearly")
        assert result['Period'].iloc[0] == '2020-H1'
        assert result['Period'].iloc[1] == '2020-H2'

    def test_quarterly(self, sample_df):
        result = _assign_period(sample_df.copy(), "Quarterly")
        assert str(result['Period'].iloc[0]) == '2020Q1'
        assert str(result['Period'].iloc[2]) == '2021Q1'

    def test_monthly(self, sample_df):
        result = _assign_period(sample_df.copy(), "Monthly")
        assert str(result['Period'].iloc[0]) == '2020-01'
        assert str(result['Period'].iloc[1]) == '2020-07'


# ---------- _build_full_index ----------

class TestBuildFullIndex:
    def test_returns_none_when_years_missing(self):
        assert _build_full_index("Yearly", None, None) is None
        assert _build_full_index("Yearly", 2020, None) is None

    def test_yearly_range(self):
        idx = _build_full_index("Yearly", 2020, 2022)
        assert len(idx) == 3  # 2020, 2021, 2022

    def test_half_yearly_range(self):
        idx = _build_full_index("Half-Yearly", 2020, 2021)
        assert len(idx) == 4  # 2020-H1, 2020-H2, 2021-H1, 2021-H2
        assert idx[0] == '2020-H1'

    def test_quarterly_range(self):
        idx = _build_full_index("Quarterly", 2020, 2020)
        assert len(idx) == 4  # Q1 through Q4

    def test_monthly_range(self):
        idx = _build_full_index("Monthly", 2020, 2020)
        assert len(idx) == 12


# ---------- _count_by_period ----------

class TestCountByPeriod:
    def test_yearly_counts_with_gap(self):
        df = pd.DataFrame({
            'Original Issue Date': pd.to_datetime([
                '2020-01-01', '2020-06-01', '2022-01-01'
            ])
        })
        counts = _count_by_period(df, "Yearly", 2020, 2022)
        # 2020 should have 2, 2021 should have 0, 2022 should have 1
        assert len(counts) == 3
        assert counts.iloc[0] == 2  # 2020
        assert counts.iloc[1] == 0  # 2021 (gap filled)
        assert counts.iloc[2] == 1  # 2022


# ---------- parse_dates ----------

class TestParseDates:
    def test_parses_date_columns(self):
        df = pd.DataFrame({
            'Original Issue Date': ['2020-01-15', '2021-06-20'],
            'Expiration Date': ['2025-01-15', '2026-06-20']
        })
        result = parse_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result['Original Issue Date'])
        assert pd.api.types.is_datetime64_any_dtype(result['Expiration Date'])

    def test_handles_none(self):
        assert parse_dates(None) is None

    def test_handles_invalid_dates(self):
        df = pd.DataFrame({
            'Original Issue Date': ['not-a-date', '2020-01-15']
        })
        result = parse_dates(df)
        assert pd.isna(result['Original Issue Date'].iloc[0])
        assert pd.notna(result['Original Issue Date'].iloc[1])
