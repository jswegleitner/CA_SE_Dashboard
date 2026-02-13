# -*- coding: utf-8 -*-
"""Unit tests for data processor helper functions."""

import sys
from pathlib import Path

# Add project root and Offline Processing to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Offline Processing'))

import pandas as pd
import pytest

from SE_Data_Process_V9 import (
    _sanitize_excel_string,
    clean_df_for_excel,
    find_structural_engineers,
    clean_and_standardize_data,
)


# ---------- _sanitize_excel_string ----------

class TestSanitizeExcelString:
    def test_removes_control_characters(self):
        assert _sanitize_excel_string("hello\x00world") == "helloworld"
        assert _sanitize_excel_string("test\x0Bvalue") == "testvalue"

    def test_preserves_normal_strings(self):
        assert _sanitize_excel_string("normal text") == "normal text"

    def test_non_string_passthrough(self):
        assert _sanitize_excel_string(42) == 42
        assert _sanitize_excel_string(None) is None


# ---------- clean_df_for_excel ----------

class TestCleanDfForExcel:
    def test_cleans_headers(self):
        df = pd.DataFrame({"col\x00name": [1, 2]})
        result = clean_df_for_excel(df)
        assert "colname" in result.columns

    def test_cleans_object_cells(self):
        df = pd.DataFrame({"text": ["hello\x00", "world\x0B"]})
        result = clean_df_for_excel(df)
        assert result["text"].iloc[0] == "hello"
        assert result["text"].iloc[1] == "world"

    def test_returns_none_for_none(self):
        assert clean_df_for_excel(None) is None

    def test_returns_empty_for_empty(self):
        df = pd.DataFrame()
        result = clean_df_for_excel(df)
        assert result.empty


# ---------- find_structural_engineers ----------

class TestFindStructuralEngineers:
    def test_finds_exact_match(self):
        df = pd.DataFrame({
            'License Type': ['Structural Engineer', 'Civil Engineer', 'Structural Engineer'],
            'License Number': ['S001', 'C001', 'S002']
        })
        result = find_structural_engineers(df)
        assert result is not None
        assert len(result) == 2

    def test_finds_partial_match(self):
        df = pd.DataFrame({
            'License Type': ['structural engineering', 'Other'],
            'License Number': ['S001', 'O001']
        })
        result = find_structural_engineers(df)
        assert result is not None
        assert len(result) == 1

    def test_returns_none_no_license_type_col(self):
        df = pd.DataFrame({'Name': ['John'], 'Number': [1]})
        result = find_structural_engineers(df)
        assert result is None

    def test_returns_none_no_matches(self):
        df = pd.DataFrame({
            'License Type': ['Civil Engineer', 'Mechanical Engineer'],
            'License Number': ['C001', 'M001']
        })
        result = find_structural_engineers(df)
        assert result is None


# ---------- clean_and_standardize_data ----------

class TestCleanAndStandardizeData:
    def test_state_uppercased(self):
        df = pd.DataFrame({
            'State': ['ca', 'ny'],
            'Original Issue Date': ['2020-01-01', '2021-01-01']
        })
        result = clean_and_standardize_data(df)
        assert result['State'].iloc[0] == 'CA'

    def test_removes_null_bytes(self):
        df = pd.DataFrame({
            'State': ['CA\x00'],
            'Original Issue Date': ['2020-01-01']
        })
        result = clean_and_standardize_data(df)
        assert '\x00' not in result['State'].iloc[0]

    def test_drops_invalid_dates(self):
        df = pd.DataFrame({
            'Original Issue Date': ['2020-01-01', 'not-a-date', '2021-06-15']
        })
        result = clean_and_standardize_data(df)
        assert len(result) == 2

    def test_deduplicates_by_license_number(self):
        df = pd.DataFrame({
            'License Number': ['S001', 'S001', 'S002'],
            'Original Issue Date': ['2020-01-01', '2021-01-01', '2020-06-01'],
            'State': ['CA', 'CA', 'NY']
        })
        result = clean_and_standardize_data(df)
        assert len(result) == 2
        # Should keep the most recent (2021) for S001
        s001 = result[result['License Number'] == 'S001']
        assert len(s001) == 1

    def test_fills_missing_expiration(self):
        df = pd.DataFrame({
            'Original Issue Date': ['2020-01-01'],
            'Expiration Date': [None]
        })
        result = clean_and_standardize_data(df)
        assert pd.notna(result['Expiration Date'].iloc[0])
