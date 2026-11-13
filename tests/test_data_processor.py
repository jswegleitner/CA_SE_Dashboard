# -*- coding: utf-8 -*-
"""Unit tests for data processor helper functions."""

import sys
from pathlib import Path

# Add project root and Offline Processing to path. The latter is gitignored, so
# importorskip below cleanly skips this whole module on a fresh clone where it
# isn't present.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Offline Processing'))

import pandas as pd
import pytest

pytest.importorskip(
    "SE_Data_Process_V9",
    reason="Offline Processing module not present (gitignored)",
)
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


# ---------- key normalization / stamp detection (pipeline integrity) ----------

from SE_Data_Process_V9 import (  # noqa: E402
    normalize_license_key,
    invalid_issue_date_licenses,
    _has_stamp,
    save_or_append_master_excel,
)


class TestNormalizeLicenseKey:
    def test_strips_float_suffix_and_whitespace(self):
        s = pd.Series([1234.0, ' 56 ', '78.0', None, float('nan')])
        assert normalize_license_key(s).tolist() == ['1234', '56', '78', '', '']

    def test_int_and_str_keys_match(self):
        a = normalize_license_key(pd.Series([5, 6]))
        b = normalize_license_key(pd.Series(['5.0', '6']))
        assert a.tolist() == b.tolist()


class TestHasStamp:
    def test_missing_values_are_not_stamps(self):
        # pandas 3 keeps NaN as missing through astype(str); NaN != '' is
        # True, which made every blank count as "stamped".
        s = pd.Series(['2026-05-13', None, float('nan'), '', 'nan', 'NaT'])
        assert _has_stamp(s).tolist() == [True, False, False, False, False, False]


class TestInvalidIssueDateLicenses:
    def test_lists_unparseable(self):
        df = pd.DataFrame({'License Number': [1.0, 2.0, 3.0],
                           'Original Issue Date': ['2020-01-01', 'garbage', None]})
        assert invalid_issue_date_licenses(df) == ['2', '3']


def _master_frame():
    return pd.DataFrame({
        'License Number': [str(n) for n in range(1, 101)],
        'City': ['Oakland'] * 100,
        'County': ['Alameda'] * 100,
        'Original Issue Date': pd.to_datetime(['2000-01-01'] * 100),
        'Expiration Date': pd.to_datetime(['2027-06-30'] * 100),
        'License Status': ['Active'] * 100,
        'Status_Changed_Date': [None] * 100,
        'Last_Updated': ['2026-01-01 00:00:00'] * 100,
    })


class TestMasterUpsert:
    def test_html_partial_parse_keeps_master_values(self, tmp_path):
        path = tmp_path / 'master.xlsx'
        _master_frame().to_excel(path, index=False)
        incoming = pd.DataFrame({
            'License Number': ['7'],
            'City': [None],                       # address regex missed
            'County': [None],
            'Original Issue Date': [pd.NaT],      # page lacked issue date
            'Expiration Date': pd.to_datetime(['2099-12-31']),  # placeholder
            'License Status': ['Retired'],
        })
        out = save_or_append_master_excel(incoming, str(path), source='html')
        row = out[out['License Number'] == '7'].iloc[0]
        assert row['License Status'] == 'Retired'
        assert row['City'] == 'Oakland'
        assert row['County'] == 'Alameda'
        assert pd.Timestamp(row['Expiration Date']) == pd.Timestamp('2027-06-30')
        assert pd.Timestamp(row['Original Issue Date']) == pd.Timestamp('2000-01-01')
        # Untouched rows keep their Last_Updated (no spurious mass stamping)
        assert (out['Last_Updated'] == '2026-01-01 00:00:00').sum() == 99
        # Backup written
        assert list((tmp_path / 'master_backups').glob('*.xlsx'))

    def test_feed_reconciliation_skips_bad_date_rows_and_stamps(self, tmp_path):
        path = tmp_path / 'master.xlsx'
        _master_frame().to_excel(path, index=False)
        feed = _master_frame().drop(columns=['Status_Changed_Date', 'Last_Updated'])
        feed = feed[~feed['License Number'].isin(['10', '11'])]  # 10 bad date, 11 truly gone
        out = save_or_append_master_excel(feed, str(path), source='feed',
                                          present_in_feed=['10'])
        by = out.set_index('License Number')
        assert by.loc['10', 'License Status'] == 'Active'
        assert by.loc['11', 'License Status'] == 'Cancelled'
        assert str(by.loc['11', 'Status_Changed_Date']).strip() not in ('', 'nan')
        assert (out['Last_Updated'] == '2026-01-01 00:00:00').sum() == 99

    def test_feed_with_float_keys_still_matches(self, tmp_path):
        path = tmp_path / 'master.xlsx'
        _master_frame().to_excel(path, index=False)
        feed = _master_frame().drop(columns=['Status_Changed_Date', 'Last_Updated'])
        feed['License Number'] = feed['License Number'].astype(float)
        out = save_or_append_master_excel(feed, str(path), source='feed')
        assert (out['License Status'] == 'Active').all()
        assert len(out) == 100

    def test_mass_dropoff_aborts_without_saving(self, tmp_path):
        path = tmp_path / 'master.xlsx'
        _master_frame().to_excel(path, index=False)
        before = path.read_bytes()
        feed = _master_frame().drop(columns=['Status_Changed_Date', 'Last_Updated']).head(10)
        assert save_or_append_master_excel(feed, str(path), source='feed') is None
        assert path.read_bytes() == before


# ---------- HTML parser fallbacks ----------

parser_mod = pytest.importorskip("DCA_License_HTML_Parser")


class TestParserFallbacks:
    def test_unknown_month_is_none(self):
        assert parser_mod.parse_date("Smarch 3, 1999") is None
        assert parser_mod.parse_date("March 3, 1999") == "1999-03-03"

    def test_status_fallback_keeps_full_status(self):
        html = ("<html>LICENSING DETAILS <div>Name: DOE, JANE</div>"
                "<p>License Status: VOLUNTARY SURRENDER OF LICENSE</p></html>")
        rec, _ = parser_mod.parse_html(html, 1)
        assert rec["License Status"] == "Voluntary Surrender Of License"
