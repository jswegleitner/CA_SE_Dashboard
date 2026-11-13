"""Tests for dashboard_lib.exam_results and the shipped data/se_exam_results.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from dashboard_lib.exam_results import (
    ERA_KEYS,
    EXAM_ERAS,
    REQUIRED_COLUMNS,
    STRANDS,
    STRAND_LABELS,
    era_bounds,
    era_of,
    era_rate_summary,
    estimated_passed,
    exam_results_path,
    licenses_after_sittings,
    read_exam_results_csv,
    sitting_date,
    sittings,
)
from dashboard_lib.theme import EXAM_STRAND_COLORS, EXAM_STRAND_SYMBOLS

HEADER = ','.join(REQUIRED_COLUMNS)


def _write(tmp_path, *rows):
    p = tmp_path / 'results.csv'
    p.write_text('\n'.join([HEADER, *rows]) + '\n', encoding='utf-8')
    return p


def _row(sitting='2013-04', era='sixteen', strand='vertical', scope='California',
         population='all', tested='100', passed='40', rate='40', evidence='primary', note='', source=''):
    return ','.join([sitting, era, strand, 'Exam', scope, population, tested, passed, rate, evidence, note, source])


class TestReader:
    def test_missing_file_returns_none(self, tmp_path):
        assert read_exam_results_csv(tmp_path / 'nope.csv') is None

    def test_reads_and_derives(self, tmp_path):
        df = read_exam_results_csv(_write(tmp_path, _row(), _row(sitting='2024-10', era='cbt', passed='', rate='20')))
        assert list(df['sitting']) == ['2013-04', '2024-10']
        assert df.loc[0, 'date'] == pd.Timestamp('2013-04-15')
        assert df.loc[0, 'passed_est'] == 40
        assert df.loc[1, 'passed_est'] == 20  # 100 tested x 20%

    def test_blank_result_allowed_without_evidence(self, tmp_path):
        df = read_exam_results_csv(_write(tmp_path, _row(tested='', passed='', rate='', evidence='')))
        assert pd.isna(df.loc[0, 'pass_rate_pct'])

    @pytest.mark.parametrize('kwargs, message', [
        (dict(sitting='2013-4'), 'YYYY-MM'),
        (dict(era='paper'), 'era must be'),
        (dict(strand='gravity'), 'strand must be'),
        (dict(scope='Oregon'), 'scope must be'),
        (dict(population='repeat'), 'population must be'),
        (dict(evidence='rumor'), 'evidence must be'),
        (dict(rate='forty'), 'not a number'),
        (dict(rate='140'), 'outside 0-100'),
        (dict(passed='120'), 'passed exceeds tested'),
        (dict(evidence=''), 'needs an evidence flag'),
    ])
    def test_rejects_bad_rows(self, tmp_path, kwargs, message):
        with pytest.raises(ValueError, match=message):
            read_exam_results_csv(_write(tmp_path, _row(**kwargs)))

    def test_missing_column(self, tmp_path):
        p = tmp_path / 'bad.csv'
        p.write_text('sitting,era\n2013-04,sixteen\n', encoding='utf-8')
        with pytest.raises(ValueError, match='missing required column'):
            read_exam_results_csv(p)


class TestHelpers:
    def test_sitting_date_mid_month(self):
        assert sitting_date('2024-10') == pd.Timestamp('2024-10-15')

    def test_era_of_boundaries(self):
        assert era_of('2003-12-31') == 'state'
        assert era_of('2004-01-01') == 'split'
        assert era_of('2011-03-31') == 'split'
        assert era_of('2011-04-15') == 'sixteen'
        assert era_of('2024-04-15') == 'cbt'

    def test_era_bounds_open_era_ends_today(self):
        lo, hi = era_bounds('cbt', today=pd.Timestamp('2026-11-13'))
        assert lo == pd.Timestamp('2024-01-01') and hi == pd.Timestamp('2026-11-13')
        lo, hi = era_bounds('split')
        assert hi == pd.Timestamp('2011-03-31')

    def test_estimated_passed_prefers_published_count(self):
        df = pd.DataFrame({'tested': [100, 82, None], 'passed': [41, None, None], 'pass_rate_pct': [40, 20, 30]})
        assert estimated_passed(df).tolist()[:2] == [41, 16]
        assert pd.isna(estimated_passed(df).iloc[2])

    def test_licenses_after_sittings_windows(self):
        lic = pd.DataFrame({'Original Issue Date': [
            '2013-03-01',  # before the first window
            '2013-06-10', '2013-06-11', '2013-10-31',  # April window (May-Oct)
            '2013-12-05',  # October window
            '2014-05-02',  # last window, still open
        ]})
        dates = [pd.Timestamp('2013-04-15'), pd.Timestamp('2013-10-15'), pd.Timestamp('2014-04-15')]
        w = licenses_after_sittings(lic, dates, today=pd.Timestamp('2014-06-01'))
        assert w['licenses'].tolist() == [3, 1, 1]
        assert w['partial'].tolist() == [False, False, True]
        assert w.loc[0, 'window_start'] == pd.Timestamp('2013-05-01')
        assert w.loc[0, 'window_end'] == pd.Timestamp('2013-10-31')

    def test_licenses_after_sittings_skips_future(self):
        dates = [pd.Timestamp('2026-10-15')]
        w = licenses_after_sittings(pd.DataFrame({'Original Issue Date': []}), dates, today=pd.Timestamp('2026-10-20'))
        assert w.empty

    def test_licenses_after_sittings_no_data(self):
        assert licenses_after_sittings(None, [pd.Timestamp('2013-04-15')]).empty

    def test_era_rate_summary_drops_whole_exam_rows_after_2011(self, tmp_path):
        df = read_exam_results_csv(_write(
            tmp_path,
            _row(sitting='2011-04', strand='whole', rate='24', passed='24'),
            _row(sitting='2012-04', rate='30', passed='30'),
            _row(sitting='2012-04', strand='lateral', rate='50', passed='50'),
        ))
        s = era_rate_summary(df).set_index('era')
        assert s.loc['sixteen', 'n'] == 2
        assert s.loc['sixteen', 'median'] == 40


@pytest.fixture(scope='module')
def results():
    df = read_exam_results_csv(exam_results_path())
    assert df is not None, 'data/se_exam_results.csv is missing'
    return df


class TestShippedFile:
    def test_every_strand_has_a_color_and_symbol(self):
        assert set(STRANDS) == set(EXAM_STRAND_COLORS) == set(EXAM_STRAND_SYMBOLS) == set(STRAND_LABELS)

    def test_eras_are_ordered_and_contiguous(self):
        assert ERA_KEYS == ['state', 'split', 'sixteen', 'cbt']
        for (_, _, end, *_), (_, start, *_rest) in zip(EXAM_ERAS, EXAM_ERAS[1:]):
            assert (pd.Timestamp(f"{end}-01") + pd.offsets.MonthBegin(1)) == pd.Timestamp(f"{start}-01")

    def test_row_era_matches_its_date(self, results):
        # The October 2011 SE III sitting is the one planned exception: it is the
        # last sitting of the two-exam era, given after the 16-hour exam began.
        mismatched = results[results['date'].map(era_of) != results['era']]
        assert set(mismatched['sitting']) <= {'2011-10'}
        assert (mismatched['exam'].str.contains('SE III')).all()

    def test_one_row_per_sitting_piece_and_population(self, results):
        key = ['sitting', 'scope', 'strand', 'population', 'exam']
        assert not results.duplicated(key).any()

    def test_california_covers_every_era(self, results):
        ca = results[results['scope'] == 'California'].dropna(subset=['pass_rate_pct'])
        assert set(ca['era']) == set(ERA_KEYS)

    def test_known_figures(self, results):
        def rate(sitting, scope, strand, exam_part=''):
            r = results[(results['sitting'] == sitting) & (results['scope'] == scope)
                        & (results['strand'] == strand) & results['exam'].str.contains(exam_part)]
            assert len(r) == 1, (sitting, scope, strand, exam_part)
            return r.iloc[0]

        oct03 = rate('2003-10', 'California', 'whole')
        assert (oct03['tested'], oct03['passed']) == (290, 73)
        assert rate('2009-10', 'California', 'lateral', 'SE III')['passed'] == 177
        assert rate('2013-04', 'California', 'lateral')['passed'] == 78
        assert rate('2025-04', 'California', 'vertical')['pass_rate_pct'] == 12
        assert rate('2026-04', 'National', 'lateral')['pass_rate_pct'] == 63

    def test_fall_2017_marked_unpublished(self, results):
        rows = results[(results['sitting'] == '2017-10') & (results['scope'] == 'California')]
        assert len(rows) == 2 and rows['pass_rate_pct'].isna().all()
        assert rows['note'].str.contains('never published').all()

    def test_sittings_list(self, results):
        s = sittings(results)
        assert s[0] == pd.Timestamp('2003-10-15')
        assert pd.Timestamp('2020-04-15') not in s  # cancelled
        assert s == sorted(set(s))
