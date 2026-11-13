"""Tests for dashboard_lib.history, the licensure-history CSV, and its chart overlay feed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from dashboard_lib.history import (
    ERAS,
    HISTORY_CATEGORIES,
    early_record_coverage,
    era_index,
    era_label,
    event_card_html,
    format_event_date,
    fractional_year,
    history_path,
    lane_offsets,
    license_number_milestones,
    parse_sources,
    plot_position,
    read_history_csv,
)
from dashboard_lib.theme import HISTORY_CATEGORY_COLORS, history_category_color
from dashboard_lib.timeline import history_to_chart_events, render_event_descriptions


def _write(tmp_path, body):
    path = tmp_path / 'licensure_history.csv'
    path.write_text(body, encoding='utf-8')
    return path


HEADER = "date,precision,category,title,summary,significance,sources\n"


class TestFormatting:
    def test_date_precision(self):
        ts = pd.Timestamp('1933-04-10')
        assert format_event_date(ts, 'day') == 'April 10, 1933'
        assert format_event_date(ts, 'month') == 'April 1933'
        assert format_event_date(ts, 'year') == '1933'

    def test_plot_position_centers_imprecise_dates(self):
        assert plot_position('1931-01-01', 'year') == pd.Timestamp('1931-07-01')
        assert plot_position('1988-04-01', 'month') == pd.Timestamp('1988-04-15')
        assert plot_position('1933-03-10', 'day') == pd.Timestamp('1933-03-10')

    def test_fractional_year_is_monotonic_within_a_year(self):
        assert fractional_year('1985-02-01') < fractional_year('1985-10-01') < 1986


class TestParseSources:
    def test_label_and_url(self):
        assert parse_sources('Board history <https://example.org/a.pdf>') == [
            ('Board history', 'https://example.org/a.pdf')
        ]

    def test_mixed_and_blank(self):
        out = parse_sources('An oral history (EERI, 1994); Page <https://example.org/x?y=1&z=2>; ')
        assert out == [
            ('An oral history (EERI, 1994)', None),
            ('Page', 'https://example.org/x?y=1&z=2'),
        ]

    def test_empty(self):
        assert parse_sources('') == []
        assert parse_sources(None) == []


class TestEras:
    def test_every_year_lands_in_exactly_one_era(self):
        for year in range(1850, 2101):
            idx = era_index(year)
            start, end, _, _ = ERAS[idx]
            assert start is None or year >= start
            assert end is None or year <= end

    def test_eras_are_contiguous(self):
        for (_, end, _, _), (start, _, _, _) in zip(ERAS, ERAS[1:]):
            assert start == end + 1

    def test_labels(self):
        assert era_label(0).endswith('(to 1928)')
        assert '1929 to 1933' in era_label(1)
        assert era_label(len(ERAS) - 1).endswith('to today)')


class TestLaneOffsets:
    def test_isolated_events_stay_centered(self):
        assert lane_offsets([1931.0, 1947.0, 1982.0]) == [0.0, 0.0, 0.0]

    def test_close_pair_straddles_the_line(self):
        assert lane_offsets([1929.0, 1929.1]) == [-0.2, 0.2]

    def test_run_of_close_events_alternates(self):
        offsets = lane_offsets([1985.1, 1985.8, 1987.9])
        assert offsets == [-0.2, 0.2, -0.2]

    def test_gap_resets(self):
        assert lane_offsets([1985.0, 1986.0, 2004.0]) == [-0.2, 0.2, 0.0]

    def test_empty(self):
        assert lane_offsets([]) == []


class TestReadHistoryCsv:
    def test_missing_file_returns_none(self, tmp_path):
        assert read_history_csv(tmp_path / 'nope.csv') is None

    def test_minimal_file(self, tmp_path):
        path = _write(tmp_path, HEADER + '1933-04-10,day,Legislation,Field Act,What,Why,Src\n')
        df = read_history_csv(path)
        assert list(df['title']) == ['Field Act']
        assert df['year'].iloc[0] == 1933
        # optional columns are filled in
        assert df['data_note'].iloc[0] == '' and not df['on_chart'].iloc[0]

    def test_sorted_by_date(self, tmp_path):
        path = _write(
            tmp_path,
            HEADER
            + '1971-02-09,day,Earthquake or disaster,B,s,w,x\n'
            + '1931-01-01,year,Licensure,A,s,w,x\n',
        )
        assert list(read_history_csv(path)['title']) == ['A', 'B']

    def test_blank_precision_defaults_to_day(self, tmp_path):
        path = _write(tmp_path, HEADER + '1933-04-10,,Legislation,Field Act,s,w,x\n')
        assert read_history_csv(path)['precision'].iloc[0] == 'day'

    @pytest.mark.parametrize('row, message', [
        ('not-a-date,day,Legislation,T,s,w,x\n', 'unparseable date'),
        ('1933-04-10,decade,Legislation,T,s,w,x\n', 'precision'),
        ('1933-04-10,day,Legislatoin,T,s,w,x\n', 'unknown category'),
    ])
    def test_bad_rows_raise(self, tmp_path, row, message):
        path = _write(tmp_path, HEADER + row)
        with pytest.raises(ValueError, match=message):
            read_history_csv(path)

    def test_missing_required_column_raises(self, tmp_path):
        path = _write(tmp_path, 'date,category,title\n1933-04-10,Legislation,T\n')
        with pytest.raises(ValueError, match='missing required column'):
            read_history_csv(path)


class TestShippedHistoryFile:
    """Guards on the real data/licensure_history.csv so a hand edit can't quietly break the tab."""

    @pytest.fixture
    def events(self):
        df = read_history_csv(history_path())
        assert df is not None, 'data/licensure_history.csv is missing'
        return df

    def test_has_events_in_every_category(self, events):
        assert len(events) >= 30
        assert set(events['category']) == set(HISTORY_CATEGORIES)

    def test_text_fields_are_filled(self, events):
        for col in ('title', 'summary', 'significance', 'sources'):
            blank = events[events[col] == '']
            assert blank.empty, f"blank {col}: {list(blank['title'])}"

    def test_titles_are_unique(self, events):
        assert not events['title'].duplicated().any()

    def test_every_event_cites_a_source(self, events):
        for title, text in zip(events['title'], events['sources']):
            assert parse_sources(text), f'no source for {title}'

    def test_source_urls_are_wellformed(self, events):
        for text in events['sources']:
            for label, url in parse_sources(text):
                assert label
                # an unparsed "<http..." means a typo in the "Label <url>" syntax
                assert '<' not in label and '>' not in label, label
                assert url is None or url.startswith('https://')

    def test_chart_events_have_a_hover_note(self, events):
        on_chart = events[events['on_chart']]
        assert 10 <= len(on_chart) <= 25
        assert (on_chart['chart_note'] != '').all()

    def test_every_category_has_a_color(self, events):
        assert set(HISTORY_CATEGORY_COLORS) == set(HISTORY_CATEGORIES)
        assert history_category_color('Exam') == HISTORY_CATEGORY_COLORS['Exam']
        assert history_category_color('something else').startswith('#')

    def test_no_emoji_or_control_characters(self, events):
        for col in ('title', 'summary', 'significance', 'data_note', 'chart_note'):
            for text in events[col]:
                assert all(ord(ch) < 0x2190 and (ch == '\n' or ord(ch) >= 32) for ch in text), text


class TestChartEvents:
    def _history(self, tmp_path):
        path = _write(
            tmp_path,
            'date,precision,category,title,summary,significance,sources,on_chart,chart_note,exam_format\n'
            '1931-01-01,year,Licensure,Title created,Long summary.,Why,x,1,Short note.,\n'
            '1933-05-01,month,Legislation,Riley Act,Summary.,Why,x,0,,\n'
            '2011-04-01,month,Exam,National exam,Exam summary.,Why,x,yes,,Two 8-hour components\n',
        )
        return read_history_csv(path)

    def test_only_flagged_rows_are_overlaid(self, tmp_path):
        ev = history_to_chart_events(self._history(tmp_path))
        assert list(ev['label']) == ['Title created', 'National exam']
        assert list(ev['type']) == ['Licensure', 'Exam']

    def test_note_falls_back_to_summary(self, tmp_path):
        ev = history_to_chart_events(self._history(tmp_path))
        assert list(ev['notes']) == ['Short note.', 'Exam summary.']

    def test_date_text_respects_precision(self, tmp_path):
        ev = history_to_chart_events(self._history(tmp_path))
        assert list(ev['date_text']) == ['1931', 'April 2011']

    def test_hover_description_skips_empty_fields(self, tmp_path):
        ev = history_to_chart_events(self._history(tmp_path))
        desc = render_event_descriptions(ev)
        assert 'Codes:' not in desc.iloc[0] and 'Format:' not in desc.iloc[0]
        assert 'Format:' in desc.iloc[1]

    def test_shipped_file_feeds_the_overlay(self):
        ev = history_to_chart_events(read_history_csv(history_path()))
        assert pd.api.types.is_datetime64_any_dtype(ev['start_date'])
        assert ev['start_date'].is_monotonic_increasing
        assert {'label', 'type', 'code', 'format', 'notes', 'description', 'date_text'} <= set(ev.columns)


class TestLabelPositions:
    """Direct labels on the History timeline (SE_Dashboard_v7._history_label_positions)."""

    @staticmethod
    def _positions(years, labels, offsets):
        from SE_Dashboard_v7 import _history_label_positions
        xs = [pd.Timestamp(year=int(y), month=7, day=1) for y in years]
        return _history_label_positions(xs, labels, offsets)

    def test_nudged_markers_label_on_the_outside(self):
        out = self._positions([1933, 1933], ['Field Act', 'Riley Act'], [-0.2, 0.2])
        assert out == ['bottom center', 'top center']

    def test_close_centered_labels_alternate(self):
        out = self._positions([1989, 1994, 2030], ['Loma Prieta', 'Northridge', 'Later'], [0, 0, 0])
        assert out == ['top center', 'bottom center', 'top center']

    def test_unlabelled_markers_do_not_affect_alternation(self):
        out = self._positions([1906, 1925, 1933], ['San Francisco', '', 'Long Beach'], [0, 0, 0])
        assert out[0] == 'top center' and out[2] == 'top center'


def _licenses(rows):
    df = pd.DataFrame(rows, columns=['License Number', 'Original Issue Date'])
    df['Original Issue Date'] = pd.to_datetime(df['Original Issue Date'])
    return df


class TestLicenseNumberMilestones:
    def test_first_and_round_numbers(self):
        rows = [(n, f'{1931 + n // 100}-06-01') for n in range(1, 2051)]
        out = dict(license_number_milestones(_licenses(rows), step=1000))
        assert out[1] == pd.Timestamp('1931-06-01')
        assert out[1000].year == 1941
        assert out[2000].year == 1951
        assert 3000 not in out

    def test_one_redated_record_does_not_move_a_milestone(self):
        rows = [(n, '1959-03-20') for n in range(980, 1021)]
        rows[20] = (1000, '2002-07-26')  # the round number itself carries a reinstatement date
        out = dict(license_number_milestones(_licenses(rows), step=1000))
        assert out[1000] == pd.Timestamp('1959-03-20')

    def test_number_stored_as_text(self):
        rows = [(str(n), '1960-01-01') for n in range(990, 1011)]
        assert 1000 in dict(license_number_milestones(_licenses(rows), step=1000))

    def test_empty_or_missing_columns(self):
        assert license_number_milestones(pd.DataFrame()) == []
        assert license_number_milestones(pd.DataFrame({'x': [1]})) == []
        assert license_number_milestones(None) == []


class TestEarlyRecordCoverage:
    def test_counts_surviving_records_against_numbering(self):
        rows = [(n, '1935-01-01') for n in (1, 50, 120, 300, 400)] + [(900, '1955-01-01')]
        present, numbered = early_record_coverage(_licenses(rows), before_year=1940)
        assert present == 5
        assert 300 <= numbered <= 400

    def test_none_when_no_early_records(self):
        assert early_record_coverage(_licenses([(900, '1955-01-01')]), before_year=1940) is None
        assert early_record_coverage(None) is None


class TestEventCardHtml:
    ROW = {
        'date': pd.Timestamp('1933-04-10'), 'precision': 'day', 'category': 'Legislation',
        'title': 'Field Act <script>', 'summary': 'Schools & safety.', 'significance': 'It "mattered".',
        'data_note': '', 'exam_format': '', 'codes': '',
        'sources': 'Statute <https://example.org/a?b=1&c=2>; Oral history',
    }

    def test_escapes_content(self):
        out = event_card_html(self.ROW, '#6D28D9')
        assert '<script>' not in out
        assert 'Field Act &lt;script&gt;' in out
        assert 'Schools &amp; safety.' in out

    def test_links_and_plain_sources(self):
        out = event_card_html(self.ROW, '#6D28D9')
        assert 'href="https://example.org/a?b=1&amp;c=2"' in out
        assert 'rel="noopener"' in out
        assert 'Oral history' in out

    def test_optional_sections_only_when_present(self):
        out = event_card_html(self.ROW, '#6D28D9')
        assert 'In the license data' not in out and 'Exam format' not in out
        row = dict(self.ROW, data_note='Only 26 licenses.', exam_format='16 hours')
        out = event_card_html(row, '#6D28D9')
        assert 'In the license data' in out and 'Exam format' in out

    def test_single_line_for_markdown(self):
        # A newline or leading indent would make st.markdown treat the card as a code block.
        assert '\n' not in event_card_html(self.ROW, '#6D28D9')
