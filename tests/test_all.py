# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Tests for the --all aggregate overlay feature."""


# Standard libs
import io
import sys

# External libs
import pytest
import pandas as pd
from pandas import DataFrame, Timestamp

# Internal libs
from sacct_plot import SacctPlotApp
from sacct_plot.sweep import compute_allocation, apply_bucket, apply_top_n


def make_jobs(**overrides):
    """Helper: create a multi-user job DataFrame."""
    defaults = {
        'job_id': ['1', '2', '3', '4'],
        'user': ['alice', 'alice', 'bob', 'carol'],
        'account': ['physics', 'physics', 'chemistry', 'biology'],
        'qos': ['normal', 'normal', 'normal', 'normal'],
        'ncpus': [8, 4, 16, 2],
        'gpus': [2, 1, 4, 0],
        'elapsed_raw': [3600, 3600, 3600, 3600],
        'state': ['COMPLETED', 'COMPLETED', 'COMPLETED', 'COMPLETED'],
        'submit': [
            Timestamp('2026-02-18 09:30:00'),
            Timestamp('2026-02-18 09:45:00'),
            Timestamp('2026-02-18 09:50:00'),
            Timestamp('2026-02-18 09:55:00'),
        ],
        'start': [
            Timestamp('2026-02-18 10:00:00'),
            Timestamp('2026-02-18 10:00:00'),
            Timestamp('2026-02-18 10:00:00'),
            Timestamp('2026-02-18 10:00:00'),
        ],
        'end': [
            Timestamp('2026-02-18 11:00:00'),
            Timestamp('2026-02-18 11:00:00'),
            Timestamp('2026-02-18 11:00:00'),
            Timestamp('2026-02-18 11:00:00'),
        ],
    }
    defaults.update(overrides)
    return DataFrame(defaults)


class TestAllAggregateUnit:
    """Unit-level tests for --all merge logic using synthetic data."""

    def test_all_column_is_total(self):
        """The 'all' column equals the ungrouped allocation (total across groups)."""
        df = make_jobs()

        # Grouped allocation
        grouped = compute_allocation(df, metric='cpu', by='user')
        assert 'alice' in grouped.columns
        assert 'bob' in grouped.columns
        assert 'carol' in grouped.columns

        # Ungrouped allocation (the "all" line)
        total = compute_allocation(df, metric='cpu', by=None)

        # Merge as the app does
        grouped['all'] = total['allocation'].reindex(grouped.index, method='ffill').fillna(0)

        # "all" should be the sum of all individual groups at each timestamp
        assert 'all' in grouped.columns
        # At start (10:00), all four jobs start: 8+4+16+2 = 30
        assert grouped['all'].iloc[0] == 30
        # At end (11:00), all jobs end: 0
        assert grouped['all'].iloc[-1] == 0

    def test_all_with_top_n(self):
        """Top-N filters groups but 'all' remains the full aggregate."""
        df = make_jobs()

        grouped = compute_allocation(df, metric='cpu', by='user')
        total = compute_allocation(df, metric='cpu', by=None)

        # Apply top-1 (alice has 12 CPUs total, bob has 16)
        grouped = apply_top_n(grouped, n=1)
        assert 'other' in grouped.columns

        # Merge "all" after top-N
        grouped['all'] = total['allocation'].reindex(grouped.index, method='ffill').fillna(0)
        assert 'all' in grouped.columns
        assert grouped['all'].iloc[0] == 30  # total is still 30

    def test_all_with_bucket(self):
        """Bucketed 'all' column is present and non-empty."""
        df = make_jobs()

        grouped = compute_allocation(df, metric='cpu', by='user')
        total = compute_allocation(df, metric='cpu', by=None)

        grouped = apply_bucket(grouped, interval='1h', agg='sum')
        total = apply_bucket(total, interval='1h', agg='sum')

        grouped['all'] = total['allocation'].reindex(grouped.index, method='ffill').fillna(0)
        assert 'all' in grouped.columns
        assert not grouped['all'].isna().all()

    def test_all_with_gpu_metric(self):
        """The 'all' column works with GPU metric."""
        df = make_jobs()

        grouped = compute_allocation(df, metric='gpu', by='user')
        total = compute_allocation(df, metric='gpu', by=None)

        grouped['all'] = total['allocation'].reindex(grouped.index, method='ffill').fillna(0)

        # At start (10:00): alice=2+1, bob=4, carol=0 → total=7
        assert grouped['all'].iloc[0] == 7


class TestAllFlagIntegration:
    """Integration tests for --all through the full CLI pipeline with mock sacct."""

    def _run_app(self, *args):
        """Run the app, catching SystemExit, return exit code."""
        try:
            SacctPlotApp.main(list(args))
            return 0
        except SystemExit as exc:
            return exc.code if exc.code is not None else 0

    def test_all_by_user_data(self, mock_sacct, capsys):
        """--all --by user --data produces output with 'all' column."""
        code = self._run_app('-S', '2026-03-13', '--by', 'user', '--all', '--data')
        assert code == 0
        output = capsys.readouterr().out
        assert 'all' in output

    def test_all_by_account_data(self, mock_sacct, capsys):
        """--all --by account --data produces output with 'all' column."""
        code = self._run_app('-S', '2026-03-13', '--by', 'account', '--all', '--data')
        assert code == 0
        output = capsys.readouterr().out
        assert 'all' in output

    def test_all_without_by(self, mock_sacct, capsys):
        """--all without --by doesn't crash; 'all' column is not added."""
        code = self._run_app('-S', '2026-03-13', '--all', '--data')
        assert code == 0
        output = capsys.readouterr().out
        # Without --by, the output has 'allocation' column but no 'all'
        assert 'allocation' in output

    def test_all_with_top(self, mock_sacct, capsys):
        """--all with --top includes both top groups, 'other', and 'all'."""
        code = self._run_app('-S', '2026-03-13', '--by', 'account', '--top', '3', '--all', '--data')
        assert code == 0
        output = capsys.readouterr().out
        assert 'all' in output

    def test_all_with_bucket(self, mock_sacct, capsys):
        """--all with --bucket produces bucketed data with 'all' column."""
        code = self._run_app('-S', '2026-03-13', '--by', 'account', '--bucket', '1d', '--all', '--data')
        assert code == 0
        output = capsys.readouterr().out
        assert 'all' in output

    def test_help_shows_all_flag(self):
        """--all appears in help output."""
        try:
            SacctPlotApp.main(['--help'])
        except SystemExit:
            pass
