# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Integration tests for wait time rendering (--wait mode)."""


# Standard libs
from unittest.mock import patch, MagicMock

# External libs
import pytest
import pandas as pd
from pandas import DataFrame, Timestamp

# Internal libs
from sacct_plot import SacctPlotApp
from sacct_plot.plot import render_wait, UNIT_DIVISORS


def _run_app(*args):
    """Run the app, catching SystemExit, return exit code."""
    try:
        SacctPlotApp.main(list(args))
        return 0
    except SystemExit as exc:
        return exc.code if exc.code is not None else 0


class TestRenderWaitUnit:
    """Unit tests for render_wait with synthetic data."""

    def _make_wait_data(self, n=20, by=None, wait_range=(60, 7200)):
        """Create synthetic per-job wait data."""
        starts = pd.date_range('2026-02-18 10:00:00', periods=n, freq='15min')
        lo, hi = wait_range
        waits = [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
        data = {'wait': waits}
        if by:
            data[by] = ['alpha' if i % 2 == 0 else 'beta' for i in range(n)]
        result = DataFrame(data, index=starts)
        result.index.name = 'start'
        return result

    @patch('sacct_plot.plot.TimeSeriesFigure')
    def test_unbucketed_calls_scatter(self, MockFig):
        """Unbucketed mode calls scatter and draw, no line calls."""
        mock_fig = MagicMock()
        MockFig.return_value = mock_fig

        wait_data = self._make_wait_data()
        render_wait(wait_data, summary=None, unit='minutes')

        assert mock_fig.scatter.called
        assert not mock_fig.line.called
        mock_fig.draw.assert_called_once()

    @patch('sacct_plot.plot.TimeSeriesFigure')
    def test_bucketed_calls_scatter_and_line(self, MockFig):
        """Bucketed mode calls both scatter and line."""
        mock_fig = MagicMock()
        MockFig.return_value = mock_fig

        wait_data = self._make_wait_data()
        from sacct_plot.wait import apply_wait_bucket
        summary = apply_wait_bucket(wait_data, '1h', agg='median')

        render_wait(wait_data, summary=summary, unit='minutes')

        assert mock_fig.scatter.called
        assert mock_fig.line.called
        mock_fig.draw.assert_called_once()

    @patch('sacct_plot.plot.TimeSeriesFigure')
    def test_bucketed_median_has_envelope(self, MockFig):
        """Median bucketing produces center + p25 + p75 = 3 line calls."""
        mock_fig = MagicMock()
        MockFig.return_value = mock_fig

        wait_data = self._make_wait_data()
        from sacct_plot.wait import apply_wait_bucket
        summary = apply_wait_bucket(wait_data, '1h', agg='median')

        render_wait(wait_data, summary=summary, unit='minutes')

        # 1 center + 1 p25 + 1 p75 for the single ungrouped 'wait' column
        assert mock_fig.line.call_count == 3

    @patch('sacct_plot.plot.TimeSeriesFigure')
    def test_bucketed_mean_no_envelope(self, MockFig):
        """Mean bucketing produces only 1 center line (no p25/p75)."""
        mock_fig = MagicMock()
        MockFig.return_value = mock_fig

        wait_data = self._make_wait_data()
        from sacct_plot.wait import apply_wait_bucket
        summary = apply_wait_bucket(wait_data, '1h', agg='mean')

        render_wait(wait_data, summary=summary, unit='minutes')

        assert mock_fig.line.call_count == 1

    @patch('sacct_plot.plot.TimeSeriesFigure')
    def test_grouped_scatter_per_group(self, MockFig):
        """Grouped unbucketed scatter creates one scatter call per group."""
        mock_fig = MagicMock()
        MockFig.return_value = mock_fig

        wait_data = self._make_wait_data(by='user')
        render_wait(wait_data, summary=None, unit='hours', by='user')

        # 2 groups: alpha and beta
        assert mock_fig.scatter.call_count == 2

    @patch('sacct_plot.plot.TimeSeriesFigure')
    def test_unit_scaling(self, MockFig):
        """Y-values are scaled by the unit divisor."""
        mock_fig = MagicMock()
        MockFig.return_value = mock_fig

        wait_data = self._make_wait_data(n=1, wait_range=(3600, 3600))
        render_wait(wait_data, summary=None, unit='hours')

        # The scatter y-value should be 1.0 (3600s / 3600)
        call_args = mock_fig.scatter.call_args
        y_values = call_args.kwargs.get('y') or call_args[1].get('y')
        assert y_values == [pytest.approx(1.0)]

    @patch('sacct_plot.plot.TimeSeriesFigure')
    def test_empty_data_no_draw(self, MockFig):
        """Empty wait data does not draw anything."""
        mock_fig = MagicMock()
        MockFig.return_value = mock_fig

        render_wait(DataFrame(), summary=None, unit='seconds')

        assert not mock_fig.draw.called


class TestWaitRenderIntegration:
    """Full CLI integration tests for --wait rendering with mock sacct."""

    def test_wait_scatter_no_crash(self, mock_sacct, capsys):
        """--wait renders without crashing."""
        code = _run_app('-S', '2026-03-13', '--wait', '--size', '80,24')
        assert code == 0
        output = capsys.readouterr().out
        assert len(output) > 0

    def test_wait_bucketed_no_crash(self, mock_sacct, capsys):
        """--wait --bucket 1h renders without crashing."""
        code = _run_app('-S', '2026-03-13', '--wait', '--bucket', '1h', '--size', '80,24')
        assert code == 0
        output = capsys.readouterr().out
        assert len(output) > 0

    def test_wait_grouped_no_crash(self, mock_sacct, capsys):
        """--wait --by account --top 5 renders without crashing."""
        code = _run_app(
            '-S', '2026-03-13', '--wait', '--by', 'account',
            '--top', '5', '--size', '80,24',
        )
        assert code == 0
        output = capsys.readouterr().out
        assert len(output) > 0

    def test_wait_bucketed_grouped_no_crash(self, mock_sacct, capsys):
        """--wait --bucket 1h --by account --top 5 renders without crashing."""
        code = _run_app(
            '-S', '2026-03-13', '--wait', '--bucket', '1h',
            '--by', 'account', '--top', '5', '--size', '80,24',
        )
        assert code == 0
        output = capsys.readouterr().out
        assert len(output) > 0

    def test_wait_bucketed_mean_no_crash(self, mock_sacct, capsys):
        """--wait --bucket 1h --mean renders without crashing."""
        code = _run_app(
            '-S', '2026-03-13', '--wait', '--bucket', '1h',
            '--mean', '--size', '80,24',
        )
        assert code == 0

    def test_wait_bucketed_max_no_crash(self, mock_sacct, capsys):
        """--wait --bucket 1h --max renders without crashing."""
        code = _run_app(
            '-S', '2026-03-13', '--wait', '--bucket', '1h',
            '--max', '--size', '80,24',
        )
        assert code == 0

    def test_wait_data_mode(self, mock_sacct, capsys):
        """--wait --data outputs CSV with wait column."""
        code = _run_app('-S', '2026-03-13', '--wait', '--data')
        assert code == 0
        output = capsys.readouterr().out
        assert 'wait' in output

    def test_wait_bucketed_data_mode(self, mock_sacct, capsys):
        """--wait --bucket 1h --data outputs CSV."""
        code = _run_app('-S', '2026-03-13', '--wait', '--bucket', '1h', '--data')
        assert code == 0
        output = capsys.readouterr().out
        assert 'wait' in output
        assert len(output.strip().split('\n')) > 1  # header + data rows

    def test_wait_with_top(self, mock_sacct, capsys):
        """--wait --by account --top 3 renders without crashing."""
        code = _run_app(
            '-S', '2026-03-13', '--wait', '--by', 'account',
            '--top', '3', '--size', '80,24',
        )
        assert code == 0
