# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Unit tests for the event-sweep algorithm."""

import pytest
from datetime import datetime

import pandas as pd
from pandas import DataFrame, Timestamp

from sacct_plot.sweep import compute_allocation, apply_bucket, apply_cumulative, apply_top_n


def make_jobs(**overrides):
    """Helper: create a minimal job DataFrame."""
    defaults = {
        'job_id': ['1'],
        'user': ['user1'],
        'account': ['acct1'],
        'qos': ['normal'],
        'ncpus': [8],
        'gpus': [0],
        'elapsed_raw': [3600],
        'state': ['COMPLETED'],
        'start': [Timestamp('2026-02-18 10:00:00')],
        'end': [Timestamp('2026-02-18 11:00:00')],
    }
    defaults.update(overrides)
    return DataFrame(defaults)


class TestComputeAllocation:
    """Tests for the core event-sweep algorithm."""

    def test_single_job(self):
        """Single job: allocation goes up at start, back to 0 at end."""
        df = make_jobs()
        result = compute_allocation(df, metric='cpu')
        assert result.index[0] == Timestamp('2026-02-18 10:00:00')
        assert result.index[-1] == Timestamp('2026-02-18 11:00:00')
        assert result['allocation'].iloc[0] == 8
        assert result['allocation'].iloc[-1] == 0

    def test_two_overlapping_jobs(self):
        """Two overlapping jobs: allocation stacks during overlap."""
        df = make_jobs(
            job_id=['1', '2'],
            ncpus=[8, 4],
            start=[Timestamp('2026-02-18 10:00:00'), Timestamp('2026-02-18 10:30:00')],
            end=[Timestamp('2026-02-18 11:00:00'), Timestamp('2026-02-18 11:30:00')],
            user=['u1', 'u1'],
            account=['a1', 'a1'],
            qos=['normal', 'normal'],
            gpus=[0, 0],
            elapsed_raw=[3600, 3600],
            state=['COMPLETED', 'COMPLETED'],
        )
        result = compute_allocation(df, metric='cpu')
        # 10:00 -> +8, 10:30 -> +4 (total 12), 11:00 -> -8 (total 4), 11:30 -> -4 (total 0)
        assert len(result) == 4
        vals = result['allocation'].tolist()
        assert vals == [8, 12, 4, 0]

    def test_gpu_metric(self):
        """GPU metric uses the gpus column."""
        df = make_jobs(gpus=[4])
        result = compute_allocation(df, metric='gpu')
        assert result['allocation'].iloc[0] == 4
        assert result['allocation'].iloc[-1] == 0

    def test_groupby_account(self):
        """Grouping by account produces wide-format columns."""
        df = make_jobs(
            job_id=['1', '2'],
            account=['physics', 'chemistry'],
            ncpus=[8, 4],
            start=[Timestamp('2026-02-18 10:00:00'), Timestamp('2026-02-18 10:00:00')],
            end=[Timestamp('2026-02-18 11:00:00'), Timestamp('2026-02-18 11:00:00')],
            user=['u1', 'u2'],
            qos=['normal', 'normal'],
            gpus=[0, 0],
            elapsed_raw=[3600, 3600],
            state=['COMPLETED', 'COMPLETED'],
        )
        result = compute_allocation(df, metric='cpu', by='account')
        assert 'physics' in result.columns
        assert 'chemistry' in result.columns
        # At start, both go up
        assert result['physics'].iloc[0] == 8
        assert result['chemistry'].iloc[0] == 4

    def test_empty_after_filter(self):
        """Jobs with no start/end yield empty result."""
        df = make_jobs(start=[pd.NaT], end=[pd.NaT])
        result = compute_allocation(df, metric='cpu')
        assert result.empty

    def test_simultaneous_start_end(self):
        """Two jobs starting and ending at the same time are handled."""
        df = make_jobs(
            job_id=['1', '2'],
            ncpus=[8, 4],
            start=[Timestamp('2026-02-18 10:00:00'), Timestamp('2026-02-18 10:00:00')],
            end=[Timestamp('2026-02-18 11:00:00'), Timestamp('2026-02-18 11:00:00')],
            user=['u1', 'u2'],
            account=['a1', 'a1'],
            qos=['normal', 'normal'],
            gpus=[0, 0],
            elapsed_raw=[3600, 3600],
            state=['COMPLETED', 'COMPLETED'],
        )
        result = compute_allocation(df, metric='cpu')
        # Both start at 10:00 -> 12, both end at 11:00 -> 0
        assert result['allocation'].iloc[0] == 12
        assert result['allocation'].iloc[-1] == 0


class TestApplyBucket:
    """Tests for bucket aggregation with proper step-function integration."""

    def test_hourly_bucket_max(self):
        """Max aggregation picks the peak allocation in each bucket."""
        index = pd.DatetimeIndex([
            '2026-02-18 10:00:00',
            '2026-02-18 10:30:00',
            '2026-02-18 11:00:00',
        ])
        df = DataFrame({'allocation': [8, 12, 0]}, index=index)
        result = apply_bucket(df, '1h', agg='max')
        assert len(result) == 2  # 10:00 and 11:00 buckets
        assert result['allocation'].iloc[0] == 12  # max in first hour

    def test_hourly_bucket_sum_integration(self):
        """Sum aggregation computes the time-weighted integral in resource-hours.

        Step function: 8 GPUs from 10:00-10:30 (0.5h), 12 GPUs from 10:30-11:00 (0.5h).
        Integral = 8*0.5 + 12*0.5 = 10 GPU-hours for the 10:00 bucket.
        """
        index = pd.DatetimeIndex([
            '2026-02-18 10:00:00',
            '2026-02-18 10:30:00',
            '2026-02-18 11:00:00',
        ])
        df = DataFrame({'allocation': [8, 12, 0]}, index=index)
        result = apply_bucket(df, '1h', agg='sum')
        assert result['allocation'].iloc[0] == pytest.approx(10.0)  # GPU-hours

    def test_hourly_bucket_mean(self):
        """Mean aggregation gives the time-weighted average allocation level.

        Same step function as above: mean = 10 GPU-h / 1 h = 10 GPUs avg.
        """
        index = pd.DatetimeIndex([
            '2026-02-18 10:00:00',
            '2026-02-18 10:30:00',
            '2026-02-18 11:00:00',
        ])
        df = DataFrame({'allocation': [8, 12, 0]}, index=index)
        result = apply_bucket(df, '1h', agg='mean')
        assert result['allocation'].iloc[0] == pytest.approx(10.0)  # avg GPUs

    def test_constant_level_full_day(self):
        """Constant 4 GPUs for a full day = 4*24 = 96 GPU-hours with --sum."""
        index = pd.DatetimeIndex([
            '2026-02-18 00:00:00',
            '2026-02-19 00:00:00',
        ])
        df = DataFrame({'allocation': [4, 0]}, index=index)
        result = apply_bucket(df, '1D', agg='sum')
        assert result['allocation'].iloc[0] == pytest.approx(96.0)

    def test_max_sees_carried_forward_level(self):
        """Max in a bucket should see levels carried forward from prior events."""
        # Level goes to 20 at 09:30, drops to 5 at 10:30.
        # The 10:00 bucket should see max=20 (carried forward from 09:30).
        index = pd.DatetimeIndex([
            '2026-02-18 09:30:00',
            '2026-02-18 10:30:00',
        ])
        df = DataFrame({'allocation': [20, 5]}, index=index)
        result = apply_bucket(df, '1h', agg='max')
        # 09:00 bucket has the event at 09:30 → max=20
        assert result['allocation'].iloc[0] == 20
        # 10:00 bucket: starts at 20 (carried forward), drops to 5 at 10:30 → max=20
        assert result['allocation'].iloc[1] == 20

    def test_grouped_columns(self):
        """Bucketing preserves grouped (wide-format) columns."""
        index = pd.DatetimeIndex([
            '2026-02-18 10:00:00',
            '2026-02-18 11:00:00',
        ])
        df = DataFrame({'alice': [4, 0], 'bob': [2, 0]}, index=index)
        result = apply_bucket(df, '1h', agg='sum')
        assert 'alice' in result.columns
        assert 'bob' in result.columns
        # alice: 4 GPUs × 1 hour = 4 GPU-hours
        assert result['alice'].iloc[0] == pytest.approx(4.0)
        assert result['bob'].iloc[0] == pytest.approx(2.0)

    def test_empty_input(self):
        result = apply_bucket(DataFrame(), '1h')
        assert result.empty


class TestApplyCumulative:
    """Tests for cumulative sum of bucketed data."""

    def test_cumulative_sum(self):
        """Running total of bucketed values."""
        index = pd.DatetimeIndex([
            '2026-02-18 10:00:00',
            '2026-02-18 11:00:00',
            '2026-02-18 12:00:00',
        ])
        df = DataFrame({'allocation': [10.0, 5.0, 8.0]}, index=index)
        result = apply_cumulative(df)
        assert result['allocation'].tolist() == [10.0, 15.0, 23.0]

    def test_cumulative_grouped(self):
        """Cumulative sum works independently per group column."""
        index = pd.DatetimeIndex([
            '2026-02-18 10:00:00',
            '2026-02-18 11:00:00',
        ])
        df = DataFrame({'alice': [4.0, 6.0], 'bob': [2.0, 3.0]}, index=index)
        result = apply_cumulative(df)
        assert result['alice'].tolist() == [4.0, 10.0]
        assert result['bob'].tolist() == [2.0, 5.0]

    def test_empty_input(self):
        result = apply_cumulative(DataFrame())
        assert result.empty


class TestApplyTopN:
    """Tests for top-N group filtering."""

    def test_top_2_of_4(self):
        """Keep top 2 groups, collapse rest into 'other'."""
        index = pd.DatetimeIndex([
            '2026-02-18 10:00:00',
            '2026-02-18 11:00:00',
        ])
        df = DataFrame({
            'alpha': [100, 100],
            'beta': [50, 50],
            'gamma': [10, 10],
            'delta': [5, 5],
        }, index=index)
        result = apply_top_n(df, n=2)
        assert 'alpha' in result.columns
        assert 'beta' in result.columns
        assert 'gamma' not in result.columns
        assert 'delta' not in result.columns
        assert 'other' in result.columns
        assert result['other'].iloc[0] == 15  # gamma + delta

    def test_n_exceeds_columns(self):
        """If N >= number of columns, return unchanged."""
        index = pd.DatetimeIndex(['2026-02-18 10:00:00'])
        df = DataFrame({'a': [10], 'b': [5]}, index=index)
        result = apply_top_n(df, n=5)
        assert list(result.columns) == ['a', 'b']

    def test_empty_input(self):
        result = apply_top_n(DataFrame(), n=3)
        assert result.empty
