# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Unit tests for wait time computation and bucketed aggregation."""


# External libs
import pytest
import pandas as pd
from pandas import DataFrame, Timestamp

# Internal libs
from sacct_plot.wait import compute_wait_time, apply_wait_bucket, apply_wait_top_n


def make_wait_jobs(**overrides):
    """Helper: create a minimal job DataFrame with submit/start/end."""
    defaults = {
        'job_id': ['1'],
        'user': ['user1'],
        'account': ['acct1'],
        'qos': ['normal'],
        'ncpus': [8],
        'gpus': [0],
        'elapsed_raw': [3600],
        'state': ['COMPLETED'],
        'submit': [Timestamp('2026-02-18 09:00:00')],
        'start': [Timestamp('2026-02-18 10:00:00')],
        'end': [Timestamp('2026-02-18 11:00:00')],
    }
    defaults.update(overrides)
    return DataFrame(defaults)


class TestComputeWaitTime:
    """Tests for per-job wait time computation."""

    def test_single_job(self):
        """Single job with 1-hour wait."""
        df = make_wait_jobs()
        result = compute_wait_time(df)
        assert len(result) == 1
        assert result['wait'].iloc[0] == 3600.0  # 1 hour in seconds

    def test_multiple_jobs(self):
        """Multiple jobs with different wait times."""
        df = make_wait_jobs(
            job_id=['1', '2', '3'],
            submit=[
                Timestamp('2026-02-18 09:00:00'),
                Timestamp('2026-02-18 09:30:00'),
                Timestamp('2026-02-18 10:00:00'),
            ],
            start=[
                Timestamp('2026-02-18 10:00:00'),
                Timestamp('2026-02-18 10:00:00'),
                Timestamp('2026-02-18 10:00:00'),
            ],
            end=[Timestamp('2026-02-18 11:00:00')] * 3,
            user=['u1', 'u2', 'u3'],
            account=['a1', 'a1', 'a1'],
            qos=['normal'] * 3,
            ncpus=[8, 4, 2],
            gpus=[0, 0, 0],
            elapsed_raw=[3600, 3600, 3600],
            state=['COMPLETED'] * 3,
        )
        result = compute_wait_time(df)
        assert len(result) == 3
        waits = sorted(result['wait'].tolist())
        assert waits == [0.0, 1800.0, 3600.0]  # 0s, 30min, 1h

    def test_zero_wait(self):
        """Job that starts immediately has zero wait."""
        df = make_wait_jobs(
            submit=[Timestamp('2026-02-18 10:00:00')],
            start=[Timestamp('2026-02-18 10:00:00')],
        )
        result = compute_wait_time(df)
        assert result['wait'].iloc[0] == 0.0

    def test_grouped_by_user(self):
        """Grouping includes the group column."""
        df = make_wait_jobs(
            job_id=['1', '2'],
            user=['alice', 'bob'],
            submit=[
                Timestamp('2026-02-18 09:00:00'),
                Timestamp('2026-02-18 09:30:00'),
            ],
            start=[
                Timestamp('2026-02-18 10:00:00'),
                Timestamp('2026-02-18 10:00:00'),
            ],
            end=[Timestamp('2026-02-18 11:00:00')] * 2,
            account=['a1', 'a1'],
            qos=['normal'] * 2,
            ncpus=[8, 4],
            gpus=[0, 0],
            elapsed_raw=[3600, 3600],
            state=['COMPLETED'] * 2,
        )
        result = compute_wait_time(df, by='user')
        assert 'user' in result.columns
        assert 'wait' in result.columns
        assert set(result['user']) == {'alice', 'bob'}

    def test_grouped_by_account(self):
        """Grouping by account works."""
        df = make_wait_jobs(
            job_id=['1', '2'],
            account=['physics', 'chemistry'],
            submit=[
                Timestamp('2026-02-18 09:00:00'),
                Timestamp('2026-02-18 09:30:00'),
            ],
            start=[
                Timestamp('2026-02-18 10:00:00'),
                Timestamp('2026-02-18 10:00:00'),
            ],
            end=[Timestamp('2026-02-18 11:00:00')] * 2,
            user=['u1', 'u2'],
            qos=['normal'] * 2,
            ncpus=[8, 4],
            gpus=[0, 0],
            elapsed_raw=[3600, 3600],
            state=['COMPLETED'] * 2,
        )
        result = compute_wait_time(df, by='account')
        assert 'account' in result.columns
        assert set(result['account']) == {'physics', 'chemistry'}

    def test_missing_submit_dropped(self):
        """Jobs with NaT submit are excluded."""
        df = make_wait_jobs(
            job_id=['1', '2'],
            submit=[Timestamp('2026-02-18 09:00:00'), pd.NaT],
            start=[Timestamp('2026-02-18 10:00:00'), Timestamp('2026-02-18 10:30:00')],
            end=[Timestamp('2026-02-18 11:00:00')] * 2,
            user=['u1', 'u2'],
            account=['a1', 'a1'],
            qos=['normal'] * 2,
            ncpus=[8, 4],
            gpus=[0, 0],
            elapsed_raw=[3600, 3600],
            state=['COMPLETED'] * 2,
        )
        result = compute_wait_time(df)
        assert len(result) == 1

    def test_missing_start_dropped(self):
        """Jobs with NaT start are excluded."""
        df = make_wait_jobs(
            job_id=['1', '2'],
            submit=[Timestamp('2026-02-18 09:00:00')] * 2,
            start=[Timestamp('2026-02-18 10:00:00'), pd.NaT],
            end=[Timestamp('2026-02-18 11:00:00'), pd.NaT],
            user=['u1', 'u2'],
            account=['a1', 'a1'],
            qos=['normal'] * 2,
            ncpus=[8, 4],
            gpus=[0, 0],
            elapsed_raw=[3600, 0],
            state=['COMPLETED', 'PENDING'],
        )
        result = compute_wait_time(df)
        assert len(result) == 1

    def test_empty_input(self):
        """Empty DataFrame yields empty result."""
        df = make_wait_jobs(submit=[pd.NaT], start=[pd.NaT])
        result = compute_wait_time(df)
        assert result.empty

    def test_index_is_start_time(self):
        """Result is indexed by start time."""
        df = make_wait_jobs()
        result = compute_wait_time(df)
        assert result.index.name == 'start'
        assert result.index[0] == Timestamp('2026-02-18 10:00:00')

    def test_sorted_by_start(self):
        """Result is sorted by start time."""
        df = make_wait_jobs(
            job_id=['1', '2', '3'],
            submit=[
                Timestamp('2026-02-18 09:00:00'),
                Timestamp('2026-02-18 08:00:00'),
                Timestamp('2026-02-18 09:30:00'),
            ],
            start=[
                Timestamp('2026-02-18 12:00:00'),
                Timestamp('2026-02-18 10:00:00'),
                Timestamp('2026-02-18 11:00:00'),
            ],
            end=[Timestamp('2026-02-18 13:00:00')] * 3,
            user=['u1', 'u2', 'u3'],
            account=['a1', 'a1', 'a1'],
            qos=['normal'] * 3,
            ncpus=[8, 4, 2],
            gpus=[0, 0, 0],
            elapsed_raw=[3600] * 3,
            state=['COMPLETED'] * 3,
        )
        result = compute_wait_time(df)
        assert list(result.index) == sorted(result.index)


class TestApplyWaitBucket:
    """Tests for bucketed wait time aggregation."""

    def _make_wait_data(self, n=10, by=None):
        """Helper: create synthetic wait data spanning several hours."""
        starts = pd.date_range('2026-02-18 10:00:00', periods=n, freq='15min')
        waits = [60.0 * (i + 1) for i in range(n)]  # 60s, 120s, ..., n*60s
        data = {'wait': waits}
        if by:
            # Alternate between two groups
            data[by] = ['alpha' if i % 2 == 0 else 'beta' for i in range(n)]
        result = DataFrame(data, index=starts)
        result.index.name = 'start'
        return result

    def test_median_ungrouped(self):
        """Median aggregation returns center, p25, p75."""
        df = self._make_wait_data(n=12)
        summary = apply_wait_bucket(df, '1h', agg='median')
        assert 'center' in summary
        assert 'p25' in summary
        assert 'p75' in summary
        assert summary['p25'] is not None
        assert summary['p75'] is not None
        assert 'wait' in summary['center'].columns
        assert not summary['center'].empty

    def test_mean_ungrouped(self):
        """Mean aggregation returns center only."""
        df = self._make_wait_data(n=12)
        summary = apply_wait_bucket(df, '1h', agg='mean')
        assert not summary['center'].empty
        assert summary['p25'] is None
        assert summary['p75'] is None

    def test_max_ungrouped(self):
        """Max aggregation returns center only."""
        df = self._make_wait_data(n=12)
        summary = apply_wait_bucket(df, '1h', agg='max')
        assert not summary['center'].empty
        assert summary['p25'] is None
        assert summary['p75'] is None

    def test_median_values(self):
        """Verify median computation for a known distribution."""
        # 4 jobs in the same hour with waits: 100, 200, 300, 400
        starts = pd.date_range('2026-02-18 10:00:00', periods=4, freq='10min')
        df = DataFrame({'wait': [100.0, 200.0, 300.0, 400.0]}, index=starts)
        df.index.name = 'start'
        summary = apply_wait_bucket(df, '1h', agg='median')
        # Median of [100, 200, 300, 400] = 250
        assert summary['center']['wait'].iloc[0] == pytest.approx(250.0)
        # p25 = 175, p75 = 325
        assert summary['p25']['wait'].iloc[0] == pytest.approx(175.0)
        assert summary['p75']['wait'].iloc[0] == pytest.approx(325.0)

    def test_mean_values(self):
        """Verify mean computation for a known distribution."""
        starts = pd.date_range('2026-02-18 10:00:00', periods=4, freq='10min')
        df = DataFrame({'wait': [100.0, 200.0, 300.0, 400.0]}, index=starts)
        df.index.name = 'start'
        summary = apply_wait_bucket(df, '1h', agg='mean')
        # Mean of [100, 200, 300, 400] = 250
        assert summary['center']['wait'].iloc[0] == pytest.approx(250.0)

    def test_max_values(self):
        """Verify max computation for a known distribution."""
        starts = pd.date_range('2026-02-18 10:00:00', periods=4, freq='10min')
        df = DataFrame({'wait': [100.0, 200.0, 300.0, 400.0]}, index=starts)
        df.index.name = 'start'
        summary = apply_wait_bucket(df, '1h', agg='max')
        assert summary['center']['wait'].iloc[0] == pytest.approx(400.0)

    def test_grouped_median(self):
        """Grouped median produces wide-format columns."""
        df = self._make_wait_data(n=12, by='account')
        summary = apply_wait_bucket(df, '1h', agg='median', by='account')
        center = summary['center']
        assert 'alpha' in center.columns
        assert 'beta' in center.columns
        assert summary['p25'] is not None
        assert 'alpha' in summary['p25'].columns

    def test_grouped_mean(self):
        """Grouped mean produces wide-format columns."""
        df = self._make_wait_data(n=12, by='user')
        summary = apply_wait_bucket(df, '1h', agg='mean', by='user')
        center = summary['center']
        assert 'alpha' in center.columns
        assert 'beta' in center.columns

    def test_empty_input(self):
        """Empty DataFrame yields empty result."""
        summary = apply_wait_bucket(DataFrame(), '1h')
        assert summary['center'].empty
        assert summary['p25'] is None
        assert summary['p75'] is None

    def test_invalid_agg(self):
        """Unknown aggregation raises ValueError."""
        df = self._make_wait_data()
        with pytest.raises(ValueError, match='Unknown aggregation'):
            apply_wait_bucket(df, '1h', agg='invalid')


class TestApplyWaitTopN:
    """Tests for top-N group filtering on raw wait data."""

    def test_top_1_of_3(self):
        """Keep top 1 group, collapse rest into 'other'."""
        starts = pd.date_range('2026-02-18 10:00:00', periods=6, freq='10min')
        df = DataFrame({
            'wait': [1000.0, 900.0, 500.0, 400.0, 100.0, 50.0],
            'user': ['alice', 'alice', 'bob', 'bob', 'carol', 'carol'],
        }, index=starts)
        df.index.name = 'start'
        result = apply_wait_top_n(df, n=1, by='user')
        groups = set(result['user'])
        assert 'alice' in groups  # highest total wait
        assert 'other' in groups
        assert 'bob' not in groups
        assert 'carol' not in groups

    def test_top_2_of_3(self):
        """Keep top 2 groups, collapse rest into 'other'."""
        starts = pd.date_range('2026-02-18 10:00:00', periods=6, freq='10min')
        df = DataFrame({
            'wait': [1000.0, 900.0, 500.0, 400.0, 100.0, 50.0],
            'user': ['alice', 'alice', 'bob', 'bob', 'carol', 'carol'],
        }, index=starts)
        df.index.name = 'start'
        result = apply_wait_top_n(df, n=2, by='user')
        groups = set(result['user'])
        assert 'alice' in groups
        assert 'bob' in groups
        assert 'other' in groups
        assert 'carol' not in groups

    def test_n_exceeds_groups(self):
        """If N >= number of groups, return unchanged."""
        starts = pd.date_range('2026-02-18 10:00:00', periods=4, freq='10min')
        df = DataFrame({
            'wait': [100.0, 200.0, 300.0, 400.0],
            'user': ['alice', 'alice', 'bob', 'bob'],
        }, index=starts)
        df.index.name = 'start'
        result = apply_wait_top_n(df, n=5, by='user')
        assert set(result['user']) == {'alice', 'bob'}
        assert 'other' not in set(result['user'])

    def test_empty_input(self):
        """Empty DataFrame returns empty."""
        result = apply_wait_top_n(DataFrame(), n=3, by='user')
        assert result.empty

    def test_missing_by_column(self):
        """If by column not present, return unchanged."""
        starts = pd.date_range('2026-02-18 10:00:00', periods=2, freq='10min')
        df = DataFrame({'wait': [100.0, 200.0]}, index=starts)
        df.index.name = 'start'
        result = apply_wait_top_n(df, n=1, by='user')
        assert len(result) == 2  # unchanged


class TestWaitWithMockSacct:
    """Integration tests using the mock sacct fixture."""

    def test_wait_data_mode(self, mock_sacct):
        """--wait --data produces output with wait times."""
        from sacct_plot.sacct import SacctData
        data = SacctData.from_sacct(starttime='2026-03-13')
        wait_data = compute_wait_time(data.data)
        assert not wait_data.empty
        assert 'wait' in wait_data.columns
        assert (wait_data['wait'] >= 0).all()

    def test_wait_grouped(self, mock_sacct):
        """--wait --by account with fixture data."""
        from sacct_plot.sacct import SacctData
        data = SacctData.from_sacct(starttime='2026-03-13')
        wait_data = compute_wait_time(data.data, by='account')
        assert 'account' in wait_data.columns
        assert len(wait_data['account'].unique()) > 0

    def test_wait_bucketed(self, mock_sacct):
        """--wait --bucket 1h with fixture data."""
        from sacct_plot.sacct import SacctData
        data = SacctData.from_sacct(starttime='2026-03-13')
        wait_data = compute_wait_time(data.data)
        summary = apply_wait_bucket(wait_data, '1h', agg='median')
        assert not summary['center'].empty
        assert summary['p25'] is not None
        assert summary['p75'] is not None

    def test_wait_bucketed_grouped(self, mock_sacct):
        """--wait --bucket 1h --by account with fixture data."""
        from sacct_plot.sacct import SacctData
        data = SacctData.from_sacct(starttime='2026-03-13')
        wait_data = compute_wait_time(data.data, by='account')
        summary = apply_wait_bucket(wait_data, '1h', agg='median', by='account')
        assert not summary['center'].empty
        assert len(summary['center'].columns) > 1  # multiple accounts
