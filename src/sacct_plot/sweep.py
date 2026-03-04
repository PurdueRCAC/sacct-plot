# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Event-based sweep algorithm for computing instantaneous resource allocation."""


# Type annotations
from __future__ import annotations
from typing import Optional

# Standard libs
from datetime import datetime

# External libs
import pandas as pd
from pandas import DataFrame, concat


def compute_allocation(df: DataFrame, metric: str = 'cpu', by: Optional[str] = None) -> DataFrame:
    """Compute instantaneous allocation time-series from job records.

    Uses the event-sweep approach: each job emits +resources at start and
    -resources at end. Sorting by time and computing the cumulative sum
    yields the exact step function. O(N log N), fully vectorized.

    Args:
        df: Job records with columns: start, end, ncpus, gpus, and optionally
            account/user for grouping.
        metric: 'cpu' or 'gpu' — which resource column to use.
        by: Optional grouping column ('account', 'user', 'qos').

    Returns:
        DataFrame indexed by timestamp. If ``by`` is set, columns are group
        names (wide format). Otherwise a single 'allocation' column.
    """
    resource_col = 'ncpus' if metric == 'cpu' else 'gpus'

    # Drop jobs missing start timestamps; treat running jobs (end=NaT) as
    # still allocating resources through "now".
    valid = df.dropna(subset=['start']).copy()
    if valid.empty:
        return DataFrame()
    now = pd.Timestamp(datetime.now())
    valid['end'] = valid['end'].fillna(now)

    # Build event pairs: (time, +delta, [group]) and (time, -delta, [group])
    cols = ['time', 'delta']
    if by:
        cols.append(by)

    start_events = DataFrame({
        'time': valid['start'],
        'delta': valid[resource_col],
    })
    end_events = DataFrame({
        'time': valid['end'],
        'delta': -valid[resource_col],
    })

    if by:
        start_events[by] = valid[by].values
        end_events[by] = valid[by].values

    events = concat([start_events, end_events], ignore_index=True)
    events = events.sort_values('time').reset_index(drop=True)

    if by:
        # Per-group cumulative sum, then pivot to wide format
        events['allocation'] = events.groupby(by)['delta'].cumsum()
        # Keep only the columns we need for pivoting; handle duplicate timestamps
        # by taking the last value per (time, group) pair
        pivoted = events.pivot_table(
            index='time', columns=by, values='allocation', aggfunc='last',
        )
        pivoted = pivoted.sort_index()
        pivoted = pivoted.ffill().fillna(0)
        return pivoted
    else:
        # Single series cumulative sum
        events['allocation'] = events['delta'].cumsum()
        # Collapse duplicate timestamps (take last value at each time)
        result = events.groupby('time')['allocation'].last().to_frame()
        result = result.sort_index()
        return result


def _fill_at_boundaries(df: DataFrame, interval: str) -> DataFrame:
    """Insert bucket boundaries into the step-function index and forward-fill.

    Ensures every bucket has a valid level at its start, so segments never
    cross bucket boundaries.
    """
    start = df.index.min().floor(interval)
    end = df.index.max().ceil(interval)
    boundaries = pd.date_range(start, end, freq=interval)
    combined = df.index.union(boundaries)
    return df.reindex(combined).ffill().fillna(0)


def _integrate_buckets(df: DataFrame, interval: str) -> DataFrame:
    """Compute the time-weighted integral of a step function per bucket.

    Each constant segment contributes level × duration (in GPU-seconds).
    Returns resource-hours per bucket (integral / 3600).
    """
    filled = _fill_at_boundaries(df, interval)

    # Forward-looking duration (seconds) from each point to the next
    idx = filled.index
    durations = pd.Series(0.0, index=idx)
    durations.iloc[:-1] = (idx[1:] - idx[:-1]).total_seconds()

    # GPU-seconds per segment = level × duration
    weighted = filled.multiply(durations, axis=0)

    # Sum per bucket, convert to resource-hours
    return weighted.resample(interval).sum() / 3600


def apply_bucket(df: DataFrame, interval: str, agg: str = 'sum') -> DataFrame:
    """Aggregate the allocation step function into time buckets.

    For ``sum`` and ``mean`` the step function is properly integrated
    (level × duration) so values reflect actual resource-time consumed.

    * ``sum``  — resource-hours per bucket (e.g. GPU·h).
    * ``mean`` — time-weighted average allocation level (e.g. avg GPUs).
    * ``max``  — peak allocation within the bucket.
    * ``min``  — minimum allocation within the bucket.

    Args:
        df: Time-indexed allocation DataFrame (from compute_allocation).
        interval: Pandas-compatible frequency string (e.g. '1h', '1D').
        agg: Aggregation method ('sum', 'mean', 'max', 'min').
    """
    if df.empty:
        return df

    if agg == 'sum':
        result = _integrate_buckets(df, interval)
    elif agg == 'mean':
        integral = _integrate_buckets(df, interval)
        bucket_hours = pd.Timedelta(interval).total_seconds() / 3600
        result = integral / bucket_hours
    else:
        # max / min — forward-fill at boundaries so carried-forward levels
        # are visible, then take the standard aggregate.
        filled = _fill_at_boundaries(df, interval)
        result = filled.resample(interval).agg(agg)

    return result.fillna(0)


def apply_cumulative(df: DataFrame) -> DataFrame:
    """Compute the running cumulative sum of a bucketed DataFrame."""
    if df.empty:
        return df
    return df.cumsum()


def apply_top_n(df: DataFrame, n: int) -> DataFrame:
    """Keep only the top N groups by total area; collapse the rest into "other".

    Args:
        df: Wide-format allocation DataFrame (columns = group names).
        n: Number of top groups to keep.
    """
    if df.empty or len(df.columns) <= n:
        return df

    # Rank columns by total area (sum of values over time)
    totals = df.sum().sort_values(ascending=False)
    top_cols = totals.index[:n].tolist()
    other_cols = totals.index[n:].tolist()

    result = df[top_cols].copy()
    if other_cols:
        result.insert(0, 'other', df[other_cols].sum(axis=1))
    return result
