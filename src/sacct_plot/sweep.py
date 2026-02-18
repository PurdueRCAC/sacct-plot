# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Event-based sweep algorithm for computing instantaneous resource allocation."""


# Type annotations
from __future__ import annotations
from typing import Optional

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

    # Drop jobs missing start/end timestamps
    valid = df.dropna(subset=['start', 'end']).copy()
    if valid.empty:
        return DataFrame()

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


def apply_bucket(df: DataFrame, interval: str, agg: str = 'sum') -> DataFrame:
    """Resample the allocation step function to a coarser time grid.

    Forward-fills the sparse event data to the bucket grid, then applies
    the specified aggregation. Never expands to per-second resolution.

    Args:
        df: Time-indexed allocation DataFrame (from compute_allocation).
        interval: Pandas-compatible frequency string (e.g. '1h', '1D').
        agg: Aggregation method ('sum', 'mean', 'max', 'min').
    """
    if df.empty:
        return df
    resampled = df.resample(interval).agg(agg)
    resampled = resampled.ffill().fillna(0)
    return resampled


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
        result['other'] = df[other_cols].sum(axis=1)
    return result
