# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Wait time computation and bucketed aggregation."""


# Type annotations
from __future__ import annotations
from typing import Dict, Optional

# External libs
import pandas as pd
from pandas import DataFrame


def compute_wait_time(df: DataFrame, by: Optional[str] = None) -> DataFrame:
    """Compute per-job wait time (start − submit).

    Each job's wait time is the difference between when it started running
    and when it was submitted, measured in seconds.

    Args:
        df: Job records with 'start' and 'submit' datetime columns.
        by: Optional grouping column ('account', 'user', 'qos').

    Returns:
        DataFrame indexed by start time with a 'wait' column (seconds).
        If ``by`` is set, includes the grouping column.
    """
    valid = df.dropna(subset=['start', 'submit']).copy()
    if valid.empty:
        return DataFrame()

    wait_seconds = (valid['start'] - valid['submit']).dt.total_seconds()

    cols = {'wait': wait_seconds.values}
    if by and by in valid.columns:
        cols[by] = valid[by].values

    result = DataFrame(cols, index=valid['start'].values)
    result.index.name = 'start'
    result = result.sort_index()
    return result


def apply_wait_bucket(
    df: DataFrame,
    interval: str,
    agg: str = 'median',
    by: Optional[str] = None,
) -> Dict[str, Optional[DataFrame]]:
    """Aggregate wait times into time buckets with summary statistics.

    For ``median`` aggregation, also computes the p25/p75 percentile envelope.
    For ``mean`` and ``max``, only the center statistic is returned.

    Args:
        df: Per-job wait DataFrame from ``compute_wait_time``.
        interval: Pandas-compatible frequency string (e.g. '1h', '1D').
        agg: Aggregation method ('median', 'mean', 'max').
        by: Grouping column name (must be present in df if provided).

    Returns:
        Dict with keys 'center', 'p25', 'p75'. Each value is a DataFrame
        indexed by bucket timestamps. For ungrouped data, a single 'wait'
        column. For grouped data, columns are group names (wide format).
        'p25' and 'p75' are None unless ``agg='median'``.
    """
    if df.empty:
        return {'center': DataFrame(), 'p25': None, 'p75': None}

    has_groups = by and by in df.columns

    if has_groups:
        grouped = df.groupby([pd.Grouper(freq=interval), by])['wait']
    else:
        grouped = df.resample(interval)['wait']

    if agg == 'median':
        center = grouped.median()
        p25 = grouped.quantile(0.25)
        p75 = grouped.quantile(0.75)
    elif agg == 'mean':
        center = grouped.mean()
        p25 = None
        p75 = None
    elif agg == 'max':
        center = grouped.max()
        p25 = None
        p75 = None
    else:
        raise ValueError(f'Unknown aggregation: {agg!r}')

    # Pivot grouped results to wide format
    if has_groups:
        center = center.unstack(by).fillna(0)
        if p25 is not None:
            p25 = p25.unstack(by).fillna(0)
            p75 = p75.unstack(by).fillna(0)
    else:
        center = center.to_frame('wait')
        if p25 is not None:
            p25 = p25.to_frame('wait')
            p75 = p75.to_frame('wait')

    return {
        'center': center.fillna(0),
        'p25': p25.fillna(0) if p25 is not None else None,
        'p75': p75.fillna(0) if p75 is not None else None,
    }


def apply_wait_top_n(df: DataFrame, n: int, by: str) -> DataFrame:
    """Keep only jobs from the top N groups by total wait time.

    Remaining groups are collapsed into 'other'.

    Args:
        df: Per-job wait DataFrame with a grouping column.
        n: Number of top groups to keep.
    """
    if df.empty or by not in df.columns:
        return df

    totals = df.groupby(by)['wait'].sum().sort_values(ascending=False)
    if len(totals) <= n:
        return df

    top_groups = set(totals.index[:n])
    result = df.copy()
    result.loc[~result[by].isin(top_groups), by] = 'other'
    return result
