# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Terminal rendering of allocation and wait-time plots using tplot via plot-cli."""


# Type annotations
from __future__ import annotations
from typing import Dict, Optional, List, Tuple

# External libs
import pandas as pd
from pandas import DataFrame
from plot_cli.plot import TimeSeriesFigure, generate_time_ticks


# Color cycle for overlaid series (tplot color names)
COLORS: List[str] = [
    'blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'white',
]

# Unit conversion divisors from seconds
UNIT_DIVISORS: Dict[str, float] = {
    'seconds': 1.0,
    'minutes': 60.0,
    'hours': 3600.0,
    'days': 86400.0,
}


def render(
    df: DataFrame,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    stacked: bool = False,
    colors: Optional[List[str]] = None,
    size: Optional[Tuple[int, int]] = None,
    grouped: bool = False,
    legend: str = 'bottomright',
) -> None:
    """Render allocation time-series to the terminal.

    Args:
        df: Time-indexed allocation DataFrame. Single column ('allocation')
            for ungrouped data, or multiple columns (group names) for grouped.
        title: Plot title.
        ylabel: Y-axis label (e.g. 'CPUs' or 'GPUs').
        stacked: If True, render stacked area (cumulative); otherwise overlaid lines.
        colors: Optional list of color names to cycle through.
        size: Optional (width, height) in characters.
        grouped: If True, always show legend labels (even for a single series).
        legend: Legend position ('topleft', 'topright', 'bottomleft', 'bottomright').
    """
    if df.empty:
        return

    color_cycle = colors if colors else COLORS

    # Convert datetime index to epoch seconds for tplot
    # Use precision-agnostic conversion (works with datetime64[ns], [us], [ms])
    epochs = (df.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    min_epoch = float(epochs.min())
    max_epoch = float(epochs.max())

    # Generate smart time ticks
    ticks = generate_time_ticks(min_epoch, max_epoch, max_ticks=10)

    # Build tick formatter: map epoch -> label from precomputed ticks
    tick_map = dict(zip(ticks.tick_epochs, ticks.tick_labels))

    def tick_formatter(value: float) -> str:
        """Format epoch tick value using precomputed labels."""
        # Find nearest tick
        if not tick_map:
            return str(int(value))
        closest = min(tick_map, key=lambda t: abs(t - value))
        if abs(closest - value) < 1:
            return tick_map[closest]
        return ''

    # Build secondary x-label from secondary labels (e.g. date markers)
    secondary_xlabel = None
    if ticks.secondary_labels:
        secondary_xlabel = '  '.join(label for _, label in ticks.secondary_labels)

    # Build xlabel
    xlabel = 'Time'
    if ticks.xlabel_suffix:
        xlabel = f'Time {ticks.xlabel_suffix}'

    fig = TimeSeriesFigure(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        size=size,
        legend=legend,
        x_tick_formatter=tick_formatter,
        x_tick_values=ticks.tick_epochs,
        secondary_xlabel=secondary_xlabel,
    )

    columns = [c for c in df.columns]
    x = epochs.tolist()

    if stacked and len(columns) > 1:
        # Stacked: plot cumulative sums bottom-up
        cumulative = df[columns].fillna(0).cumsum(axis=1)
        for i, col in enumerate(columns):
            color = color_cycle[i % len(color_cycle)]
            fig.line(x=x, y=cumulative[col].tolist(), color=color, label=str(col))
    else:
        # Overlaid lines (or single series)
        for i, col in enumerate(columns):
            color = color_cycle[i % len(color_cycle)]
            label = str(col) if (grouped or len(columns) > 1) else None
            fig.line(x=x, y=df[col].fillna(0).tolist(), color=color, label=label)

    fig.draw()


def _make_time_figure(
    epochs: pd.Index,
    title: Optional[str],
    xlabel_base: str,
    ylabel: Optional[str],
    size: Optional[Tuple[int, int]],
    legend: str,
) -> Tuple[TimeSeriesFigure, List[float]]:
    """Create a TimeSeriesFigure with smart time-axis ticks.

    Returns the figure and epoch x-values as a list.
    """
    min_epoch = float(epochs.min())
    max_epoch = float(epochs.max())
    ticks = generate_time_ticks(min_epoch, max_epoch, max_ticks=10)

    tick_map = dict(zip(ticks.tick_epochs, ticks.tick_labels))

    def tick_formatter(value: float) -> str:
        if not tick_map:
            return str(int(value))
        closest = min(tick_map, key=lambda t: abs(t - value))
        if abs(closest - value) < 1:
            return tick_map[closest]
        return ''

    secondary_xlabel = None
    if ticks.secondary_labels:
        secondary_xlabel = '  '.join(label for _, label in ticks.secondary_labels)

    xlabel = xlabel_base
    if ticks.xlabel_suffix:
        xlabel = f'{xlabel_base} {ticks.xlabel_suffix}'

    fig = TimeSeriesFigure(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        size=size,
        legend=legend,
        x_tick_formatter=tick_formatter,
        x_tick_values=ticks.tick_epochs,
        secondary_xlabel=secondary_xlabel,
    )
    return fig, epochs.tolist()


def render_wait(
    wait_data: DataFrame,
    summary: Optional[Dict[str, Optional[DataFrame]]],
    unit: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    by: Optional[str] = None,
    colors: Optional[List[str]] = None,
    size: Optional[Tuple[int, int]] = None,
    legend: str = 'bottomright',
) -> None:
    """Render wait-time scatter / summary to the terminal.

    Without ``summary`` (unbucketed): scatter plot of per-job wait times.
    With ``summary`` (bucketed): scatter dots overlaid with summary lines
    (center line + p25/p75 percentile envelope when available).

    Args:
        wait_data: Per-job wait DataFrame (index=start time, 'wait' column
            in seconds, optional grouping column).
        summary: Bucketed summary dict with keys 'center', 'p25', 'p75'
            (from ``apply_wait_bucket``).  None for unbucketed mode.
        unit: Display unit ('seconds', 'minutes', 'hours', 'days').
        title: Plot title.
        ylabel: Y-axis label.
        by: Grouping column name (e.g. 'user', 'account').
        colors: Optional color cycle override.
        size: Optional (width, height) in characters.
        legend: Legend position.
    """
    if wait_data.empty:
        return

    color_cycle = colors if colors else COLORS
    divisor = UNIT_DIVISORS[unit]

    # --- Scatter: per-job dots ---
    scatter_epochs = (wait_data.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

    if summary is not None:
        # Bucketed mode — build figure from the summary index
        center = summary['center']
        summary_epochs = (center.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
        # Use the union of scatter and summary ranges for the axis
        all_epochs = scatter_epochs.union(summary_epochs)
        fig, _ = _make_time_figure(all_epochs, title, 'Time', ylabel, size, legend)
        sx = scatter_epochs.tolist()
        lx = summary_epochs.tolist()
    else:
        # Unbucketed mode
        fig, sx = _make_time_figure(scatter_epochs, title, 'Time', ylabel, size, legend)

    # Draw scatter points
    has_groups = by and by in wait_data.columns
    if has_groups:
        groups = wait_data[by].unique()
        for i, group in enumerate(groups):
            color = color_cycle[i % len(color_cycle)]
            mask = wait_data[by] == group
            gx = scatter_epochs[mask].tolist()
            gy = (wait_data.loc[mask, 'wait'] / divisor).tolist()
            fig.scatter(x=gx, y=gy, marker='⠂', color=color, label=str(group))
    else:
        sy = (wait_data['wait'] / divisor).tolist()
        fig.scatter(x=sx, y=sy, marker='⠂', color=color_cycle[0])

    # --- Overlay summary lines (bucketed mode) ---
    if summary is not None:
        center = summary['center']
        p25 = summary.get('p25')
        p75 = summary.get('p75')
        columns = [c for c in center.columns]

        for i, col in enumerate(columns):
            color = color_cycle[i % len(color_cycle)]
            label = str(col) if has_groups or len(columns) > 1 else None
            # Center line
            fig.line(x=lx, y=(center[col] / divisor).tolist(), color=color, label=label)
            # Percentile envelope (dashed-like: same color, thinner visual presence)
            if p25 is not None and p75 is not None and col in p25.columns:
                fig.line(x=lx, y=(p25[col] / divisor).tolist(), color=color)
                fig.line(x=lx, y=(p75[col] / divisor).tolist(), color=color)

    fig.draw()
