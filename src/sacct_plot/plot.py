# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Terminal rendering of allocation time-series using tplot via plot-cli."""


# Type annotations
from __future__ import annotations
from typing import Optional, List

# External libs
from pandas import DataFrame
from plot_cli.plot import TimeSeriesFigure, generate_time_ticks


# Color cycle for overlaid series (tplot color names)
COLORS: List[str] = [
    'blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'white',
]


def render(
    df: DataFrame,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    stacked: bool = False,
) -> None:
    """Render allocation time-series to the terminal.

    Args:
        df: Time-indexed allocation DataFrame. Single column ('allocation')
            for ungrouped data, or multiple columns (group names) for grouped.
        title: Plot title.
        ylabel: Y-axis label (e.g. 'CPUs' or 'GPUs').
        stacked: If True, render stacked area (cumulative); otherwise overlaid lines.
    """
    if df.empty:
        return

    # Convert datetime index to epoch seconds for tplot
    epochs = df.index.astype('int64') // 10**9
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
            color = COLORS[i % len(COLORS)]
            fig.line(x=x, y=cumulative[col].tolist(), color=color, label=str(col))
    else:
        # Overlaid lines (or single series)
        for i, col in enumerate(columns):
            color = COLORS[i % len(COLORS)]
            label = str(col) if len(columns) > 1 else None
            fig.line(x=x, y=df[col].fillna(0).tolist(), color=color, label=label)

    fig.draw()
