# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Visualize instantaneous allocated resources (CPUs/GPUs) on Slurm clusters over time."""


# Type annotations
from __future__ import annotations
from typing import Final, List, Optional, Tuple

# Standard libs
import sys
from os import path
from importlib.metadata import version as get_version
from platform import python_version, python_implementation

# External libs
from cmdkit.app import Application, exit_status
from cmdkit.cli import Interface
from cmdkit.config import Configuration, Namespace
from cmdkit.logging import Logger, level_by_name, logging_styles

# Internal libs
from sacct_plot.sacct import SacctData
from sacct_plot.sweep import compute_allocation, apply_bucket, apply_cumulative, apply_top_n
from sacct_plot.wait import compute_wait_time, apply_wait_bucket, apply_wait_top_n
from sacct_plot.plot import render, render_wait


# Public interface
__all__ = ['main', 'SacctPlotApp', '__version__']
__version__ = get_version('sacct-plot')


# Global logger and configuration
default_config: Final[Namespace] = Namespace({
    'log': {
        'level': 'info',
        'style': 'default',
    },
})

try:
    config = Configuration.from_local(env=True, prefix='SACCT_PLOT', default=default_config)
    log = Logger.default('sacct-plot',
                         level=level_by_name[config.log.level.upper()],
                         **logging_styles[config.log.style.lower()])
except Exception as exc:
    print(f'Error [{exc.__class__.__name__}] {exc}')
    sys.exit(exit_status.bad_config)


PROGRAM: Final[str] = path.basename(sys.argv[0])
USAGE: Final[str] = f"""\
Usage:
    {PROGRAM} [-hv] [-u USER] [-A ACCOUNT] [-r PARTITION] [-q QOS] [-s STATE]
    {'':>{len(PROGRAM)}} [-S STARTTIME] [-E ENDTIME]
    {'':>{len(PROGRAM)}} [--by {{account,user,qos}}] [--gpu] [--wait] [--bucket INTERVAL]
    {'':>{len(PROGRAM)}} [--all] [--sum | --mean | --max | --min | --median] [--cumulative] [--top N]
    {'':>{len(PROGRAM)}} [--stacked] [-c COLORS] [--size W,H] [--data]
    {__doc__}\
"""

VERSION: Final[str] = f'{PROGRAM} v{__version__} ({python_implementation()} {python_version()})'

HELP: Final[str] = f"""\
{USAGE}

Sacct Filters:
  -u, --user       USER        Filter by user.
  -A, --account    ACCOUNT     Filter by account.
  -r, --partition  PARTITION   Filter by partition.
  -q, --qos        QOS         Filter by quality of service.
  -s, --state      STATE       Filter by job state.
  -S, --starttime  STARTTIME   Filter by start time (YYYY-MM-DD[THH:MM[:SS]]).
  -E, --endtime    ENDTIME     Filter by end time (YYYY-MM-DD[THH:MM[:SS]]).

Analysis:
  --by             GROUP       Overlay series by {{account,user,qos}}.
  --gpu                        Plot GPU allocation instead of CPU.
  --wait                       Analyse job wait time (start − submit) instead of allocation.
  --bucket         INTERVAL    Resample to interval (e.g. 1h, 1d).
  --top            N           Show only top N groups; collapse rest to "other".
  --all                        Overlay full aggregate as "all" line (requires --by).

Aggregation (with --bucket):
  --sum                        Resource-hours per bucket (allocation default).
  --mean                       Time-weighted average / mean wait time.
  --max                        Peak allocation / longest wait in the bucket.
  --min                        Minimum allocation within the bucket.
  --median                     Median wait time per bucket (wait default).
  --cumulative                 Show running cumulative total.

Formatting:
  -c, --color        COLORS    Comma-separated color names (e.g. 'blue,red,green').
  -l, --legend       POS       Legend position (topleft, topright, bottomleft, bottomright).
      --size         W,H       Plot width and height in characters (default: terminal size).

Output:
  --stacked                    Stacked area view instead of overlaid lines.
  --data                       Dump processed DataFrame instead of plotting.

General:
  -d, --debug                  Enable debug logging.
  -v, --version                Show version information and exit.
  -h, --help                   Show this help message and exit.\
"""


def _color_list(spec: str, sep: str = ',') -> List[str]:
    """Split comma-separated color names."""
    return spec.strip().split(sep)


def _split_size(spec: str, sep: str = ',') -> Tuple[int, int]:
    """Split size spec (e.g. '100,20') into (width, height)."""
    width, height = map(int, spec.strip().split(sep))
    return width, height


class SacctPlotApp(Application):
    """Application interface for sacct-plot."""

    interface = Interface(PROGRAM, USAGE, HELP)
    interface.add_argument('-v', '--version', action='version', version=VERSION)

    # Sacct filter flags
    user: str = None
    interface.add_argument('-u', '--user', type=str, default=None)

    account: str = None
    interface.add_argument('-A', '--account', type=str, default=None)

    partition: str = None
    interface.add_argument('-r', '--partition', type=str, default=None)

    qos: str = None
    interface.add_argument('-q', '--qos', type=str, default=None)

    state: str = None
    interface.add_argument('-s', '--state', type=str, default=None)

    starttime: str = None
    interface.add_argument('-S', '--starttime', type=str, default=None)

    endtime: str = None
    interface.add_argument('-E', '--endtime', type=str, default=None)

    # Analysis flags
    by: str = None
    interface.add_argument('--by', type=str, default=None, choices=['account', 'user', 'qos'])

    gpu: bool = False
    interface.add_argument('--gpu', action='store_true', default=False)

    wait: bool = False
    interface.add_argument('--wait', action='store_true', default=False)

    bucket: str = None
    interface.add_argument('--bucket', type=str, default=None)

    top: int = None
    interface.add_argument('--top', type=int, default=None)

    all_groups: bool = False
    interface.add_argument('--all', action='store_true', default=False, dest='all_groups')

    # Aggregation flags (mutually exclusive)
    agg: str = 'sum'
    agg_interface = interface.add_mutually_exclusive_group()
    agg_interface.add_argument('--sum', action='store_const', const='sum', default='sum', dest='agg')
    agg_interface.add_argument('--mean', action='store_const', const='mean', dest='agg')
    agg_interface.add_argument('--max', action='store_const', const='max', dest='agg')
    agg_interface.add_argument('--min', action='store_const', const='min', dest='agg')
    agg_interface.add_argument('--median', action='store_const', const='median', dest='agg')

    cumulative: bool = False
    interface.add_argument('--cumulative', action='store_true', default=False)

    # Output flags
    stacked: bool = False
    interface.add_argument('--stacked', action='store_true', default=False)

    data_mode: bool = False
    interface.add_argument('--data', action='store_true', default=False, dest='data_mode')

    # Formatting
    colors: Optional[List[str]] = None
    interface.add_argument('-c', '--color', type=_color_list, default=None, dest='colors')

    legend: str = 'bottomright'
    interface.add_argument('-l', '--legend', type=str, default='bottomright',
                           choices=['topleft', 'topright', 'bottomleft', 'bottomright'])

    size: Optional[Tuple[int, int]] = None
    interface.add_argument('--size', type=_split_size, default=None)

    # Logging
    log_level: str = config.log.level.lower()
    log_interface = interface.add_mutually_exclusive_group()
    log_interface.add_argument('-d', '--debug', action='store_const', const='debug',
                               default=log_level, dest='log_level')

    def run(self: SacctPlotApp) -> None:
        """Run the application."""
        log.setLevel(level_by_name[self.log_level.upper()])

        # Build sacct filter options
        options = {
            'user': self.user,
            'account': self.account,
            'partition': self.partition,
            'qos': self.qos,
            'state': self.state,
            'starttime': self.starttime,
            'endtime': self.endtime,
        }
        options_info = ', '.join(f'{k}={v}' for k, v in options.items() if v is not None) or 'no filters'
        log.info(f'Scanning jobs with sacct ({options_info})')

        # Fetch data
        sacct_data = SacctData.from_sacct(**options)
        log.info(f'Loaded {len(sacct_data.data)} job records')

        if self.wait:
            self._run_wait(sacct_data, options)
        else:
            self._run_allocation(sacct_data, options)

    def _run_wait(self: SacctPlotApp, sacct_data: SacctData, options: dict) -> None:
        """Wait time analysis mode."""
        # Default to median for wait mode (sum is allocation default)
        wait_agg = self.agg if self.agg in ('median', 'mean', 'max') else 'median'

        wait_data = compute_wait_time(sacct_data.data, by=self.by)
        if wait_data.empty:
            log.warning('No valid job records for wait time analysis')
            return
        log.info(f'Computed wait time for {len(wait_data)} jobs')

        # Optional top-N filtering on raw data
        if self.top and self.by:
            wait_data = apply_wait_top_n(wait_data, n=self.top, by=self.by)
            log.debug(f'Filtered to top {self.top} groups by total wait')

        # Optional bucket rollup
        summary = None
        if self.bucket:
            summary = apply_wait_bucket(wait_data, self.bucket, agg=wait_agg, by=self.by)
            result = summary['center']
            log.debug(f'Bucketed to {self.bucket} with {wait_agg} aggregation')

            # Optional --all aggregate overlay
            if self.all_groups and self.by:
                if options.get(self.by) is not None:
                    all_options = {k: v for k, v in options.items() if k != self.by}
                    log.info('Fetching full aggregate for --all')
                    all_data = SacctData.from_sacct(**all_options)
                else:
                    log.info('Reusing dataset for --all (no filter on %s)', self.by)
                    all_data = sacct_data
                all_wait = compute_wait_time(all_data.data, by=None)
                if not all_wait.empty:
                    all_summary = apply_wait_bucket(all_wait, self.bucket, agg=wait_agg, by=None)
                    result['all'] = all_summary['center']['wait'].reindex(result.index).fillna(0)
                    log.debug('Merged "all" aggregate column')

            if self.data_mode:
                result.to_csv(sys.stdout)
                return
        else:
            if self.all_groups:
                log.warning('--all in wait mode requires --bucket; ignoring')
            if self.data_mode:
                wait_data.to_csv(sys.stdout)
                return

        # Build title and ylabel with auto-scaled units
        max_wait = wait_data['wait'].max()
        if max_wait < 120:
            unit = 'seconds'
        elif max_wait < 7200:
            unit = 'minutes'
        elif max_wait < 172800:
            unit = 'hours'
        else:
            unit = 'days'

        ylabel = f'Wait ({unit})'
        if self.bucket:
            title = f'{wait_agg.title()} Wait Time (per {self.bucket})'
        else:
            title = 'Job Wait Time'
        if self.by:
            title += f' (by {self.by})'

        render_wait(
            wait_data=wait_data,
            summary=summary,
            unit=unit,
            title=title,
            ylabel=ylabel,
            by=self.by,
            colors=self.colors,
            size=self.size,
            legend=self.legend,
        )

    def _run_allocation(self: SacctPlotApp, sacct_data: SacctData, options: dict) -> None:
        """Allocation analysis mode (default)."""
        # Compute allocation time-series
        metric = 'gpu' if self.gpu else 'cpu'
        alloc = compute_allocation(sacct_data.data, metric=metric, by=self.by)
        if alloc.empty:
            log.warning('No valid job records to plot')
            return

        # Optional bucket rollup
        if self.bucket:
            alloc = apply_bucket(alloc, interval=self.bucket, agg=self.agg)
            log.debug(f'Bucketed to {self.bucket} with {self.agg} aggregation')

        # Optional cumulative sum (requires --bucket)
        if self.cumulative:
            if not self.bucket:
                log.warning('--cumulative requires --bucket; ignoring')
            else:
                alloc = apply_cumulative(alloc)
                log.debug('Applied cumulative sum')

        # Optional top-N filtering
        if self.top and self.by:
            alloc = apply_top_n(alloc, n=self.top)
            log.debug(f'Filtered to top {self.top} groups')

        # Optional --all aggregate overlay
        if self.all_groups:
            if not self.by:
                log.warning('--all requires --by; ignoring')
            else:
                # Reuse already-fetched data when the by-dimension has no filter,
                # otherwise fetch the full dataset without the by-dimension filter.
                if options.get(self.by) is not None:
                    all_options = {k: v for k, v in options.items() if k != self.by}
                    log.info('Fetching full aggregate for --all')
                    all_data = SacctData.from_sacct(**all_options)
                else:
                    log.info('Reusing dataset for --all (no filter on %s)', self.by)
                    all_data = sacct_data
                all_alloc = compute_allocation(all_data.data, metric=metric, by=None)
                if not all_alloc.empty:
                    if self.bucket:
                        all_alloc = apply_bucket(all_alloc, interval=self.bucket, agg=self.agg)
                    if self.cumulative and self.bucket:
                        all_alloc = apply_cumulative(all_alloc)
                    # Align step functions and merge
                    alloc['all'] = all_alloc['allocation'].reindex(alloc.index, method='ffill').fillna(0)
                    log.debug('Merged "all" aggregate column')

        if self.data_mode:
            alloc.to_csv(sys.stdout)
            return

        # Build title and labels
        resource = 'GPUs' if self.gpu else 'CPUs'
        resource_unit = resource  # e.g. 'GPUs'
        if self.bucket:
            if self.agg == 'sum':
                ylabel = f'{resource[:-1]}\u00b7h'
                title = f'{ylabel} Allocated (per {self.bucket})'
            elif self.agg == 'mean':
                ylabel = f'{resource} (avg)'
                title = f'Average {resource} Allocated (per {self.bucket})'
            else:
                ylabel = resource
                title = f'{self.agg.title()} {resource} Allocated (per {self.bucket})'
            if self.cumulative:
                title = f'Cumulative {title.split("(")[0].strip()}'
        else:
            title = f'Instantaneous {resource} Allocated'
            ylabel = resource
        if self.by:
            title += f' (by {self.by})'

        render(alloc, title=title, ylabel=ylabel, stacked=self.stacked,
               colors=self.colors, size=self.size, grouped=bool(self.by),
               legend=self.legend)


def main() -> int:
    """Entry point for sacct-plot."""
    return SacctPlotApp.main(sys.argv[1:])
