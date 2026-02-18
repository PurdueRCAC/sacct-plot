# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Visualize instantaneous allocated resources (CPUs/GPUs) on Slurm clusters over time."""


# Type annotations
from __future__ import annotations
from typing import Final

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
from sacct_plot.sweep import compute_allocation, apply_bucket, apply_top_n


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
    {'':>{len(PROGRAM)}} [--by {{account,user,qos}}] [--gpu] [--bucket INTERVAL]
    {'':>{len(PROGRAM)}} [--sum | --mean | --max | --min] [--top N]
    {'':>{len(PROGRAM)}} [--stacked] [--data]
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
  --bucket         INTERVAL    Resample to interval (e.g. 1h, 1d).
  --top            N           Show only top N groups; collapse rest to "other".

Aggregation (with --bucket):
  --sum                        Aggregate by sum (default).
  --mean                       Aggregate by mean.
  --max                        Aggregate by max.
  --min                        Aggregate by min.

Output:
  --stacked                    Stacked area view instead of overlaid lines.
  --data                       Dump processed DataFrame instead of plotting.

General:
  -d, --debug                  Enable debug logging.
  -v, --version                Show version information and exit.
  -h, --help                   Show this help message and exit.\
"""


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

    bucket: str = None
    interface.add_argument('--bucket', type=str, default=None)

    top: int = None
    interface.add_argument('--top', type=int, default=None)

    # Aggregation flags (mutually exclusive)
    agg: str = 'sum'
    agg_interface = interface.add_mutually_exclusive_group()
    agg_interface.add_argument('--sum', action='store_const', const='sum', default='sum', dest='agg')
    agg_interface.add_argument('--mean', action='store_const', const='mean', dest='agg')
    agg_interface.add_argument('--max', action='store_const', const='max', dest='agg')
    agg_interface.add_argument('--min', action='store_const', const='min', dest='agg')

    # Output flags
    stacked: bool = False
    interface.add_argument('--stacked', action='store_true', default=False)

    data_mode: bool = False
    interface.add_argument('--data', action='store_true', default=False, dest='data_mode')

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

        # Optional top-N filtering
        if self.top and self.by:
            alloc = apply_top_n(alloc, n=self.top)
            log.debug(f'Filtered to top {self.top} groups')

        if self.data_mode:
            print(alloc.to_string())
            return

        # Rendering will be wired in Phase 4
        log.info('Rendering not yet implemented')


def main() -> int:
    """Entry point for sacct-plot."""
    return SacctPlotApp.main(sys.argv[1:])
