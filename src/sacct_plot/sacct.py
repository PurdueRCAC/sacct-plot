# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Sacct data acquisition, parsing, and caching."""


# Type annotations
from __future__ import annotations
from typing import Final, Dict, List, Optional

# Standard libs
import os
from datetime import datetime
from dataclasses import dataclass
from subprocess import check_output

# External libs
from pandas import DataFrame, read_parquet, to_datetime
from cmdkit.logging import Logger


# Module logger
log = Logger.default(name=__name__)


# Sacct command configuration
SACCT_BASE: Final[List[str]] = [
    '/usr/bin/sacct', '-aX', '-o',
    'JobID,User,Account,QOS,NCPUS,AllocTRES,ElapsedRaw,State,Start,End',
    '--parsable2', '--noheader', '--duplicates', '--array',
]

SACCT_OPTIONS: Final[Dict[str, str]] = {
    'starttime': '-S',
    'endtime': '-E',
    'account': '-A',
    'user': '-u',
    'partition': '-r',
    'qos': '-q',
    'state': '-s',
}

SACCT_FIELDS: Final[List[str]] = [
    'job_id', 'user', 'account', 'qos', 'ncpus', 'alloc_tres',
    'elapsed_raw', 'state', 'start', 'end',
]

# Cache configuration
CACHE_DIR: Final[str] = os.path.join(os.path.expanduser('~'), '.cache', 'sacct')
CACHE_TTL: Final[int] = 600  # 10 minutes


def _parse_gpus(alloc_tres: str) -> int:
    """Extract GPU count from AllocTRES field (e.g. 'billing=8,cpu=8,gres/gpu=4,mem=64G')."""
    for item in alloc_tres.split(','):
        key, _, value = item.partition('=')
        if key.startswith('gres/gpu'):
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def _parse_timestamp(value: str) -> Optional[datetime]:
    """Parse sacct timestamp, returning None for placeholder values."""
    if not value or value in ('Unknown', 'None'):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class JobInfo:
    """Parsed sacct job record."""

    job_id: str
    user: str
    account: str
    qos: str
    ncpus: int
    alloc_tres: str
    elapsed_raw: int
    state: str
    start: Optional[datetime]
    end: Optional[datetime]

    @property
    def gpus(self) -> int:
        """GPU count parsed from AllocTRES."""
        return _parse_gpus(self.alloc_tres)

    @classmethod
    def from_line(cls, line: str) -> JobInfo:
        """Create a JobInfo from a pipe-delimited sacct output line."""
        parts = line.strip().split('|')
        if len(parts) != 10:
            raise ValueError(f'Expected 10 fields, got {len(parts)}: {line!r}')
        return cls(
            job_id=parts[0].split('.')[0],
            user=parts[1],
            account=parts[2],
            qos=parts[3],
            ncpus=int(parts[4]),
            alloc_tres=parts[5],
            elapsed_raw=int(parts[6]) if parts[6] else 0,
            state=parts[7],
            start=_parse_timestamp(parts[8]),
            end=_parse_timestamp(parts[9]),
        )

    def to_dict(self) -> dict:
        """Convert to dict suitable for DataFrame construction."""
        return {
            'job_id': self.job_id,
            'user': self.user,
            'account': self.account,
            'qos': self.qos,
            'ncpus': self.ncpus,
            'gpus': self.gpus,
            'elapsed_raw': self.elapsed_raw,
            'state': self.state,
            'start': self.start,
            'end': self.end,
        }


@dataclass
class SacctData:
    """Container for sacct job data as a DataFrame."""

    data: DataFrame

    @classmethod
    def from_sacct(cls, cache: bool = True, **options: Optional[str]) -> SacctData:
        """Query sacct and return parsed job data (with parquet caching)."""
        cmd = SACCT_BASE.copy()
        for key, value in options.items():
            if value is not None:
                cmd.extend([SACCT_OPTIONS[key], value])

        # Check cache
        cache_file = _cache_path(options)
        if cache and _cache_valid(cache_file):
            log.info(f'Loading cached data from {cache_file}')
            return cls.from_local(cache_file)

        # Run sacct
        log.debug(f'Running: {" ".join(cmd)}')
        try:
            output = check_output(cmd).decode('utf-8')
        except Exception as exc:
            log.error(f'sacct failed: {exc}')
            raise

        # Parse lines
        jobs: List[dict] = []
        for line in output.strip().split('\n'):
            if not line:
                continue
            try:
                job = JobInfo.from_line(line)
                jobs.append(job.to_dict())
            except ValueError as exc:
                log.warning(f'Skipping invalid line: {exc}')

        data = cls(data=DataFrame(jobs))

        # Ensure datetime columns
        if not data.data.empty:
            data.data['start'] = to_datetime(data.data['start'])
            data.data['end'] = to_datetime(data.data['end'])

        # Write cache
        if cache:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            log.info(f'Caching data to {cache_file}')
            data.to_local(cache_file)

        return data

    def to_local(self, filename: str) -> None:
        """Save data to parquet."""
        self.data.to_parquet(filename, index=False)

    @classmethod
    def from_local(cls, filename: str) -> SacctData:
        """Load data from parquet."""
        return cls(data=read_parquet(filename))


def _cache_path(options: dict) -> str:
    """Build a deterministic cache file path from filter options."""
    parts = '.'.join(f'{k}={v}' for k, v in sorted(options.items()) if v is not None)
    fn = f'sacct.{parts}.parquet' if parts else 'sacct.parquet'
    return os.path.join(CACHE_DIR, fn)


def _cache_valid(filepath: str) -> bool:
    """Check if cache file exists and is within TTL."""
    if not os.path.exists(filepath):
        return False
    age = datetime.now().timestamp() - os.path.getmtime(filepath)
    return age < CACHE_TTL
