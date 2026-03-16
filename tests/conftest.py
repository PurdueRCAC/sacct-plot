# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Shared test fixtures for sacct-plot."""


# Type annotations
from __future__ import annotations
from typing import Dict, Final, List, Tuple

# Standard libs
from pathlib import Path
import gzip

# External libs
import pytest


# Path to compressed fixture data
FIXTURES_DIR: Final[Path] = Path(__file__).parent / 'fixtures'
FIXTURE_FILE: Final[Path] = FIXTURES_DIR / 'sacct_3months.txt.gz'

# Sacct flag → field index in pipe-delimited output
# Fields: JobID(0) User(1) Account(2) QOS(3) NCPUS(4) AllocTRES(5)
#         ElapsedRaw(6) State(7) Submit(8) Start(9) End(10)
_EXACT_FILTERS: Final[Dict[str, int]] = {
    '-u': 1,   # user
    '-A': 2,   # account
    '-q': 3,   # qos
    '-s': 7,   # state
}

_PLACEHOLDER_VALUES: Final[frozenset] = frozenset(('Unknown', 'None', ''))


def _parse_cmd_filters(cmd: List[str]) -> Dict[str, str]:
    """Extract filter options from a sacct command list.

    Returns a dict with keys like '-u', '-A', '-S', etc. mapped to their
    argument values.
    """
    known_flags = {'-u', '-A', '-q', '-s', '-S', '-E', '-r'}
    filters: Dict[str, str] = {}
    i = 0
    while i < len(cmd):
        if cmd[i] in known_flags and i + 1 < len(cmd):
            filters[cmd[i]] = cmd[i + 1]
            i += 2
        else:
            i += 1
    return filters


def _matches(fields: List[str], filters: Dict[str, str]) -> bool:
    """Return True if a pre-split sacct record matches all filters.

    Exact-match filters (user, account, qos, state) support comma-separated
    multi-values (e.g. ``-u alice,bob``).  Timestamp filters use lexicographic
    comparison which is correct for ISO-8601 strings.
    """
    # Exact-match filters
    for flag, idx in _EXACT_FILTERS.items():
        if flag in filters:
            allowed = set(filters[flag].split(','))
            if fields[idx] not in allowed:
                return False

    # -S starttime: include jobs whose Start >= value
    if '-S' in filters:
        start = fields[9]
        if start in _PLACEHOLDER_VALUES:
            return False
        if start < filters['-S']:
            return False

    # -E endtime: include jobs whose End <= value
    if '-E' in filters:
        end = fields[10]
        if end in _PLACEHOLDER_VALUES:
            return False
        if end > filters['-E']:
            return False

    # -r partition: not in output fields — skip (can't filter)
    return True


@pytest.fixture(scope='session')
def sacct_fixture_bytes() -> bytes:
    """Load the full anonymized sacct fixture (decompressed) as bytes.

    This matches the return type of ``subprocess.check_output``.
    Loaded once per test session for performance.
    """
    with gzip.open(FIXTURE_FILE, 'rb') as f:
        return f.read()


@pytest.fixture(scope='session')
def sacct_fixture_text(sacct_fixture_bytes: bytes) -> str:
    """Load the full anonymized sacct fixture as a string."""
    return sacct_fixture_bytes.decode('utf-8')


@pytest.fixture(scope='session')
def sacct_fixture_records(sacct_fixture_text: str) -> List[Tuple[str, List[str]]]:
    """Pre-split fixture lines into (raw_line, fields) tuples.

    Splitting is done once per session so that per-call filtering in
    ``mock_sacct`` only needs to compare pre-split fields.
    """
    records: List[Tuple[str, List[str]]] = []
    for line in sacct_fixture_text.strip().split('\n'):
        if not line:
            continue
        fields = line.split('|')
        if len(fields) == 11:
            records.append((line, fields))
    return records


@pytest.fixture
def mock_sacct(
    sacct_fixture_records: List[Tuple[str, List[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch ``subprocess.check_output`` to return filtered fixture data.

    Inspects the sacct command to extract filter flags and returns only
    matching records — mimicking what real ``sacct`` does server-side.
    Also disables caching so tests always go through the mock.

    Usage::

        def test_something(mock_sacct):
            data = SacctData.from_sacct(starttime='2026-03-13')
            assert len(data.data) > 0
    """
    def _mock_check_output(cmd: List[str]) -> bytes:
        filters = _parse_cmd_filters(cmd)
        if filters:
            matching = [line for line, fields in sacct_fixture_records
                        if _matches(fields, filters)]
        else:
            matching = [line for line, _ in sacct_fixture_records]
        return '\n'.join(matching).encode('utf-8')

    import sacct_plot.sacct as sacct_module
    monkeypatch.setattr(sacct_module, 'check_output', _mock_check_output)
    monkeypatch.setattr(sacct_module, 'CACHE_TTL', 0)  # Disable caching
