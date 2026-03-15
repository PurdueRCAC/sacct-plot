# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Shared test fixtures for sacct-plot."""

from __future__ import annotations
from typing import Final
from pathlib import Path
import gzip

import pytest


# Path to compressed fixture data
FIXTURES_DIR: Final[Path] = Path(__file__).parent / 'fixtures'
FIXTURE_FILE: Final[Path] = FIXTURES_DIR / 'sacct_3months.txt.gz'


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


@pytest.fixture
def mock_sacct(sacct_fixture_bytes: bytes, monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Patch ``subprocess.check_output`` to return fixture data instead of calling sacct.

    Also disables caching so tests always go through the mock.

    Usage::

        def test_something(mock_sacct):
            data = SacctData.from_sacct(starttime='2026-01-01')
            assert len(data.data) > 0
    """
    import sacct_plot.sacct as sacct_module

    monkeypatch.setattr(sacct_module, 'check_output', lambda cmd: sacct_fixture_bytes)
    monkeypatch.setattr(sacct_module, 'CACHE_TTL', 0)  # Disable caching

    return sacct_fixture_bytes
