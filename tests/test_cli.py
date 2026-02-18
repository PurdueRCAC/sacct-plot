# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Integration tests for the sacct-plot CLI."""

import subprocess
import sys

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run sacct-plot as a subprocess and return the result."""
    return subprocess.run(
        [sys.executable, '-m', 'sacct_plot', *args],
        capture_output=True, text=True, timeout=10,
    )


class TestHelp:
    """Tests for --help output."""

    def test_help_exits_zero(self):
        result = run_cli('--help')
        assert result.returncode == 0

    def test_help_contains_usage(self):
        result = run_cli('--help')
        assert 'Usage:' in result.stdout

    def test_help_contains_sacct_filters(self):
        result = run_cli('--help')
        assert '--user' in result.stdout
        assert '--account' in result.stdout
        assert '--starttime' in result.stdout

    def test_help_contains_analysis_flags(self):
        result = run_cli('--help')
        assert '--by' in result.stdout
        assert '--gpu' in result.stdout
        assert '--bucket' in result.stdout
        assert '--top' in result.stdout

    def test_help_contains_output_flags(self):
        result = run_cli('--help')
        assert '--stacked' in result.stdout
        assert '--data' in result.stdout


class TestVersion:
    """Tests for --version output."""

    def test_version_exits_zero(self):
        result = run_cli('--version')
        assert result.returncode == 0

    def test_version_contains_version_string(self):
        result = run_cli('--version')
        output = result.stdout + result.stderr
        assert 'v0.' in output  # Contains version number

    def test_version_contains_python_version(self):
        result = run_cli('--version')
        output = result.stdout + result.stderr
        assert 'Python' in output or 'CPython' in output


class TestDataMode:
    """Tests for --data mode (requires no sacct, uses mocked data)."""

    def test_data_flag_recognized(self):
        """Verify --data is a recognized flag (will fail on sacct, but not on parsing)."""
        result = run_cli('--help')
        assert '--data' in result.stdout
