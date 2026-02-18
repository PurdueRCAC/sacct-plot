# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Unit tests for sacct data parsing."""

import pytest
from datetime import datetime

from sacct_plot.sacct import JobInfo, _parse_gpus, _parse_timestamp


class TestParseGpus:
    """Tests for GPU count extraction from AllocTRES."""

    def test_no_gpu(self):
        assert _parse_gpus('billing=8,cpu=8,mem=64G') == 0

    def test_single_gpu(self):
        assert _parse_gpus('billing=8,cpu=8,gres/gpu=1,mem=64G') == 1

    def test_multiple_gpus(self):
        assert _parse_gpus('billing=32,cpu=32,gres/gpu=4,mem=256G') == 4

    def test_gpu_with_type(self):
        assert _parse_gpus('billing=8,cpu=8,gres/gpu:a100=2,mem=128G') == 2

    def test_empty_string(self):
        assert _parse_gpus('') == 0

    def test_gpu_invalid_value(self):
        assert _parse_gpus('gres/gpu=abc') == 0


class TestParseTimestamp:
    """Tests for sacct timestamp parsing."""

    def test_valid_datetime(self):
        result = _parse_timestamp('2026-02-18T12:30:00')
        assert result == datetime(2026, 2, 18, 12, 30, 0)

    def test_unknown(self):
        assert _parse_timestamp('Unknown') is None

    def test_none_string(self):
        assert _parse_timestamp('None') is None

    def test_empty(self):
        assert _parse_timestamp('') is None


class TestJobInfo:
    """Tests for JobInfo.from_line parsing."""

    def test_basic_job(self):
        line = '12345|user1|account1|8|billing=8,cpu=8,mem=64G|3600|COMPLETED|2026-02-18T10:00:00|2026-02-18T11:00:00'
        job = JobInfo.from_line(line)
        assert job.job_id == '12345'
        assert job.user == 'user1'
        assert job.account == 'account1'
        assert job.ncpus == 8
        assert job.elapsed_raw == 3600
        assert job.state == 'COMPLETED'
        assert job.start == datetime(2026, 2, 18, 10, 0, 0)
        assert job.end == datetime(2026, 2, 18, 11, 0, 0)
        assert job.gpus == 0

    def test_gpu_job(self):
        line = '67890|user2|gpu_account|32|billing=32,cpu=32,gres/gpu=4,mem=256G|7200|COMPLETED|2026-02-18T08:00:00|2026-02-18T10:00:00'
        job = JobInfo.from_line(line)
        assert job.ncpus == 32
        assert job.gpus == 4
        assert job.elapsed_raw == 7200

    def test_array_job_id(self):
        """Array job IDs with underscores should be split at the dot."""
        line = '12345_10|user1|acct|4|cpu=4|1800|COMPLETED|2026-02-18T10:00:00|2026-02-18T10:30:00'
        job = JobInfo.from_line(line)
        assert job.job_id == '12345_10'

    def test_step_job_id(self):
        """Job step IDs (12345.batch) should be truncated to the base job ID."""
        line = '12345.batch|user1|acct|4|cpu=4|1800|COMPLETED|2026-02-18T10:00:00|2026-02-18T10:30:00'
        job = JobInfo.from_line(line)
        assert job.job_id == '12345'

    def test_pending_job_no_end(self):
        line = '99999|user1|acct|16|cpu=16|0|PENDING|Unknown|Unknown'
        job = JobInfo.from_line(line)
        assert job.start is None
        assert job.end is None
        assert job.elapsed_raw == 0

    def test_wrong_field_count(self):
        with pytest.raises(ValueError, match='Expected 9 fields'):
            JobInfo.from_line('12345|user1|acct|8|cpu=8|3600|COMPLETED')

    def test_to_dict(self):
        line = '12345|user1|acct|8|billing=8,cpu=8,gres/gpu=2,mem=64G|3600|COMPLETED|2026-02-18T10:00:00|2026-02-18T11:00:00'
        job = JobInfo.from_line(line)
        d = job.to_dict()
        assert d['job_id'] == '12345'
        assert d['ncpus'] == 8
        assert d['gpus'] == 2
        assert 'alloc_tres' not in d  # raw field not in dict
        assert d['start'] == datetime(2026, 2, 18, 10, 0, 0)
        assert d['end'] == datetime(2026, 2, 18, 11, 0, 0)
