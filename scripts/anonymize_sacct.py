#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Anonymize sacct output for use as test fixture data.

Reads pipe-delimited sacct output (with header), anonymizes identifying fields
(user, account, job ID, CANCELLED-by UIDs), and writes headerless output
matching the format produced by `sacct --parsable2 --noheader`.

Preserved as-is: timestamps, NCPUS, AllocTRES, ElapsedRaw, State (base),
and QOS (generic values kept, specific ones anonymized).

Usage:
    python scripts/anonymize_sacct.py INPUT_FILE OUTPUT_FILE
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict


# QOS values that are generic enough to keep as-is
GENERIC_QOS = {'gpu', 'cpu', 'normal', 'ai', 'wide-highmem', 'benchmarking',
               'array-submission', 'rcac'}

# Pattern for "CANCELLED by NNNNNN"
CANCELLED_BY_RE = re.compile(r'^CANCELLED by (\d+)$')


def main() -> None:
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} INPUT_FILE OUTPUT_FILE', file=sys.stderr)
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]

    # Mapping registries
    user_map: dict[str, str] = {}
    account_map: dict[str, str] = {}
    qos_map: dict[str, str] = {}
    uid_map: dict[str, str] = {}

    # Counters for anonymous names
    user_counter = 0
    account_counter = 0
    qos_counter = 0
    uid_counter = 1000  # Start UIDs at 1000

    # Job ID offset (shift all job IDs by a fixed amount)
    job_id_offset = 1_000_000

    def anon_user(name: str) -> str:
        nonlocal user_counter
        if name not in user_map:
            user_counter += 1
            user_map[name] = f'user{user_counter:04d}'
        return user_map[name]

    def anon_account(name: str) -> str:
        nonlocal account_counter
        if name not in account_map:
            account_counter += 1
            # Preserve -gpu / -ai suffix as a hint for partition association
            suffix = ''
            base = name
            for tag in ('-gpu', '-ai'):
                if name.endswith(tag):
                    suffix = tag.replace('-', '_')
                    base = name[:-len(tag)]
                    break
            account_map[name] = f'acct{account_counter:04d}{suffix}'
        return account_map[name]

    def anon_qos(name: str) -> str:
        nonlocal qos_counter
        if name in GENERIC_QOS:
            return name
        if name not in qos_map:
            qos_counter += 1
            qos_map[name] = f'qos{qos_counter:02d}'
        return qos_map[name]

    def anon_uid(uid_str: str) -> str:
        nonlocal uid_counter
        if uid_str not in uid_map:
            uid_counter += 1
            uid_map[uid_str] = str(uid_counter)
        return uid_map[uid_str]

    def anon_job_id(jid: str) -> str:
        """Shift numeric job IDs; preserve array suffixes like _10."""
        # Handle array jobs (12345_10) and step jobs (12345.batch)
        for sep in ('_', '.'):
            if sep in jid:
                base, rest = jid.split(sep, 1)
                try:
                    return f'{int(base) + job_id_offset}{sep}{rest}'
                except ValueError:
                    return jid
        try:
            return str(int(jid) + job_id_offset)
        except ValueError:
            return jid

    def anon_state(state: str) -> str:
        """Anonymize UIDs in CANCELLED by NNNNNN."""
        m = CANCELLED_BY_RE.match(state)
        if m:
            return f'CANCELLED by {anon_uid(m.group(1))}'
        return state

    lines_written = 0

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for i, line in enumerate(fin):
            line = line.rstrip('\n')
            if not line:
                continue

            # Skip header line
            if i == 0 and line.startswith('JobID|'):
                continue

            fields = line.split('|')
            if len(fields) != 11:
                print(f'WARNING: skipping line {i+1} with {len(fields)} fields',
                      file=sys.stderr)
                continue

            job_id, user, account, qos, ncpus, alloc_tres, elapsed_raw, \
                state, submit, start, end = fields

            anon_fields = [
                anon_job_id(job_id),
                anon_user(user),
                anon_account(account),
                anon_qos(qos),
                ncpus,
                alloc_tres,
                elapsed_raw,
                anon_state(state),
                submit,
                start,
                end,
            ]

            fout.write('|'.join(anon_fields) + '\n')
            lines_written += 1

    print(f'Anonymized {lines_written:,} records', file=sys.stderr)
    print(f'  Users:    {len(user_map):,} -> user0001..user{user_counter:04d}',
          file=sys.stderr)
    print(f'  Accounts: {len(account_map):,} -> acct0001..acct{account_counter:04d}',
          file=sys.stderr)
    print(f'  QOS:      {len(qos_map)} anonymized, {len(GENERIC_QOS)} kept as-is',
          file=sys.stderr)
    print(f'  UIDs:     {len(uid_map):,} in CANCELLED states',
          file=sys.stderr)
    print(f'  Job IDs:  offset by +{job_id_offset:,}', file=sys.stderr)


if __name__ == '__main__':
    main()
