---
status: in-progress
current_phase: 6
last_updated: 2026-03-14
---

# sacct-plot Roadmap

## Overview

sacct-plot is a command-line tool that visualizes instantaneous allocated resources
(CPUs or GPUs) on Slurm clusters over time. It queries `sacct` for job records, uses
an event-based sweep algorithm to compute a 1-second-accurate allocation step function,
and renders overlaid time-series in the terminal via `tplot`.

Designed for HPC administrators and leadership to visualize fair-share usage patterns
across accounts, users, and QOS levels.

## Phase 1: Project Scaffolding

Set up the installable package skeleton with hatchling build, entry points, and
the basic Application class. No functional logic yet — just the shell.

Detailed implementation plan: <plan:44d0b156-7fab-4fc2-9849-4810d949b806>

- [x] `.gitignore` (rcac-mcp style, concise Python/venv/IDE)
- [x] `LICENSE` (MIT, "2026 Purdue RCAC")
- [x] `pyproject.toml` (hatchling, entry point, dependencies)
- [x] `README.md` (brief description, install, usage sketch)
- [x] `src/sacct_plot/__init__.py` (SPDX, SacctPlotApp skeleton, CLI flags, main())
- [x] `src/sacct_plot/__main__.py` (python -m support)
- [x] Verify: `uv sync && uv run sacct-plot --help`
- [x] Commit: "WIP: project scaffolding and package skeleton"

## Phase 2: Data Acquisition

Implement sacct subprocess integration, job record parsing, and parquet caching.
Add `Start` and `End` timestamps to the sacct output fields.

- [x] `src/sacct_plot/sacct.py` (JobInfo dataclass, SacctData with from_sacct/from_local/to_local)
- [x] Wire `SacctData.from_sacct()` into `SacctPlotApp.run()` with `--data` dump mode
- [x] Unit tests: `tests/test_sacct.py` (JobInfo.from_line parsing, GPU extraction from AllocTRES)
- [x] Commit: "WIP: sacct data acquisition and caching"

## Phase 3: Event Sweep

Core algorithm — transform job records into an instantaneous allocation time-series
using the event-based sweep approach.

- [x] `src/sacct_plot/sweep.py` (compute_allocation, apply_bucket, apply_top_n)
- [x] Wire sweep into `SacctPlotApp.run()` (--by, --gpu, --bucket, --top, aggregation flags)
- [x] Unit tests: `tests/test_sweep.py` (known jobs → expected step function, groupby, bucket, top-N)
- [x] Commit: "WIP: event sweep algorithm"

## Phase 4: Rendering

Terminal plotting with tplot via plot-cli's TimeSeriesFigure. Overlaid and stacked
line modes with smart time-axis labels.

- [x] `src/sacct_plot/plot.py` (render function using TimeSeriesFigure + generate_time_ticks)
- [x] Wire render into `SacctPlotApp.run()` (--stacked flag)
- [x] Integration tests: `tests/test_cli.py` (help, version, --data mode)
- [x] Commit: "WIP: terminal rendering with tplot"

## Phase 5: Polish & Release Prep

End-to-end testing on a live cluster, README expansion, and preparing for merge
from wip to main.

- [x] Manual testing with real sacct data on cluster (Gautschi, Anvil)
- [x] README: full usage examples
- [ ] Squash WIP commits and open PR against main

## Phase 6: Test Fixtures

Set up anonymized sacct fixture data and mock infrastructure so all tests
(existing and new) can run deterministically without a live cluster.

Detailed implementation plan: <plan:bf259d9b-7dc8-4b79-b373-95d0f739bc31>

- [ ] Create `scripts/anonymize_sacct.py` — reads raw sacct output, maps
      real usernames/accounts/job IDs to anonymized values, preserves timestamps
      and resource fields, writes pipe-delimited fixture file
- [ ] Generate fixture: `tests/fixtures/sacct_gpu_3months.txt` (anonymized
      ~3 months of a busy cluster)
- [ ] Add `tests/conftest.py` with `mock_sacct` fixture that patches
      `subprocess.check_output` to return fixture data
- [ ] Refactor existing tests to use fixture data where applicable
- [ ] Commit: "WIP: anonymized test fixtures and mock sacct"

## Phase 7: Data Acquisition Update

Add `Submit` timestamp to sacct query for wait time analysis.

- [ ] Add `Submit` to `SACCT_BASE` fields and `SACCT_FIELDS`
- [ ] Parse `Submit` in `JobInfo.from_line()` (11 fields)
- [ ] Include `submit` in `JobInfo.to_dict()`
- [ ] Update test fixtures for 11-field format
- [ ] Commit: "WIP: add Submit timestamp to sacct data"

## Phase 8: `--all` Aggregate Overlay

Add `--all` flag to overlay the full partition aggregate alongside grouped series.
Requires `--by`. Fetches the full dataset (dropping the filter for the `--by`
dimension), computes both per-group and aggregate series, and merges them.

- [ ] Add `--all` CLI flag to `SacctPlotApp`
- [ ] Implement dual-path computation in `run()` (full + filtered, merge "all" column)
- [ ] Validate `--all` requires `--by`, emit warning otherwise
- [ ] Unit tests: `--all` with `--by user`, interaction with `--top`
- [ ] Commit: "WIP: --all aggregate overlay"

## Phase 9: Wait Time Computation

New `wait.py` module. Computes per-job wait time (start − submit) and optional
bucketed aggregation with percentile envelope (p25, center, p75).

- [ ] `src/sacct_plot/wait.py` (`compute_wait_time`, `apply_wait_bucket`)
- [ ] Add `--wait` and `--median` CLI flags
- [ ] Wire `--wait` mode into `SacctPlotApp.run()` with title/ylabel auto-scaling
- [ ] Unit tests: `tests/test_wait.py` (wait computation, bucketing, grouped)
- [ ] Commit: "WIP: wait time computation"

## Phase 10: Wait Time Rendering

Scatter plot for raw wait time, line chart with percentile envelope for bucketed
mode. Scatter uses braille markers via tplot.

- [ ] Add `scatter()` method to `plot_cli.Figure`
- [ ] New `render_wait()` in `plot.py` (scatter for unbucketed, scatter + envelope for bucketed)
- [ ] Y-axis auto-scaling (minutes / hours / days)
- [ ] Integration tests for `--wait` mode
- [ ] Commit: "WIP: wait time rendering"

## Phase 11: Polish & v0.2 Release Prep

- [ ] Update AGENTS.md with new architecture
- [ ] Manual testing on cluster with real sacct data
- [ ] Squash WIP commits and open PR against main

---

## Design Considerations

### Event-Based Sweep

Each job emits two events: `(Start, +NCPUS)` and `(End, -NCPUS)`. Sorting all events
by timestamp and computing a cumulative sum yields the exact instantaneous allocation
at every boundary — O(N log N), fully vectorized in pandas, no per-second expansion.

With `--by account`, separate cumulative sums per group via `groupby().cumsum()`,
then pivot to wide format for overlaid plotting.

### Bucket Rollup

Optional `--bucket INTERVAL` resamples the step function to a coarser grid via
forward-fill + aggregation (default: sum; also --mean, --max, --min). This never
expands to per-second resolution — it operates on the sparse event boundaries.

### `--all` vs "other"

`--all` is an *inclusive* aggregate — it shows the total across all groups, including
the named ones. The "all" line sits above/around the individual group lines. This is
semantically opposite to "other" (produced by `--top N`), which is the *exclusive*
complement of the top N groups.

### Wait Time Analysis

Wait time (start − submit) is a per-job scalar, not a step function. Without
`--bucket`, it renders as a scatter plot (braille markers). With `--bucket`, it
shows both the scatter and summary envelope lines (center line + p25/p75). Default
aggregation is median (robust to outliers); `--mean`/`--max` available as overrides.

### plot-cli Dependency

The `TimeSeriesFigure` and `generate_time_ticks` classes live in plot-cli (currently
on its `wip` branch). During development, depend on the wip branch or a commit SHA.
Before release, merge/tag plot-cli's main branch and pin a stable version.

---

## Bootstrap Prompt

Use this prompt to resume development on this project:

```
I'm working on sacct-plot — a command-line tool that visualizes instantaneous
allocated resources (CPUs/GPUs) on Slurm clusters over time, built in Python
with cmdkit, pandas, and tplot.

Please read:
- ROADMAP.md for current status and next tasks
- The implementation plan referenced in ROADMAP.md for architectural details
- src/sacct_plot/__init__.py for the Application class and CLI
- src/sacct_plot/sacct.py for data acquisition
- src/sacct_plot/sweep.py for the event sweep algorithm
- src/sacct_plot/wait.py for wait time computation (if it exists)

Check the ROADMAP.md YAML frontmatter for the current phase. Implement the next
unchecked item(s) in the current phase, then:
1. Update ROADMAP.md to check off completed items
2. Update the frontmatter (current_phase, last_updated)
3. Commit with "WIP: <description>"

When a phase is complete, check in before proceeding to the next phase.
```
