---
status: in-progress
current_phase: 7
last_updated: 2026-03-15
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
- [x] Commit: "Merge wip to main, tag v0.1.0"

## Phase 6: Test Fixtures

Set up anonymized sacct fixture data and mock infrastructure so all tests
(existing and new) can run deterministically without a live cluster.

Detailed implementation plan: <plan:bf259d9b-7dc8-4b79-b373-95d0f739bc31>

- [x] Create `scripts/anonymize_sacct.py` — reads raw sacct output, maps
      real usernames/accounts/job IDs to anonymized values, preserves timestamps
      and resource fields, writes pipe-delimited fixture file
- [x] Generate fixture: `tests/fixtures/sacct_3months.txt.gz` (anonymized
      ~3 months of a busy cluster, gzip-compressed)
- [x] Add `tests/conftest.py` with `mock_sacct` fixture that patches
      `subprocess.check_output` to return fixture data
- [x] Commit: "WIP: anonymized test fixtures and mock sacct"

## Phase 7: Data Acquisition Update

Add `Submit` timestamp to sacct query for wait time analysis.

- [x] Add `Submit` to `SACCT_BASE` fields and `SACCT_FIELDS`
- [x] Parse `Submit` in `JobInfo.from_line()` (11 fields)
- [x] Include `submit` in `JobInfo.to_dict()`
- [x] Update test fixtures for 11-field format
- [x] Commit: "WIP: add Submit timestamp to sacct data"

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

## Phase 12: Comprehensive Testing Infrastructure

Build layered, automated end-to-end testing that exercises the full CLI pipeline
— from mock sacct data through sweep/wait computation to terminal rendering —
without requiring a live cluster. Each layer catches a different class of bug;
together they give high confidence that flag combinations, data shapes, and visual
output remain correct across refactors and dependency updates.

### Layer 1 — E2E Data Pipeline (`--data` mode)

The `--data` flag prints the processed DataFrame to stdout, exercising 100% of
the logic except the final render call. These tests run the full CLI with
`mock_sacct`, capture stdout, and assert on shape, columns, and values.

- [ ] `tests/test_pipeline.py` — allocation mode scenarios:
      ungrouped CPU, ungrouped GPU, `--by account`, `--by user`,
      `--bucket 1d --sum`, `--bucket 1h --mean`, `--top 3`,
      `--cumulative`, `--all --by account`
- [ ] `tests/test_pipeline.py` — wait mode scenarios:
      `--wait` (raw per-job), `--wait --bucket 1h`,
      `--wait --by user`, `--wait --bucket 1d --mean`
- [ ] Each test invokes `SacctPlotApp.main()` with `mock_sacct` patched in,
      captures stdout via `capsys`, parses the table, and asserts:
      exit code 0, expected column names, non-empty rows, value ranges
      (e.g. GPU counts ≥ 0, wait times ≥ 0)
- [ ] Commit: "WIP: e2e data pipeline tests"

### Layer 2 — Render Smoke Tests

Minimal integration tests that run the full CLI *including* the render path.
They don't validate what the plot looks like — only that the pipeline doesn't
crash for any supported flag combination.

- [ ] `tests/test_render_smoke.py` — for each major mode (allocation ungrouped,
      allocation grouped, allocation bucketed, allocation stacked,
      wait scatter, wait bucketed), invoke the CLI with `mock_sacct`
      and `--size 80,24` to force a fixed canvas
- [ ] Assert: exit code 0, stdout is non-empty, no exceptions on stderr
- [ ] Cover edge cases: empty result set (filter that matches no jobs),
      single-job dataset, `--top N` where N > number of groups
- [ ] Commit: "WIP: render smoke tests"

### Layer 3 — Render Structure Tests

Mock `TimeSeriesFigure` (and `render_wait`'s scatter/envelope path once it
exists) to validate that the render functions receive correct inputs without
actually drawing anything.

- [ ] `tests/test_render.py` — patch `plot_cli.plot.TimeSeriesFigure` with
      a recording mock that captures every `.line()` / `.scatter()` call
- [ ] Allocation mode assertions: number of `.line()` calls matches number
      of DataFrame columns, each call's `label` matches a column name,
      color cycle wraps correctly, x-values are epoch seconds,
      y-values match DataFrame column values
- [ ] Wait mode assertions: `.scatter()` called for raw dots,
      `.line()` called for median/p25/p75 envelope when bucketed,
      y-axis label reflects auto-scaled unit (minutes/hours/days)
- [ ] Stacked mode assertion: cumulative y-values are monotonically
      non-decreasing across series at each x-point
- [ ] Assert `fig.draw()` is called exactly once per render invocation
- [ ] Commit: "WIP: render structure tests"

### Layer 4 — Golden Snapshot Tests

Capture the full terminal-rendered output (character grid) for key scenarios
and diff against checked-in golden files. Since tplot renders to a fixed
character grid at a fixed `--size`, output is deterministic — diffs are
plain text and show exactly what changed (axis labels, legend entries,
series shapes).

- [ ] `tests/conftest.py` — add `golden_dir` fixture pointing to
      `tests/golden/`, add `--update-golden` pytest CLI flag
      (via `conftest.py` hook) that regenerates golden files in-place
- [ ] `tests/test_golden.py` — parameterized test: each case is a tuple
      of (name, cli_args). Invokes CLI with `mock_sacct` + `--size 80,24`,
      captures stdout, compares against `tests/golden/{name}.txt`.
      When `--update-golden` is set, writes stdout to the golden file
      instead of comparing
- [ ] Initial golden scenarios (allocation mode):
      `ungrouped_cpu`, `ungrouped_gpu`, `by_account`, `by_user_top3`,
      `bucketed_1d_sum`, `bucketed_1h_mean`, `cumulative`, `stacked`,
      `all_by_account`
- [ ] Initial golden scenarios (wait mode):
      `wait_scatter`, `wait_bucketed_1h`, `wait_by_user`,
      `wait_bucketed_mean`
- [ ] Generate and commit golden files: `tests/golden/*.txt`
- [ ] Commit: "WIP: golden snapshot test infrastructure and initial snapshots"

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
- README.md for a high-level overview
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
