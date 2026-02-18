---
status: in-progress
current_phase: 2
last_updated: 2026-02-18
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

- [ ] `src/sacct_plot/sacct.py` (JobInfo dataclass, SacctData with from_sacct/from_local/to_local)
- [ ] Wire `SacctData.from_sacct()` into `SacctPlotApp.run()` with `--data` dump mode
- [ ] Unit tests: `tests/test_sacct.py` (JobInfo.from_line parsing, GPU extraction from AllocTRES)
- [ ] Commit: "WIP: sacct data acquisition and caching"

## Phase 3: Event Sweep

Core algorithm — transform job records into an instantaneous allocation time-series
using the event-based sweep approach.

- [ ] `src/sacct_plot/sweep.py` (compute_allocation, apply_bucket, apply_top_n)
- [ ] Wire sweep into `SacctPlotApp.run()` (--by, --gpu, --bucket, --top, aggregation flags)
- [ ] Unit tests: `tests/test_sweep.py` (known jobs → expected step function, groupby, bucket, top-N)
- [ ] Commit: "WIP: event sweep algorithm"

## Phase 4: Rendering

Terminal plotting with tplot via plot-cli's TimeSeriesFigure. Overlaid and stacked
line modes with smart time-axis labels.

- [ ] `src/sacct_plot/plot.py` (render function using TimeSeriesFigure + generate_time_ticks)
- [ ] Wire render into `SacctPlotApp.run()` (--stacked flag)
- [ ] Integration tests: `tests/test_cli.py` (help, version, --data mode)
- [ ] Commit: "WIP: terminal rendering with tplot"

## Phase 5: Polish & Release Prep

End-to-end testing on a live cluster, README expansion, and preparing for merge
from wip to main.

- [ ] Manual testing with real sacct data on cluster
- [ ] README: full usage examples, screenshots
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

Check the ROADMAP.md YAML frontmatter for the current phase. Implement the next
unchecked item(s) in the current phase, then:
1. Update ROADMAP.md to check off completed items
2. Update the frontmatter (current_phase, last_updated)
3. Commit with "WIP: <description>"

When a phase is complete, check in before proceeding to the next phase.
```
