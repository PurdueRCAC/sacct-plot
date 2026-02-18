# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

sacct-plot is a command-line tool that visualizes instantaneous allocated resources
(CPUs or GPUs) on Slurm clusters over time. It queries `sacct` for job records, uses
an event-based sweep algorithm to compute a 1-second-accurate allocation step function,
and renders overlaid time-series in the terminal via `tplot`.

Designed for HPC administrators and leadership to visualize fair-share usage patterns
across accounts, users, and QOS levels.

## Development Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run the CLI
uv run sacct-plot --help
uv run sacct-plot -S 2026-02-11
uv run sacct-plot -S 2026-02-11 --by account --top 5
uv run sacct-plot -S 2026-02-11 --gpu --bucket 1h

# Install as a tool
uv tool install .

# Run tests
uv run pytest
```

Note: `plot-cli` is installed from GitHub (see `[tool.uv.sources]` in pyproject.toml).

## Architecture

### CLI Structure (cmdkit-based)
- `SacctPlotApp` (Application) → single command with sacct filter flags, analysis flags, and output flags
- Sacct filters (`-u`, `-A`, `-r`, `-q`, `-s`, `-S`, `-E`) map directly to `sacct` CLI options
- Analysis flags (`--by`, `--gpu`, `--bucket`, `--top`) control grouping and aggregation
- Output flags (`--stacked`, `--data`) control rendering mode

### Data Pipeline
1. **Acquisition** (`sacct.py`): Runs `sacct` subprocess, parses pipe-delimited output into `JobInfo` dataclass records, caches as parquet in `~/.cache/sacct/`.
2. **Sweep** (`sweep.py`): Event-based algorithm — each job emits `(start, +N)` and `(end, -N)` events. Sorting + cumulative sum yields exact allocation step function. Optional `--bucket` resamples, `--top` filters to top N groups.
3. **Rendering** (`plot.py`): Converts DatetimeIndex to epoch seconds, uses plot-cli's `generate_time_ticks` for smart axis labels, renders via `TimeSeriesFigure`.

### Key Patterns
- All CLI apps use cmdkit's `Interface` for argument parsing
- Configuration merges env vars (prefix `SACCT_PLOT`) with defaults via `cmdkit.config`
- Parquet caching with 10-minute TTL avoids repeated `sacct` calls
- Sweep is fully vectorized in pandas — no per-second expansion

## File Layout

```
src/sacct_plot/
├── __init__.py      # SacctPlotApp, main(), CLI definition
├── __main__.py      # python -m support
├── sacct.py         # JobInfo, SacctData, sacct subprocess + parquet caching
├── sweep.py         # compute_allocation, apply_bucket, apply_top_n
└── plot.py          # render() using plot-cli TimeSeriesFigure
```
