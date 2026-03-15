# sacct-plot

Visualize resource allocation and job wait times on Slurm clusters, directly
in the terminal.

Queries `sacct` for job records and renders time-series plots via
[tplot](https://github.com/JeroenDelwortel/tplot). Designed for HPC
administrators and leadership to understand fair-share usage patterns and
scheduler contention across accounts, users, and QOS levels.

## Install

```
uv tool install git+https://github.com/PurdueRCAC/sacct-plot
```

## Quick Start

```bash
# CPU allocation over the past week
sacct-plot -S 2026-03-07

# GPU allocation on a specific partition, grouped by user
sacct-plot -S 2026-03-07 -r gpu --gpu --by user --top 5

# Wait time analysis — how long are jobs waiting?
sacct-plot -S 2026-03-07 -r gpu --wait --bucket 1h
```

## Concepts

sacct-plot has two analysis modes that answer different questions about your cluster.

### Allocation Mode (default)

*How many resources are in use right now, and by whom?*

Computes the **instantaneous allocation** — at every point in time, exactly how
many CPUs or GPUs are allocated. Each job contributes +N resources at its start
time and −N at its end time; sorting these events and computing the cumulative
sum yields a step function accurate to the second.

```bash
# Total CPU allocation across the cluster
sacct-plot -S 2026-03-01

# GPU allocation on the gpu partition
sacct-plot -S 2026-03-01 -r gpu --gpu
```

### Wait Time Mode (`--wait`)

*How long are jobs waiting in the queue before they start?*

Computes the **wait time** (start − submit) for each job and plots it against
the job's start time. Without `--bucket`, this is a scatter plot where each dot
is a single job. With `--bucket`, the scatter is overlaid with summary lines
(median and p25/p75 percentile envelope) to show trends and spread.

```bash
# Raw scatter of wait times on the gpu partition
sacct-plot -S 2026-03-01 -r gpu --wait

# Hourly summary with percentile envelope
sacct-plot -S 2026-03-01 -r gpu --wait --bucket 1h
```

## Grouping, Filtering, and Aggregation

These options compose freely with both analysis modes.

### Sacct Filters

Filter jobs before analysis using the same flags as `sacct`:

| Flag | Meaning |
|------|------|
| `-u USER` | Filter by user (comma-separated) |
| `-A ACCOUNT` | Filter by account |
| `-r PARTITION` | Filter by partition |
| `-q QOS` | Filter by quality of service |
| `-s STATE` | Filter by job state |
| `-S STARTTIME` | Jobs starting after this time |
| `-E ENDTIME` | Jobs ending before this time |

### Grouping (`--by`)

Break the analysis down by `account`, `user`, or `qos`. Each group becomes a
separate line (or scatter series).

```bash
# GPU allocation broken down by account
sacct-plot -S 2026-03-01 -r gpu --gpu --by account

# Wait time by user
sacct-plot -S 2026-03-01 -r gpu --wait --by user
```

### `--top N`

When grouping, show only the top N groups by total allocation (or total wait).
Remaining groups are collapsed into an "other" line.

```bash
# Top 5 accounts on the gpu partition, everything else as "other"
sacct-plot -S 2026-03-01 -r gpu --gpu --by account --top 5
```

### `--all`

Overlay the **full partition aggregate** as an "all" line alongside the named
groups. This is the inclusive total (not the complement) — it shows the big
picture that the individual group lines add up to.

Requires `--by`.

```bash
# Alice and Bob's GPU usage against the full partition total
sacct-plot -S 2026-03-01 -u alice,bob --by user --all -r gpu --gpu

# Same idea for wait time
sacct-plot -S 2026-03-01 -u alice,bob --by user --all -r gpu --wait --bucket 1h
```

### Bucketing (`--bucket INTERVAL`)

Resample into fixed time intervals (e.g., `1h`, `1d`, `15min`).

**In allocation mode**, this aggregates the step function per bucket:

| Aggregation | Meaning |
|---|---|
| `--sum` (default) | Resource-hours per bucket (e.g., GPU·h) |
| `--mean` | Time-weighted average allocation level |
| `--max` | Peak allocation within the bucket |
| `--min` | Minimum allocation within the bucket |
| `--cumulative` | Running total of the bucketed values |

**In wait mode**, bucketing computes summary statistics per interval:

| Aggregation | Meaning |
|---|---|
| `--median` (default) | Median wait time per bucket |
| `--mean` | Mean wait time per bucket |
| `--max` | Longest wait in the bucket |

The scatter dots are always shown; the summary lines overlay them.

```bash
# GPU-hours consumed per day by account
sacct-plot -S 2026-03-01 -r gpu --gpu --by account --bucket 1d

# Cumulative GPU-hours over time
sacct-plot -S 2026-03-01 -r gpu --gpu --bucket 1d --cumulative

# Hourly median wait time with percentile envelope
sacct-plot -S 2026-03-01 -r gpu --wait --bucket 1h

# Mean wait time instead of median
sacct-plot -S 2026-03-01 -r gpu --wait --bucket 1h --mean
```

## Putting It All Together

A few real-world scenarios that combine multiple options:

```bash
# "Is the gpu partition oversubscribed?"
# Show instantaneous GPU allocation vs. time for the past month
sacct-plot -S 2026-02-14 -r gpu --gpu

# "Who's using all the GPUs?"
# Break down by account, show top 5
sacct-plot -S 2026-02-14 -r gpu --gpu --by account --top 5

# "How does my group compare to the whole partition?"
# Overlay physics account against the full gpu partition total
sacct-plot -S 2026-02-14 -A physics --by account --all -r gpu --gpu

# "Are wait times getting worse?"
# Daily median wait time with spread, broken down by QOS
sacct-plot -S 2026-02-14 -r gpu --wait --bucket 1d --by qos

# "How long are my users waiting vs. everyone else?"
# Wait time for specific users against the cluster-wide trend
sacct-plot -S 2026-02-14 -u alice,bob --by user --all -r gpu --wait --bucket 1h

# "Show me the raw data"
# Dump the processed DataFrame for external analysis
sacct-plot -S 2026-02-14 -r gpu --gpu --by account --data
```

## Formatting Options

```
--stacked              Stacked area view instead of overlaid lines
-c, --color COLORS     Comma-separated color names (e.g. 'blue,red,green')
-l, --legend POS       Legend position (topleft, topright, bottomleft, bottomright)
    --size W,H         Plot width and height in characters
```

### Visuals

<img width="1125" height="513" alt="Screenshot 2026-03-04 at 3 48 35 PM" src="https://github.com/user-attachments/assets/7643e8b8-1250-4725-a493-2fcc4acac02c" />

## License

MIT — see [LICENSE](LICENSE).
