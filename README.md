# sacct-plot

Visualize instantaneous allocated resources (CPUs or GPUs) on Slurm clusters over time.

Queries `sacct` for job records, uses an event-based sweep algorithm to compute a
1-second-accurate allocation step function, and renders overlaid time-series in the
terminal via [tplot](https://github.com/JeroenDelwortel/tplot).

Designed for HPC administrators and leadership to visualize fair-share usage patterns
across accounts, users, and QOS levels.

## Install

```
uv tool install .
```

## Usage

```
sacct-plot [-h] [-v] [-u USER] [-A ACCOUNT] [-r PARTITION] [-q QOS]
           [-s STATE] [-S STARTTIME] [-E ENDTIME]
           [--by {account,user,qos}] [--gpu] [--bucket INTERVAL]
           [--sum | --mean | --max | --min] [--top N] [--stacked] [--data]
```

### Examples

```
# Show CPU allocation over the past 7 days
sacct-plot -S 2026-02-11

# Overlay by account, show top 5
sacct-plot -S 2026-02-11 --by account --top 5

# GPU allocation, bucketed by hour
sacct-plot -S 2026-02-11 --gpu --bucket 1h

# Dump data instead of plotting
sacct-plot -S 2026-02-11 --by user --data
```

### Visuals

<img width="1125" height="513" alt="Screenshot 2026-03-04 at 3 48 35 PM" src="https://github.com/user-attachments/assets/7643e8b8-1250-4725-a493-2fcc4acac02c" />

## License

MIT — see [LICENSE](LICENSE).
