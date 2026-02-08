# Batch Analysis Example

Generates 5 recordings with varying conditions, runs batch diagnosis, and exports to CSV.

## Run

```bash
python examples/batch_analysis.py
```

No external dependencies required.

## What It Does

1. **Generates 5 recordings** with different parameters:
    - `run_clean_1` — Low noise, no failure
    - `run_clean_2` — Medium noise, no failure
    - `run_noisy` — High noise, no failure
    - `run_fail_early` — Failure at step 80
    - `run_fail_late` — Failure at step 160

2. **Batch diagnosis** — Runs `diagnose()` on all recordings, prints summary table

3. **CSV export** — Exports all recordings to CSV for further analysis

## Output Structure

```
batch_output/
├── run_clean_1.rrp
├── run_clean_2.rrp
├── run_noisy.rrp
├── run_fail_early.rrp
├── run_fail_late.rrp
└── csv/
    ├── run_clean_1_channels.csv
    ├── run_clean_1_metadata.csv
    ├── ...
```

## Use Case

This pattern is useful for:

- Comparing policy versions across multiple seeds
- Automated test pipelines that check for regressions
- Data collection campaigns with quality validation
