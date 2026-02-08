# Comparing Recordings

Compare two recordings side-by-side to find where they diverge.

## Basic Usage

```python
from roboreplay import compare

diff = compare("run_success.rrp", "run_failure.rrp")

print(diff.summary())           # Formatted comparison table
print(diff.divergence_step)     # Step where they start differing
print(diff.channel_diffs)       # Per-channel statistics
```

## CompareResult

The result contains:

- **`name_a`/`name_b`** — Recording names
- **`steps_a`/`steps_b`** — Step counts
- **`shared_channels`** — Channels present in both recordings
- **`channel_diffs`** — Per-channel diff stats (mean, std, max abs diff)
- **`divergence_step`** — First step where recordings diverge significantly

## Per-Channel Diffs

```python
for name, diff in result.channel_diffs.items():
    print(f"{name}: mean_a={diff.mean_a:.4f}, mean_b={diff.mean_b:.4f}")
    print(f"  Change: {diff.mean_change_pct:.1f}%")
    print(f"  Max abs diff: {diff.max_abs_diff:.4f}")
    print(f"  Divergence: step {diff.divergence_step}")
```

## CLI

```bash
roboreplay compare run_success.rrp run_failure.rrp
```
