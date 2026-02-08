# CLI Reference

RoboReplay provides a command-line interface for quick analysis.

## Commands

### `roboreplay info`

Show recording summary.

```bash
roboreplay info recording.rrp
```

Displays: name, robot, task, steps, channels, statistics, and events.

### `roboreplay diagnose`

Run anomaly detection on a recording.

```bash
roboreplay diagnose recording.rrp
roboreplay diagnose recording.rrp --llm
roboreplay diagnose recording.rrp --llm --api-key sk-ant-...
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--drop-threshold` | 0.5 | Sensitivity for drop detection (0-1) |
| `--spike-threshold` | 3.0 | Standard deviations for spike detection |
| `--flatline-duration` | 20 | Minimum steps for flatline detection |
| `--llm` | false | Use LLM for enhanced diagnosis |
| `--api-key` | env var | Anthropic API key |

### `roboreplay compare`

Compare two recordings side-by-side.

```bash
roboreplay compare run_a.rrp run_b.rrp
```

### `roboreplay export`

Export recording to CSV or HTML.

```bash
roboreplay export recording.rrp --format csv
roboreplay export recording.rrp --format html
roboreplay export recording.rrp --format csv -o exports/
roboreplay export recording.rrp --format html -o report.html
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--format, -f` | csv | Export format: `csv` or `html` |
| `--output, -o` | auto | Output directory (csv) or file (html) |
| `--channel, -c` | all | Channels to export (repeatable) |

### `roboreplay plot`

Plot channels from a recording.

```bash
roboreplay plot recording.rrp
roboreplay plot recording.rrp -c force -c position
roboreplay plot recording.rrp -o plots.png
```

### `roboreplay --version`

```bash
roboreplay --version
```
