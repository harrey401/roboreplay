# Exporting

RoboReplay supports exporting recordings to CSV and interactive HTML formats.

## CSV Export

```python
from roboreplay.export import export_csv

# Export all channels
files = export_csv("recording.rrp")

# Export specific channels to a directory
files = export_csv(
    "recording.rrp",
    output_dir="exports/",
    channels=["position", "force"],
    include_metadata=True,
    include_events=True,
)

for f in files:
    print(f"Created: {f}")
```

### CSV Output Files

- `{name}_channels.csv` — All channel data, one row per step. Multi-dimensional channels are flattened (e.g., `position_0`, `position_1`, `position_2`).
- `{name}_events.csv` — Event log with step, type, wall_time, and data.
- `{name}_metadata.csv` — Key-value pairs of recording metadata.

## HTML Export

Generates a self-contained HTML file with interactive Chart.js charts:

```python
from roboreplay.export import export_html

path = export_html("recording.rrp")
path = export_html("recording.rrp", output="my_report.html", max_points=5000)
```

The HTML file includes:

- Interactive time-series charts for each channel
- Metadata summary header
- Statistics table
- Event timeline table
- LTTB downsampling for large recordings

## CLI

```bash
# CSV export
roboreplay export recording.rrp --format csv
roboreplay export recording.rrp --format csv -o exports/

# HTML export
roboreplay export recording.rrp --format html
roboreplay export recording.rrp --format html -o report.html
```
