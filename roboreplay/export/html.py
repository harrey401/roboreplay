"""HTML export for RoboReplay recordings.

Generates a self-contained HTML file with interactive Chart.js charts.
No new Python dependencies required — Chart.js loaded from CDN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roboreplay.replay import Replay


def _lttb_downsample(data: np.ndarray, threshold: int) -> tuple[list[int], list[float]]:
    """Largest-Triangle-Three-Buckets downsampling for large time series.

    Returns (indices, values) of the downsampled data.
    """
    if len(data) <= threshold:
        return list(range(len(data))), data.tolist()

    # Always keep first and last point
    sampled_indices = [0]
    sampled_values = [float(data[0])]

    bucket_size = (len(data) - 2) / (threshold - 2)

    a_index = 0
    a_value = float(data[0])

    for i in range(1, threshold - 1):
        # Calculate bucket range
        bucket_start = int((i - 1) * bucket_size) + 1
        bucket_end = int(i * bucket_size) + 1
        bucket_end = min(bucket_end, len(data) - 1)

        # Calculate next bucket average
        next_start = int(i * bucket_size) + 1
        next_end = int((i + 1) * bucket_size) + 1
        next_end = min(next_end, len(data))
        avg_x = (next_start + next_end - 1) / 2
        avg_y = float(np.mean(data[next_start:next_end]))

        # Find point in current bucket with max triangle area
        max_area = -1.0
        max_index = bucket_start
        for j in range(bucket_start, bucket_end):
            area = abs(
                (a_index - avg_x) * (float(data[j]) - a_value)
                - (a_index - j) * (avg_y - a_value)
            )
            if area > max_area:
                max_area = area
                max_index = j

        sampled_indices.append(max_index)
        sampled_values.append(float(data[max_index]))
        a_index = max_index
        a_value = float(data[max_index])

    sampled_indices.append(len(data) - 1)
    sampled_values.append(float(data[-1]))

    return sampled_indices, sampled_values


def export_html(
    path: str | Path,
    output: str | Path | None = None,
    channels: list[str] | None = None,
    max_points: int = 2000,
) -> Path:
    """Export a .rrp recording to a self-contained HTML file.

    Args:
        path: Path to the .rrp file.
        output: Output HTML file path. Defaults to {name}.html next to input.
        channels: Specific channels to include. None means all.
        max_points: Max data points per chart (LTTB downsampling above this).

    Returns:
        Path to the created HTML file.
    """
    replay = Replay(path)
    path = Path(path)

    if output is None:
        out_path = path.with_suffix(".html")
    else:
        out_path = Path(output)

    channel_names = channels if channels else replay.channels

    # Prepare chart data
    chart_data: dict[str, dict] = {}
    for ch_name in channel_names:
        if ch_name not in replay.channels:
            continue
        data = replay.channel(ch_name)
        if data.ndim > 1:
            data = np.linalg.norm(data, axis=-1)
        else:
            data = data.flatten()

        indices, values = _lttb_downsample(data, max_points)
        chart_data[ch_name] = {"labels": indices, "values": values}

    # Prepare events
    events_data = []
    for event in replay.events.events:
        events_data.append({
            "step": event.step,
            "type": event.event_type,
            "data": event.data,
        })

    # Prepare stats
    stats_data = {}
    for name, stat in replay.stats.items():
        if name in channel_names:
            stats_data[name] = {
                "min": round(stat.min, 4),
                "max": round(stat.max, 4),
                "mean": round(stat.mean, 4),
                "std": round(stat.std, 4),
            }

    # Chart colors
    colors = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
        "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
        "#9c755f", "#bab0ac",
    ]

    html = _generate_html(
        name=replay.name,
        robot=replay.robot,
        task=replay.task,
        num_steps=replay.num_steps,
        channel_names=[c for c in channel_names if c in chart_data],
        chart_data=chart_data,
        events=events_data,
        stats=stats_data,
        colors=colors,
        metadata=replay.metadata,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    replay.close()
    return out_path


def _generate_html(
    name: str,
    robot: str,
    task: str,
    num_steps: int,
    channel_names: list[str],
    chart_data: dict,
    events: list[dict],
    stats: dict,
    colors: list[str],
    metadata: object,
) -> str:
    """Generate the complete HTML string."""
    chart_data_json = json.dumps(chart_data)
    events_json = json.dumps(events)
    colors_json = json.dumps(colors)

    # Build chart canvases
    chart_canvases = ""
    for ch_name in channel_names:
        chart_canvases += (
            f'    <div class="chart-container">'
            f'<canvas id="chart-{ch_name}"></canvas></div>\n'
        )

    # Build stats table rows
    stats_rows = ""
    for ch_name in channel_names:
        if ch_name in stats:
            s = stats[ch_name]
            stats_rows += (
                f"      <tr><td>{ch_name}</td>"
                f"<td>{s['min']}</td><td>{s['max']}</td>"
                f"<td>{s['mean']}</td><td>{s['std']}</td></tr>\n"
            )

    # Build events table rows
    events_rows = ""
    for e in events:
        data_str = ", ".join(f"{k}={v}" for k, v in e["data"].items()) if e["data"] else ""
        events_rows += (
            f"      <tr><td>{e['step']}</td>"
            f"<td>{e['type']}</td><td>{data_str}</td></tr>\n"
        )

    channel_names_json = json.dumps(channel_names)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RoboReplay — {name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f5f5; color: #333; padding: 20px; }}
        .header {{ background: #1a1a2e; color: white; padding: 24px; border-radius: 8px;
                   margin-bottom: 20px; }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header .meta {{ color: #aaa; font-size: 14px; }}
        .header .meta span {{ margin-right: 16px; }}
        .section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .section h2 {{ font-size: 18px; margin-bottom: 12px; color: #1a1a2e; }}
        .chart-container {{ position: relative; height: 250px; margin-bottom: 16px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{name}</h1>
        <div class="meta">
            <span>Robot: {robot or "N/A"}</span>
            <span>Task: {task or "N/A"}</span>
            <span>Steps: {num_steps}</span>
            <span>Channels: {len(channel_names)}</span>
        </div>
    </div>

    <div class="section">
        <h2>Channel Plots</h2>
{chart_canvases}
    </div>

    <div class="section">
        <h2>Statistics</h2>
        <table>
            <thead><tr><th>Channel</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th></tr></thead>
            <tbody>
{stats_rows}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Events ({len(events)})</h2>
        <table>
            <thead><tr><th>Step</th><th>Type</th><th>Data</th></tr></thead>
            <tbody>
{events_rows}
            </tbody>
        </table>
    </div>

    <div class="footer">Generated by RoboReplay</div>

    <script>
        const chartData = {chart_data_json};
        const events = {events_json};
        const channelNames = {channel_names_json};
        const colors = {colors_json};

        channelNames.forEach((name, idx) => {{
            const ctx = document.getElementById('chart-' + name);
            if (!ctx || !chartData[name]) return;
            const color = colors[idx % colors.length];
            const annotations = {{}};
            events.forEach((e, i) => {{
                annotations['event' + i] = {{
                    type: 'line',
                    xMin: e.step,
                    xMax: e.step,
                    borderColor: 'rgba(255, 0, 0, 0.3)',
                    borderWidth: 1,
                    borderDash: [4, 4],
                }};
            }});
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: chartData[name].labels,
                    datasets: [{{
                        label: name,
                        data: chartData[name].values,
                        borderColor: color,
                        backgroundColor: color + '20',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: true,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: true, position: 'top' }},
                    }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Step' }} }},
                        y: {{ title: {{ display: true, text: name }} }},
                    }},
                    interaction: {{ intersect: false, mode: 'index' }},
                }}
            }});
        }});
    </script>
</body>
</html>"""
