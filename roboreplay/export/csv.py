"""CSV export for RoboReplay recordings.

Exports recording data to CSV files:
  - {name}_channels.csv — all channel data, one row per step
  - {name}_events.csv — event log
  - {name}_metadata.csv — recording metadata
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from roboreplay.replay import Replay


def export_csv(
    path: str | Path,
    output_dir: str | Path | None = None,
    channels: list[str] | None = None,
    include_metadata: bool = True,
    include_events: bool = True,
) -> list[Path]:
    """Export a .rrp recording to CSV files.

    Args:
        path: Path to the .rrp file.
        output_dir: Directory for output files. Defaults to same directory as input.
        channels: Specific channels to export. None means all.
        include_metadata: Whether to write a metadata CSV.
        include_events: Whether to write an events CSV.

    Returns:
        List of paths to created CSV files.
    """
    replay = Replay(path)
    path = Path(path)

    if output_dir is None:
        out = path.parent
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    name = replay.name
    channel_names = channels if channels else replay.channels
    created: list[Path] = []

    # --- Channels CSV ---
    channels_path = out / f"{name}_channels.csv"
    _write_channels_csv(replay, channels_path, channel_names)
    created.append(channels_path)

    # --- Events CSV ---
    if include_events and len(replay.events) > 0:
        events_path = out / f"{name}_events.csv"
        _write_events_csv(replay, events_path)
        created.append(events_path)

    # --- Metadata CSV ---
    if include_metadata:
        meta_path = out / f"{name}_metadata.csv"
        _write_metadata_csv(replay, meta_path)
        created.append(meta_path)

    replay.close()
    return created


def _build_column_headers(replay: Replay, channel_names: list[str]) -> list[str]:
    """Build flattened column headers for multi-dimensional channels."""
    headers = ["step"]
    for ch_name in channel_names:
        schema = replay.schema.channels.get(ch_name)
        if schema is None:
            continue
        shape = schema.shape
        n_elements = int(np.prod(shape))
        if n_elements == 1:
            headers.append(ch_name)
        else:
            for i in range(n_elements):
                headers.append(f"{ch_name}_{i}")
    return headers


def _write_channels_csv(
    replay: Replay, path: Path, channel_names: list[str]
) -> None:
    """Write channel data to a CSV file."""
    headers = _build_column_headers(replay, channel_names)

    # Load channel data
    channel_data: dict[str, np.ndarray] = {}
    for ch_name in channel_names:
        if ch_name in replay.channels:
            channel_data[ch_name] = replay.channel(ch_name)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for step in range(replay.num_steps):
            row: list[str] = [str(step)]
            for ch_name in channel_names:
                if ch_name not in channel_data:
                    continue
                values = channel_data[ch_name][step].flatten()
                row.extend(f"{v:.6g}" for v in values)
            writer.writerow(row)


def _write_events_csv(replay: Replay, path: Path) -> None:
    """Write events to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "event_type", "wall_time", "data"])
        for event in replay.events.events:
            data_str = ", ".join(f"{k}={v}" for k, v in event.data.items()) if event.data else ""
            writer.writerow([event.step, event.event_type, event.wall_time, data_str])


def _write_metadata_csv(replay: Replay, path: Path) -> None:
    """Write metadata to a CSV file."""
    meta = replay.metadata
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["name", meta.name])
        writer.writerow(["robot", meta.robot])
        writer.writerow(["task", meta.task])
        writer.writerow(["created_at", meta.created_at])
        writer.writerow(["roboreplay_version", meta.roboreplay_version])
        writer.writerow(["python_version", meta.system_info.python_version])
        writer.writerow(["platform", meta.system_info.platform])
        writer.writerow(["hostname", meta.system_info.hostname])
        for k, v in meta.user_metadata.items():
            writer.writerow([f"user.{k}", v])
