"""Replay — the main interface for reading and navigating recordings.

Usage:
    from roboreplay import Replay

    r = Replay("experiment_042.rrp")

    print(r)                     # Summary
    print(r.channels)            # ['state', 'action', 'reward', 'gripper_force']
    print(r.num_steps)           # 620
    print(r[487])                # Dict of all channels at step 487
    print(r[450:520])            # Dict of arrays over range
    print(r.events)              # Event list

    r.plot("gripper_force")      # Matplotlib plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, overload

import numpy as np

from roboreplay.storage.reader import Reader
from roboreplay.utils.schema import ChannelStats, EventLog, RecordingMetadata, RecordingSchema


class Replay:
    """Read and navigate a .rrp recording.

    Provides indexing, slicing, channel access, event queries,
    and basic visualization.

    Args:
        path: Path to a .rrp file.
    """

    def __init__(self, path: str | Path) -> None:
        self._reader = Reader(path)
        self._reader.open()

    def close(self) -> None:
        """Close the underlying file."""
        self._reader.close()

    def __enter__(self) -> Replay:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # --- Properties ---

    @property
    def metadata(self) -> RecordingMetadata:
        """Recording metadata (name, robot, task, system info, etc.)."""
        return self._reader.metadata

    @property
    def schema(self) -> RecordingSchema:
        """Schema describing all recorded channels."""
        return self._reader.schema

    @property
    def events(self) -> EventLog:
        """All events marked during recording."""
        return self._reader.events

    @property
    def stats(self) -> dict[str, ChannelStats]:
        """Pre-computed statistics per channel."""
        return self._reader.stats

    @property
    def channels(self) -> list[str]:
        """List of recorded channel names."""
        return self._reader.channel_names

    @property
    def num_steps(self) -> int:
        """Total number of recorded steps."""
        return self._reader.num_steps

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def robot(self) -> str:
        return self.metadata.robot

    @property
    def task(self) -> str:
        return self.metadata.task

    # --- Data Access ---

    def channel(self, name: str, start: int = 0, end: int | None = None) -> np.ndarray:
        """Get data for a single channel.

        Args:
            name: Channel name.
            start: Start step (inclusive). Default 0.
            end: End step (exclusive). Default None (all steps).

        Returns:
            numpy array of shape [steps, *channel_shape]
        """
        return self._reader.get_channel(name, start, end)

    @overload
    def __getitem__(self, key: int) -> dict[str, np.ndarray]: ...

    @overload
    def __getitem__(self, key: slice) -> dict[str, np.ndarray]: ...

    def __getitem__(self, key: int | slice) -> dict[str, np.ndarray]:
        """Index or slice the recording.

        replay[step] → dict of all channel values at that step
        replay[start:end] → dict of all channel arrays over range
        """
        if isinstance(key, int):
            if key < 0:
                key = self.num_steps + key
            return self._reader.get_step(key)
        elif isinstance(key, slice):
            start, end, _ = key.indices(self.num_steps)
            return self._reader.get_slice(start, end)
        else:
            raise TypeError(f"Invalid index type: {type(key)}. Use int or slice.")

    # --- Visualization ---

    def plot(self, channel_name: str, start: int = 0, end: int | None = None) -> Any:
        """Plot a channel over time using matplotlib.

        Args:
            channel_name: Name of the channel to plot.
            start: Start step.
            end: End step.

        Returns:
            matplotlib Figure object.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install roboreplay[viz]"
            )

        data = self.channel(channel_name, start, end)
        fig, ax = plt.subplots(figsize=(12, 4))

        steps = np.arange(start, start + len(data))

        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            # Scalar channel
            values = data.flatten()
            ax.plot(steps, values, linewidth=0.8)
        elif data.ndim == 2:
            # Multi-dimensional channel — plot each dimension
            for dim in range(data.shape[1]):
                ax.plot(steps, data[:, dim], linewidth=0.8, label=f"dim_{dim}")
            ax.legend(fontsize=8)
        else:
            raise ValueError(f"Cannot plot channel with {data.ndim} dimensions")

        # Mark events on the plot
        for event in self.events.events:
            if start <= event.step < (end or self.num_steps):
                ax.axvline(x=event.step, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.text(
                    event.step, ax.get_ylim()[1] * 0.95,
                    event.event_type, fontsize=7, color="red",
                    rotation=45, ha="left", va="top",
                )

        ax.set_xlabel("Step")
        ax.set_ylabel(channel_name)
        ax.set_title(f"{self.name} — {channel_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig

    # --- Display ---

    def summary(self) -> str:
        """Generate a human-readable summary string."""
        lines = []
        lines.append(f"Recording: {self.name}")
        if self.robot:
            lines.append(f"Robot: {self.robot}")
        if self.task:
            lines.append(f"Task: {self.task}")
        lines.append(f"Steps: {self.num_steps}")
        lines.append(f"Channels: {', '.join(self.channels)}")

        if self.stats:
            lines.append("Channel Stats:")
            for name, stat in self.stats.items():
                lines.append(
                    f"  {name}: min={stat.min:.4f}, max={stat.max:.4f}, "
                    f"mean={stat.mean:.4f}, std={stat.std:.4f}"
                )

        n_events = len(self.events)
        if n_events > 0:
            lines.append(f"Events: {n_events}")
            for event in self.events.events[:5]:  # Show first 5
                lines.append(f"  [{event.step}] {event.event_type}: {event.data}")
            if n_events > 5:
                lines.append(f"  ... and {n_events - 5} more")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Replay(name='{self.name}', robot='{self.robot}', "
            f"steps={self.num_steps}, channels={self.channels})"
        )

    def __str__(self) -> str:
        return self.summary()

    def __len__(self) -> int:
        return self.num_steps
