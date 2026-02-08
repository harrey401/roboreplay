"""Random-access HDF5 reader for .rrp files.

Supports indexing, slicing, and lazy loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from roboreplay.storage.format import (
    CHANNELS_GROUP,
    EVENTS_ATTR,
    METADATA_ATTR,
    SCHEMA_ATTR,
    STATS_GROUP,
)
from roboreplay.utils.schema import (
    ChannelStats,
    EventLog,
    RecordingMetadata,
    RecordingSchema,
)


class Reader:
    """Random-access reader for .rrp recording files.

    Lazy-loads data — only reads from disk when channels are accessed.
    Caches metadata and schema on open.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Recording not found: {self.path}")

        self._file: h5py.File | None = None
        self._metadata: RecordingMetadata | None = None
        self._schema: RecordingSchema | None = None
        self._events: EventLog | None = None
        self._stats: dict[str, ChannelStats] | None = None

    def open(self) -> None:
        """Open the file for reading."""
        self._file = h5py.File(str(self.path), "r")

        # Cache metadata
        self._metadata = RecordingMetadata.from_json(self._file.attrs[METADATA_ATTR])
        self._schema = RecordingSchema.from_json(self._file.attrs[SCHEMA_ATTR])
        self._events = EventLog.from_json(self._file.attrs[EVENTS_ATTR])

        # Cache stats
        self._stats = {}
        if STATS_GROUP in self._file:
            stats_group = self._file[STATS_GROUP]
            for name in stats_group.attrs:
                self._stats[name] = ChannelStats.model_validate_json(stats_group.attrs[name])

    def close(self) -> None:
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Reader:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def metadata(self) -> RecordingMetadata:
        if self._metadata is None:
            raise RuntimeError("Reader not opened. Call .open() first.")
        return self._metadata

    @property
    def schema(self) -> RecordingSchema:
        if self._schema is None:
            raise RuntimeError("Reader not opened. Call .open() first.")
        return self._schema

    @property
    def events(self) -> EventLog:
        if self._events is None:
            raise RuntimeError("Reader not opened. Call .open() first.")
        return self._events

    @property
    def stats(self) -> dict[str, ChannelStats]:
        if self._stats is None:
            raise RuntimeError("Reader not opened. Call .open() first.")
        return self._stats

    @property
    def channel_names(self) -> list[str]:
        """List of all channel names in the recording."""
        return list(self.schema.channels.keys())

    @property
    def num_steps(self) -> int:
        """Total number of recorded steps."""
        if self._file is None:
            raise RuntimeError("Reader not opened.")
        # Get length from first channel
        channels_group = self._file[CHANNELS_GROUP]
        for name in channels_group:
            return channels_group[name].shape[0]
        return 0

    def get_channel(self, name: str, start: int = 0, end: int | None = None) -> np.ndarray:
        """Read a channel's data, optionally sliced.

        Args:
            name: Channel name.
            start: Start step (inclusive).
            end: End step (exclusive). None means all remaining steps.

        Returns:
            numpy array of shape [steps, *channel_shape]
        """
        if self._file is None:
            raise RuntimeError("Reader not opened.")
        if name not in self._file[CHANNELS_GROUP]:
            raise KeyError(
                f"Channel '{name}' not found. "
                f"Available: {list(self._file[CHANNELS_GROUP].keys())}"
            )

        ds = self._file[CHANNELS_GROUP][name]
        if end is None:
            end = ds.shape[0]
        return ds[start:end]

    def get_step(self, step: int) -> dict[str, np.ndarray]:
        """Get all channel data at a single step.

        Args:
            step: Step index.

        Returns:
            dict mapping channel name to value at that step.
        """
        if self._file is None:
            raise RuntimeError("Reader not opened.")

        result = {}
        channels_group = self._file[CHANNELS_GROUP]
        for name in channels_group:
            result[name] = channels_group[name][step]
        return result

    def get_slice(self, start: int, end: int) -> dict[str, np.ndarray]:
        """Get all channel data over a range of steps.

        Args:
            start: Start step (inclusive).
            end: End step (exclusive).

        Returns:
            dict mapping channel name to array of shape [end-start, *channel_shape]
        """
        if self._file is None:
            raise RuntimeError("Reader not opened.")

        result = {}
        channels_group = self._file[CHANNELS_GROUP]
        for name in channels_group:
            result[name] = channels_group[name][start:end]
        return result
