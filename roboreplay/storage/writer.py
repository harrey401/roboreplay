"""Streaming HDF5 writer for .rrp files.

Writes data incrementally using chunked, resizable HDF5 datasets.
Survives crashes by flushing after each chunk.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from roboreplay.storage.format import (
    CHANNELS_GROUP,
    CHUNK_STEPS,
    COMPRESSION,
    COMPRESSION_OPTS,
    EVENTS_ATTR,
    FORMAT_VERSION,
    INITIAL_STEPS,
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


class StreamingWriter:
    """Writes recording data incrementally to an HDF5 file.

    Thread-safe. Uses chunked datasets that grow dynamically.
    Flushes to disk periodically to survive crashes.
    """

    def __init__(self, path: str | Path, metadata: RecordingMetadata) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.schema = RecordingSchema()
        self.events = EventLog()

        self._file: h5py.File | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._step_count = 0
        self._lock = threading.Lock()
        self._initialized = False

        # Running stats accumulators
        self._stats_sum: dict[str, np.ndarray] = {}
        self._stats_sq_sum: dict[str, np.ndarray] = {}
        self._stats_min: dict[str, np.ndarray] = {}
        self._stats_max: dict[str, np.ndarray] = {}

    def open(self) -> None:
        """Open the HDF5 file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(str(self.path), "w")
        self._file.attrs["format_version"] = FORMAT_VERSION
        self._file.create_group(CHANNELS_GROUP)
        self._file.create_group(STATS_GROUP)

    def _init_channel(self, name: str, data: np.ndarray) -> None:
        """Initialize a dataset for a new channel on first write."""
        if self._file is None:
            raise RuntimeError("Writer not opened. Call .open() first.")

        dtype = data.dtype
        shape_per_step = data.shape

        # Register in schema
        self.schema.add_channel(name, str(dtype), shape_per_step)

        # Create resizable chunked dataset
        maxshape = (None, *shape_per_step)
        initial_shape = (INITIAL_STEPS, *shape_per_step)
        chunks = (min(CHUNK_STEPS, INITIAL_STEPS), *shape_per_step)

        ds = self._file[CHANNELS_GROUP].create_dataset(
            name,
            shape=initial_shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
        )
        self._datasets[name] = ds

        # Init stats accumulators
        flat = data.flatten().astype(np.float64)
        self._stats_sum[name] = flat.copy()
        self._stats_sq_sum[name] = (flat**2).copy()
        self._stats_min[name] = flat.copy()
        self._stats_max[name] = flat.copy()

    def write_step(self, channels: dict[str, np.ndarray]) -> None:
        """Write one step of data across all channels.

        Args:
            channels: dict mapping channel name to numpy array for this step.
                      On the first call, this defines the schema.
                      Subsequent calls must match the same channel names and shapes.
        """
        with self._lock:
            if not self._initialized:
                # First step — initialize all channels
                for name, data in channels.items():
                    arr = np.asarray(data)
                    self._init_channel(name, arr)
                self._initialized = True

            step = self._step_count

            for name, data in channels.items():
                arr = np.asarray(data)

                if name not in self._datasets:
                    # New channel added after first step — init it
                    self._init_channel(name, arr)
                    # Backfill with zeros
                    ds = self._datasets[name]
                    if step > 0:
                        ds.resize(max(step, INITIAL_STEPS), axis=0)

                ds = self._datasets[name]

                # Resize if needed
                if step >= ds.shape[0]:
                    new_size = max(ds.shape[0] * 2, step + CHUNK_STEPS)
                    ds.resize(new_size, axis=0)

                ds[step] = arr

                # Update running stats
                flat = arr.flatten().astype(np.float64)
                self._stats_sum[name] += flat
                self._stats_sq_sum[name] += flat**2
                np.minimum(self._stats_min[name], flat, out=self._stats_min[name])
                np.maximum(self._stats_max[name], flat, out=self._stats_max[name])

            self._step_count += 1

            # Periodic flush
            if self._step_count % CHUNK_STEPS == 0:
                self._flush()

    def write_event(self, step: int, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Record an event at the given step."""
        with self._lock:
            self.events.add(step, event_type, data)

    def _flush(self) -> None:
        """Flush data to disk."""
        if self._file is not None:
            self._file.flush()

    def _compute_stats(self) -> dict[str, ChannelStats]:
        """Compute final statistics for all channels."""
        stats = {}
        n = max(self._step_count, 1)
        for name in self._stats_sum:
            s = self._stats_sum[name]
            sq = self._stats_sq_sum[name]
            mean = float(np.mean(s / n))
            variance = float(np.mean(sq / n - (s / n) ** 2))
            stats[name] = ChannelStats(
                name=name,
                min=float(np.min(self._stats_min[name])),
                max=float(np.max(self._stats_max[name])),
                mean=mean,
                std=float(np.sqrt(max(variance, 0.0))),
                num_steps=self._step_count,
            )
        return stats

    def close(self) -> None:
        """Finalize and close the file.

        Truncates datasets to actual size, writes metadata/schema/events.
        """
        if self._file is None:
            return

        with self._lock:
            # Truncate datasets to actual step count
            for name, ds in self._datasets.items():
                ds.resize(self._step_count, axis=0)

            # Write metadata
            self._file.attrs[METADATA_ATTR] = self.metadata.to_json()
            self._file.attrs[SCHEMA_ATTR] = self.schema.to_json()
            self._file.attrs[EVENTS_ATTR] = self.events.to_json()

            # Write stats
            stats = self._compute_stats()
            stats_group = self._file[STATS_GROUP]
            for name, stat in stats.items():
                stats_group.attrs[name] = stat.model_dump_json()

            self._file.close()
            self._file = None

    @property
    def step_count(self) -> int:
        return self._step_count
