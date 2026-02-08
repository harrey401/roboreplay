"""Recorder — the main interface for recording robot execution data.

Usage:
    from roboreplay import Recorder

    rec = Recorder("experiment_042", metadata={"robot": "panda", "task": "pick_place"})
    rec.start()

    for step in range(1000):
        state = env.get_state()
        action = policy(state)
        obs, reward, done, info = env.step(action)

        rec.step(state=state, action=action, reward=reward, gripper_force=info["force"])

        if done:
            rec.mark_event("episode_end", {"success": reward > 0})
            break

    rec.save()

Or as a context manager:

    with Recorder("experiment_042") as rec:
        for step in range(1000):
            rec.step(state=state, action=action)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from roboreplay.storage.format import FILE_EXTENSION
from roboreplay.storage.writer import StreamingWriter
from roboreplay.utils.schema import RecordingMetadata


class Recorder:
    """Records robot execution data to a .rrp file.

    Accepts arbitrary numpy-compatible data channels. Schema is inferred
    from the first step() call. Subsequent calls must provide the same
    channels with matching shapes.

    Args:
        name: Name for this recording (used as filename if path not specified).
        path: Explicit output path. If None, saves to current directory as {name}.rrp.
        metadata: Dict of recording metadata (robot, task, etc.).
    """

    def __init__(
        self,
        name: str,
        path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        meta = metadata or {}

        if path is None:
            self._path = Path(f"{name}{FILE_EXTENSION}")
        else:
            self._path = Path(path)
            if not self._path.suffix:
                self._path = self._path.with_suffix(FILE_EXTENSION)

        self._metadata = RecordingMetadata(
            name=name,
            robot=meta.pop("robot", ""),
            task=meta.pop("task", ""),
            user_metadata=meta,
        )

        self._writer: StreamingWriter | None = None
        self._step_count = 0
        self._start_time: float | None = None
        self._started = False
        self._saved = False

    def start(self) -> None:
        """Start recording. Opens the output file."""
        if self._started:
            raise RuntimeError("Recording already started.")

        self._writer = StreamingWriter(self._path, self._metadata)
        self._writer.open()
        self._start_time = time.time()
        self._started = True

    def step(self, **channels: Any) -> None:
        """Record one step of data.

        Pass each data channel as a keyword argument. Values must be
        numpy arrays or array-like (lists, floats, etc.).

        On the first call, this defines the recording schema.
        Subsequent calls must provide the same channel names.

        Args:
            **channels: Keyword arguments mapping channel names to data.
                        e.g. step(state=np.array([...]), action=np.array([...]), reward=0.5)

        Example:
            rec.step(
                state=env.get_state(),
                action=policy_output,
                reward=reward,
                gripper_force=np.array([8.2]),
            )
        """
        if not self._started:
            self.start()

        if not channels:
            raise ValueError("step() requires at least one channel. e.g. rec.step(state=arr)")

        # Convert all values to numpy arrays
        np_channels: dict[str, np.ndarray] = {}
        for name, value in channels.items():
            arr = np.asarray(value, dtype=np.float32)
            # Scalars become 1D arrays
            if arr.ndim == 0:
                arr = arr.reshape(1)
            np_channels[name] = arr

        assert self._writer is not None
        self._writer.write_step(np_channels)
        self._step_count += 1

    def mark_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Mark an event at the current step.

        Events are timestamped markers that annotate the recording.
        Use them to flag milestones, failures, phase transitions, etc.

        Args:
            event_type: Type of event (e.g. "failure", "grasp_start", "episode_end").
            data: Optional dict of event-specific data.

        Example:
            rec.mark_event("grasp_slip", {"force_at_slip": 1.2, "expected_force": 8.0})
        """
        if not self._started:
            raise RuntimeError("Cannot mark events before recording starts. Call start() first.")

        assert self._writer is not None
        self._writer.write_event(self._step_count, event_type, data)

    def save(self) -> Path:
        """Finalize and save the recording.

        Returns:
            Path to the saved .rrp file.
        """
        if not self._started:
            raise RuntimeError("Nothing to save — recording was never started.")
        if self._saved:
            raise RuntimeError("Recording already saved.")

        assert self._writer is not None
        self._writer.close()
        self._saved = True
        return self._path

    @property
    def num_steps(self) -> int:
        """Number of steps recorded so far."""
        return self._step_count

    @property
    def duration(self) -> float:
        """Elapsed time since recording started, in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def path(self) -> Path:
        """Output file path."""
        return self._path

    # Context manager support
    def __enter__(self) -> Recorder:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._started and not self._saved:
            self.save()

    def __repr__(self) -> str:
        if self._started and not self._saved:
            status = "recording"
        elif self._saved:
            status = "saved"
        else:
            status = "idle"
        return f"Recorder(name='{self.name}', steps={self._step_count}, status={status})"
