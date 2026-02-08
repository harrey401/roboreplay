"""Pydantic models for RoboReplay data structures."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


class SystemInfo(BaseModel):
    """Captured system information at recording time."""

    python_version: str = Field(default_factory=lambda: platform.python_version())
    platform: str = Field(default_factory=lambda: platform.platform())
    hostname: str = Field(default_factory=lambda: platform.node())
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RecordingMetadata(BaseModel):
    """Metadata for a recording session."""

    name: str
    robot: str = ""
    task: str = ""
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    system_info: SystemInfo = Field(default_factory=SystemInfo)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    roboreplay_version: str = "0.1.0"

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> RecordingMetadata:
        return cls.model_validate_json(data)


class ChannelSchema(BaseModel):
    """Schema for a single data channel."""

    name: str
    dtype: str  # numpy dtype string, e.g. "float32"
    shape: tuple[int, ...]  # shape per step, e.g. (7,) for 7-DOF

    @field_validator("shape", mode="before")
    @classmethod
    def coerce_shape(cls, v: Any) -> tuple[int, ...]:
        if isinstance(v, list):
            return tuple(v)
        return v


class RecordingSchema(BaseModel):
    """Schema describing all channels in a recording."""

    channels: dict[str, ChannelSchema] = Field(default_factory=dict)

    def add_channel(self, name: str, dtype: str, shape: tuple[int, ...]) -> None:
        self.channels[name] = ChannelSchema(name=name, dtype=dtype, shape=shape)

    def validate_step(self, channel_name: str, data: np.ndarray) -> None:
        """Validate that data matches the expected schema for this channel."""
        if channel_name not in self.channels:
            raise KeyError(f"Unknown channel '{channel_name}'. Known: {list(self.channels.keys())}")

        expected = self.channels[channel_name]
        if data.shape != expected.shape:
            raise ValueError(
                f"Channel '{channel_name}' shape mismatch: "
                f"expected {expected.shape}, got {data.shape}"
            )

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> RecordingSchema:
        return cls.model_validate_json(data)


class Event(BaseModel):
    """A marked event during recording."""

    step: int
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    wall_time: float = Field(default_factory=time.time)

    def to_json(self) -> str:
        return self.model_dump_json()


class EventLog(BaseModel):
    """Collection of events for a recording."""

    events: list[Event] = Field(default_factory=list)

    def add(self, step: int, event_type: str, data: dict[str, Any] | None = None) -> None:
        self.events.append(Event(step=step, event_type=event_type, data=data or {}))

    def where(self, event_type: str | None = None) -> list[Event]:
        """Filter events by type."""
        if event_type is None:
            return self.events
        return [e for e in self.events if e.event_type == event_type]

    def to_json(self) -> str:
        return json.dumps([e.model_dump() for e in self.events])

    @classmethod
    def from_json(cls, data: str) -> EventLog:
        items = json.loads(data)
        return cls(events=[Event.model_validate(item) for item in items])

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Event:
        return self.events[idx]


class ChannelStats(BaseModel):
    """Pre-computed statistics for a channel."""

    name: str
    min: float
    max: float
    mean: float
    std: float
    num_steps: int
