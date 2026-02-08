"""Statistical anomaly detection for robot execution data.

Detects common failure patterns using pure numpy.
No LLM required — works entirely offline.

Design principle: fewer, higher-confidence detections beat many noisy ones.
We'd rather miss a subtle anomaly than flood the user with false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Anomaly:
    """A detected anomaly in a recording channel."""

    channel: str
    step: int
    anomaly_type: str
    severity: float  # 0.0 to 1.0
    description: str
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Anomaly(step={self.step}, type='{self.anomaly_type}', channel='{self.channel}')"


def detect_sudden_drops(
    data: np.ndarray,
    channel_name: str,
    threshold_pct: float = 0.6,
    window: int = 10,
    min_baseline: float = 0.01,
) -> list[Anomaly]:
    """Detect sudden drops in a signal.

    A sudden drop is when the value decreases by more than threshold_pct
    of its recent average within a short window. Deduplicates nearby detections.
    """
    if data.ndim > 1:
        data = np.linalg.norm(data, axis=-1)

    anomalies = []
    last_step = -window * 2

    for i in range(window, len(data)):
        if i - last_step < window:
            continue

        recent_avg = np.mean(data[i - window : i])
        if abs(recent_avg) < min_baseline:
            continue

        drop = (recent_avg - data[i]) / abs(recent_avg)
        if drop > threshold_pct:
            severity = min(drop, 1.0)
            anomalies.append(
                Anomaly(
                    channel=channel_name,
                    step=i,
                    anomaly_type="sudden_drop",
                    severity=severity,
                    description=(
                        f"{channel_name} dropped {drop:.0%} at step {i} "
                        f"(from {recent_avg:.4f} → {data[i]:.4f})"
                    ),
                    details={
                        "recent_avg": float(recent_avg),
                        "value_at_drop": float(data[i]),
                        "drop_pct": float(drop),
                    },
                )
            )
            last_step = i
    return anomalies


def detect_sudden_spikes(
    data: np.ndarray,
    channel_name: str,
    threshold_std: float = 5.0,
    window: int = 30,
    min_std: float = 0.001,
) -> list[Anomaly]:
    """Detect sudden spikes (outliers) in a signal.

    A spike is when the value exceeds threshold_std standard deviations
    from the rolling mean. Deduplicates within half-window.
    """
    if data.ndim > 1:
        data = np.linalg.norm(data, axis=-1)

    anomalies = []
    last_step = -window

    for i in range(window, len(data)):
        if i - last_step < window // 2:
            continue

        segment = data[i - window : i]
        mean = np.mean(segment)
        std = np.std(segment)
        if std < min_std:
            continue

        z_score = abs(data[i] - mean) / std
        if z_score > threshold_std:
            severity = min(z_score / (threshold_std * 3), 1.0)
            anomalies.append(
                Anomaly(
                    channel=channel_name,
                    step=i,
                    anomaly_type="spike",
                    severity=severity,
                    description=(
                        f"{channel_name} spiked at step {i} "
                        f"(value={data[i]:.4f}, {z_score:.1f}σ from mean)"
                    ),
                    details={
                        "value": float(data[i]),
                        "rolling_mean": float(mean),
                        "rolling_std": float(std),
                        "z_score": float(z_score),
                    },
                )
            )
            last_step = i
    return anomalies


def detect_flatlines(
    data: np.ndarray,
    channel_name: str,
    min_duration: int = 50,
    tolerance: float = 1e-6,
) -> list[Anomaly]:
    """Detect periods where a signal goes completely flat unexpectedly.

    Only flags flatlines that occur AFTER the signal had variation — not
    constant-from-start signals which are usually just inactive channels.
    """
    if data.ndim > 1:
        data = np.linalg.norm(data, axis=-1)

    if np.std(data) < tolerance * 10:
        return []

    anomalies = []
    i = 0
    while i < len(data):
        j = i + 1
        while j < len(data) and abs(data[j] - data[i]) < tolerance:
            j += 1

        duration = j - i
        if duration >= min_duration and i > 5:
            pre_std = np.std(data[max(0, i - 20) : i])
            if pre_std > tolerance * 10:
                anomalies.append(
                    Anomaly(
                        channel=channel_name,
                        step=i,
                        anomaly_type="flatline",
                        severity=min(duration / (min_duration * 5), 1.0),
                        description=(
                            f"{channel_name} flatlined for {duration} steps "
                            f"(steps {i}→{j-1}, stuck at {data[i]:.4f})"
                        ),
                        details={
                            "start_step": i,
                            "end_step": j - 1,
                            "duration": duration,
                            "flat_value": float(data[i]),
                        },
                    )
                )
        i = j
    return anomalies


def _merge_nearby(anomalies: list[Anomaly], merge_window: int = 10) -> list[Anomaly]:
    """Merge anomalies close together in time on the same channel.

    When multiple detectors fire on the same event, keep the highest severity.
    """
    if not anomalies:
        return []

    sorted_a = sorted(anomalies, key=lambda a: (a.channel, a.step))
    merged: list[Anomaly] = []
    current = sorted_a[0]

    for a in sorted_a[1:]:
        if a.channel == current.channel and abs(a.step - current.step) <= merge_window:
            if a.severity > current.severity:
                current = a
        else:
            merged.append(current)
            current = a
    merged.append(current)
    return merged


def _is_noisy_channel(data: np.ndarray) -> bool:
    """Check if a channel is too noisy for standard thresholds.

    Noisy channels (like raw action outputs) need higher thresholds
    to avoid false positives.
    """
    if data.size == 0:
        return False
    channel_range = float(np.ptp(data))
    if channel_range == 0:
        return False
    channel_std = float(np.std(data))
    return (channel_std / channel_range) > 0.35


def detect_all(
    channels: dict[str, np.ndarray],
    drop_threshold: float = 0.6,
    spike_threshold: float = 5.0,
    flatline_duration: int = 50,
    skip_channels: set[str] | None = None,
) -> list[Anomaly]:
    """Run all anomaly detectors on all channels.

    Automatically adjusts thresholds for noisy channels and deduplicates
    nearby detections.

    Args:
        channels: Dict mapping channel name to numpy array [T, *shape].
        drop_threshold: Threshold for sudden drop detection (0-1).
        spike_threshold: Std deviation threshold for spike detection.
        flatline_duration: Min steps for flatline detection.
        skip_channels: Channel names to exclude from analysis.

    Returns:
        List of detected anomalies, deduplicated and sorted by severity.
    """
    skip = skip_channels or set()
    all_anomalies: list[Anomaly] = []

    for name, data in channels.items():
        if name in skip:
            continue

        if _is_noisy_channel(data):
            all_anomalies.extend(
                detect_sudden_drops(data, name, threshold_pct=0.85, min_baseline=0.1)
            )
            all_anomalies.extend(
                detect_sudden_spikes(data, name, threshold_std=8.0)
            )
        else:
            all_anomalies.extend(
                detect_sudden_drops(data, name, threshold_pct=drop_threshold)
            )
            all_anomalies.extend(
                detect_sudden_spikes(data, name, threshold_std=spike_threshold)
            )

        all_anomalies.extend(
            detect_flatlines(data, name, min_duration=flatline_duration)
        )

    all_anomalies = _merge_nearby(all_anomalies)
    all_anomalies.sort(key=lambda a: (-a.severity, a.step))
    return all_anomalies
