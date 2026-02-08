"""Compare — side-by-side analysis of two recordings.

Usage:
    from roboreplay import Replay
    from roboreplay.compare import compare

    diff = compare("run_success.rrp", "run_failure.rrp")
    print(diff.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from roboreplay.replay import Replay


@dataclass
class ChannelDiff:
    """Difference in a single channel between two recordings."""

    name: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    max_abs_diff: float
    divergence_step: int | None = None  # Step where they start diverging

    @property
    def mean_change_pct(self) -> float:
        if abs(self.mean_a) < 1e-9:
            return 0.0
        return (self.mean_b - self.mean_a) / abs(self.mean_a) * 100


@dataclass
class CompareResult:
    """Result of comparing two recordings."""

    name_a: str
    name_b: str
    steps_a: int
    steps_b: int
    shared_channels: list[str]
    channel_diffs: dict[str, ChannelDiff] = field(default_factory=dict)
    divergence_step: int | None = None

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = []
        lines.append(f"Comparison: {self.name_a} vs {self.name_b}")
        lines.append("")

        # Header
        col_a = self.name_a[:20]
        col_b = self.name_b[:20]
        lines.append(f"{'':20s}  {col_a:>20s}  {col_b:>20s}  {'Change':>10s}")
        lines.append("-" * 76)

        lines.append(f"{'Steps':20s}  {self.steps_a:>20d}  {self.steps_b:>20d}")

        for name, diff in self.channel_diffs.items():
            pct = diff.mean_change_pct
            marker = " ⚠" if abs(pct) > 25 else ""
            lines.append(
                f"{name + ' (mean)':20s}  {diff.mean_a:>20.4f}  {diff.mean_b:>20.4f}  "
                f"{pct:>+8.1f}%{marker}"
            )

        if self.divergence_step is not None:
            lines.append("")
            lines.append(f"Divergence point: step {self.divergence_step}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CompareResult('{self.name_a}' vs '{self.name_b}', "
            f"divergence={self.divergence_step})"
        )


def _find_divergence(
    data_a: np.ndarray,
    data_b: np.ndarray,
    threshold_std: float = 3.0,
) -> int | None:
    """Find the step where two signals start diverging.

    Uses a rolling comparison — divergence is where the difference
    exceeds threshold_std * baseline noise.
    """
    min_len = min(len(data_a), len(data_b))
    if min_len < 10:
        return None

    # Flatten to 1D for comparison
    if data_a.ndim > 1:
        data_a = np.linalg.norm(data_a, axis=-1)
    if data_b.ndim > 1:
        data_b = np.linalg.norm(data_b, axis=-1)

    diff = np.abs(data_a[:min_len] - data_b[:min_len])

    # Baseline noise from first 10% of recording
    baseline_end = max(10, min_len // 10)
    baseline_std = np.std(diff[:baseline_end])
    if baseline_std < 1e-9:
        baseline_std = 1e-9

    # Find first sustained divergence (3+ consecutive steps above threshold)
    threshold = baseline_std * threshold_std
    for i in range(baseline_end, min_len - 3):
        if all(diff[i + j] > threshold for j in range(3)):
            return i

    return None


def compare(
    path_a: str | Path,
    path_b: str | Path,
) -> CompareResult:
    """Compare two recordings side-by-side.

    Analyzes per-channel statistics and finds the divergence point
    where the two runs start behaving differently.

    Args:
        path_a: Path to first recording.
        path_b: Path to second recording.

    Returns:
        CompareResult with detailed comparison.
    """
    replay_a = Replay(path_a)
    replay_b = Replay(path_b)

    shared = sorted(set(replay_a.channels) & set(replay_b.channels))

    channel_diffs: dict[str, ChannelDiff] = {}
    earliest_divergence: int | None = None

    for name in shared:
        data_a = replay_a.channel(name)
        data_b = replay_b.channel(name)

        mean_a = float(np.mean(data_a))
        mean_b = float(np.mean(data_b))
        std_a = float(np.std(data_a))
        std_b = float(np.std(data_b))

        min_len = min(len(data_a), len(data_b))
        max_abs_diff = float(np.max(np.abs(data_a[:min_len] - data_b[:min_len])))

        div_step = _find_divergence(data_a, data_b)

        channel_diffs[name] = ChannelDiff(
            name=name,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            max_abs_diff=max_abs_diff,
            divergence_step=div_step,
        )

        if div_step is not None:
            if earliest_divergence is None or div_step < earliest_divergence:
                earliest_divergence = div_step

    result = CompareResult(
        name_a=replay_a.name,
        name_b=replay_b.name,
        steps_a=replay_a.num_steps,
        steps_b=replay_b.num_steps,
        shared_channels=shared,
        channel_diffs=channel_diffs,
        divergence_step=earliest_divergence,
    )

    replay_a.close()
    replay_b.close()
    return result
