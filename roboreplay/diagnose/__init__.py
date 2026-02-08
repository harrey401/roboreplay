"""Diagnosis engine for RoboReplay recordings.

Provides both offline (statistical) and AI-powered failure analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboreplay.diagnose.anomaly import Anomaly, detect_all
from roboreplay.replay import Replay


@dataclass
class DiagnosisResult:
    """Result of diagnosing a recording."""

    recording_name: str
    num_steps: int
    anomalies: list[Anomaly] = field(default_factory=list)
    summary: str = ""
    llm_result: Any = None  # LLMDiagnosisResult when use_llm=True

    @property
    def failures(self) -> list[Anomaly]:
        """Anomalies with severity > 0.5."""
        return [a for a in self.anomalies if a.severity > 0.5]

    @property
    def warnings(self) -> list[Anomaly]:
        """Anomalies with severity 0.2 - 0.5."""
        return [a for a in self.anomalies if 0.2 <= a.severity <= 0.5]

    @property
    def has_failures(self) -> bool:
        return len(self.failures) > 0

    def __repr__(self) -> str:
        return (
            f"DiagnosisResult(recording='{self.recording_name}', "
            f"failures={len(self.failures)}, warnings={len(self.warnings)})"
        )


def diagnose(
    path: str | Path,
    drop_threshold: float = 0.5,
    spike_threshold: float = 3.0,
    flatline_duration: int = 20,
    use_llm: bool = False,
    api_key: str | None = None,
) -> DiagnosisResult:
    """Diagnose a recording for anomalies and failures.

    Runs statistical anomaly detection on all channels. Optionally
    uses an LLM for deeper analysis.

    Args:
        path: Path to a .rrp file.
        drop_threshold: Sensitivity for sudden drop detection (0-1).
        spike_threshold: Std deviations for spike detection.
        flatline_duration: Min steps for flatline detection.
        use_llm: Whether to use LLM for enhanced diagnosis (requires API key).
        api_key: Anthropic API key for LLM diagnosis. If None, reads from env.

    Returns:
        DiagnosisResult with detected anomalies and summary.
    """
    replay = Replay(path)

    # Load all channels
    channels = {}
    for name in replay.channels:
        channels[name] = replay.channel(name)

    # Run anomaly detection
    anomalies = detect_all(
        channels,
        drop_threshold=drop_threshold,
        spike_threshold=spike_threshold,
        flatline_duration=flatline_duration,
    )

    # Build summary
    n_failures = sum(1 for a in anomalies if a.severity > 0.5)
    n_warnings = sum(1 for a in anomalies if 0.2 <= a.severity <= 0.5)

    if n_failures == 0 and n_warnings == 0:
        summary = "No anomalies detected. Recording looks clean."
    else:
        lines = []
        if n_failures > 0:
            lines.append(f"{n_failures} failure(s) detected:")
            for a in anomalies:
                if a.severity > 0.5:
                    lines.append(f"  \u2022 {a.description}")
        if n_warnings > 0:
            lines.append(f"{n_warnings} warning(s):")
            for a in anomalies:
                if 0.2 <= a.severity <= 0.5:
                    lines.append(f"  \u2022 {a.description}")
        summary = "\n".join(lines)

    # LLM diagnosis (optional)
    llm_result = None
    if use_llm:
        from roboreplay.diagnose.llm import llm_diagnose

        llm_result = llm_diagnose(
            metadata=replay.metadata,
            anomalies=anomalies,
            stats=replay.stats,
            events=replay.events,
            api_key=api_key,
        )

    result = DiagnosisResult(
        recording_name=replay.name,
        num_steps=replay.num_steps,
        anomalies=anomalies,
        summary=summary,
        llm_result=llm_result,
    )

    replay.close()
    return result
