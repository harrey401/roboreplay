"""LLM-powered diagnosis using Claude for natural-language failure analysis.

Requires the `anthropic` package: pip install roboreplay[diagnose]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from roboreplay.diagnose.anomaly import Anomaly
from roboreplay.utils.schema import ChannelStats, EventLog, RecordingMetadata


@dataclass
class LLMDiagnosisResult:
    """Result of LLM-powered diagnosis."""

    explanation: str
    root_causes: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    raw_response: str = ""

    def __repr__(self) -> str:
        return (
            f"LLMDiagnosisResult(causes={len(self.root_causes)}, "
            f"recommendations={len(self.recommendations)})"
        )


def _build_prompt(
    metadata: RecordingMetadata,
    anomalies: list[Anomaly],
    stats: dict[str, ChannelStats],
    events: EventLog,
) -> str:
    """Build a structured prompt for the LLM."""
    lines = [
        "You are analyzing a robot execution recording for failures and anomalies.",
        "Provide a clear diagnosis with root causes and actionable recommendations.",
        "",
        "## Recording Info",
        f"- Name: {metadata.name}",
        f"- Robot: {metadata.robot or 'unspecified'}",
        f"- Task: {metadata.task or 'unspecified'}",
    ]

    if metadata.user_metadata:
        for k, v in metadata.user_metadata.items():
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Channel Statistics")
    for name, stat in stats.items():
        lines.append(
            f"- {name}: min={stat.min:.4f}, max={stat.max:.4f}, "
            f"mean={stat.mean:.4f}, std={stat.std:.4f}, steps={stat.num_steps}"
        )

    lines.append("")
    lines.append("## Detected Anomalies")
    if anomalies:
        for a in anomalies:
            lines.append(
                f"- [{a.anomaly_type}] step {a.step}, channel '{a.channel}', "
                f"severity={a.severity:.2f}: {a.description}"
            )
    else:
        lines.append("- None detected")

    lines.append("")
    lines.append("## Event Timeline")
    if len(events) > 0:
        for event in events.events:
            data_str = ", ".join(f"{k}={v}" for k, v in event.data.items()) if event.data else ""
            lines.append(f"- step {event.step}: {event.event_type} {data_str}")
    else:
        lines.append("- No events recorded")

    lines.append("")
    lines.append("## Instructions")
    lines.append("Respond with exactly three sections:")
    lines.append("EXPLANATION: A 2-3 sentence summary of what happened.")
    lines.append("ROOT CAUSES: A numbered list of likely root causes.")
    lines.append("RECOMMENDATIONS: A numbered list of actionable recommendations.")

    return "\n".join(lines)


def _parse_response(text: str) -> LLMDiagnosisResult:
    """Parse the LLM response into structured fields."""
    explanation = ""
    root_causes: list[str] = []
    recommendations: list[str] = []

    current_section = ""
    for line in text.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("EXPLANATION"):
            current_section = "explanation"
            # Check if content is on the same line after colon
            after = stripped.split(":", 1)
            if len(after) > 1 and after[1].strip():
                explanation = after[1].strip()
            continue
        elif upper.startswith("ROOT CAUSE"):
            current_section = "root_causes"
            continue
        elif upper.startswith("RECOMMENDATION"):
            current_section = "recommendations"
            continue

        if not stripped:
            continue

        # Remove list markers
        clean = stripped.lstrip("0123456789.-) ").strip()
        if not clean:
            continue

        if current_section == "explanation":
            explanation = (explanation + " " + clean).strip() if explanation else clean
        elif current_section == "root_causes":
            root_causes.append(clean)
        elif current_section == "recommendations":
            recommendations.append(clean)

    return LLMDiagnosisResult(
        explanation=explanation,
        root_causes=root_causes,
        recommendations=recommendations,
        raw_response=text,
    )


def llm_diagnose(
    metadata: RecordingMetadata,
    anomalies: list[Anomaly],
    stats: dict[str, ChannelStats],
    events: EventLog,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
) -> LLMDiagnosisResult:
    """Run LLM-powered diagnosis on recording data.

    Args:
        metadata: Recording metadata.
        anomalies: Detected anomalies from statistical analysis.
        stats: Per-channel statistics.
        events: Event log.
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model: Claude model to use.

    Returns:
        LLMDiagnosisResult with explanation, root causes, and recommendations.

    Raises:
        ImportError: If anthropic package is not installed.
        RuntimeError: If API call fails.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for LLM diagnosis. "
            "Install it with: pip install roboreplay[diagnose]"
        )

    prompt = _build_prompt(metadata, anomalies, stats, events)

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
    except Exception as e:
        raise RuntimeError(f"LLM diagnosis failed: {e}") from e

    return _parse_response(response_text)
