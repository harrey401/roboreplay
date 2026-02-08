"""RoboReplay CLI — command-line interface for recording analysis.

Commands:
    roboreplay info <file>        Show recording summary
    roboreplay diagnose <file>    Run anomaly detection and diagnosis
    roboreplay compare <a> <b>    Compare two recordings
    roboreplay export <file>      Export to HTML/GIF/CSV
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="roboreplay")
def cli() -> None:
    """RoboReplay — The DVR for robot behavior.

    Record, replay, diagnose, and share robot execution data.
    """
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def info(file: Path) -> None:
    """Show recording summary."""
    from roboreplay import Replay

    try:
        replay = Replay(file)
    except Exception as e:
        console.print(f"[red]Error opening {file}: {e}[/red]")
        raise SystemExit(1)

    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold]{replay.name}[/bold]",
        subtitle=f"{file}",
    ))

    # Metadata table
    meta_table = Table(show_header=False, box=None, padding=(0, 2))
    meta_table.add_column("Key", style="dim")
    meta_table.add_column("Value")

    if replay.robot:
        meta_table.add_row("Robot", replay.robot)
    if replay.task:
        meta_table.add_row("Task", replay.task)
    meta_table.add_row("Steps", str(replay.num_steps))
    meta_table.add_row("Channels", ", ".join(replay.channels))
    meta_table.add_row("Events", str(len(replay.events)))
    meta_table.add_row("Created", replay.metadata.created_at)

    console.print(meta_table)

    # Channel stats
    if replay.stats:
        console.print()
        stats_table = Table(title="Channel Statistics")
        stats_table.add_column("Channel")
        stats_table.add_column("Min", justify="right")
        stats_table.add_column("Max", justify="right")
        stats_table.add_column("Mean", justify="right")
        stats_table.add_column("Std", justify="right")

        for name, stat in replay.stats.items():
            stats_table.add_row(
                name,
                f"{stat.min:.4f}",
                f"{stat.max:.4f}",
                f"{stat.mean:.4f}",
                f"{stat.std:.4f}",
            )
        console.print(stats_table)

    # Events
    if len(replay.events) > 0:
        console.print()
        events_table = Table(title="Events")
        events_table.add_column("Step", justify="right")
        events_table.add_column("Type")
        events_table.add_column("Data")

        for event in replay.events.events[:10]:
            data_str = ", ".join(f"{k}={v}" for k, v in event.data.items()) if event.data else ""
            events_table.add_row(str(event.step), event.event_type, data_str)

        if len(replay.events) > 10:
            events_table.add_row("...", f"({len(replay.events) - 10} more)", "")

        console.print(events_table)

    replay.close()
    console.print()


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--drop-threshold", default=0.5, help="Sensitivity for drop detection (0-1)")
@click.option("--spike-threshold", default=3.0, help="Std devs for spike detection")
@click.option("--flatline-duration", default=20, help="Min steps for flatline detection")
@click.option("--llm", is_flag=True, default=False, help="Use LLM for enhanced diagnosis")
@click.option("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
def diagnose(
    file: Path,
    drop_threshold: float,
    spike_threshold: float,
    flatline_duration: int,
    llm: bool,
    api_key: str | None,
) -> None:
    """Run anomaly detection and diagnosis on a recording."""
    from roboreplay.diagnose import diagnose as run_diagnosis

    console.print()
    console.print(f"[dim]Analyzing {file}...[/dim]")

    result = run_diagnosis(
        file,
        drop_threshold=drop_threshold,
        spike_threshold=spike_threshold,
        flatline_duration=flatline_duration,
        use_llm=llm,
        api_key=api_key,
    )

    if not result.has_failures and len(result.warnings) == 0:
        console.print(Panel(
            "[green]\u2713 No anomalies detected. Recording looks clean.[/green]",
            title="Diagnosis",
            border_style="green",
        ))
    else:
        # Failures
        if result.failures:
            failure_lines = []
            for a in result.failures:
                line = f"[red]\u2717[/red] [bold]{a.anomaly_type}[/bold] at step {a.step}"
                failure_lines.append(line)
                failure_lines.append(f"  {a.description}")
                if a.details:
                    for k, v in a.details.items():
                        if isinstance(v, float):
                            failure_lines.append(f"  [dim]{k}: {v:.4f}[/dim]")
                        else:
                            failure_lines.append(f"  [dim]{k}: {v}[/dim]")
                failure_lines.append("")

            console.print(Panel(
                "\n".join(failure_lines),
                title=f"[red]Failures ({len(result.failures)})[/red]",
                border_style="red",
            ))

        # Warnings
        if result.warnings:
            warning_lines = []
            for a in result.warnings:
                warning_lines.append(f"[yellow]\u26a0[/yellow] {a.description}")

            console.print(Panel(
                "\n".join(warning_lines),
                title=f"[yellow]Warnings ({len(result.warnings)})[/yellow]",
                border_style="yellow",
            ))

    # LLM diagnosis output
    if result.llm_result is not None:
        llm = result.llm_result
        llm_lines = []
        if llm.explanation:
            llm_lines.append(f"[bold]Explanation:[/bold] {llm.explanation}")
            llm_lines.append("")
        if llm.root_causes:
            llm_lines.append("[bold]Root Causes:[/bold]")
            for i, cause in enumerate(llm.root_causes, 1):
                llm_lines.append(f"  {i}. {cause}")
            llm_lines.append("")
        if llm.recommendations:
            llm_lines.append("[bold]Recommendations:[/bold]")
            for i, rec in enumerate(llm.recommendations, 1):
                llm_lines.append(f"  {i}. {rec}")

        console.print(Panel(
            "\n".join(llm_lines),
            title="[blue]LLM Diagnosis[/blue]",
            border_style="blue",
        ))

    console.print()


@cli.command()
@click.argument("file_a", type=click.Path(exists=True, path_type=Path))
@click.argument("file_b", type=click.Path(exists=True, path_type=Path))
def compare(file_a: Path, file_b: Path) -> None:
    """Compare two recordings side-by-side."""
    from roboreplay.compare import compare as run_compare

    console.print()
    result = run_compare(file_a, file_b)
    console.print(result.summary())

    if result.divergence_step is not None:
        console.print()
        console.print(
            f"[yellow]⚠ Recordings diverge at step {result.divergence_step}[/yellow]"
        )
    console.print()


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "-f", "fmt", type=click.Choice(["csv", "html"]), default="csv",
              help="Export format")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory or file")
@click.option("--channel", "-c", multiple=True, help="Channels to export (default: all)")
def export(file: Path, fmt: str, output: Path | None, channel: tuple[str, ...]) -> None:
    """Export recording to CSV or HTML."""
    channels = list(channel) if channel else None

    if fmt == "csv":
        from roboreplay.export.csv import export_csv

        created = export_csv(file, output_dir=output, channels=channels)
        for p in created:
            console.print(f"  Created: {p}")
        console.print(f"[green]Exported {len(created)} CSV file(s)[/green]")

    elif fmt == "html":
        from roboreplay.export.html import export_html

        out_path = output or file.with_suffix(".html")
        result = export_html(file, output=out_path, channels=channels)
        console.print(f"  Created: {result}")
        console.print("[green]HTML export complete[/green]")


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--channel", "-c", multiple=True, help="Channels to plot (default: all)")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Save plot to file")
def plot(file: Path, channel: tuple[str, ...], output: Path | None) -> None:
    """Plot channels from a recording."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[red]matplotlib required. Install with: pip install roboreplay[viz][/red]")
        raise SystemExit(1)

    from roboreplay import Replay

    replay = Replay(file)
    channels_to_plot = list(channel) if channel else replay.channels

    for ch in channels_to_plot:
        if ch not in replay.channels:
            console.print(f"[yellow]Warning: channel '{ch}' not found, skipping[/yellow]")
            continue
        fig = replay.plot(ch)
        if output:
            suffix = output.suffix or ".png"
            out_path = output.with_name(f"{output.stem}_{ch}{suffix}")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            console.print(f"Saved: {out_path}")
        else:
            plt.show()

    replay.close()


if __name__ == "__main__":
    cli()
