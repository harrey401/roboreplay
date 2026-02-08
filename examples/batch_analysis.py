"""RoboReplay Example: Batch Analysis

Generates 5 recordings with varying conditions, runs batch diagnosis,
exports to CSV, and prints a summary table.

No external dependencies required.

Run:
    python examples/batch_analysis.py

Output:
    - Creates 5 .rrp files in batch_output/
    - Creates CSV exports for each
    - Prints diagnosis summary table
"""

from pathlib import Path

import numpy as np

from roboreplay import Recorder, Replay
from roboreplay.diagnose import diagnose
from roboreplay.export.csv import export_csv


def generate_recording(
    name: str,
    output_dir: Path,
    seed: int,
    failure_step: int | None = None,
    noise_level: float = 0.1,
) -> Path:
    """Generate a synthetic recording with controllable parameters.

    Args:
        name: Recording name.
        output_dir: Directory for output files.
        seed: Random seed for reproducibility.
        failure_step: Step at which to inject a failure (None = no failure).
        noise_level: Standard deviation of sensor noise.

    Returns:
        Path to the saved recording.
    """
    rng = np.random.default_rng(seed)
    path = output_dir / f"{name}.rrp"

    rec = Recorder(
        name,
        path=path,
        metadata={
            "robot": "simulated_arm",
            "task": "batch_test",
            "seed": str(seed),
            "noise_level": str(noise_level),
        },
    )
    rec.start()

    for step in range(200):
        # Sinusoidal position trajectory
        t = step / 200.0
        position = np.array([
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t),
            0.5 + 0.3 * np.sin(4 * np.pi * t),
        ]) + rng.normal(0, noise_level, 3)

        # Force signal: stable at 5N, drops on failure
        if failure_step is not None and step >= failure_step:
            force = 0.5 + rng.normal(0, 0.1)
            reward = -1.0
        else:
            force = 5.0 + rng.normal(0, 0.2)
            reward = 1.0 - t

        velocity = rng.normal(0, noise_level, 3)

        rec.step(
            position=position,
            velocity=velocity,
            force=np.array([max(0, force)]),
            reward=reward,
        )

        if failure_step is not None and step == failure_step:
            rec.mark_event("failure", {
                "type": "force_drop",
                "force_before": 5.0,
                "force_after": 0.5,
            })

    rec.mark_event("episode_end", {
        "success": failure_step is None,
        "noise_level": noise_level,
    })

    return rec.save()


def main() -> None:
    print("=" * 60)
    print("  RoboReplay Example: Batch Analysis")
    print("=" * 60)
    print()

    output_dir = Path("batch_output")
    output_dir.mkdir(exist_ok=True)

    # Generate 5 recordings with different conditions
    configs = [
        {"name": "run_clean_1", "seed": 10, "failure_step": None, "noise_level": 0.05},
        {"name": "run_clean_2", "seed": 20, "failure_step": None, "noise_level": 0.1},
        {"name": "run_noisy", "seed": 30, "failure_step": None, "noise_level": 0.5},
        {"name": "run_fail_early", "seed": 40, "failure_step": 80, "noise_level": 0.1},
        {"name": "run_fail_late", "seed": 50, "failure_step": 160, "noise_level": 0.1},
    ]

    print("[1/3] Generating recordings...")
    paths: list[Path] = []
    for cfg in configs:
        p = generate_recording(output_dir=output_dir, **cfg)
        paths.append(p)
        print(f"  Created: {p}")
    print()

    # Batch diagnosis
    print("[2/3] Running batch diagnosis...")
    print()
    print(f"  {'Name':<20s} {'Steps':>6s} {'Failures':>9s} {'Warnings':>9s}  Status")
    print("  " + "-" * 62)

    for p in paths:
        result = diagnose(p)
        n_f = len(result.failures)
        n_w = len(result.warnings)
        status = "FAIL" if n_f > 0 else "WARN" if n_w > 0 else "OK"
        print(f"  {result.recording_name:<20s} {result.num_steps:>6d} {n_f:>9d} {n_w:>9d}  {status}")
    print()

    # Export all to CSV
    print("[3/3] Exporting to CSV...")
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    total_files = 0
    for p in paths:
        created = export_csv(p, output_dir=csv_dir)
        total_files += len(created)
    print(f"  Exported {total_files} CSV files to {csv_dir}/")
    print()

    print("=" * 60)
    print("  Done! Try:")
    print(f"    roboreplay info {paths[0]}")
    print(f"    roboreplay diagnose {paths[3]}")
    print(f"    roboreplay export {paths[0]} --format csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
