"""RoboReplay Example: Custom 6-DOF Robot Arm Simulation

Demonstrates recording two runs of a 6-DOF robot arm performing
a reaching task — one successful, one with a joint limit violation —
then comparing them side-by-side.

No external dependencies required.

Run:
    python examples/custom_robot.py

Output:
    - Creates custom_arm_success.rrp and custom_arm_failure.rrp
    - Compares both recordings and prints divergence analysis
"""

import numpy as np

from roboreplay import Recorder, Replay, compare
from roboreplay.diagnose import diagnose


def simulate_arm(
    name: str,
    seed: int,
    inject_failure: bool = False,
) -> str:
    """Simulate a 6-DOF arm performing a reaching task.

    Args:
        name: Recording name.
        seed: Random seed.
        inject_failure: If True, injects a joint limit violation.

    Returns:
        Path to the saved recording.
    """
    rng = np.random.default_rng(seed)

    rec = Recorder(
        name,
        metadata={
            "robot": "custom_6dof_arm",
            "task": "reach_target",
            "notes": "Failure run" if inject_failure else "Success run",
        },
    )
    rec.start()

    # Joint limits (radians)
    joint_limits_lower = np.array([-2.96, -2.09, -2.96, -2.09, -2.96, -2.09])
    joint_limits_upper = np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09])

    # Initial position (centered)
    joint_pos = np.zeros(6)
    target = np.array([0.5, 0.3, 0.4])

    rec.mark_event("phase_start", {"phase": "reach"})

    for step in range(300):
        t = step / 300.0

        # Simple trajectory toward target
        desired_vel = rng.normal(0, 0.02, 6)
        desired_vel[:3] += 0.01 * (1 - t)  # Gradually slow down

        if inject_failure and step == 150:
            # Inject a joint limit violation: sudden large command
            desired_vel[2] = 0.5  # Way too fast
            rec.mark_event("failure", {
                "type": "joint_limit_violation",
                "joint": 2,
                "velocity_command": 0.5,
            })

        # Integrate position
        joint_pos = joint_pos + desired_vel * 0.01  # dt = 0.01

        # Compute end-effector (simplified forward kinematics)
        ee_pos = np.array([
            0.3 * np.cos(joint_pos[0]) + 0.2 * np.cos(joint_pos[0] + joint_pos[1]),
            0.3 * np.sin(joint_pos[0]) + 0.2 * np.sin(joint_pos[0] + joint_pos[1]),
            0.1 + 0.2 * np.sin(joint_pos[2]),
        ])

        # Compute distance to target
        error = np.linalg.norm(ee_pos - target)

        # Torque estimate (proportional to velocity + noise)
        torque = np.abs(desired_vel) * 10 + rng.normal(0, 0.5, 6)

        # Check joint limits
        violations = np.any(joint_pos < joint_limits_lower) or np.any(
            joint_pos > joint_limits_upper
        )

        rec.step(
            joint_positions=joint_pos,
            joint_velocities=desired_vel,
            ee_position=ee_pos,
            target_error=np.array([error]),
            torque=torque,
            reward=float(-error),
        )

        if violations and inject_failure:
            rec.mark_event("joint_limit_hit", {
                "step": step,
                "joints_violated": [
                    int(j) for j in range(6)
                    if joint_pos[j] < joint_limits_lower[j]
                    or joint_pos[j] > joint_limits_upper[j]
                ],
            })

    rec.mark_event("episode_end", {"success": not inject_failure})
    path = rec.save()
    return str(path)


def main() -> None:
    print("=" * 60)
    print("  RoboReplay Example: Custom 6-DOF Robot Arm")
    print("=" * 60)
    print()

    # Run 1: Success
    print("[1/4] Recording successful run...")
    path_success = simulate_arm("custom_arm_success", seed=42, inject_failure=False)
    print(f"  Saved: {path_success}")

    # Run 2: Failure
    print("[2/4] Recording failure run...")
    path_failure = simulate_arm("custom_arm_failure", seed=42, inject_failure=True)
    print(f"  Saved: {path_failure}")
    print()

    # Replay both
    print("[3/4] Replaying recordings...")
    r_success = Replay(path_success)
    r_failure = Replay(path_failure)
    print(f"  Success: {r_success.num_steps} steps, channels: {r_success.channels}")
    print(f"  Failure: {r_failure.num_steps} steps, channels: {r_failure.channels}")
    r_success.close()
    r_failure.close()
    print()

    # Compare
    print("[4/4] Comparing recordings...")
    diff = compare(path_success, path_failure)
    print(diff.summary())
    print()

    if diff.divergence_step is not None:
        print(f"Divergence detected at step {diff.divergence_step}")
    print()

    # Diagnose failure run
    print("Diagnosing failure run...")
    result = diagnose(path_failure)
    if result.has_failures:
        print(f"  Found {len(result.failures)} failure(s):")
        for a in result.failures:
            print(f"    [{a.step}] {a.anomaly_type}: {a.description}")
    else:
        print("  No failures detected in statistical analysis")

    print()
    print("=" * 60)
    print("  Done! Try:")
    print(f"    roboreplay info {path_success}")
    print(f"    roboreplay compare {path_success} {path_failure}")
    print("=" * 60)


if __name__ == "__main__":
    main()
