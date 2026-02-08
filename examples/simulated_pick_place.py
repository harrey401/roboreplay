"""RoboReplay Example: Simulated Pick-and-Place with Failure

This example demonstrates RoboReplay without any external dependencies
(no MuJoCo required). It simulates a 7-DOF robot arm performing a
pick-and-place task, with a realistic gripper slip failure partway through.

Run:
    python examples/simulated_pick_place.py

Output:
    - Creates pick_place_demo.rrp
    - Prints recording summary
    - Runs diagnosis
"""

import numpy as np

from roboreplay import Recorder, Replay
from roboreplay.diagnose import diagnose


def simulate_pick_place(seed: int = 42) -> str:
    """Simulate a pick-and-place with a mid-task grasp slip.

    Returns the path to the saved recording.
    """
    rng = np.random.default_rng(seed)

    rec = Recorder(
        "pick_place_demo",
        metadata={
            "robot": "simulated_panda_7dof",
            "task": "pick_and_place",
            "scene": "tabletop_single_object",
            "notes": "Demo recording with simulated grasp slip failure",
        },
    )
    rec.start()

    # Simulation parameters
    home_pos = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
    target_pos = np.array([0.4, 0.18, 0.02])  # Object on table
    bin_pos = np.array([0.6, -0.3, 0.1])       # Bin location

    # Phase 1: Approach (steps 0-99)
    # Robot moves from home to above the object
    rec.mark_event("phase_start", {"phase": "approach"})
    for i in range(100):
        t = i / 100.0
        joint_pos = home_pos + t * rng.normal(0, 0.01, 7)  # Slight noise
        joint_pos[0] += t * 0.3  # Shoulder moves toward target
        joint_pos[1] += t * 0.2

        gripper_width = 0.08  # Open
        gripper_force = 0.0
        ee_z = 0.3 - t * 0.15  # Descending toward table

        rec.step(
            joint_positions=joint_pos,
            action=rng.normal(0, 0.05, 7),
            gripper_width=np.array([gripper_width]),
            gripper_force=np.array([gripper_force]),
            ee_position=np.array([target_pos[0], target_pos[1], ee_z]),
            reward=0.0,
        )

    # Phase 2: Grasp (steps 100-149)
    # Gripper closes on object, force builds up
    rec.mark_event("phase_start", {"phase": "grasp"})
    for i in range(50):
        t = i / 50.0
        joint_pos = home_pos + 0.3 + rng.normal(0, 0.005, 7)

        gripper_width = 0.08 * (1 - t)  # Closing
        gripper_force = t * 8.5 + rng.normal(0, 0.2)  # Force building
        gripper_force = max(0, gripper_force)

        rec.step(
            joint_positions=joint_pos,
            action=rng.normal(0, 0.02, 7),
            gripper_width=np.array([gripper_width]),
            gripper_force=np.array([gripper_force]),
            ee_position=np.array([target_pos[0], target_pos[1], 0.02]),
            reward=0.1,
        )

    rec.mark_event("grasp_acquired", {"force": 8.5, "width": 0.01})

    # Phase 3: Lift (steps 150-249)
    # Lifting object — FAILURE happens at step 200
    rec.mark_event("phase_start", {"phase": "lift"})
    for i in range(100):
        t = i / 100.0
        joint_pos = home_pos + 0.3 + rng.normal(0, 0.005, 7)
        joint_pos[1] -= t * 0.3  # Shoulder lifts

        ee_z = 0.02 + t * 0.25  # Rising
        lift_accel = 0.15 + t * 0.25  # Acceleration increasing

        step_num = 150 + i
        if step_num < 200:
            # Normal: force stable around 8N
            gripper_force = 8.0 + rng.normal(0, 0.3)
            gripper_width = 0.01
            reward = 0.5
        elif step_num == 200:
            # THE SLIP: force drops suddenly
            rec.mark_event("failure", {
                "type": "grasp_slip",
                "force_before": 7.8,
                "force_after": 1.2,
                "lift_acceleration": float(lift_accel),
            })
            gripper_force = 1.2  # Dropped!
            gripper_width = 0.04  # Gripper opened
            reward = -1.0
        else:
            # After slip: force stays low, object falling
            gripper_force = 0.5 + rng.normal(0, 0.1)
            gripper_force = max(0, gripper_force)
            gripper_width = 0.06
            ee_z = max(0.02, ee_z - (step_num - 200) * 0.005)
            reward = -0.5

        rec.step(
            joint_positions=joint_pos,
            action=rng.normal(0, 0.02, 7),
            gripper_width=np.array([gripper_width]),
            gripper_force=np.array([max(0, gripper_force)]),
            ee_position=np.array([target_pos[0], target_pos[1], ee_z]),
            reward=reward,
        )

    # Phase 4: Recovery attempt (steps 250-299)
    rec.mark_event("phase_start", {"phase": "recovery"})
    for i in range(50):
        joint_pos = home_pos + rng.normal(0, 0.005, 7)
        rec.step(
            joint_positions=joint_pos,
            action=rng.normal(0, 0.01, 7),
            gripper_width=np.array([0.08]),
            gripper_force=np.array([0.0]),
            ee_position=np.array([target_pos[0], target_pos[1], 0.15 - i * 0.002]),
            reward=-0.1,
        )

    rec.mark_event("episode_end", {"success": False, "total_reward": -12.5})
    path = rec.save()
    return str(path)


def main() -> None:
    print("=" * 60)
    print("  RoboReplay Demo: Simulated Pick-and-Place")
    print("=" * 60)
    print()

    # 1. Record
    print("[1/3] Recording simulated pick-and-place...")
    path = simulate_pick_place()
    print(f"  Saved: {path}")
    print()

    # 2. Replay and inspect
    print("[2/3] Replaying recording...")
    replay = Replay(path)
    print(replay)
    print()

    # Show data at failure point
    print("Data at failure point (step 200):")
    frame = replay[200]
    for name, value in frame.items():
        print(f"  {name}: {value}")
    print()

    replay.close()

    # 3. Diagnose
    print("[3/3] Running diagnosis...")
    result = diagnose(path)
    print()
    if result.has_failures:
        print(f"Found {len(result.failures)} failure(s):")
        for a in result.failures:
            print(f"  [{a.step}] {a.anomaly_type}: {a.description}")
    print()
    if result.warnings:
        print(f"Found {len(result.warnings)} warning(s):")
        for a in result.warnings:
            print(f"  [{a.step}] {a.anomaly_type}: {a.description}")

    print()
    print("=" * 60)
    print("  Demo complete! Try the CLI:")
    print(f"    roboreplay info {path}")
    print(f"    roboreplay diagnose {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
