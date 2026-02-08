# Custom Robot Example

Simulates a 6-DOF robot arm with two runs (success and failure), then compares them.

## Run

```bash
python examples/custom_robot.py
```

No external dependencies required.

## What It Does

1. Records a **success run** — smooth reaching task
2. Records a **failure run** — injects a joint limit violation at step 150
3. Compares both recordings to find the divergence point
4. Diagnoses the failure run

## Channels Recorded

- `joint_positions` — 6-DOF joint angles
- `joint_velocities` — 6-DOF velocities
- `ee_position` — End-effector position (simplified FK)
- `target_error` — Distance to target
- `torque` — Estimated joint torques
- `reward` — Negative error as reward

## Key Takeaway

The comparison finds exactly where the two runs diverge, making it easy to identify what went wrong and when.
