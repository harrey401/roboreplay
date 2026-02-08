# Pick and Place Demo

The flagship example simulates a 7-DOF robot arm performing a pick-and-place task with a realistic gripper slip failure.

## Run

```bash
python examples/simulated_pick_place.py
```

## What It Does

1. **Approach** (steps 0-99) — Robot moves from home to above the object
2. **Grasp** (steps 100-149) — Gripper closes, force builds to ~8.5N
3. **Lift** (steps 150-249) — Lifts object; **slip failure at step 200** (force drops from 8N to 1.2N)
4. **Recovery** (steps 250-299) — Robot attempts recovery

## Channels Recorded

- `joint_positions` — 7-DOF joint angles
- `action` — 7-DOF action commands
- `gripper_width` — Gripper opening
- `gripper_force` — Gripper force (key signal for failure)
- `ee_position` — End-effector position
- `reward` — Scalar reward

## Events

- `phase_start` — Phase transitions
- `grasp_acquired` — Successful grasp
- `failure` — Grasp slip at step 200
- `episode_end` — End of episode

## After Recording

```bash
roboreplay info pick_place_demo.rrp
roboreplay diagnose pick_place_demo.rrp
roboreplay export pick_place_demo.rrp --format html
```

The diagnosis should detect the sudden drop in `gripper_force` at step 200.
