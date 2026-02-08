# MuJoCo Panda Reaching Task

A real MuJoCo simulation example using the Franka Emika Panda robot arm performing a reaching task with a PD controller.

## Requirements

```bash
pip install roboreplay[mujoco]
```

## Run

```bash
python examples/mujoco_panda.py
```

## What It Does

1. **Loads a 7-DOF Panda arm** — Simplified MJCF model with joint limits and actuator ranges
2. **PD controller** — Tracks a target joint configuration to reach a 3D target position
3. **Records at 50 Hz** — Every 10th physics step (timestep = 0.002s)
4. **Reaches target** — End-effector converges to within 5cm of the target

## Channels Recorded

- `qpos` — 7-DOF joint positions (radians)
- `qvel` — 7-DOF joint velocities (rad/s)
- `ctrl` — 7-DOF control torques (Nm)
- `ee_position` — End-effector XYZ position (meters)
- `ee_error` — Scalar distance to target (meters)

## Events

- `phase_start` — Start of reaching phase with target position
- `target_reached` — End-effector within 5cm of target
- `episode_end` — End of simulation with success flag

## Key Code

The PD controller computes torques from joint position error:

```python
q_error = TARGET_QPOS - data.qpos[:7]
ctrl = KP * q_error - KD * data.qvel[:7]
data.ctrl[:7] = np.clip(ctrl, -ctrl_range, ctrl_range)
```

Recording happens every 10th physics step:

```python
rec.step(
    qpos=data.qpos[:7].copy(),
    qvel=data.qvel[:7].copy(),
    ctrl=data.ctrl[:7].copy(),
    ee_position=ee_pos,
    ee_error=np.array([ee_error]),
)
```

## After Recording

```bash
roboreplay info mujoco_panda_reach.rrp
roboreplay diagnose mujoco_panda_reach.rrp
roboreplay export mujoco_panda_reach.rrp --format html
```

The diagnosis should show a clean run with the end-effector converging smoothly to the target.
