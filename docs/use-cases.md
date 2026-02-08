# Use Cases

## Why RoboReplay?

| What exists | What's wrong with it |
|---|---|
| **rosbag** | ROS-only. Huge files. No analysis. No querying. |
| **Weights & Biases** | For ML training metrics, not physical robot behavior. |
| **Video recording** | Loses all the data. You see *what* happened but not *why*. |
| **Manual logging** | Everyone builds their own. Everyone hates it. |

RoboReplay is the first **framework-agnostic, pip-installable tool** for recording, analyzing, and sharing robot execution data with built-in anomaly detection.

---

## Works With Any Robotics Stack

| Platform | What you record |
|---|---|
| **MuJoCo** | `data.qpos`, `data.qvel`, `data.ctrl`, `data.sensordata` |
| **Isaac Sim** | Joint states, rigid body poses, sensor readings |
| **PyBullet** | `getJointStates()`, `getContactPoints()`, `getBasePositionAndOrientation()` |
| **ROS2** | Any topic data converted to numpy in a subscriber callback |
| **Gymnasium** | One-line: `roboreplay.wrap(env)` — automatic obs/action/reward |
| **Real hardware** | Serial data, force/torque sensors, encoders, cameras — anything as numpy |
| **Custom simulation** | Any loop that produces numpy arrays |

---

## Who It's For

**RL researchers** — Record every training episode. Debug why episode 4,847 failed. Compare policy v2 against v3. Wrap Gymnasium envs in one line.

**Manipulation researchers** — Capture joint states, end-effector poses, gripper forces, contact forces. Diagnose grasp failures automatically. Compare successful vs failed grasps to find the exact divergence step.

**Real robot operators** — Black-box recorder for teleoperation, autonomous runs, and calibration. Crash-safe streaming writes mean you never lose overnight data. Post-run anomaly detection catches hardware issues.

**Robotics teams** — Batch diagnosis across fleet runs. Export HTML reports for non-technical stakeholders. CSV export for downstream analysis in pandas, MATLAB, R, or Excel.

**Students and educators** — Students submit `.rrp` files as deliverables. Instructors replay, diagnose, and batch-compare submissions. HTML export for presentations without Python.

---

## Concrete Workflows

### Debugging a failure

```python
r = Replay("failed_run.rrp")
r[200]                         # All sensors at the failure step
r.events.where("failure")     # What events were marked?
result = diagnose("failed_run.rrp", use_llm=True)
print(result.llm_result.explanation)  # "The robot dropped the object because..."
```

### A/B testing policy versions

```python
diff = compare("policy_v2.rrp", "policy_v3.rrp")
print(f"Divergence at step {diff.divergence_step}")
print(f"Reward change: {diff.channel_diffs['reward'].mean_change_pct:+.1f}%")
```

### Batch fleet analysis

```python
from pathlib import Path
for rrp in Path("fleet_data/").glob("*.rrp"):
    result = diagnose(rrp)
    if result.has_failures:
        export_html(rrp, output=f"reports/{rrp.stem}.html")
```

### Overnight experiment safety net

```python
with Recorder("overnight_run", path="/data/experiments/") as rec:
    for step in range(1_000_000):
        rec.step(state=state, action=action)
        # Crash at step 500,000? You still have 499,900+ steps safely on disk.
```

### Sim-to-real gap analysis

```python
diff = compare("sim_run.rrp", "real_run.rrp")
# See exactly which channels diverge and by how much between sim and real
```

### Imitation learning dataset

```python
for demo_id in range(100):
    with Recorder(f"demo_{demo_id:03d}", path="dataset/") as rec:
        while demonstrating:
            rec.step(observation=obs, action=human_action)
# Then: export_csv for PyTorch dataloaders
```

### CI regression testing

```python
def test_grasp_succeeds():
    with Recorder("test_grasp", path=tmp / "test.rrp") as rec:
        run_grasp_policy(rec)
    result = diagnose(tmp / "test.rrp")
    assert not result.has_failures
```
