# RoboReplay

**The DVR for robot behavior — record, replay, diagnose, and share robot execution data.**

<!-- TODO: Replace with actual demo GIF -->
<p align="center">
  <em>[ demo GIF coming soon — record, diagnose, compare in one workflow ]</em>
</p>

```bash
pip install roboreplay
```

Framework-agnostic. Works with MuJoCo, Isaac Sim, ROS2, Gymnasium, PyBullet, real hardware — anything with a Python API.

---

## Quick Start

### Record

```python
from roboreplay import Recorder

with Recorder("experiment_042", metadata={"robot": "panda", "task": "pick_place"}) as rec:
    for step in range(episode_length):
        obs, reward, done, info = env.step(action)
        rec.step(state=obs, action=action, reward=reward, gripper_force=info["force"])

        if done:
            rec.mark_event("episode_end", {"success": reward > 0})
            break
# Saved to experiment_042.rrp
```

### Replay

```python
from roboreplay import Replay

r = Replay("experiment_042.rrp")
print(r)                               # Summary: channels, steps, events
frame = r[487]                         # All channels at step 487
chunk = r[450:520]                     # Slice a range
force = r.channel("gripper_force")     # One channel, all steps
r.plot("gripper_force")                # matplotlib figure with events marked
```

### Diagnose

```python
from roboreplay.diagnose import diagnose

result = diagnose("experiment_042.rrp")
for failure in result.failures:
    print(f"[step {failure.step}] {failure.description}")
# [step 200] gripper_force dropped 85% at step 200 (from 7.94 -> 1.20)

# Optional: LLM-powered analysis with Claude
result = diagnose("experiment_042.rrp", use_llm=True)
print(result.llm_result.explanation)      # Natural-language failure analysis
print(result.llm_result.root_causes)      # Likely causes
print(result.llm_result.recommendations)  # Actionable fixes
```

### CLI

```bash
roboreplay info experiment_042.rrp                  # Pretty-printed summary
roboreplay diagnose experiment_042.rrp              # Anomaly detection report
roboreplay diagnose experiment_042.rrp --llm        # + LLM analysis
roboreplay compare run_a.rrp run_b.rrp              # Side-by-side comparison
roboreplay export experiment_042.rrp --format csv   # CSV export
roboreplay export experiment_042.rrp --format html  # Interactive HTML viewer
roboreplay plot experiment_042.rrp -c gripper_force # Channel plot
```

---

## Features

**Recording** — One-line `rec.step(...)`, schema inference, event marking, thread-safe, crash-safe streaming writes

**Replay** — Random-access indexing/slicing, channel queries, matplotlib plotting with event annotations

**Diagnosis** — Automatic anomaly detection (drops, spikes, flatlines) + optional LLM-powered analysis via Claude

**Comparison** — Side-by-side recording diff with divergence detection and per-channel stats

**Export** — CSV (flattened channels, events, metadata) and HTML (self-contained interactive viewer with Chart.js)

**Gymnasium** — `roboreplay.wrap(env)` for automatic recording of observations, actions, and rewards

**CLI** — Rich terminal output: info, diagnose, compare, export, plot commands

**Storage** — `.rrp` files (HDF5 + gzip). Compact, random-access, self-contained, portable, streamable.

---

## Installation

```bash
pip install roboreplay               # Core (record, replay, compare, diagnose, export)
pip install roboreplay[viz]          # + matplotlib plotting
pip install roboreplay[gym]          # + Gymnasium wrapper
pip install roboreplay[diagnose]     # + LLM diagnosis (Claude API)
pip install roboreplay[mujoco]       # + MuJoCo bindings
pip install roboreplay[all]          # Everything
```

**Requirements:** Python 3.10+. Core depends only on numpy, h5py, click, rich, and pydantic.

---

## Learn More

| Topic | Link |
|---|---|
| Full quickstart (compare, export, gym) | [Quickstart Guide](https://gow.github.io/roboreplay/getting-started/quickstart/) |
| Recording guide | [Recording](https://gow.github.io/roboreplay/guide/recording/) |
| Diagnosis guide | [Diagnosis](https://gow.github.io/roboreplay/guide/diagnosis/) |
| Use cases & workflows | [Use Cases](https://gow.github.io/roboreplay/use-cases/) |
| API reference | [API Docs](https://gow.github.io/roboreplay/api/recorder/) |
| CLI reference | [CLI](https://gow.github.io/roboreplay/cli/) |

---

## Examples

```bash
python examples/simulated_pick_place.py   # 7-DOF pick-and-place with grasp slip failure
python examples/custom_robot.py           # 6-DOF arm, success vs failure comparison
python examples/batch_analysis.py         # 5 recordings, batch diagnosis, CSV export
python examples/gymnasium_cartpole.py     # CartPole with automatic recording
python examples/mujoco_panda.py           # MuJoCo Panda arm reaching task
```

---

## Contributing

Contributions welcome! Easy on-ramps:

- **New anomaly detectors** — add detection rules in `roboreplay/diagnose/`
- **New export formats** — add exporters in `roboreplay/export/`
- **New examples** — show RoboReplay with your favorite simulator
- **Bug reports** — file issues with a `.rrp` file if possible

```bash
git clone https://github.com/harrey401/roboreplay
cd roboreplay
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
