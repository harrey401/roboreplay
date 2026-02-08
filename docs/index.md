# RoboReplay

**The DVR for robot behavior** — record, replay, diagnose, and share robot execution data.

RoboReplay is a lightweight, framework-agnostic Python library that works with MuJoCo, Isaac Sim, ROS2, Gymnasium, real hardware — anything with a Python API.

## Features

- **Record** — Stream robot data to compressed `.rrp` files with automatic schema inference
- **Replay** — Random-access indexing, slicing, and channel queries
- **Diagnose** — Statistical anomaly detection (drops, spikes, flatlines) + optional LLM-powered analysis via Claude
- **Compare** — Side-by-side recording diff with divergence detection
- **Export** — CSV files and self-contained interactive HTML viewer with Chart.js
- **Gymnasium Wrapper** — `roboreplay.wrap(env)` for automatic recording of observations, actions, and rewards
- **Rich CLI** — `info`, `diagnose`, `compare`, `export`, `plot` commands with beautiful terminal output

## Quick Start

```python
from roboreplay import Recorder, Replay

# Record
with Recorder("my_experiment", metadata={"robot": "panda"}) as rec:
    for step in range(100):
        rec.step(state=get_state(), action=get_action(), reward=get_reward())

# Replay
r = Replay("my_experiment.rrp")
print(r)            # Summary
print(r[50])        # Data at step 50
print(r.channels)   # Channel names
```

## Installation

```bash
pip install roboreplay                  # Core
pip install roboreplay[viz]             # + matplotlib plotting
pip install roboreplay[gym]             # + Gymnasium wrapper
pip install roboreplay[diagnose]        # + LLM diagnosis (Claude API)
pip install roboreplay[all]             # Everything
```

See the [Installation guide](getting-started/installation.md) for full details.

## Why RoboReplay?

- **Minimal dependencies** — Core: numpy, h5py, click, rich, pydantic
- **Framework-agnostic** — Pass numpy arrays, we store them
- **Streaming writes** — Never accumulates full episodes in RAM
- **Offline-first** — Everything works without internet
- **64 tests passing** — Comprehensive test coverage across all features

## Examples

```bash
python examples/simulated_pick_place.py   # 7-DOF pick-and-place with grasp slip failure
python examples/custom_robot.py           # 6-DOF arm, success vs failure comparison
python examples/batch_analysis.py         # 5 recordings, batch diagnosis, CSV export
python examples/gymnasium_cartpole.py     # CartPole with automatic recording (needs roboreplay[gym])
```
