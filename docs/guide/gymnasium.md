# Gymnasium Integration

RoboReplay integrates with Gymnasium via `roboreplay.wrap()` for automatic recording.

## Setup

```bash
pip install roboreplay[gym]
```

## Basic Usage

```python
import gymnasium as gym
import roboreplay

env = gym.make("CartPole-v1")
wrapped = roboreplay.wrap(env, name="cartpole_run")

obs, info = wrapped.reset()
for _ in range(1000):
    action = wrapped.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped.step(action)
    if terminated or truncated:
        obs, info = wrapped.reset()

wrapped.close()  # Saves recording to cartpole_run.rrp
```

## What Gets Recorded

The wrapper automatically records:

- **`observation`** — Environment observation at each step
- **`action`** — Action taken at each step
- **`reward`** — Reward received at each step

Events are automatically marked for:

- **`episode_reset`** — Each call to `reset()`
- **`episode_terminated`** — When `terminated=True`
- **`episode_truncated`** — When `truncated=True`
- **`recording_end`** — On `close()`, with total steps/episodes count

## Custom Metadata

```python
wrapped = roboreplay.wrap(
    env,
    name="experiment_42",
    path="data/experiment_42.rrp",
    metadata={
        "robot": "cartpole",
        "task": "balance",
        "policy": "random",
    },
)
```

## Environment Passthrough

The wrapper delegates all attribute access to the underlying environment:

```python
wrapped.action_space    # Works — delegated to env
wrapped.observation_space  # Works — delegated to env
wrapped.spec            # Works — delegated to env
```

## After Recording

```python
from roboreplay import Replay
from roboreplay.diagnose import diagnose

r = Replay("cartpole_run.rrp")
print(r)

result = diagnose("cartpole_run.rrp")
print(result.summary)
```
