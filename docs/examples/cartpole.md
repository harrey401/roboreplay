# CartPole Example

Demonstrates using `roboreplay.wrap()` with Gymnasium's CartPole environment.

## Requirements

```bash
pip install roboreplay[gym]
```

## Run

```bash
python examples/gymnasium_cartpole.py
```

## What It Does

1. Wraps `CartPole-v1` with `roboreplay.wrap()`
2. Runs 5 episodes with random actions
3. Replays and inspects the recording
4. Runs diagnosis

## Recorded Channels

- `observation` — Cart position, velocity, pole angle, angular velocity
- `action` — Discrete action (left/right, stored as float)
- `reward` — Step reward (always 1.0 for CartPole)

## Automatic Events

- `episode_reset` — On each `reset()` call
- `episode_terminated` — On each episode termination
- `recording_end` — On `close()`, with total counts
