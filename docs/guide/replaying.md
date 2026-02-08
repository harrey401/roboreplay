# Replaying

The `Replay` class provides random-access reading and navigation of recordings.

## Opening a Recording

```python
from roboreplay import Replay

r = Replay("experiment.rrp")
print(r)  # Human-readable summary
```

## Properties

```python
r.name          # Recording name
r.robot         # Robot name
r.task          # Task name
r.num_steps     # Total steps
r.channels      # List of channel names
r.events        # Event log
r.stats         # Per-channel statistics
r.metadata      # Full metadata
```

## Indexing and Slicing

```python
# Single step → dict of arrays
frame = r[50]
print(frame["position"])  # array at step 50

# Negative indexing
last = r[-1]

# Slicing → dict of arrays
chunk = r[40:60]
print(chunk["position"].shape)  # (20, 3)
```

## Channel Access

```python
# Full channel
all_pos = r.channel("position")  # shape: (num_steps, 3)

# Partial range
subset = r.channel("position", start=100, end=200)
```

## Events

```python
for event in r.events.events:
    print(f"Step {event.step}: {event.event_type} {event.data}")

# Filter by type
failures = r.events.where("failure")
```

## Plotting

Requires `matplotlib`: `pip install roboreplay[viz]`

```python
fig = r.plot("gripper_force")        # Returns matplotlib Figure
fig = r.plot("position", start=100, end=200)  # Partial range
```

## Context Manager

```python
with Replay("experiment.rrp") as r:
    print(r.num_steps)
# Automatically closed
```
