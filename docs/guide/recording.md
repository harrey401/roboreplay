# Recording

The `Recorder` class is the main interface for capturing robot execution data.

## Basic Usage

```python
from roboreplay import Recorder
import numpy as np

rec = Recorder("experiment_001", metadata={"robot": "panda", "task": "pick_place"})
rec.start()

for step in range(1000):
    state = get_state()       # Your robot's state
    action = get_action()     # Your policy output
    reward = get_reward()

    rec.step(state=state, action=action, reward=reward)

rec.save()
```

## Context Manager

The recommended pattern uses a context manager for automatic save:

```python
with Recorder("experiment_001") as rec:
    for step in range(1000):
        rec.step(state=state, action=action, reward=reward)
# Automatically saved on exit
```

## Schema Inference

RoboReplay infers the schema from your first `step()` call. Subsequent calls must provide the same channel names with matching shapes:

```python
with Recorder("demo") as rec:
    # First call defines the schema
    rec.step(pos=np.array([1.0, 2.0, 3.0]), vel=np.array([0.1]))

    # Must match: pos is (3,), vel is (1,)
    rec.step(pos=np.array([1.1, 2.1, 3.1]), vel=np.array([0.2]))
```

## Events

Mark notable moments during recording:

```python
rec.mark_event("grasp_start", {"force": 5.0})
rec.mark_event("failure", {"type": "slip", "force_at_fail": 1.2})
rec.mark_event("episode_end", {"success": True})
```

## Data Types

Values are automatically converted to `float32` numpy arrays:

```python
rec.step(
    position=np.array([1.0, 2.0, 3.0]),  # numpy array
    reward=0.5,                            # Python float → array([0.5])
    flag=1,                                # Python int → array([1.0])
)
```

## Metadata

Pass metadata as a dict. `robot` and `task` are special fields:

```python
rec = Recorder("exp", metadata={
    "robot": "panda",          # Stored as top-level field
    "task": "pick_place",      # Stored as top-level field
    "scene": "tabletop",       # Stored in user_metadata
    "policy_version": "v2.1",  # Stored in user_metadata
})
```
