# Quickstart

## Record Your First Session

```python
import numpy as np
from roboreplay import Recorder

with Recorder("my_first_recording", metadata={"robot": "my_robot"}) as rec:
    for i in range(100):
        rec.step(
            position=np.random.randn(3),
            velocity=np.random.randn(3),
            reward=float(i) / 100,
        )
        if i == 50:
            rec.mark_event("halfway", {"note": "midpoint reached"})
```

This creates `my_first_recording.rrp` in the current directory.

## Replay and Inspect

```python
from roboreplay import Replay

r = Replay("my_first_recording.rrp")
print(r)                    # Summary
print(r.channels)           # ['position', 'velocity', 'reward']
print(r[50])                # Data at step 50
print(r[40:60])             # Slice of data
print(r.channel("reward"))  # All reward values
r.close()
```

## Run Diagnosis

```python
from roboreplay.diagnose import diagnose

result = diagnose("my_first_recording.rrp")
print(result.summary)
print(f"Failures: {len(result.failures)}")
print(f"Warnings: {len(result.warnings)}")
```

## Compare Two Runs

```python
from roboreplay import compare

diff = compare("run_a.rrp", "run_b.rrp")
print(diff.summary())
print(f"Divergence at step: {diff.divergence_step}")
```

## Export

```python
from roboreplay.export import export_csv, export_html

export_csv("my_first_recording.rrp")    # Creates CSV files
export_html("my_first_recording.rrp")   # Creates interactive HTML
```

## Use the CLI

```bash
roboreplay info my_first_recording.rrp
roboreplay diagnose my_first_recording.rrp
roboreplay export my_first_recording.rrp --format csv
roboreplay export my_first_recording.rrp --format html
```
