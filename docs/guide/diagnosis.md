# Diagnosis

RoboReplay provides both statistical and LLM-powered failure analysis.

## Statistical Diagnosis

```python
from roboreplay.diagnose import diagnose

result = diagnose("recording.rrp")

print(result.summary)           # Human-readable summary
print(result.failures)          # High-severity anomalies (severity > 0.5)
print(result.warnings)          # Medium-severity anomalies (0.2 - 0.5)
print(result.has_failures)      # Quick check
```

## Anomaly Types

### Sudden Drops
Detects when a signal drops significantly from its recent average.

### Spikes
Detects outlier values that exceed several standard deviations from the rolling mean.

### Flatlines
Detects periods where a previously-varying signal goes completely flat.

## Tuning Thresholds

```python
result = diagnose(
    "recording.rrp",
    drop_threshold=0.5,      # Sensitivity for drops (0-1)
    spike_threshold=3.0,     # Std devs for spikes
    flatline_duration=20,    # Min steps for flatlines
)
```

## LLM-Powered Diagnosis

For deeper analysis, enable LLM diagnosis (requires `pip install roboreplay[diagnose]`):

```python
result = diagnose("recording.rrp", use_llm=True)

if result.llm_result:
    print(result.llm_result.explanation)
    print(result.llm_result.root_causes)
    print(result.llm_result.recommendations)
```

Set your API key via environment variable or parameter:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
result = diagnose("recording.rrp", use_llm=True, api_key="sk-ant-...")
```

## CLI

```bash
roboreplay diagnose recording.rrp
roboreplay diagnose recording.rrp --llm
roboreplay diagnose recording.rrp --llm --api-key sk-ant-...
roboreplay diagnose recording.rrp --drop-threshold 0.3 --spike-threshold 5.0
```
