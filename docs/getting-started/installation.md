# Installation

## Basic Install

```bash
pip install roboreplay
```

This installs core functionality: recording, replaying, diagnosis, comparison, and CLI.

## Optional Dependencies

RoboReplay has several optional dependency groups:

```bash
# Visualization (matplotlib)
pip install roboreplay[viz]

# Gymnasium wrapper
pip install roboreplay[gym]

# LLM-powered diagnosis (requires Anthropic API key)
pip install roboreplay[diagnose]

# Documentation tools
pip install roboreplay[docs]

# Development (testing, linting, type checking)
pip install roboreplay[dev]

# Everything
pip install roboreplay[all]
```

## Development Setup

```bash
git clone https://github.com/gow/roboreplay.git
cd roboreplay
pip install -e ".[dev]"
pytest tests/ -v
```

## Requirements

- Python 3.10+
- NumPy >= 1.24
- h5py >= 3.9
- Click >= 8.0
- Rich >= 13.0
- Pydantic >= 2.0
