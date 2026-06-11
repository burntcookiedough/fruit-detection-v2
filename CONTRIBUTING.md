# Contributing

Thank you for your interest in contributing to **fruit-detector**!

## Development Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/anshu/fruit-detection-v2.git
cd fruit-detection-v2
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Quality

We use automated tools to maintain code quality:

| Tool | Purpose | Run |
|------|---------|-----|
| **Ruff** | Linting + formatting | `ruff check src/` / `ruff format src/` |
| **Mypy** | Type checking | `mypy src/fruit_detector/` |
| **Pytest** | Testing | `pytest tests/ -v` |

All of these run automatically via pre-commit hooks on each commit.

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Make your changes following the existing code style
3. Add or update tests for any new functionality
4. Run the full test suite: `pytest tests/ -v`
5. Run linting: `ruff check src/ && ruff format --check src/`
6. Submit a PR with a clear description of the changes

## Architecture Overview

```
src/fruit_detector/
├── config.py       # Environment-driven configuration
├── model/          # Backbone, neck, heads, detector
├── ops/            # Anchor points, loss, assignment, inference
├── data/           # Dataset and augmentations
├── engine/         # Training loop and validation
├── utils/          # Checkpoint I/O, TTA, visualization
└── cli/            # CLI entry points (train, infer, webcam, etc.)
```

See [docs/architecture.md](docs/architecture.md) for a deeper dive.

## Reporting Issues

Please include:
- Python version and OS
- PyTorch/CUDA version
- Full error traceback
- Steps to reproduce
