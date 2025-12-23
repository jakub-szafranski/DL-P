# SimCLR

Adaptation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
https://arxiv.org/pdf/2002.05709

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### 1. Install uv

```bash
# Homebrew
brew install uv

# pip (in any Python environment)
pip install uv
```

### 2. Create virtual environment and install dependencies

```bash
# Create venv and sync dependencies (uv.lock will be created automatically)
uv sync
```

That's it! The `uv sync` command will:
- Create a `.venv` virtual environment (if it doesn't exist)
- Generate/update `uv.lock` (lockfile with exact versions)
- Install all dependencies from the lockfile
> **Note:** When using `uv run`, activation is not required - uv automatically uses the project's virtual environment.

### Running the project

```bash
uv run torchrun --nproc_per_node=4 run.py
```

### Adding new dependencies

Add the package to `pyproject.toml` under `[project.dependencies]`, then run `uv sync`.

## Configuration

Edit `config.yaml` to adjust training parameters.

## Project Structure

```
├── config.yaml          # Training configuration
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock              # Locked dependency versions (auto-generated)
├── run.py               # Main training script
├── sim_clr/             # SimCLR implementation
│   ├── encoder.py       # Encoder architectures
│   ├── lars.py          # LARS optimizer
│   └── sim_clr.py       # SimCLR model and NT-Xent loss
└── utils/               # Utilities
    ├── config.py        # Config loader
    ├── data.py          # Dataset utilities
    ├── distributed.py   # torch.distributed utilities
    └── fine_tuning.py   # Fine-tuning evaluation
```
