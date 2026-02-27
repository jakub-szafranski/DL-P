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
uv sync
```

That's it! The `uv sync` command will:
- Create a `.venv` virtual environment (if it doesn't exist)
- Generate/update `uv.lock` (lockfile with exact versions)
- Install all dependencies from the lockfile
> **Note:** When using `uv run`, activation is not required - uv automatically uses the project's virtual environment.

### Running the project

**SimCLR only:**
```bash
uv run torchrun --nproc_per_node=4 run_simclr.py
```

**SimCLR + SoftMatch:**
```bash
uv run torchrun --nproc_per_node=4 run_simclr_softmatch.py
```

**SimCLR linear eval + fine tuning on saved models:**
```bash
uv run torchrun --nproc_per_node=4 simclr_eval.py
```

### Adding new dependencies

Add the package to `pyproject.toml` under `[project.dependencies]`, then run `uv sync`.

## Configuration

Edit config files to adjust training parameters.