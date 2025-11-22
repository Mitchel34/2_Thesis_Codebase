# Contributing Guidelines

Thanks for helping improve the Watauga residual-correction toolkit! These conventions keep the
repository reproducible and pleasant to collaborate on.

## Getting Started

1. Create a Python 3.11 virtual environment and install dependencies:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Export the repository root to `PYTHONPATH` so scripts can discover shared modules:
   ```bash
   export PYTHONPATH="$(pwd)"
   ```
3. For data-acquisition work, populate the credentials listed in `README.md` (Copernicus, optional
   AWS profile, etc.).

## Branching & Pull Requests

- Create feature branches from `master` and keep pull requests focused on a single change set.
- Update documentation (`README.md`, `docs/*.md`, notebooks) whenever you add new capabilities or
  adjust workflows.
- Reference relevant issues or TODOs in your PR description.

## Testing & Quality Checks

- Run `pytest --maxfail=1 --disable-warnings -q` before opening a PR. The CI workflow mirrors this
  command on every push.
- Keep smoke tests lightweight—heavy data downloads or training runs should remain opt-in scripts.
- When adding dependencies, update `requirements.txt` and explain why the package is needed.
- Install the repo's pre-commit hooks to automatically format/lint files:
  ```bash
  pre-commit install
  pre-commit run --all-files  # optional before committing
  ```

## Code Style

- Prefer readable, well-documented functions. Use docstrings for public helpers and include
  high-level comments for complex logic (avoid inline restatements of trivial statements).
- Follow standard Python formatting (Black-compatible) and type annotate new utility functions when
  practical.
- Keep LaTeX and Markdown files wrapped at ~100 characters to simplify diff reviews.

## Data & Artefacts

- Large artefacts (Hydra sweeps, checkpoints, results) belong under `local_only/`. Never commit raw
  data or long-lived binaries—describe regeneration steps instead.
- If you introduce new generated directories, extend `.gitignore` accordingly and document the
  policy in `local_only/README.md`.

## Questions?

Open a discussion or ping Mitchel in the thesis workspace when in doubt. Clear, reproducible
contributions mean we can spend more time on science and less on repo archaeology.
