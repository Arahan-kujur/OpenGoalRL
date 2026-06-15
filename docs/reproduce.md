# Reproduce Our Benchmarks

The goal of this page: **let a stranger reproduce a number from the README in
under 10 minutes.** Pick whichever path matches your machine.

## Option A — Colab (zero local setup)

Open the quickstart notebook in Google Colab, run all cells, and watch an agent
learn to score on `empty_goal_close` with an inline learning curve.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arahan-kujur/OpenGoalRL/blob/main/notebooks/quickstart.ipynb)

## Option B — Docker (one command, reproducible Linux env)

The Docker image bundles all GRF system dependencies, removing the #1
onboarding failure on non-Linux machines.

```bash
make docker-build
make docker-train      # trains empty_goal_close inside the container
```

Or directly:

```bash
docker build -t opengoalrl .
docker run --rm opengoalrl \
    python -m opengoalrl.scripts.train --config opengoalrl/configs/empty_goal_close.yaml
```

## Option C — Native + benchmark harness

With GRF installed (see [Installation](installation.md)):

```bash
python benchmarks/run_benchmarks.py --seeds 3 && python benchmarks/aggregate.py
```

This trains seeded runs, evaluates them, and writes
`benchmarks/results/summary.csv` with mean and 95% confidence intervals. See
[Benchmarks](benchmarks.md) for the full schema.

## Pretrained checkpoints (skip training)

Don't want to train at all? Download a pretrained checkpoint and run the
analysis tools directly:

```bash
opengoalrl-download-models --scenario empty_goal_close
opengoalrl-diagnose models/zoo/empty_goal_close --config opengoalrl/configs/empty_goal_close.yaml
```

See the [Model Zoo](https://github.com/Arahan-kujur/OpenGoalRL/blob/main/MODEL_ZOO.md)
for the full checkpoint table.

## Makefile shortcuts

| Target              | What it does                                 |
|---------------------|----------------------------------------------|
| `make test-fast`    | GRF-free test suite                          |
| `make test`         | Full test suite (requires GRF)               |
| `make benchmark`    | Run harness + aggregate to `summary.csv`     |
| `make docker-build` | Build the reproducible Docker image          |
| `make docker-train` | Train `empty_goal_close` inside Docker       |
| `make docs`         | Build the docs site (`mkdocs build --strict`)|
| `make docs-serve`   | Serve the docs locally                       |
