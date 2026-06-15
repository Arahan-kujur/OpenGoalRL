# OpenGoalRL Benchmarks

Reproducible, seeded benchmarks for OpenGoalRL scenarios. The harness trains an
agent per seed, evaluates it, and aggregates results into a published
`summary.csv` with mean and 95% confidence intervals.

## Reproduce (one command path)

> Requires GRF system dependencies (Linux/WSL) and the full install
> (`pip install -e .`). See the project README for the system packages.

```bash
python benchmarks/run_benchmarks.py --seeds 3 && python benchmarks/aggregate.py
```

This writes:

- `benchmarks/results/<scenario>/seed_<n>.csv` — per-seed evaluation episodes
- `benchmarks/results/<scenario>/meta.json` — algorithm + timestep budget
- `benchmarks/results/summary.csv` — aggregated mean / 95% CI table

## What runs

| Step                   | Script                          | Needs GRF? |
|------------------------|---------------------------------|------------|
| Train + evaluate seeds | `benchmarks/run_benchmarks.py`  | Yes        |
| Aggregate to summary   | `benchmarks/aggregate.py`       | No         |

The aggregation step is pure-Python (numpy only) and is unit-tested in CI via
`opengoalrl/tests/test_benchmarks.py`, so the schema stays verified even though
the training half cannot run in CI.

## Configs

Standardized, seeded configs live in `benchmarks/configs/`:

- `empty_goal_close.yaml` — trivial scoring scenario (50K steps)
- `corner.yaml` — hard corner-kick set piece (200K steps)

Pass your own with `--configs path/to/a.yaml path/to/b.yaml`.

## Schema

See [`results/SCHEMA.md`](results/SCHEMA.md) for the full column documentation.
`results/summary.template.csv` shows the structure before real numbers exist.

## Aggregate only (no training)

If you already have per-seed CSVs (e.g. from `opengoalrl-eval`), drop them under
`benchmarks/results/<scenario>/seed_<n>.csv` and run:

```bash
python benchmarks/aggregate.py --results-dir benchmarks/results --output benchmarks/results/summary.csv
```
