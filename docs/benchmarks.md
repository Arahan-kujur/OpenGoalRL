# Benchmarks

OpenGoalRL ships a seeded, reproducible benchmark harness. A maintainer runs it
once on Linux/GPU and commits `benchmarks/results/summary.csv`; the aggregation
step is GRF-free and unit-tested in CI.

## Reproduce

```bash
python benchmarks/run_benchmarks.py --seeds 3 && python benchmarks/aggregate.py
```

| Step                   | Script                          | Needs GRF? |
|------------------------|---------------------------------|------------|
| Train + evaluate seeds | `benchmarks/run_benchmarks.py`  | Yes        |
| Aggregate to summary   | `benchmarks/aggregate.py`       | No         |

## Summary schema

`aggregate.py` writes `benchmarks/results/summary.csv` with these columns:

| Column              | Meaning                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `scenario`          | Scenario name (matches the results subdirectory)                        |
| `algorithm`         | Training algorithm (default `ppo`)                                      |
| `seeds`             | Number of seeds aggregated                                              |
| `timesteps`         | Training budget per seed                                                |
| `scoring_rate_mean` | Mean across seeds of the per-seed scoring rate (% episodes with a goal) |
| `scoring_rate_ci`   | 95% CI half-width (normal approx; `0.0` for `<2` seeds)                 |
| `mean_reward`       | Mean across seeds of the per-seed mean episode reward                   |
| `sample_efficiency` | `scoring_rate_mean` per 1M timesteps                                    |

The 95% confidence interval uses a normal approximation:

```
scoring_rate_ci = 1.96 * std(scoring_rates, ddof=1) / sqrt(n_seeds)
```

See `benchmarks/results/SCHEMA.md` in the repository for the per-seed CSV layout
and `benchmarks/README.md` for the full reproduce guide.
