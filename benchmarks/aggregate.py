"""Aggregate per-seed benchmark CSVs into a published-results summary.

This module is intentionally import-light (standard library + numpy only) so it
stays GRF-free and unit-testable in CI. It reads the per-seed episode CSVs that
``run_benchmarks.py`` (or ``opengoalrl-eval``) writes and produces
``summary.csv`` with mean / 95% CI across seeds.

Input layout (produced by the harness)::

    benchmarks/results/<scenario>/seed_<n>.csv      # episode-level eval CSV
    benchmarks/results/<scenario>/meta.json         # optional run metadata

Each ``seed_<n>.csv`` uses the shared episode schema written by
``opengoalrl.utils.rollout.write_episode_csv``:
``episode, reward, steps, goals, shots, ball_in_box[, tactical...]``.

The optional ``meta.json`` carries non-episode metadata::

    {"algorithm": "ppo", "timesteps": 50000}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SUMMARY_COLUMNS = [
    "scenario",
    "algorithm",
    "seeds",
    "timesteps",
    "scoring_rate_mean",
    "scoring_rate_ci",
    "mean_reward",
    "sample_efficiency",
]

DEFAULT_ALGORITHM = "ppo"


def read_episode_csv(path: str | Path) -> list[dict[str, float]]:
    """Read an episode-level eval CSV into a list of float-coerced row dicts."""
    rows: list[dict[str, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, float] = {}
            for key, val in raw.items():
                if key is None:
                    continue
                try:
                    row[key] = float(val)
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


def seed_scoring_rate(episodes: list[dict[str, float]]) -> float:
    """Percentage of episodes that scored at least one goal."""
    if not episodes:
        return 0.0
    scored = sum(1 for ep in episodes if ep.get("goals", 0.0) > 0)
    return 100.0 * scored / len(episodes)


def seed_mean_reward(episodes: list[dict[str, float]]) -> float:
    if not episodes:
        return 0.0
    return float(np.mean([ep.get("reward", 0.0) for ep in episodes]))


def confidence_interval_95(values: list[float]) -> float:
    """Half-width of the 95% confidence interval (normal approximation).

    Returns 0.0 for fewer than two samples (CI is undefined / degenerate).
    """
    n = len(values)
    if n < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    return 1.96 * std / math.sqrt(n)


def _load_meta(scenario_dir: Path) -> dict:
    meta_path = scenario_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def aggregate_scenario(scenario_dir: Path) -> dict | None:
    """Aggregate all ``seed_*.csv`` files in one scenario directory."""
    seed_files = sorted(scenario_dir.glob("seed_*.csv"))
    if not seed_files:
        return None

    scoring_rates: list[float] = []
    mean_rewards: list[float] = []
    for seed_file in seed_files:
        episodes = read_episode_csv(seed_file)
        scoring_rates.append(seed_scoring_rate(episodes))
        mean_rewards.append(seed_mean_reward(episodes))

    meta = _load_meta(scenario_dir)
    algorithm = meta.get("algorithm", DEFAULT_ALGORITHM)
    timesteps = int(meta.get("timesteps", 0))

    scoring_rate_mean = float(np.mean(scoring_rates))
    scoring_rate_ci = confidence_interval_95(scoring_rates)
    mean_reward = float(np.mean(mean_rewards))

    # Scoring rate gained per 1M environment steps; 0 when timesteps unknown.
    sample_efficiency = (
        scoring_rate_mean / (timesteps / 1_000_000.0) if timesteps > 0 else 0.0
    )

    return {
        "scenario": scenario_dir.name,
        "algorithm": algorithm,
        "seeds": len(seed_files),
        "timesteps": timesteps,
        "scoring_rate_mean": round(scoring_rate_mean, 4),
        "scoring_rate_ci": round(scoring_rate_ci, 4),
        "mean_reward": round(mean_reward, 4),
        "sample_efficiency": round(sample_efficiency, 6),
    }


def aggregate_results(results_dir: str | Path) -> list[dict]:
    """Aggregate every scenario subdirectory under ``results_dir``."""
    root = Path(results_dir)
    rows: list[dict] = []
    if not root.exists():
        return rows
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        row = aggregate_scenario(scenario_dir)
        if row is not None:
            rows.append(row)
    return rows


def write_summary(rows: list[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})
    return out


def main(argv: list[str] | None = None) -> None:
    default_results = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description="Aggregate OpenGoalRL benchmark seeds")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(default_results),
        help="Directory containing <scenario>/seed_<n>.csv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write summary.csv (default: <results-dir>/summary.csv)",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_path = Path(args.output) if args.output else results_dir / "summary.csv"

    rows = aggregate_results(results_dir)
    write_summary(rows, out_path)
    print(f"Aggregated {len(rows)} scenario(s) -> {out_path}")
    for row in rows:
        print(
            f"  {row['scenario']:<20} "
            f"scoring_rate={row['scoring_rate_mean']:.1f}% "
            f"(+/-{row['scoring_rate_ci']:.1f}) over {row['seeds']} seed(s)"
        )


if __name__ == "__main__":
    main()
