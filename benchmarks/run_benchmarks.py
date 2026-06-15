"""Seeded benchmark harness for OpenGoalRL.

For each ``(config, seed)`` pair this trains an agent, evaluates it, and writes
the per-seed episode CSV to ``benchmarks/results/<scenario>/seed_<n>.csv`` plus
a ``meta.json`` with the algorithm and timestep budget. ``aggregate.py`` then
turns those into a published ``summary.csv``.

This script *runs GRF* (training + evaluation), so it is meant to be run once by
a maintainer on Linux/GPU -- it is intentionally not exercised in CI. The
GRF-free half (``aggregate.py``) is unit-tested in ``test_benchmarks.py``.

Reproduce::

    python benchmarks/run_benchmarks.py --seeds 3
    python benchmarks/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from opengoalrl.scripts import evaluate as evaluate_script
from opengoalrl.scripts import train as train_script

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGS = [
    BENCH_DIR / "configs" / "empty_goal_close.yaml",
    BENCH_DIR / "configs" / "corner.yaml",
]


def _write_seed_config(base_config: dict, seed: int, save_dir: Path) -> Path:
    """Materialise a per-seed config with overridden seed + save_dir."""
    cfg = json.loads(json.dumps(base_config))  # deep copy via round-trip
    cfg.setdefault("training", {})["seed"] = seed
    cfg.setdefault("logging", {})["save_dir"] = str(save_dir)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_seed{seed}.yaml", delete=False
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_one(config_path: Path, seed: int, results_dir: Path) -> Path:
    """Train + evaluate one seed and write its per-episode CSV."""
    base_config = yaml.safe_load(config_path.read_text())
    scenario = base_config.get("environment", {}).get("scenario", config_path.stem)

    scenario_dir = results_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    save_dir = scenario_dir / f"seed_{seed}_run"

    seed_config = _write_seed_config(base_config, seed, save_dir)
    try:
        train_script.main(["--config", str(seed_config)])

        model_path = save_dir / "ppo_opengoalrl"
        seed_csv = scenario_dir / f"seed_{seed}.csv"
        evaluate_script.main([
            "--config", str(seed_config),
            "--model", str(model_path),
            "--output", str(seed_csv),
        ])
    finally:
        seed_config.unlink(missing_ok=True)

    meta = {
        "algorithm": base_config.get("training", {}).get("algorithm", "ppo"),
        "timesteps": base_config.get("training", {}).get("total_timesteps", 0),
        "config": config_path.name,
    }
    (scenario_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return seed_csv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run OpenGoalRL seeded benchmarks")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[str(p) for p in DEFAULT_CONFIGS],
        help="Benchmark config YAML files",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per config (uses seeds 0..N-1)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(BENCH_DIR / "results"),
        help="Where to write per-seed CSVs",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    for config_str in args.configs:
        config_path = Path(config_str)
        for seed in range(args.seeds):
            print(f"=== {config_path.name} | seed {seed} ===")
            out = run_one(config_path, seed, results_dir)
            print(f"  wrote {out}")

    print("\nDone. Aggregate with: python benchmarks/aggregate.py")


if __name__ == "__main__":
    main()
