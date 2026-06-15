"""Unit tests for the benchmark aggregation -- no GRF required.

Feeds synthetic per-seed episode CSVs into ``benchmarks/aggregate.py`` and
asserts the mean / 95% CI math and the published ``summary.csv`` schema.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

# benchmarks/ lives at the repo root (it is not part of the installed package),
# so make sure the repo root is importable regardless of how pytest is invoked.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import aggregate  # noqa: E402


def _write_seed_csv(path: Path, rewards: list[float], goals: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "goals", "shots", "ball_in_box"])
        for i, (reward, goal) in enumerate(zip(rewards, goals), start=1):
            writer.writerow([i, reward, 100, goal, 0, 0])


def test_seed_scoring_rate():
    episodes = [{"goals": 1.0}, {"goals": 0.0}, {"goals": 2.0}, {"goals": 0.0}]
    assert aggregate.seed_scoring_rate(episodes) == pytest.approx(50.0)


def test_seed_scoring_rate_empty():
    assert aggregate.seed_scoring_rate([]) == 0.0


def test_seed_mean_reward():
    episodes = [{"reward": 2.0}, {"reward": 4.0}]
    assert aggregate.seed_mean_reward(episodes) == pytest.approx(3.0)


def test_confidence_interval_single_sample_is_zero():
    assert aggregate.confidence_interval_95([42.0]) == 0.0


def test_confidence_interval_matches_normal_approx():
    values = [40.0, 60.0]
    # std(ddof=1) of [40,60] = 14.142..., CI = 1.96 * std / sqrt(2)
    expected = 1.96 * (20.0 / math.sqrt(2)) / math.sqrt(2)
    assert aggregate.confidence_interval_95(values) == pytest.approx(expected)


def test_aggregate_scenario_math(tmp_path):
    scenario = tmp_path / "empty_goal_close"
    # seed 1: 100% scoring (both episodes score), mean reward 5
    _write_seed_csv(scenario / "seed_1.csv", rewards=[4.0, 6.0], goals=[1, 1])
    # seed 2: 50% scoring, mean reward 3
    _write_seed_csv(scenario / "seed_2.csv", rewards=[2.0, 4.0], goals=[1, 0])
    (scenario / "meta.json").write_text(
        json.dumps({"algorithm": "ppo", "timesteps": 50000})
    )

    row = aggregate.aggregate_scenario(scenario)

    assert row["scenario"] == "empty_goal_close"
    assert row["algorithm"] == "ppo"
    assert row["seeds"] == 2
    assert row["timesteps"] == 50000
    assert row["scoring_rate_mean"] == pytest.approx(75.0)  # mean(100, 50)
    assert row["mean_reward"] == pytest.approx(4.0)  # mean(5, 3)
    # CI across [100, 50]: 1.96 * std(ddof=1)/sqrt(2)
    expected_ci = 1.96 * (50.0 / math.sqrt(2)) / math.sqrt(2)
    assert row["scoring_rate_ci"] == pytest.approx(expected_ci, abs=1e-3)
    # sample efficiency: 75% per 0.05M steps = 1500
    assert row["sample_efficiency"] == pytest.approx(75.0 / 0.05, abs=1e-3)


def test_aggregate_scenario_without_meta_defaults(tmp_path):
    scenario = tmp_path / "corner"
    _write_seed_csv(scenario / "seed_1.csv", rewards=[1.0], goals=[0])
    row = aggregate.aggregate_scenario(scenario)
    assert row["algorithm"] == aggregate.DEFAULT_ALGORITHM
    assert row["timesteps"] == 0
    assert row["sample_efficiency"] == 0.0


def test_aggregate_scenario_empty_returns_none(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert aggregate.aggregate_scenario(empty) is None


def test_aggregate_results_and_summary_columns(tmp_path):
    results = tmp_path / "results"
    _write_seed_csv(results / "a" / "seed_1.csv", rewards=[1.0], goals=[1])
    _write_seed_csv(results / "b" / "seed_1.csv", rewards=[2.0], goals=[0])

    rows = aggregate.aggregate_results(results)
    assert [r["scenario"] for r in rows] == ["a", "b"]

    out = aggregate.write_summary(rows, results / "summary.csv")
    assert out.exists()
    with open(out, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)
    assert header == aggregate.SUMMARY_COLUMNS
    assert len(data_rows) == 2


def test_aggregate_results_missing_dir_returns_empty(tmp_path):
    assert aggregate.aggregate_results(tmp_path / "does_not_exist") == []
