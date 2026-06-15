"""Shared rollout and evaluation utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from opengoalrl.metrics.tactical import EpisodeTacticalMetrics, TacticalMetricsTracker


class ActionPolicy(Protocol):
    def __call__(self, obs: np.ndarray) -> int: ...


@dataclass
class EpisodeResult:
    episode: int
    reward: float
    steps: int
    goals: int
    shots: int
    ball_in_box: int
    tactical: EpisodeTacticalMetrics | None = None
    trajectory_summary: dict[str, Any] = field(default_factory=dict)


def extract_step_stats(info: dict[str, Any]) -> tuple[int, int, int]:
    """Return (goals_delta, shots_delta, ball_in_box_delta) from step info."""
    goals = 1 if info.get("score_reward", 0.0) > 0 else 0
    shots = 0
    ball_in_box = 0
    rc = info.get("reward_components", {})
    for key, val in rc.items():
        if "ShotReward" in key and val > 0:
            shots = 1
        if "BallInBoxReward" in key and val > 0:
            ball_in_box = 1
    return goals, shots, ball_in_box


def run_episode(
    env,
    policy: ActionPolicy,
    *,
    episode: int = 1,
    collect_tactical: bool = False,
    collect_trajectory: bool = False,
) -> EpisodeResult:
    """Run one episode and aggregate standard (+ optional tactical) metrics."""
    obs, info = env.reset()
    episode_reward = 0.0
    episode_goals = 0
    episode_shots = 0
    episode_ball_in_box = 0
    steps = 0
    terminated = truncated = False

    tracker = TacticalMetricsTracker() if collect_tactical else None
    trajectory: dict[str, Any] = {
        "actions": [],
        "ball_x": [],
        "ball_y": [],
        "possession_lost": [],
        "scenario": info.get("scenario", ""),
    }

    while not (terminated or truncated):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

        g, s, b = extract_step_stats(info)
        episode_goals += g
        episode_shots += s
        episode_ball_in_box += b

        if tracker is not None:
            tracker.step(obs, action, info)

        if collect_trajectory:
            trajectory["actions"].append(int(action))
            trajectory["ball_x"].append(float(obs[88]))
            trajectory["ball_y"].append(float(obs[89]))
            if info.get("possession_lost"):
                trajectory["possession_lost"].append(steps)

    tactical = tracker.finalize() if tracker is not None else None
    if collect_trajectory:
        trajectory["steps"] = steps
        trajectory["goals"] = episode_goals
        trajectory["reward"] = episode_reward

    return EpisodeResult(
        episode=episode,
        reward=episode_reward,
        steps=steps,
        goals=episode_goals,
        shots=episode_shots,
        ball_in_box=episode_ball_in_box,
        tactical=tactical,
        trajectory_summary=trajectory if collect_trajectory else {},
    )


def run_rollouts(
    env,
    policy: ActionPolicy,
    n_episodes: int,
    *,
    collect_tactical: bool = False,
    collect_trajectory: bool = False,
) -> list[EpisodeResult]:
    return [
        run_episode(
            env,
            policy,
            episode=ep,
            collect_tactical=collect_tactical,
            collect_trajectory=collect_trajectory,
        )
        for ep in range(1, n_episodes + 1)
    ]


BASIC_CSV_HEADER = [
    "episode", "reward", "steps", "goals", "shots", "ball_in_box",
]

TACTICAL_CSV_FIELDS = [
    "distance_advanced",
    "box_entries",
    "possession_losses",
    "max_possession_steps",
    "approx_xg",
    "pressure_proxy",
    "shot_timing_step",
]


def episode_to_row(result: EpisodeResult, *, include_tactical: bool = False) -> list:
    row = [
        result.episode,
        result.reward,
        result.steps,
        result.goals,
        result.shots,
        result.ball_in_box,
    ]
    if include_tactical and result.tactical is not None:
        t = result.tactical
        row.extend([
            t.distance_advanced,
            t.box_entries,
            t.possession_losses,
            t.max_possession_steps,
            t.approx_xg,
            t.pressure_proxy,
            t.shot_timing_step if t.shot_timing_step is not None else -1,
        ])
    return row


def write_episode_csv(
    results: list[EpisodeResult],
    output: str | Path,
    *,
    include_tactical: bool = False,
) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = list(BASIC_CSV_HEADER)
    if include_tactical:
        header.extend(TACTICAL_CSV_FIELDS)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result in results:
            writer.writerow(episode_to_row(result, include_tactical=include_tactical))
    return out


def summarize_results(results: list[EpisodeResult]) -> dict[str, float]:
    rewards = [r.reward for r in results]
    goals = [r.goals for r in results]
    shots = [r.shots for r in results]
    ball_in_box = [r.ball_in_box for r in results]
    steps = [r.steps for r in results]
    n = len(results) or 1
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "scoring_rate": 100.0 * sum(g > 0 for g in goals) / n,
        "shot_rate": 100.0 * sum(s > 0 for s in shots) / n,
        "ball_in_box_rate": 100.0 * sum(b > 0 for b in ball_in_box) / n,
        "mean_steps": float(np.mean(steps)),
        "total_goals": float(sum(goals)),
    }


def random_policy(env) -> ActionPolicy:
    def _policy(_obs: np.ndarray) -> int:
        return int(env.action_space.sample())
    return _policy
