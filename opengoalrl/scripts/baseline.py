"""Run a random-action baseline and save per-episode metrics to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from opengoalrl.scripts.train import build_env
from opengoalrl.utils.config_loader import load_config
from opengoalrl.utils.logger import get_logger, log_episode

_SHOT_ACTION = 12


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Random-action baseline for OpenGoalRL")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "corner.yaml"),
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--output",
        type=str,
        default="models/baseline_metrics.csv",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    log_cfg = config.get("logging", {})
    logger = get_logger(log_dir=log_cfg.get("log_dir"))
    logger.info("Running random baseline for %d episodes", args.episodes)

    env = build_env(config)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rewards: list[float] = []
    all_goals: list[int] = []
    all_steps: list[int] = []
    all_shots: list[int] = []
    all_ball_in_box: list[int] = []

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "steps", "goals", "shots", "ball_in_box",
        ])

        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            ep_reward = 0.0
            ep_goals = 0
            ep_shots = 0
            ep_ball_in_box = 0
            steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1

                if info.get("score_reward", 0.0) > 0:
                    ep_goals += 1

                rc = info.get("reward_components", {})
                for key, val in rc.items():
                    if "ShotReward" in key and val > 0:
                        ep_shots += 1
                    if "BallInBoxReward" in key and val > 0:
                        ep_ball_in_box += 1

            writer.writerow([ep, ep_reward, steps, ep_goals, ep_shots, ep_ball_in_box])
            log_episode(logger, ep, ep_reward, steps, ep_goals)

            all_rewards.append(ep_reward)
            all_goals.append(ep_goals)
            all_steps.append(steps)
            all_shots.append(ep_shots)
            all_ball_in_box.append(ep_ball_in_box)

    env.close()

    n = args.episodes
    logger.info("--- Random Baseline Summary (%d episodes) ---", n)
    logger.info("Mean reward       : %.2f (+/-%.2f)", np.mean(all_rewards), np.std(all_rewards))
    logger.info("Goals scored      : %d / %d (%.1f%%)", sum(all_goals), n, 100 * sum(g > 0 for g in all_goals) / n)
    logger.info("Episodes with shot: %d / %d (%.1f%%)", sum(s > 0 for s in all_shots), n, 100 * sum(s > 0 for s in all_shots) / n)
    logger.info("Ball in box       : %d / %d (%.1f%%)", sum(b > 0 for b in all_ball_in_box), n, 100 * sum(b > 0 for b in all_ball_in_box) / n)
    logger.info("Mean steps        : %.1f", np.mean(all_steps))
    logger.info("Results saved to  : %s", out)


if __name__ == "__main__":
    main()
