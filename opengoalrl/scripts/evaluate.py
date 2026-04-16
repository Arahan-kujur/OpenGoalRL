"""Evaluation entry-point for OpenGoalRL."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from opengoalrl.agents.ppo_agent import PPOAgent
from opengoalrl.scripts.train import build_env
from opengoalrl.utils.config_loader import load_config
from opengoalrl.utils.logger import get_logger, log_episode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenGoalRL agent")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "corner.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_opengoalrl",
        help="Path to saved SB3 model (without .zip extension)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/eval_metrics.csv",
        help="Path to write per-episode CSV",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    log_cfg = config.get("logging", {})

    logger = get_logger(log_dir=log_cfg.get("log_dir"))
    logger.info("Evaluating model: %s", args.model)

    env = build_env(config)
    agent = PPOAgent.load(args.model, env=env)

    n_episodes = eval_cfg.get("n_episodes", 10)
    all_rewards: list[float] = []
    all_goals: list[int] = []
    all_steps: list[int] = []
    all_shots: list[int] = []
    all_ball_in_box: list[int] = []

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "steps", "goals", "shots", "ball_in_box",
        ])

        for ep in range(1, n_episodes + 1):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_goals = 0
            episode_shots = 0
            episode_ball_in_box = 0
            steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                if info.get("score_reward", 0.0) > 0:
                    episode_goals += 1

                rc = info.get("reward_components", {})
                for key, val in rc.items():
                    if "ShotReward" in key and val > 0:
                        episode_shots += 1
                    if "BallInBoxReward" in key and val > 0:
                        episode_ball_in_box += 1

            writer.writerow([
                ep, episode_reward, steps, episode_goals,
                episode_shots, episode_ball_in_box,
            ])
            log_episode(logger, ep, episode_reward, steps, episode_goals)

            all_rewards.append(episode_reward)
            all_goals.append(episode_goals)
            all_steps.append(steps)
            all_shots.append(episode_shots)
            all_ball_in_box.append(episode_ball_in_box)

    logger.info("--- Evaluation Summary (%d episodes) ---", n_episodes)
    logger.info("Mean reward       : %.2f (+/-%.2f)", np.mean(all_rewards), np.std(all_rewards))
    logger.info("Total goals       : %d", sum(all_goals))
    logger.info("Goals scored %%    : %.1f%%", 100 * sum(g > 0 for g in all_goals) / n_episodes)
    logger.info("Episodes with shot: %.1f%%", 100 * sum(s > 0 for s in all_shots) / n_episodes)
    logger.info("Ball in box %%     : %.1f%%", 100 * sum(b > 0 for b in all_ball_in_box) / n_episodes)
    logger.info("Mean steps        : %.1f", np.mean(all_steps))
    logger.info("Results saved to  : %s", out)

    env.close()


if __name__ == "__main__":
    main()
