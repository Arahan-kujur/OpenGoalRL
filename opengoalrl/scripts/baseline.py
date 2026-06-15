"""Run a random-action baseline and save per-episode metrics to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from opengoalrl.utils.env_factory import build_env
from opengoalrl.utils.config_loader import load_config, validate_config
from opengoalrl.utils.logger import get_logger, log_episode
from opengoalrl.utils.rollout import random_policy, run_rollouts, summarize_results, write_episode_csv


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
    validate_config(config, required_sections=("environment",))
    log_cfg = config.get("logging", {})
    logger = get_logger(log_dir=log_cfg.get("log_dir"))
    logger.info("Running random baseline for %d episodes", args.episodes)

    env = build_env(config)
    results = run_rollouts(env, random_policy(env), args.episodes)
    write_episode_csv(results, args.output)

    summary = summarize_results(results)
    for ep, result in enumerate(results, 1):
        log_episode(logger, ep, result.reward, result.steps, result.goals)

    n = args.episodes
    logger.info("--- Random Baseline Summary (%d episodes) ---", n)
    logger.info("Mean reward       : %.2f (+/-%.2f)", summary["mean_reward"], summary["std_reward"])
    logger.info("Goals scored      : %.1f%%", summary["scoring_rate"])
    logger.info("Episodes with shot: %.1f%%", summary["shot_rate"])
    logger.info("Ball in box       : %.1f%%", summary["ball_in_box_rate"])
    logger.info("Mean steps        : %.1f", summary["mean_steps"])
    logger.info("Results saved to  : %s", args.output)

    env.close()


if __name__ == "__main__":
    main()
