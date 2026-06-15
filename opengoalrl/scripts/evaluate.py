"""Evaluation entry-point for OpenGoalRL."""

from __future__ import annotations

import argparse
from pathlib import Path

from opengoalrl.agents.ppo_agent import PPOAgent
from opengoalrl.metrics.tactical import aggregate_tactical
from opengoalrl.utils.env_factory import build_env
from opengoalrl.utils.config_loader import load_config, validate_config
from opengoalrl.utils.logger import get_logger, log_episode
from opengoalrl.utils.rollout import run_rollouts, summarize_results, write_episode_csv


def _wants_tactical(config: dict, cli_metrics: str | None) -> bool:
    if cli_metrics == "tactical":
        return True
    eval_cfg = config.get("evaluation", {})
    metrics = eval_cfg.get("metrics", [])
    if isinstance(metrics, str):
        return metrics == "tactical"
    return "tactical" in metrics


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
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        choices=["basic", "tactical"],
        help="Metric set to collect (overrides config)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_config(config, required_sections=("environment",))
    eval_cfg = config.get("evaluation", {})
    log_cfg = config.get("logging", {})

    logger = get_logger(log_dir=log_cfg.get("log_dir"))
    logger.info("Evaluating model: %s", args.model)

    env = build_env(config)
    agent = PPOAgent.load(args.model, env=env)

    n_episodes = eval_cfg.get("n_episodes", 10)
    include_tactical = _wants_tactical(config, args.metrics)

    def policy(obs):
        action, _ = agent.predict(obs)
        return action

    results = run_rollouts(
        env,
        policy,
        n_episodes,
        collect_tactical=include_tactical,
    )
    write_episode_csv(results, args.output, include_tactical=include_tactical)

    summary = summarize_results(results)
    for ep, result in enumerate(results, 1):
        log_episode(logger, ep, result.reward, result.steps, result.goals)

    logger.info("--- Evaluation Summary (%d episodes) ---", n_episodes)
    logger.info("Mean reward       : %.2f (+/-%.2f)", summary["mean_reward"], summary["std_reward"])
    logger.info("Total goals       : %d", int(summary["total_goals"]))
    logger.info("Goals scored %%    : %.1f%%", summary["scoring_rate"])
    logger.info("Episodes with shot: %.1f%%", summary["shot_rate"])
    logger.info("Ball in box %%     : %.1f%%", summary["ball_in_box_rate"])
    logger.info("Mean steps        : %.1f", summary["mean_steps"])

    if include_tactical:
        tactical_list = [r.tactical for r in results if r.tactical is not None]
        agg = aggregate_tactical(tactical_list)
        logger.info("Mean dist advanced: %.3f", agg.get("mean_distance_advanced", 0))
        logger.info("Mean approx xG    : %.3f", agg.get("mean_approx_xg", 0))

    logger.info("Results saved to  : %s", args.output)
    env.close()


if __name__ == "__main__":
    main()
