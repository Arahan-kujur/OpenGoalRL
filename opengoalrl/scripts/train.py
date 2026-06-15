"""Training entry-point for OpenGoalRL."""

from __future__ import annotations

import argparse
from pathlib import Path

from opengoalrl.agents.ppo_agent import PPOAgent
from opengoalrl.utils.config_loader import load_config
from opengoalrl.utils.env_factory import ENV_REGISTRY, build_env, build_vec_env, set_seed
from opengoalrl.utils.logger import get_logger, save_config_snapshot
from opengoalrl.utils.metrics_callback import MetricsCallback

# Backward-compatible aliases used by other scripts and tests.
_set_seed = set_seed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an OpenGoalRL agent")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "corner.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})
    metrics_cfg = config.get("metrics", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    logger = get_logger(log_dir=log_cfg.get("log_dir"))
    logger.info("Starting training with config: %s", args.config)

    save_dir = log_cfg.get("save_dir", "models/")
    save_config_snapshot(config, save_dir)

    n_envs = train_cfg.get("n_envs", 1)
    if n_envs > 1:
        logger.info("Using SubprocVecEnv with %d parallel environments", n_envs)
        env = build_vec_env(config)
    else:
        env = build_env(config)

    agent = PPOAgent(env, config=train_cfg)

    metrics_path = Path(save_dir) / "training_metrics.csv"
    include_tactical = (
        metrics_cfg.get("tactical", False)
        if isinstance(metrics_cfg, dict)
        else False
    )
    callback = MetricsCallback(csv_path=metrics_path, include_tactical=include_tactical)

    total_timesteps = train_cfg.get("total_timesteps", 100_000)
    logger.info("Training for %d timesteps ...", total_timesteps)
    agent.train(total_timesteps=total_timesteps, callback=callback)
    logger.info("Metrics saved to %s", metrics_path)

    model_path = Path(save_dir) / "ppo_opengoalrl"
    agent.save(model_path)
    logger.info("Model saved to %s", model_path)

    env.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
