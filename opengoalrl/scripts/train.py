"""Training entry-point for OpenGoalRL."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from opengoalrl.envs.corner_kick import CornerKickEnv
from opengoalrl.envs.penalty import PenaltyEnv
from opengoalrl.envs.empty_goal_close import EmptyGoalCloseEnv
from opengoalrl.envs.empty_goal import EmptyGoalEnv
from opengoalrl.envs.run_to_score import RunToScoreEnv
from opengoalrl.envs.pass_and_shoot import PassAndShootEnv
from opengoalrl.envs.three_vs_one import ThreeVsOneEnv
from opengoalrl.wrappers.scenario_wrapper import ScenarioWrapper
from opengoalrl.wrappers.observation_wrapper import ObservationWrapper
from opengoalrl.wrappers.reward_wrapper import RewardWrapper
from opengoalrl.wrappers.action_wrapper import ActionWrapper
from opengoalrl.agents.ppo_agent import PPOAgent
from opengoalrl.utils.config_loader import load_config, build_reward_components
from opengoalrl.utils.logger import get_logger, save_config_snapshot
from opengoalrl.utils.metrics_callback import MetricsCallback

ENV_REGISTRY: dict[str, type] = {
    "corner_kick": CornerKickEnv,
    "penalty": PenaltyEnv,
    "empty_goal_close": EmptyGoalCloseEnv,
    "empty_goal": EmptyGoalEnv,
    "run_to_score": RunToScoreEnv,
    "pass_and_shoot": PassAndShootEnv,
    "three_vs_one": ThreeVsOneEnv,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(config: dict):
    """Instantiate a single env + full wrapper stack from *config*."""
    env_cfg = config["environment"]
    scenario = env_cfg["scenario"]
    env_cls = ENV_REGISTRY.get(scenario)
    if env_cls is None:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Available: {list(ENV_REGISTRY)}"
        )

    render_mode = "human" if env_cfg.get("render", False) else None
    env = env_cls(
        max_steps=env_cfg.get("max_steps", 400),
        render_mode=render_mode,
    )

    env = ScenarioWrapper(env, scenario_name=scenario)

    reward_components = build_reward_components(config)
    if reward_components:
        env = RewardWrapper(env, components=reward_components)

    env = ObservationWrapper(env)
    env = ActionWrapper(env)
    return env


def build_vec_env(config: dict):
    """Build a vectorised environment from *config*.

    Returns a ``SubprocVecEnv`` when ``training.n_envs > 1``, otherwise
    the raw single env (SB3 wraps it in ``DummyVecEnv`` automatically).
    """
    n_envs = config.get("training", {}).get("n_envs", 1)
    if n_envs <= 1:
        return build_env(config)

    def _make_env(rank: int):
        def _init():
            env = build_env(config)
            return env
        return _init

    return SubprocVecEnv([_make_env(i) for i in range(n_envs)])


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

    seed = train_cfg.get("seed", 42)
    _set_seed(seed)

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
    callback = MetricsCallback(csv_path=metrics_path)

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
