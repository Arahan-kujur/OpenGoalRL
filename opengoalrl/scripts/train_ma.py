"""Multi-agent training entry-point (parameter-sharing PPO)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from opengoalrl.agents.ppo_agent import PPOAgent
from opengoalrl.envs.multi_agent import MultiAgentFootballEnv
from opengoalrl.wrappers.action_wrapper import ActionWrapper
from opengoalrl.wrappers.observation_wrapper import ObservationWrapper
from opengoalrl.wrappers.reward_wrapper import RewardWrapper
from opengoalrl.wrappers.scenario_wrapper import ScenarioWrapper
from opengoalrl.utils.config_loader import build_reward_components, load_config
from opengoalrl.utils.logger import get_logger, save_config_snapshot
from opengoalrl.utils.metrics_callback import MetricsCallback


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ma_env(config: dict):
    """Build multi-agent env with standard wrapper stack on single-agent view."""
    env_cfg = config["multi_agent"]
    ma = MultiAgentFootballEnv(
        scenario=env_cfg.get("scenario", "three_vs_one"),
        n_agents=int(env_cfg.get("n_agents", 3)),
        max_steps=int(env_cfg.get("max_steps", 400)),
        render_mode="human" if env_cfg.get("render", False) else None,
        team_reward=bool(env_cfg.get("team_reward", True)),
    )
    env = ma.as_single_agent()
    scenario = env_cfg.get("scenario", "three_vs_one")
    env = ScenarioWrapper(env, scenario_name=scenario)
    reward_components = build_reward_components(config)
    if reward_components:
        env = RewardWrapper(env, components=reward_components)
    env = ObservationWrapper(env)
    env = ActionWrapper(env)
    return env


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train multi-agent OpenGoalRL policy")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "configs" / "ma_3v3.yaml"
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})

    seed = train_cfg.get("seed", 42)
    _set_seed(seed)

    logger = get_logger(log_dir=log_cfg.get("log_dir"))
    save_dir = Path(log_cfg.get("save_dir", "models/ma/"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, save_dir)

    env = build_ma_env(config)
    agent = PPOAgent(env, config=train_cfg)

    metrics_path = save_dir / "ma_training_metrics.csv"
    callback = MetricsCallback(csv_path=metrics_path, include_tactical=True)

    total = train_cfg.get("total_timesteps", 100_000)
    logger.info("Multi-agent training for %d timesteps", total)
    agent.train(total_timesteps=total, callback=callback)

    model_path = save_dir / "ppo_ma"
    agent.save(model_path)
    logger.info("Model saved to %s", model_path)
    env.close()


if __name__ == "__main__":
    main()
