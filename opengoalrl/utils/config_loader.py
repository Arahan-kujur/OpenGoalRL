"""YAML config loading and reward-component registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from opengoalrl.rewards.base_reward import RewardComponent
from opengoalrl.rewards.ball_position_reward import BallInBoxReward
from opengoalrl.rewards.shot_reward import ShotReward
from opengoalrl.rewards.goal_reward import GoalReward
from opengoalrl.rewards.distance_reward import DistanceToGoalReward
from opengoalrl.utils.config_validation import ConfigValidationError, validate_config

REWARD_REGISTRY: dict[str, type[RewardComponent]] = {
    "goal": GoalReward,
    "ball_in_box": BallInBoxReward,
    "shot": ShotReward,
    "distance_to_goal": DistanceToGoalReward,
}


def load_config(path: str | Path, *, validate: bool = False) -> dict[str, Any]:
    """Load a YAML config file and optionally validate known sections."""
    with open(path, "r") as fh:
        config = yaml.safe_load(fh)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ConfigValidationError("Config root must be a mapping")
    if validate:
        validate_config(config)
    return config


def build_reward_components(config: dict[str, Any]) -> list[RewardComponent]:
    """Instantiate reward components from the ``rewards`` section of a config."""
    components: list[RewardComponent] = []
    for entry in config.get("rewards", []):
        rtype = entry["type"]
        weight = float(entry.get("weight", 1.0))
        cls = REWARD_REGISTRY.get(rtype)
        if cls is None:
            raise ValueError(
                f"Unknown reward type {rtype!r}. "
                f"Available: {list(REWARD_REGISTRY)}"
            )
        components.append(cls(weight=weight))
    return components
