"""YAML config loading and reward-component registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

from opengoalrl.rewards.base_reward import RewardComponent
from opengoalrl.rewards.ball_position_reward import BallInBoxReward
from opengoalrl.rewards.shot_reward import ShotReward
from opengoalrl.rewards.goal_reward import GoalReward
from opengoalrl.rewards.distance_reward import DistanceToGoalReward

REWARD_REGISTRY: dict[str, type[RewardComponent]] = {
    "goal": GoalReward,
    "ball_in_box": BallInBoxReward,
    "shot": ShotReward,
    "distance_to_goal": DistanceToGoalReward,
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


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
