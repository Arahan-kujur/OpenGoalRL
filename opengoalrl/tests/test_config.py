"""Unit tests for config loading and reward registry -- no GRF required."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from opengoalrl.utils.config_loader import (
    REWARD_REGISTRY,
    build_reward_components,
    load_config,
)
from opengoalrl.rewards.ball_position_reward import BallInBoxReward
from opengoalrl.rewards.shot_reward import ShotReward
from opengoalrl.rewards.goal_reward import GoalReward
from opengoalrl.rewards.distance_reward import DistanceToGoalReward


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg = {"environment": {"scenario": "test"}, "training": {"seed": 99}}
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(cfg))
        loaded = load_config(p)
        assert loaded["training"]["seed"] == 99

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestRewardRegistry:
    def test_all_types_registered(self):
        assert "goal" in REWARD_REGISTRY
        assert "ball_in_box" in REWARD_REGISTRY
        assert "shot" in REWARD_REGISTRY
        assert "distance_to_goal" in REWARD_REGISTRY

    def test_registry_maps_to_correct_classes(self):
        assert REWARD_REGISTRY["goal"] is GoalReward
        assert REWARD_REGISTRY["ball_in_box"] is BallInBoxReward
        assert REWARD_REGISTRY["shot"] is ShotReward
        assert REWARD_REGISTRY["distance_to_goal"] is DistanceToGoalReward


class TestBuildRewardComponents:
    def test_builds_from_config(self):
        config = {
            "rewards": [
                {"type": "goal", "weight": 10.0},
                {"type": "shot", "weight": 2.0},
            ]
        }
        components = build_reward_components(config)
        assert len(components) == 2
        assert isinstance(components[0], GoalReward)
        assert components[0].weight == 10.0
        assert isinstance(components[1], ShotReward)
        assert components[1].weight == 2.0

    def test_empty_rewards(self):
        assert build_reward_components({}) == []
        assert build_reward_components({"rewards": []}) == []

    def test_unknown_type_raises(self):
        config = {"rewards": [{"type": "nonexistent", "weight": 1.0}]}
        with pytest.raises(ValueError, match="Unknown reward type"):
            build_reward_components(config)

    def test_default_weight(self):
        config = {"rewards": [{"type": "goal"}]}
        components = build_reward_components(config)
        assert components[0].weight == 1.0
