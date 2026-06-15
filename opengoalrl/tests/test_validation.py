"""Tests for config validation."""

from __future__ import annotations

import pytest

from opengoalrl.utils.config_validation import (
    ConfigValidationError,
    validate_config,
    validate_environment,
)


class TestValidateEnvironment:
    def test_requires_scenario_or_spec(self):
        with pytest.raises(ConfigValidationError):
            validate_environment({})

    def test_accepts_scenario(self):
        validate_environment({"scenario": "corner_kick"})

    def test_accepts_scenario_spec(self):
        validate_environment({"scenario_spec": {"attackers": 2}})


class TestValidateConfig:
    def test_corner_config(self):
        from pathlib import Path
        from opengoalrl.utils.config_loader import load_config, validate_config
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "corner.yaml"
        config = load_config(cfg_path)
        validate_config(config)
        assert config["environment"]["scenario"] == "corner_kick"

    def test_invalid_reward_type(self):
        with pytest.raises(ConfigValidationError):
            validate_config({"rewards": [{"type": "bogus"}]})
