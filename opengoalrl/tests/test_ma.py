"""Tests for multi-agent module structure (no GRF)."""

from __future__ import annotations

import inspect

from opengoalrl.envs.multi_agent import MultiAgentFootballEnv, SingleAgentMAView


class TestMultiAgentModule:
    def test_ma_env_has_required_methods(self):
        assert hasattr(MultiAgentFootballEnv, "as_single_agent")
        assert hasattr(MultiAgentFootballEnv, "reset")
        assert hasattr(MultiAgentFootballEnv, "step")

    def test_single_agent_view_is_gym_env(self):
        assert "reset" in dir(SingleAgentMAView)
        sig = inspect.signature(SingleAgentMAView.__init__)
        assert "ma_env" in sig.parameters

    def test_ma_config_exists(self):
        from pathlib import Path
        cfg = Path(__file__).resolve().parent.parent / "configs" / "ma_3v3.yaml"
        assert cfg.exists()
