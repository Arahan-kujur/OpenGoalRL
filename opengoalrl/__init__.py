"""OpenGoalRL -- Scenario-based RL toolkit for Google Research Football."""

from __future__ import annotations

from importlib import import_module

__version__ = "0.2.0"

from opengoalrl.rewards.base_reward import RewardComponent

_LAZY_EXPORTS = {
    "CornerKickEnv": "opengoalrl.envs.corner_kick",
    "PenaltyEnv": "opengoalrl.envs.penalty",
    "EmptyGoalCloseEnv": "opengoalrl.envs.empty_goal_close",
    "EmptyGoalEnv": "opengoalrl.envs.empty_goal",
    "RunToScoreEnv": "opengoalrl.envs.run_to_score",
    "PassAndShootEnv": "opengoalrl.envs.pass_and_shoot",
    "ThreeVsOneEnv": "opengoalrl.envs.three_vs_one",
    "PPOAgent": "opengoalrl.agents.ppo_agent",
}

__all__ = [
    "__version__",
    "RewardComponent",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str):
    """Load GRF-backed exports only when callers explicitly request them."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
