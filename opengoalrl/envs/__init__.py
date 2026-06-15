"""Environment exports loaded lazily to keep GRF-free tests importable."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "BaseScenarioEnv": "opengoalrl.envs.base_env",
    "CornerKickEnv": "opengoalrl.envs.corner_kick",
    "PenaltyEnv": "opengoalrl.envs.penalty",
    "EmptyGoalCloseEnv": "opengoalrl.envs.empty_goal_close",
    "EmptyGoalEnv": "opengoalrl.envs.empty_goal",
    "RunToScoreEnv": "opengoalrl.envs.run_to_score",
    "PassAndShootEnv": "opengoalrl.envs.pass_and_shoot",
    "ThreeVsOneEnv": "opengoalrl.envs.three_vs_one",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
