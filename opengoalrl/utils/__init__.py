"""Convenience exports for OpenGoalRL utilities."""

from __future__ import annotations

from importlib import import_module

from opengoalrl.utils.config_loader import build_reward_components, load_config, validate_config
from opengoalrl.utils.logger import get_logger

_LAZY_EXPORTS = {
    "build_env": "opengoalrl.utils.env_factory",
    "build_vec_env": "opengoalrl.utils.env_factory",
    "set_seed": "opengoalrl.utils.env_factory",
    "MetricsCallback": "opengoalrl.utils.metrics_callback",
}

__all__ = [
    "build_reward_components",
    "get_logger",
    "load_config",
    "validate_config",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
