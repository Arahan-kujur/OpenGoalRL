"""Config section validation for OpenGoalRL YAML files."""

from __future__ import annotations

from typing import Any

KNOWN_SCENARIOS = frozenset({
    "corner_kick",
    "penalty",
    "empty_goal_close",
    "empty_goal",
    "run_to_score",
    "pass_and_shoot",
    "three_vs_one",
})

REWARD_TYPES = frozenset({"goal", "ball_in_box", "shot", "distance_to_goal"})


class ConfigValidationError(ValueError):
    """Raised when a config section fails validation."""


def validate_environment(env_cfg: dict[str, Any]) -> None:
    if not isinstance(env_cfg, dict):
        raise ConfigValidationError("'environment' must be a mapping")
    has_scenario = "scenario" in env_cfg
    has_spec = "scenario_spec" in env_cfg
    if not has_scenario and not has_spec:
        raise ConfigValidationError(
            "'environment' requires 'scenario' or 'scenario_spec'"
        )
    if has_scenario and not isinstance(env_cfg["scenario"], str):
        raise ConfigValidationError("'environment.scenario' must be a string")
    if has_scenario and env_cfg["scenario"] not in KNOWN_SCENARIOS:
        raise ConfigValidationError(
            f"Unknown scenario {env_cfg['scenario']!r}. "
            f"Available: {sorted(KNOWN_SCENARIOS)}"
        )
    if has_spec and not isinstance(env_cfg["scenario_spec"], dict):
        raise ConfigValidationError("'environment.scenario_spec' must be a mapping")
    if "max_steps" in env_cfg:
        _require_positive_int(env_cfg["max_steps"], "environment.max_steps")
    if "render" in env_cfg and not isinstance(env_cfg["render"], bool):
        raise ConfigValidationError("'environment.render' must be a boolean")


def validate_rewards(rewards: list[dict[str, Any]]) -> None:
    if not isinstance(rewards, list):
        raise ConfigValidationError("'rewards' must be a list")
    for i, entry in enumerate(rewards):
        if not isinstance(entry, dict):
            raise ConfigValidationError(f"rewards[{i}] must be a mapping")
        rtype = entry.get("type")
        if rtype not in REWARD_TYPES:
            raise ConfigValidationError(
                f"rewards[{i}].type {rtype!r} unknown. "
                f"Available: {sorted(REWARD_TYPES)}"
            )
        if "weight" in entry:
            _require_number(entry["weight"], f"rewards[{i}].weight")


def validate_training(train_cfg: dict[str, Any]) -> None:
    if not isinstance(train_cfg, dict):
        raise ConfigValidationError("'training' must be a mapping")
    for key in ("total_timesteps", "n_envs", "n_steps", "batch_size", "n_epochs"):
        if key in train_cfg and not isinstance(train_cfg[key], int):
            raise ConfigValidationError(f"'training.{key}' must be an integer")
        if key in train_cfg:
            _require_positive_int(train_cfg[key], f"training.{key}")
    if "seed" in train_cfg and not isinstance(train_cfg["seed"], int):
        raise ConfigValidationError("'training.seed' must be an integer")
    for key in ("learning_rate", "clip_range"):
        if key in train_cfg:
            _require_positive_number(train_cfg[key], f"training.{key}")
    if "gamma" in train_cfg:
        gamma = _require_number(train_cfg["gamma"], "training.gamma")
        if not 0 < gamma <= 1:
            raise ConfigValidationError("'training.gamma' must be in (0, 1]")


def validate_evaluation(eval_cfg: dict[str, Any]) -> None:
    if not isinstance(eval_cfg, dict):
        raise ConfigValidationError("'evaluation' must be a mapping")
    if "n_episodes" in eval_cfg:
        _require_positive_int(eval_cfg["n_episodes"], "evaluation.n_episodes")
    if "metrics" in eval_cfg:
        metrics = eval_cfg["metrics"]
        if isinstance(metrics, str):
            metrics = [metrics]
        if not isinstance(metrics, list):
            raise ConfigValidationError("'evaluation.metrics' must be a list or string")
        allowed = {"basic", "tactical"}
        for m in metrics:
            if m not in allowed:
                raise ConfigValidationError(
                    f"Unknown evaluation metric {m!r}. Available: {sorted(allowed)}"
                )


def validate_curriculum(curriculum_cfg: dict[str, Any]) -> None:
    if not isinstance(curriculum_cfg, dict):
        raise ConfigValidationError("'curriculum' must be a mapping")
    if "stages" in curriculum_cfg:
        stages = curriculum_cfg["stages"]
        if not isinstance(stages, list) or not stages:
            raise ConfigValidationError("'curriculum.stages' must be a non-empty list")
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise ConfigValidationError(f"curriculum.stages[{i}] must be a mapping")
            if "scenario" not in stage and "skill" not in stage:
                raise ConfigValidationError(
                    f"curriculum.stages[{i}] requires 'scenario' or 'skill'"
                )
            if "timesteps" not in stage:
                raise ConfigValidationError(
                    f"curriculum.stages[{i}] requires 'timesteps'"
                )
            _require_positive_int(
                stage["timesteps"],
                f"curriculum.stages[{i}].timesteps",
            )
            if "max_steps" in stage:
                _require_positive_int(
                    stage["max_steps"],
                    f"curriculum.stages[{i}].max_steps",
                )
    if "skill_graph" in curriculum_cfg:
        sg = curriculum_cfg["skill_graph"]
        if not isinstance(sg, dict):
            raise ConfigValidationError("'curriculum.skill_graph' must be a mapping")
        nodes = sg.get("nodes", [])
        if not isinstance(nodes, list) or not nodes:
            raise ConfigValidationError(
                "'curriculum.skill_graph.nodes' must be a non-empty list"
            )


def validate_report(report_cfg: dict[str, Any]) -> None:
    if not isinstance(report_cfg, dict):
        raise ConfigValidationError("'report' must be a mapping")
    fmt = report_cfg.get("format", "markdown")
    if fmt not in ("markdown", "html"):
        raise ConfigValidationError("'report.format' must be 'markdown' or 'html'")


def validate_metrics(metrics_cfg: dict[str, Any] | list[str] | str) -> None:
    if isinstance(metrics_cfg, str):
        if not metrics_cfg:
            raise ConfigValidationError("'metrics' must not be empty")
        return
    if isinstance(metrics_cfg, list):
        for i, metric in enumerate(metrics_cfg):
            if not isinstance(metric, str) or not metric:
                raise ConfigValidationError(f"metrics[{i}] must be a non-empty string")
        return
    if not isinstance(metrics_cfg, dict):
        raise ConfigValidationError("'metrics' must be a mapping, list, or string")
    enabled = metrics_cfg.get("tactical", True)
    if not isinstance(enabled, bool):
        raise ConfigValidationError("'metrics.tactical' must be a boolean")


def validate_config(
    config: dict[str, Any],
    *,
    strict: bool = False,
    required_sections: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validate known config sections. Returns the config unchanged."""
    if not isinstance(config, dict):
        raise ConfigValidationError("Config root must be a mapping")
    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required config section: {section}")
    if "environment" in config:
        validate_environment(config["environment"])
    if "rewards" in config:
        validate_rewards(config["rewards"])
    if "training" in config:
        validate_training(config["training"])
    if "evaluation" in config:
        validate_evaluation(config["evaluation"])
    if "curriculum" in config:
        validate_curriculum(config["curriculum"])
    if "report" in config:
        validate_report(config["report"])
    if "metrics" in config:
        validate_metrics(config["metrics"])
    if strict and "environment" not in config:
        raise ConfigValidationError("Config requires an 'environment' section")
    return config


def _require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"'{path}' must be a number")
    return float(value)


def _require_positive_number(value: Any, path: str) -> float:
    number = _require_number(value, path)
    if number <= 0:
        raise ConfigValidationError(f"'{path}' must be positive")
    return number


def _require_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"'{path}' must be a positive integer")
    if value <= 0:
        raise ConfigValidationError(f"'{path}' must be a positive integer")
    return value
