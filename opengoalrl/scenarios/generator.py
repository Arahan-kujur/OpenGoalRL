"""Procedural scenario generation from specs or sampled parameter ranges."""

from __future__ import annotations

import random
from typing import Any

from opengoalrl.scenarios.spec import GRF_SCENARIO_MAP, ScenarioSpec, parse_scenario_spec


class ScenarioGenerator:
    """Generate :class:`ScenarioSpec` instances from fixed or sampled parameters."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def from_fixed(self, data: dict[str, Any] | ScenarioSpec) -> ScenarioSpec:
        spec = parse_scenario_spec(data)
        spec.grf_scenario = spec.grf_scenario or _resolve_grf_scenario(spec)
        return spec

    def sample(self, ranges: dict[str, Any]) -> ScenarioSpec:
        attackers = self._sample_int(ranges.get("attackers", [1, 3]))
        defenders = self._sample_int(ranges.get("defenders", [0, 2]))
        keeper = self._rng.choice(ranges.get("keeper", [True, False]))
        ball_x = self._sample_float(ranges.get("ball_x", [0.0, 0.9]))
        ball_y = self._sample_float(ranges.get("ball_y", [-0.3, 0.3]))
        max_steps = self._sample_int(ranges.get("max_steps", [200, 400]))
        field_region = self._rng.choice(
            ranges.get("field_region", ["defensive_third", "midfield", "attacking_third"])
        )
        spec = ScenarioSpec(
            name=f"sampled_{attackers}v{defenders}",
            attackers=attackers,
            defenders=defenders,
            keeper=bool(keeper),
            ball_position=(ball_x, ball_y),
            field_region=field_region,
            max_steps=max_steps,
        )
        spec.grf_scenario = _resolve_grf_scenario(spec)
        return spec

    def _sample_int(self, bounds: list[int]) -> int:
        if len(bounds) == 1:
            return int(bounds[0])
        lo, hi = int(bounds[0]), int(bounds[1])
        return self._rng.randint(lo, hi)

    def _sample_float(self, bounds: list[float]) -> float:
        if len(bounds) == 1:
            return float(bounds[0])
        lo, hi = float(bounds[0]), float(bounds[1])
        return self._rng.uniform(lo, hi)


def generate_scenario(
    data: dict[str, Any] | ScenarioSpec,
    *,
    seed: int | None = None,
) -> ScenarioSpec:
    """Convenience wrapper: fixed spec or ``sample_ranges`` block."""
    gen = ScenarioGenerator(seed=seed)
    if isinstance(data, dict) and "sample_ranges" in data:
        return gen.sample(data["sample_ranges"])
    return gen.from_fixed(data)


def _resolve_grf_scenario(spec: ScenarioSpec) -> str:
    if spec.grf_scenario:
        return spec.grf_scenario
    if spec.attackers >= 3 and spec.defenders >= 1 and spec.keeper:
        return GRF_SCENARIO_MAP["three_vs_one"]
    if spec.attackers >= 2 and spec.keeper:
        return GRF_SCENARIO_MAP["pass_and_shoot"]
    if spec.defenders >= 1:
        return GRF_SCENARIO_MAP["run_to_score"]
    if spec.ball_position[0] > 0.7:
        return GRF_SCENARIO_MAP["empty_goal_close"]
    if spec.field_region == "attacking_third":
        return GRF_SCENARIO_MAP["corner_kick"]
    if not spec.keeper and spec.defenders == 0:
        return GRF_SCENARIO_MAP["empty_goal"]
    return GRF_SCENARIO_MAP["empty_goal_close"]
