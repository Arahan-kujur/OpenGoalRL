"""Structured scenario specification for procedural football tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ScenarioSpec:
    """Declarative description of a football scenario."""

    name: str = "generated"
    attackers: int = 1
    defenders: int = 0
    keeper: bool = True
    ball_position: tuple[float, float] = (0.0, 0.0)
    controlled_players: int = 1
    field_region: str = "midfield"
    objective: str = "score"
    max_steps: int = 400
    allowed_actions: list[int] | None = None
    grf_scenario: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ball_position"] = list(self.ball_position)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioSpec":
        ball = data.get("ball_position", [0.0, 0.0])
        if isinstance(ball, (list, tuple)) and len(ball) >= 2:
            ball_pos = (float(ball[0]), float(ball[1]))
        else:
            ball_pos = (0.0, 0.0)
        allowed = data.get("allowed_actions")
        if allowed is not None:
            allowed = [int(a) for a in allowed]
        return cls(
            name=str(data.get("name", "generated")),
            attackers=int(data.get("attackers", 1)),
            defenders=int(data.get("defenders", 0)),
            keeper=bool(data.get("keeper", True)),
            ball_position=ball_pos,
            controlled_players=int(data.get("controlled_players", 1)),
            field_region=str(data.get("field_region", "midfield")),
            objective=str(data.get("objective", "score")),
            max_steps=int(data.get("max_steps", 400)),
            allowed_actions=allowed,
            grf_scenario=data.get("grf_scenario"),
            metadata=dict(data.get("metadata", {})),
        )


def parse_scenario_spec(data: dict[str, Any] | ScenarioSpec) -> ScenarioSpec:
    if isinstance(data, ScenarioSpec):
        return data
    return ScenarioSpec.from_dict(data)


# Map OpenGoalRL scenario keys to GRF academy scenario ids.
GRF_SCENARIO_MAP: dict[str, str] = {
    "corner_kick": "academy_corner",
    "penalty": "academy_single_goal_versus_lazy",
    "empty_goal_close": "academy_empty_goal_close",
    "empty_goal": "academy_empty_goal",
    "run_to_score": "academy_run_to_score",
    "pass_and_shoot": "academy_pass_and_shoot_with_keeper",
    "three_vs_one": "academy_3_vs_1_with_keeper",
}

OPENGOAL_SCENARIO_BY_GRF: dict[str, str] = {v: k for k, v in GRF_SCENARIO_MAP.items()}
