"""Automatic curriculum discovery from scenario parameter search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from opengoalrl.curriculum.skill_graph import SkillGraph, SkillNode
from opengoalrl.scenarios.generator import ScenarioGenerator
from opengoalrl.scenarios.spec import GRF_SCENARIO_MAP


@dataclass
class AutoCurriculum:
    """Discover curriculum stages via short probe evaluations."""

    target: str
    budget: int = 20
    probe_timesteps: int = 10_000
    objective_metric: str = "scoring_rate"
    mastery_threshold: float = 15.0
    seed: int = 42
    stages: list[dict[str, Any]] = field(default_factory=list)

    def discover(self, probe_fn=None) -> "AutoCurriculum":
        """Rank candidate scenarios and build a reproducible stage list.

        *probe_fn* receives ``(scenario, timesteps)`` and returns a float
        score.  When omitted, candidates are ranked by heuristic difficulty.
        """
        gen = ScenarioGenerator(seed=self.seed)
        candidates: list[tuple[str, float, dict[str, Any]]] = []

        base_order = [
            "empty_goal_close",
            "empty_goal",
            "run_to_score",
            "pass_and_shoot",
            "three_vs_one",
            self.target,
        ]
        seen = set()
        for name in base_order:
            if name in seen:
                continue
            seen.add(name)
            spec = gen.from_fixed({"name": name, "grf_scenario": GRF_SCENARIO_MAP.get(name)})
            score = (
                probe_fn(name, self.probe_timesteps)
                if probe_fn is not None
                else _heuristic_score(name, self.target)
            )
            candidates.append((name, score, spec.to_dict()))

        while len(candidates) < self.budget:
            spec = gen.sample({
                "attackers": [1, 3],
                "defenders": [0, 2],
                "keeper": [True, False],
            })
            key = spec.name
            if key in seen:
                continue
            seen.add(key)
            score = _heuristic_score(spec.name, self.target)
            candidates.append((key, score, spec.to_dict()))

        candidates.sort(key=lambda x: x[1])
        selected = candidates[: self.budget]
        selected = [(n, s, d) for n, s, d in selected if n != self.target]
        selected.append((
            self.target,
            _heuristic_score(self.target, self.target),
            gen.from_fixed({"name": self.target}).to_dict(),
        ))
        self.stages = [
            {
                "scenario": name,
                "timesteps": self.probe_timesteps * (i + 1),
                "max_steps": 400,
                "probe_score": score,
            }
            for i, (name, score, _spec) in enumerate(selected)
        ]
        return self

    def to_skill_graph(self) -> SkillGraph:
        nodes = []
        prev: list[str] = []
        for i, stage in enumerate(self.stages):
            sid = f"stage_{i}_{stage['scenario']}"
            nodes.append(SkillNode(
                id=sid,
                scenario=stage["scenario"],
                prerequisites=list(prev),
                timesteps=int(stage.get("timesteps", self.probe_timesteps)),
                max_steps=int(stage.get("max_steps", 400)),
                mastery_threshold={"scoring_rate": self.mastery_threshold},
            ))
            prev = [sid]
        return SkillGraph(nodes)

    def to_config(self, rewards: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "rewards": rewards or [
                {"type": "goal", "weight": 10.0},
                {"type": "shot", "weight": 2.0},
            ],
            "training": {"seed": self.seed},
            "curriculum": {"stages": self.stages},
            "evaluation": {"n_episodes": 10},
            "logging": {"save_dir": f"models/auto_{self.target}/"},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_config(), f, sort_keys=False)
        meta = path.with_suffix(".meta.json")
        meta.write_text(json.dumps({
            "target": self.target,
            "budget": self.budget,
            "objective_metric": self.objective_metric,
            "stages": self.stages,
        }, indent=2))
        return path


def _heuristic_score(candidate: str, target: str) -> float:
    order = {
        "empty_goal_close": 0.1,
        "empty_goal": 0.2,
        "run_to_score": 0.4,
        "pass_and_shoot": 0.6,
        "three_vs_one": 0.8,
        "corner_kick": 0.9,
        "penalty": 0.5,
    }
    base = order.get(candidate, 0.5)
    target_val = order.get(target, 0.9)
    return abs(base - target_val * 0.7)
