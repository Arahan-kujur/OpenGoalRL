"""Graph-based football skill curricula."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opengoalrl.diagnostics.classifier import summarize_failures
from opengoalrl.metrics.tactical import EpisodeTacticalMetrics, aggregate_tactical
from opengoalrl.utils.rollout import EpisodeResult


@dataclass
class SkillNode:
    """A single skill in the curriculum graph."""

    id: str
    scenario: str
    prerequisites: list[str] = field(default_factory=list)
    reward_profile: list[dict[str, Any]] = field(default_factory=list)
    mastery_threshold: dict[str, float] = field(default_factory=dict)
    timesteps: int = 100_000
    max_steps: int = 400
    transfer_from: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillNode":
        return cls(
            id=str(data["id"]),
            scenario=str(data.get("scenario", data["id"])),
            prerequisites=list(data.get("prerequisites", [])),
            reward_profile=list(data.get("rewards", data.get("reward_profile", []))),
            mastery_threshold=dict(data.get("mastery_threshold", {})),
            timesteps=int(data.get("timesteps", 100_000)),
            max_steps=int(data.get("max_steps", 400)),
            transfer_from=list(data.get("transfer_from", [])),
        )

    def to_stage_dict(self) -> dict[str, Any]:
        return {
            "skill": self.id,
            "scenario": self.scenario,
            "timesteps": self.timesteps,
            "max_steps": self.max_steps,
        }


class SkillGraph:
    """Directed acyclic skill graph with topological ordering."""

    def __init__(self, nodes: list[SkillNode] | list[dict[str, Any]]) -> None:
        parsed = [
            n if isinstance(n, SkillNode) else SkillNode.from_dict(n)
            for n in nodes
        ]
        self.nodes: dict[str, SkillNode] = {n.id: n for n in parsed}
        self._validate()

    def _validate(self) -> None:
        for node in self.nodes.values():
            for prereq in node.prerequisites:
                if prereq not in self.nodes:
                    raise ValueError(f"Unknown prerequisite {prereq!r} for {node.id}")

    def topological_order(self) -> list[SkillNode]:
        visited: set[str] = set()
        order: list[SkillNode] = []

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for prereq in self.nodes[node_id].prerequisites:
                visit(prereq)
            order.append(self.nodes[node_id])

        for nid in self.nodes:
            visit(nid)
        return order

    def to_linear_stages(self) -> list[dict[str, Any]]:
        return [n.to_stage_dict() for n in self.topological_order()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_graph": {
                "nodes": [
                    {
                        "id": n.id,
                        "scenario": n.scenario,
                        "prerequisites": n.prerequisites,
                        "rewards": n.reward_profile,
                        "mastery_threshold": n.mastery_threshold,
                        "timesteps": n.timesteps,
                        "max_steps": n.max_steps,
                    }
                    for n in self.nodes.values()
                ]
            }
        }

    @classmethod
    def from_config(cls, curriculum_cfg: dict[str, Any]) -> "SkillGraph":
        sg = curriculum_cfg.get("skill_graph", {})
        return cls(sg.get("nodes", []))

    def is_mastered(
        self,
        skill_id: str,
        results: list[EpisodeResult],
        tactical: list[EpisodeTacticalMetrics] | None = None,
    ) -> bool:
        node = self.nodes[skill_id]
        thresholds = node.mastery_threshold or {"scoring_rate": 20.0}
        summary = summarize_failures(results)
        agg = aggregate_tactical(tactical or [r.tactical for r in results if r.tactical])
        metrics = {**summary, **agg}
        for key, threshold in thresholds.items():
            if metrics.get(key, 0.0) < threshold:
                return False
        return True

    def export_dot(self) -> str:
        lines = ["digraph skill_graph {"]
        for node in self.nodes.values():
            for prereq in node.prerequisites:
                lines.append(f'  "{prereq}" -> "{node.id}";')
            if not node.prerequisites:
                lines.append(f'  "{node.id}";')
        lines.append("}")
        return "\n".join(lines)
