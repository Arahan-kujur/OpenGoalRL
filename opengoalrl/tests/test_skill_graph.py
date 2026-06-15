"""Tests for skill graph curricula."""

from __future__ import annotations

import pytest

from opengoalrl.curriculum.skill_graph import SkillGraph, SkillNode
from opengoalrl.utils.rollout import EpisodeResult


class TestSkillGraph:
    def test_topological_order(self):
        nodes = [
            SkillNode(id="a", scenario="empty_goal_close"),
            SkillNode(id="b", scenario="empty_goal", prerequisites=["a"]),
            SkillNode(id="c", scenario="corner_kick", prerequisites=["b"]),
        ]
        graph = SkillGraph(nodes)
        order = [n.id for n in graph.topological_order()]
        assert order.index("a") < order.index("b") < order.index("c")

    def test_unknown_prerequisite_raises(self):
        with pytest.raises(ValueError):
            SkillGraph([SkillNode(id="x", scenario="a", prerequisites=["missing"])])

    def test_to_linear_stages(self):
        graph = SkillGraph([
            {"id": "shoot", "scenario": "empty_goal_close", "timesteps": 1000},
            {"id": "finish", "scenario": "corner_kick", "prerequisites": ["shoot"]},
        ])
        stages = graph.to_linear_stages()
        assert len(stages) == 2
        assert stages[0]["scenario"] == "empty_goal_close"

    def test_is_mastered(self):
        graph = SkillGraph([
            SkillNode(
                id="shoot",
                scenario="empty_goal_close",
                mastery_threshold={"scoring_rate": 50.0},
            ),
        ])
        results = [
            EpisodeResult(1, 1.0, 50, 1, 0, 0),
            EpisodeResult(2, 1.0, 50, 1, 0, 0),
        ]
        assert graph.is_mastered("shoot", results)

    def test_export_dot(self):
        graph = SkillGraph([
            SkillNode(id="a", scenario="x"),
            SkillNode(id="b", scenario="y", prerequisites=["a"]),
        ])
        dot = graph.export_dot()
        assert "a" in dot and "b" in dot
