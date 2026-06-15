"""Tests for automatic curriculum discovery."""

from __future__ import annotations

from opengoalrl.curriculum.auto import AutoCurriculum


class TestAutoCurriculum:
    def test_discover_builds_stages(self):
        ac = AutoCurriculum(target="corner_kick", budget=5, seed=0)
        ac.discover()
        assert len(ac.stages) >= 1
        assert ac.stages[-1]["scenario"] == "corner_kick"

    def test_to_skill_graph(self):
        ac = AutoCurriculum(target="corner_kick", budget=3, seed=0)
        ac.discover()
        graph = ac.to_skill_graph()
        assert len(graph.topological_order()) == len(ac.stages)

    def test_save(self, tmp_path):
        ac = AutoCurriculum(target="corner_kick", budget=3, seed=0)
        ac.discover()
        path = ac.save(tmp_path / "auto.yaml")
        assert path.exists()
        assert path.with_suffix(".meta.json").exists()
