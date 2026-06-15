"""Tests for scenario specs and generation."""

from __future__ import annotations

from opengoalrl.scenarios.generator import ScenarioGenerator, generate_scenario
from opengoalrl.scenarios.spec import ScenarioSpec, parse_scenario_spec


class TestScenarioSpec:
    def test_from_dict(self):
        spec = ScenarioSpec.from_dict({
            "name": "test",
            "attackers": 3,
            "defenders": 2,
            "ball_position": [0.8, 0.1],
        })
        assert spec.attackers == 3
        assert spec.ball_position == (0.8, 0.1)

    def test_to_dict_roundtrip(self):
        spec = ScenarioSpec(name="corner", attackers=2, keeper=True)
        restored = ScenarioSpec.from_dict(spec.to_dict())
        assert restored.name == "corner"
        assert restored.attackers == 2


class TestScenarioGenerator:
    def test_from_fixed_resolves_grf(self):
        gen = ScenarioGenerator(seed=0)
        spec = gen.from_fixed({"attackers": 3, "defenders": 1, "keeper": True})
        assert spec.grf_scenario is not None
        assert "academy" in spec.grf_scenario

    def test_sample_reproducible(self):
        g1 = ScenarioGenerator(seed=42)
        g2 = ScenarioGenerator(seed=42)
        assert g1.sample({}).to_dict() == g2.sample({}).to_dict()

    def test_generate_with_sample_ranges(self):
        spec = generate_scenario({"sample_ranges": {"attackers": [2, 2]}}, seed=1)
        assert spec.attackers == 2
