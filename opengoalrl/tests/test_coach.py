"""Tests for offline coaching."""

from __future__ import annotations

from opengoalrl.coach.coach import Coach, build_coaching_output


class TestCoach:
    def test_offline_output(self):
        data = {
            "episode": 42,
            "trajectory_summary": {
                "scenario": "corner_kick",
                "actions": [5, 6, 12],
                "steps": 100,
                "goals": 0,
            },
            "failure_label": "poor_shot_timing",
            "tactical": {"shots": 1, "approx_xg": 0.05, "distance_advanced": 0.4},
        }
        output = build_coaching_output(data)
        assert "markdown" in output
        assert "poor shot timing" in output["markdown"].lower()
        assert len(output["recommendations"]) >= 1

    def test_coach_scored_outcome(self):
        result = build_coaching_output({
            "trajectory_summary": {"goals": 1, "steps": 50, "scenario": "empty_goal"},
        })
        assert result["outcome"] == "scored"
