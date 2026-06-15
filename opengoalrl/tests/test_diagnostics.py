"""Tests for failure classifiers."""

from __future__ import annotations

from opengoalrl.diagnostics.classifier import FailureClassifier, summarize_failures
from opengoalrl.metrics.tactical import EpisodeTacticalMetrics
from opengoalrl.utils.rollout import EpisodeResult


def _result(**kwargs) -> EpisodeResult:
    defaults = dict(
        episode=1, reward=0.0, steps=200, goals=0,
        shots=0, ball_in_box=0, tactical=None, trajectory_summary={},
    )
    defaults.update(kwargs)
    return EpisodeResult(**defaults)


class TestFailureClassifier:
    def test_success_on_goal(self):
        clf = FailureClassifier()
        assert clf.classify(_result(goals=1)) == "success"

    def test_lost_possession(self):
        clf = FailureClassifier()
        tactical = EpisodeTacticalMetrics(possession_losses=3)
        assert clf.classify(_result(tactical=tactical)) == "lost_possession"

    def test_no_shot(self):
        clf = FailureClassifier()
        tactical = EpisodeTacticalMetrics(shots=0, distance_advanced=0.3)
        assert clf.classify(_result(tactical=tactical)) == "no_shot_generated"

    def test_summarize_failures(self):
        results = [
            _result(goals=1),
            _result(tactical=EpisodeTacticalMetrics(shots=0, distance_advanced=0.0)),
        ]
        summary = summarize_failures(results)
        assert summary["scoring_rate"] == 50.0
        assert "failures" in summary
