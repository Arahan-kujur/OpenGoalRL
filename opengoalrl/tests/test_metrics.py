"""Tests for tactical metrics."""

from __future__ import annotations

import numpy as np
import pytest

from opengoalrl.metrics.tactical import TacticalMetricsTracker, aggregate_tactical


def _obs(ball_x=0.0, ball_y=0.0) -> np.ndarray:
    obs = np.zeros(115, dtype=np.float32)
    obs[88] = ball_x
    obs[89] = ball_y
    return obs


class TestTacticalMetricsTracker:
    def test_distance_advanced(self):
        tracker = TacticalMetricsTracker()
        tracker.step(_obs(0.0), 5, {})
        tracker.step(_obs(0.4), 5, {})
        m = tracker.finalize()
        assert m.distance_advanced == pytest.approx(0.4)

    def test_shot_counted(self):
        tracker = TacticalMetricsTracker()
        tracker.step(_obs(0.9), 12, {})
        m = tracker.finalize()
        assert m.shots == 1
        assert m.shot_timing_step == 1

    def test_box_entry(self):
        tracker = TacticalMetricsTracker()
        tracker.step(_obs(0.5), 0, {})
        tracker.step(_obs(0.85, 0.0), 0, {})
        m = tracker.finalize()
        assert m.box_entries == 1

    def test_aggregate(self):
        t1 = TacticalMetricsTracker()
        t1.step(_obs(0.0), 12, {"score_reward": 1.0})
        t2 = TacticalMetricsTracker()
        m = aggregate_tactical([t1.finalize(), t2.finalize()])
        assert m["scoring_rate"] == 50.0
