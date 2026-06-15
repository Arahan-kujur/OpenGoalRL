"""Rule-based failure classifiers for football episodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from opengoalrl.metrics.tactical import EpisodeTacticalMetrics
from opengoalrl.utils.rollout import EpisodeResult

FAILURE_LABELS = (
    "lost_possession",
    "poor_shot_timing",
    "low_field_progression",
    "no_shot_generated",
    "incorrect_pass_choice",
    "dribble_into_pressure",
    "defensive_clearance",
    "timeout_or_random",
)

_PASS_ACTIONS = {9, 10, 11}
_DRIBBLE_ACTIONS = {5, 6, 7, 8}
_SHOT_ACTION = 12


@dataclass
class FailureClassifier:
    """Thresholds for rule-based failure classification."""

    min_distance_advanced: float = 0.15
    late_shot_step: int = 250
    high_pressure: float = 0.55
    min_possession_steps: int = 5

    def classify(
        self,
        result: EpisodeResult,
        tactical: EpisodeTacticalMetrics | None = None,
    ) -> str:
        if result.goals > 0:
            return "success"
        tactical = tactical or result.tactical
        traj = result.trajectory_summary
        actions = traj.get("actions", [])

        if result.steps >= 399 and result.goals == 0:
            return "timeout_or_random"

        if tactical is not None:
            if tactical.possession_losses >= 2:
                return "lost_possession"
            if tactical.distance_advanced < self.min_distance_advanced:
                return "low_field_progression"
            if tactical.shots == 0:
                return "no_shot_generated"
            if (
                tactical.shot_timing_step is not None
                and tactical.shot_timing_step > self.late_shot_step
            ):
                return "poor_shot_timing"
            if tactical.pressure_proxy >= self.high_pressure and any(
                a in _DRIBBLE_ACTIONS for a in actions[-5:]
            ):
                return "dribble_into_pressure"
            if tactical.box_entries == 0 and tactical.distance_advanced < 0.3:
                return "defensive_clearance"

        if actions:
            pass_count = sum(1 for a in actions if a in _PASS_ACTIONS)
            dribble_count = sum(1 for a in actions if a in _DRIBBLE_ACTIONS)
            if pass_count > dribble_count and result.shots == 0:
                return "incorrect_pass_choice"
            if result.shots == 0:
                return "no_shot_generated"
            if result.ball_in_box == 0 and result.shots > 0:
                return "poor_shot_timing"

        return "timeout_or_random"


def classify_episode(
    result: EpisodeResult,
    classifier: FailureClassifier | None = None,
) -> str:
    clf = classifier or FailureClassifier()
    return clf.classify(result)


def summarize_failures(
    results: list[EpisodeResult],
    classifier: FailureClassifier | None = None,
) -> dict[str, Any]:
    clf = classifier or FailureClassifier()
    labels = [clf.classify(r) for r in results]
    counts = Counter(labels)
    n = len(results) or 1
    scored = sum(1 for r in results if r.goals > 0)
    failures = {
        k: v for k, v in counts.items() if k != "success"
    }
    total_failures = sum(failures.values()) or 1
    breakdown = {
        label: 100.0 * count / total_failures
        for label, count in sorted(failures.items(), key=lambda x: -x[1])
    }
    return {
        "scoring_rate": 100.0 * scored / n,
        "n_episodes": n,
        "failures": breakdown,
        "failure_counts": dict(failures),
        "labels": labels,
    }
