"""Offline structured coaching and optional LLM explanations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CoachConfig:
    """Configuration for coaching output generation."""

    provider: str = "offline"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    max_actions: int = 50


@dataclass
class Coach:
    """Produce coaching explanations from episode/diagnosis artifacts."""

    config: CoachConfig = field(default_factory=CoachConfig)

    def from_episode_file(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return build_coaching_output(data, self.config)

    def explain(self, episode_data: dict[str, Any]) -> str:
        output = build_coaching_output(episode_data, self.config)
        if self.config.provider == "offline":
            return output["markdown"]
        return _llm_explain(output, self.config)


def build_coaching_output(
    episode_data: dict[str, Any],
    config: CoachConfig | None = None,
) -> dict[str, Any]:
    """Build structured JSON + Markdown coaching summary (no LLM required)."""
    config = config or CoachConfig()
    traj = episode_data.get("trajectory_summary", episode_data)
    diagnosis = episode_data.get("diagnosis", {})
    tactical = episode_data.get("tactical", {})

    actions = traj.get("actions", [])[: config.max_actions]
    failure = diagnosis.get("primary_failure") or episode_data.get("failure_label", "unknown")
    goals = traj.get("goals", episode_data.get("goals", 0))
    steps = traj.get("steps", episode_data.get("steps", 0))

    recommendations = _recommendations(failure, tactical)
    structured = {
        "episode_id": episode_data.get("episode", episode_data.get("episode_id")),
        "scenario": traj.get("scenario", episode_data.get("scenario", "")),
        "outcome": "scored" if goals > 0 else "missed",
        "steps": steps,
        "failure_label": failure,
        "tactical_summary": {
            "distance_advanced": tactical.get("distance_advanced"),
            "shots": tactical.get("shots"),
            "box_entries": tactical.get("box_entries"),
            "possession_losses": tactical.get("possession_losses"),
            "approx_xg": tactical.get("approx_xg"),
        },
        "action_sequence_length": len(actions),
        "possession_changes": len(traj.get("possession_lost", [])),
        "recommendations": recommendations,
    }

    md_lines = [
        "# Episode Coaching Summary",
        "",
        f"**Scenario:** {structured['scenario']}",
        f"**Outcome:** {structured['outcome']} in {steps} steps",
        "",
        "## Primary Issue",
        f"- {failure.replace('_', ' ')}",
        "",
        "## Tactical Snapshot",
    ]
    for k, v in structured["tactical_summary"].items():
        if v is not None:
            md_lines.append(f"- {k.replace('_', ' ')}: {v}")
    md_lines.extend(["", "## Recommendations"])
    for rec in recommendations:
        md_lines.append(f"- {rec}")

    structured["markdown"] = "\n".join(md_lines)
    return structured


def _recommendations(failure: str, tactical: dict[str, Any]) -> list[str]:
    mapping = {
        "lost_possession": [
            "Protect the ball earlier; reduce risky dribbles in midfield.",
            "Look for a simple pass before pressure arrives.",
        ],
        "poor_shot_timing": [
            "Shoot when the ball enters the box, not after defenders recover.",
            "Take the first good chance instead of extra touches.",
        ],
        "low_field_progression": [
            "Advance the ball toward the goal before attempting passes sideways.",
            "Use forward runs to create space.",
        ],
        "no_shot_generated": [
            "Enter the penalty area and attempt a shot before timeout.",
            "Prefer shoot actions when within scoring range.",
        ],
        "incorrect_pass_choice": [
            "When passing, target the open attacker closer to goal.",
            "Avoid lateral passes that do not improve xG.",
        ],
        "dribble_into_pressure": [
            "Release the ball before entering a congested zone.",
            "Use skill moves only when a lane to goal is open.",
        ],
        "defensive_clearance": [
            "Press higher to regain possession in the attacking third.",
            "Cross or shoot earlier before the ball is cleared.",
        ],
        "timeout_or_random": [
            "Increase urgency after step 200; prioritize shots or crosses.",
        ],
    }
    recs = list(mapping.get(failure, ["Review action sequence for wasted steps."]))
    if tactical.get("approx_xg", 0) and tactical["approx_xg"] < 0.1:
        recs.append("Improve shot location — current attempts have very low xG.")
    return recs


def _llm_explain(structured: dict[str, Any], config: CoachConfig) -> str:
    """Optional LLM provider hook (offline by default)."""
    return structured["markdown"]
