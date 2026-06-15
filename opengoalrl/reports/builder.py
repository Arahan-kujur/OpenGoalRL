"""Aggregate CSVs, configs, and diagnosis JSON into research reports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ReportBuilder:
    """Collect experiment artifacts and render Markdown/HTML reports."""

    experiment_dir: Path
    training_csv: Path | None = None
    eval_csv: Path | None = None
    baseline_csv: Path | None = None
    ablation_csv: Path | None = None
    diagnosis_json: Path | None = None
    config_snapshots: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.experiment_dir = Path(self.experiment_dir)
        self._discover_artifacts()

    def _discover_artifacts(self) -> None:
        if self.training_csv is None:
            candidate = self.experiment_dir / "training_metrics.csv"
            if candidate.exists():
                self.training_csv = candidate
        if self.eval_csv is None:
            for name in ("eval_metrics.csv", "curriculum_metrics.csv"):
                candidate = self.experiment_dir / name
                if candidate.exists():
                    self.eval_csv = candidate
                    break
        if self.baseline_csv is None:
            candidate = self.experiment_dir / "baseline_metrics.csv"
            if candidate.exists():
                self.baseline_csv = candidate
        if self.ablation_csv is None:
            candidate = self.experiment_dir / "ablation_metrics.csv"
            if candidate.exists():
                self.ablation_csv = candidate
        if self.diagnosis_json is None:
            candidate = self.experiment_dir / "diagnosis.json"
            if candidate.exists():
                self.diagnosis_json = candidate
        if not self.config_snapshots:
            self.config_snapshots = sorted(self.experiment_dir.glob("config_*.json"))

    def load_csv_summary(self, path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        numeric: dict[str, list[float]] = {}
        for row in rows:
            for k, v in row.items():
                try:
                    numeric.setdefault(k, []).append(float(v))
                except ValueError:
                    pass
        summary: dict[str, Any] = {"rows": len(rows)}
        for k, vals in numeric.items():
            summary[f"mean_{k}"] = float(np.mean(vals))
            summary[f"std_{k}"] = float(np.std(vals))
        if "goals" in numeric:
            summary["scoring_rate"] = 100.0 * sum(v > 0 for v in numeric["goals"]) / len(rows)
        return summary

    def load_diagnosis(self) -> dict[str, Any]:
        if self.diagnosis_json is None or not self.diagnosis_json.exists():
            return {}
        with open(self.diagnosis_json) as f:
            return json.load(f)

    def build_sections(self) -> dict[str, Any]:
        return {
            "training": self.load_csv_summary(self.training_csv),
            "evaluation": self.load_csv_summary(self.eval_csv),
            "baseline": self.load_csv_summary(self.baseline_csv),
            "ablation": self.load_csv_summary(self.ablation_csv),
            "diagnosis": self.load_diagnosis(),
            "configs": [p.name for p in self.config_snapshots],
        }

    def render_markdown(self) -> str:
        sections = self.build_sections()
        lines = [
            "# OpenGoalRL Experiment Report",
            "",
            f"**Experiment directory:** `{self.experiment_dir}`",
            "",
        ]

        if sections["training"]:
            t = sections["training"]
            lines.extend([
                "## Training Summary",
                f"- Rollout checkpoints: {t.get('rows', 0)}",
                f"- Mean reward: {t.get('mean_mean_reward', 0):.2f} "
                f"(+/- {t.get('std_mean_reward', 0):.2f})",
                "",
            ])

        if sections["evaluation"]:
            e = sections["evaluation"]
            lines.extend([
                "## Evaluation Summary",
                f"- Episodes: {e.get('rows', 0)}",
                f"- Scoring rate: {e.get('scoring_rate', 0):.1f}%",
                f"- Mean reward: {e.get('mean_reward', 0):.2f}",
                "",
            ])

        if sections["baseline"]:
            b = sections["baseline"]
            lines.extend([
                "## Baseline Comparison",
                f"- Random baseline scoring rate: {b.get('scoring_rate', 0):.1f}%",
                "",
            ])

        diag = sections["diagnosis"]
        if diag:
            lines.extend([
                "## Failure Diagnosis",
                f"- Scoring rate: {diag.get('scoring_rate', 0):.1f}%",
                "",
                "### Failure Breakdown",
            ])
            for label, pct in diag.get("failures", {}).items():
                lines.append(f"- {pct:.0f}% {label.replace('_', ' ')}")
            lines.append("")

        if sections["configs"]:
            lines.extend([
                "## Config Snapshots",
                *[f"- `{name}`" for name in sections["configs"]],
                "",
            ])

        return "\n".join(lines)

    def render_html(self) -> str:
        md = self.render_markdown()
        body = md.replace("\n", "<br>\n")
        return f"<!DOCTYPE html><html><body>{body}</body></html>"

    def save(self, output: Path, fmt: str = "markdown") -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "html":
            output.write_text(self.render_html())
        else:
            output.write_text(self.render_markdown())
        return output


def generate_report(
    experiment_dir: str | Path,
    output: str | Path | None = None,
    fmt: str = "markdown",
) -> Path:
    builder = ReportBuilder(Path(experiment_dir))
    out = Path(output) if output else Path(experiment_dir) / f"report.{fmt if fmt != 'markdown' else 'md'}"
    return builder.save(out, fmt=fmt)
