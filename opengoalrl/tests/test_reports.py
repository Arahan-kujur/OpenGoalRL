"""Tests for report generation."""

from __future__ import annotations

from pathlib import Path

from opengoalrl.reports.builder import ReportBuilder, generate_report


class TestReportBuilder:
    def test_render_markdown(self, tmp_path):
        fixtures = Path(__file__).resolve().parent / "fixtures"
        builder = ReportBuilder(
            tmp_path,
            eval_csv=fixtures / "eval_metrics.csv",
            diagnosis_json=fixtures / "diagnosis.json",
        )
        md = builder.render_markdown()
        assert "Evaluation Summary" in md
        assert "Failure Diagnosis" in md

    def test_generate_report(self, tmp_path):
        fixtures = Path(__file__).resolve().parent / "fixtures"
        (tmp_path / "eval_metrics.csv").write_text(
            (fixtures / "eval_metrics.csv").read_text()
        )
        (tmp_path / "diagnosis.json").write_text(
            (fixtures / "diagnosis.json").read_text()
        )
        out = generate_report(tmp_path)
        assert out.exists()
        assert "OpenGoalRL Experiment Report" in out.read_text()
