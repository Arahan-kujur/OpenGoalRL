"""Research report generation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from opengoalrl.reports.builder import ReportBuilder, generate_report
from opengoalrl.utils.logger import get_logger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate OpenGoalRL research report")
    parser.add_argument(
        "experiment_dir",
        type=str,
        nargs="?",
        default="models/curriculum/",
    )
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "html"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    exp = Path(args.experiment_dir)
    out = args.output
    if out is None:
        ext = "html" if args.format == "html" else "md"
        out = exp / f"report.{ext}"

    path = generate_report(exp, output=out, fmt=args.format)
    logger = get_logger()
    logger.info("Report saved to %s", path)

    builder = ReportBuilder(exp)
    plot_dir = exp / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        from opengoalrl.scripts.plot_results import (
            plot_ablation,
            plot_comparison,
            plot_curriculum,
            plot_learning_curve,
        )
        if builder.training_csv and builder.training_csv.exists():
            plot_learning_curve(builder.training_csv, plot_dir / "learning_curve.png")
        if builder.baseline_csv and builder.eval_csv:
            if builder.baseline_csv.exists() and builder.eval_csv.exists():
                plot_comparison(
                    builder.baseline_csv, builder.eval_csv, plot_dir / "comparison.png",
                )
        if builder.ablation_csv and builder.ablation_csv.exists():
            plot_ablation(builder.ablation_csv, plot_dir / "ablation.png")
        if builder.eval_csv and builder.eval_csv.name == "curriculum_metrics.csv":
            plot_curriculum(builder.eval_csv, plot_dir / "curriculum.png")
        logger.info("Plots saved to %s", plot_dir)
    except Exception as exc:
        logger.warning("Plot generation skipped: %s", exc)


if __name__ == "__main__":
    main()
