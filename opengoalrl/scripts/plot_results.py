"""Plot training curves, baseline comparisons, and ablation results.

Usage::

    python -m opengoalrl.scripts.plot_results \\
        --training models/training_metrics.csv \\
        --baseline models/baseline_metrics.csv \\
        --eval     models/eval_metrics.csv \\
        --ablation models/ablation/ablation_metrics.csv \\
        --outdir   results/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str | Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                try:
                    data.setdefault(key, []).append(float(val))
                except ValueError:
                    data.setdefault(key, []).append(val)  # type: ignore[arg-type]
    return data


def plot_learning_curve(training_csv: Path, out_path: Path) -> None:
    """Reward over training timesteps."""
    data = _read_csv(training_csv)
    timesteps = data["timestep"]
    rewards = data["mean_reward"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timesteps, rewards, linewidth=1.5, label="PPO (shaped reward)")
    if "mean_distance_advanced" in data:
        ax2 = ax.twinx()
        ax2.plot(
            timesteps, data["mean_distance_advanced"],
            color="tab:orange", alpha=0.6, label="Dist advanced",
        )
        ax2.set_ylabel("Tactical (dist advanced)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Training Learning Curve")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved learning curve to {out_path}")


def plot_comparison(baseline_csv: Path, eval_csv: Path, out_path: Path) -> None:
    """Bar chart comparing random baseline vs trained agent."""
    bl = _read_csv(baseline_csv)
    ev = _read_csv(eval_csv)

    def pct(values: list) -> float:
        return 100.0 * sum(1 for v in values if float(v) > 0) / len(values)

    metrics = ["Goals (%)", "Shots (%)", "Ball in Box (%)", "Mean Reward"]
    baseline_vals = [
        pct(bl["goals"]),
        pct(bl["shots"]),
        pct(bl["ball_in_box"]),
        float(np.mean([float(v) for v in bl["reward"]])),
    ]
    trained_vals = [
        pct(ev["goals"]),
        pct(ev["shots"]),
        pct(ev["ball_in_box"]),
        float(np.mean([float(v) for v in ev["reward"]])),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})

    pct_metrics = metrics[:3]
    x = np.arange(len(pct_metrics))
    width = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, baseline_vals[:3], width, label="Random Baseline")
    bars2 = ax.bar(x + width / 2, trained_vals[:3], width, label="Trained PPO")
    ax.set_ylabel("% of Episodes")
    ax.set_title("Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(pct_metrics)
    ax.legend()
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", alpha=0.3)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    x2 = np.arange(1)
    ax2.bar(x2 - width / 2, [baseline_vals[3]], width, label="Random Baseline")
    ax2.bar(x2 + width / 2, [trained_vals[3]], width, label="Trained PPO")
    ax2.set_ylabel("Reward")
    ax2.set_title("Mean Reward")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Reward"])
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Trained Agent vs Random Baseline", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to {out_path}")


def plot_ablation(ablation_csv: Path, out_path: Path) -> None:
    """Multi-line learning curve showing ablation variants."""
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    with open(ablation_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row.get("label") or row.get("variant", "unknown")
            data[variant]["timestep"].append(float(row["timestep"]))
            data[variant]["mean_reward"].append(float(row["mean_reward"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, vals in data.items():
        ax.plot(vals["timestep"], vals["mean_reward"], linewidth=1.5, label=variant)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward Ablation -- Shaped vs Scoring-Only vs Dense-Only")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved ablation chart to {out_path}")


def plot_curriculum(curriculum_csv: Path, out_path: Path) -> None:
    """Learning curve across curriculum stages."""
    data = _read_csv(curriculum_csv)
    if "stage" not in data:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    stages = sorted(set(int(s) for s in data["stage"]))
    for stage in stages:
        mask = [int(s) == stage for s in data["stage"]]
        ts = [data["timestep"][i] for i, m in enumerate(mask) if m]
        rw = [data["mean_reward"][i] for i, m in enumerate(mask) if m]
        scenario = next(
            (data["scenario"][i] for i, m in enumerate(mask) if m),
            str(stage),
        )
        ax.plot(ts, rw, linewidth=1.5, label=f"Stage {stage}: {scenario}")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Curriculum Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved curriculum chart to {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot OpenGoalRL results")
    parser.add_argument("--training", type=str, default="models/training_metrics.csv")
    parser.add_argument("--baseline", type=str, default="models/baseline_metrics.csv")
    parser.add_argument("--eval", type=str, default="models/eval_metrics.csv")
    parser.add_argument("--ablation", type=str, default="models/ablation/ablation_metrics.csv")
    parser.add_argument("--curriculum", type=str, default="models/curriculum/curriculum_metrics.csv")
    parser.add_argument("--outdir", type=str, default="results/")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    training_path = Path(args.training)
    baseline_path = Path(args.baseline)
    eval_path = Path(args.eval)
    ablation_path = Path(args.ablation)
    curriculum_path = Path(args.curriculum)

    if training_path.exists():
        plot_learning_curve(training_path, outdir / "learning_curve.png")
    else:
        print(f"Training CSV not found: {training_path}")

    if baseline_path.exists() and eval_path.exists():
        plot_comparison(baseline_path, eval_path, outdir / "comparison.png")
    else:
        missing = []
        if not baseline_path.exists():
            missing.append(str(baseline_path))
        if not eval_path.exists():
            missing.append(str(eval_path))
        print(f"Skipping comparison -- missing: {', '.join(missing)}")

    if ablation_path.exists():
        plot_ablation(ablation_path, outdir / "ablation.png")
    else:
        print(f"Ablation CSV not found: {ablation_path}")

    if curriculum_path.exists():
        plot_curriculum(curriculum_path, outdir / "curriculum.png")
    else:
        print(f"Curriculum CSV not found: {curriculum_path}")


if __name__ == "__main__":
    main()
