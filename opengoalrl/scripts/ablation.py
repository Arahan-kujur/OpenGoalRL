"""Reward ablation experiment -- compare shaped vs sparse vs dense-only."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

from opengoalrl.agents.ppo_agent import PPOAgent
from opengoalrl.utils.config_loader import load_config, validate_config
from opengoalrl.utils.env_factory import build_env, set_seed as _set_seed
from opengoalrl.utils.logger import get_logger, save_config_snapshot
from opengoalrl.utils.metrics_callback import MetricsCallback


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reward ablation experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "configs" / "ablation_empty_goal.yaml"
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_config(config, required_sections=("environment", "ablation"))
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})
    variants = config["ablation"]["variants"]

    seed = train_cfg.get("seed", 42)
    save_dir = Path(log_cfg.get("save_dir", "models/ablation/"))
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir=log_cfg.get("log_dir"))
    logger.info("Ablation experiment: %d variants", len(variants))
    save_config_snapshot(config, save_dir)

    combined_csv = save_dir / "ablation_metrics.csv"
    with open(combined_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "variant", "label", "timestep", "mean_reward",
            "mean_ep_length", "episodes", "total_goals",
        ])

    total_timesteps = train_cfg.get("total_timesteps", 100_000)

    for name, variant in variants.items():
        _set_seed(seed)
        label = variant.get("label", name)
        logger.info("--- Variant: %s (%s) ---", name, label)

        variant_config = copy.deepcopy(config)
        variant_config["rewards"] = variant["rewards"]

        env = build_env(variant_config)
        agent = PPOAgent(env, config=train_cfg)

        variant_csv = save_dir / f"{name}_metrics.csv"
        callback = MetricsCallback(csv_path=variant_csv)

        agent.train(total_timesteps=total_timesteps, callback=callback)

        model_path = save_dir / f"ppo_{name}"
        agent.save(model_path)

        if variant_csv.exists():
            with open(variant_csv, newline="") as src, \
                 open(combined_csv, "a", newline="") as dst:
                reader = csv.DictReader(src)
                writer = csv.writer(dst)
                for row in reader:
                    writer.writerow([
                        name, label,
                        row["timestep"], row["mean_reward"],
                        row["mean_ep_length"], row["episodes"],
                        row["total_goals"],
                    ])

        env.close()
        logger.info("Variant %s complete. Model: %s", name, model_path)

    logger.info("Ablation complete. Combined metrics: %s", combined_csv)


if __name__ == "__main__":
    main()
