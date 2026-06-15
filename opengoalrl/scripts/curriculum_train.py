"""Sequential curriculum trainer -- progress through scenarios by difficulty."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from opengoalrl.utils.env_factory import build_env, set_seed as _set_seed
from opengoalrl.agents.ppo_agent import PPOAgent
from opengoalrl.curriculum.skill_graph import SkillGraph
from opengoalrl.utils.config_loader import load_config, validate_config
from opengoalrl.utils.logger import get_logger, save_config_snapshot
from opengoalrl.utils.metrics_callback import MetricsCallback


def _resolve_stages(config: dict) -> list[dict]:
    curriculum = config["curriculum"]
    if "skill_graph" in curriculum:
        graph = SkillGraph.from_config(curriculum)
        return graph.to_linear_stages()
    return curriculum["stages"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Curriculum training for OpenGoalRL")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "configs" / "curriculum.yaml"
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_config(config, required_sections=("curriculum",))
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})
    stages = _resolve_stages(config)

    seed = train_cfg.get("seed", 42)
    _set_seed(seed)

    save_dir = Path(log_cfg.get("save_dir", "models/curriculum/"))
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir=log_cfg.get("log_dir"))
    logger.info("Curriculum training: %d stages", len(stages))
    save_config_snapshot(config, save_dir)

    combined_csv = save_dir / "curriculum_metrics.csv"
    with open(combined_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "stage", "scenario", "timestep", "mean_reward",
            "mean_ep_length", "episodes", "total_goals",
        ])

    prev_model_path: Path | None = None
    cumulative_steps = 0

    for idx, stage in enumerate(stages, 1):
        scenario = stage["scenario"]
        timesteps = stage["timesteps"]
        max_steps = stage.get("max_steps", 400)

        logger.info(
            "--- Stage %d/%d: %s (%d timesteps) ---",
            idx, len(stages), scenario, timesteps,
        )

        stage_config = {
            "environment": {
                "scenario": scenario,
                "max_steps": max_steps,
                "render": False,
            },
            "rewards": config.get("rewards", []),
            "training": train_cfg,
        }

        env = build_env(stage_config)

        if prev_model_path is not None:
            logger.info("Loading model from previous stage: %s", prev_model_path)
            agent = PPOAgent.load(prev_model_path, env=env)
        else:
            agent = PPOAgent(env, config=train_cfg)

        stage_csv = save_dir / f"stage_{idx}_{scenario}.csv"
        callback = MetricsCallback(csv_path=stage_csv)

        agent.train(total_timesteps=timesteps, callback=callback)

        model_path = save_dir / f"stage_{idx}_{scenario}"
        agent.save(model_path)
        prev_model_path = model_path

        if stage_csv.exists():
            with open(stage_csv, newline="") as src, \
                 open(combined_csv, "a", newline="") as dst:
                reader = csv.DictReader(src)
                writer = csv.writer(dst)
                for row in reader:
                    writer.writerow([
                        idx,
                        scenario,
                        cumulative_steps + int(float(row["timestep"])),
                        row["mean_reward"],
                        row["mean_ep_length"],
                        row["episodes"],
                        row["total_goals"],
                    ])
            cumulative_steps += timesteps

        env.close()
        logger.info("Stage %d complete. Model saved to %s", idx, model_path)

    logger.info("Curriculum training complete. Combined metrics: %s", combined_csv)


if __name__ == "__main__":
    main()
