"""Failure diagnosis CLI over evaluated trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opengoalrl.agents.ppo_agent import PPOAgent
from opengoalrl.diagnostics.classifier import summarize_failures
from opengoalrl.utils.env_factory import build_env
from opengoalrl.utils.config_loader import load_config
from opengoalrl.utils.logger import get_logger
from opengoalrl.utils.rollout import run_rollouts, summarize_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose agent failures")
    parser.add_argument("model", type=str, help="Path to saved model (.zip ok)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "corner.yaml"),
    )
    parser.add_argument("--output", type=str, default="models/diagnosis.json")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    n_episodes = args.episodes or eval_cfg.get("n_episodes", 10)

    logger = get_logger(log_dir=config.get("logging", {}).get("log_dir"))
    env = build_env(config)
    model_path = args.model.removesuffix(".zip")
    agent = PPOAgent.load(model_path, env=env)

    def policy(obs):
        action, _ = agent.predict(obs)
        return action

    results = run_rollouts(
        env,
        policy,
        n_episodes,
        collect_tactical=True,
        collect_trajectory=True,
    )
    diagnosis = summarize_failures(results)
    summary = summarize_results(results)
    diagnosis["summary"] = summary

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnosis, indent=2))

    logger.info("Scoring rate: %.1f%%", diagnosis["scoring_rate"])
    logger.info("Failures:")
    for label, pct in diagnosis["failures"].items():
        logger.info("- %.0f%% %s", pct, label.replace("_", " "))
    logger.info("Diagnosis saved to %s", out)
    env.close()


if __name__ == "__main__":
    main()
