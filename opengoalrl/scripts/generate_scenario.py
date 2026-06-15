"""CLI to generate and preview procedural scenario specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from opengoalrl.scenarios.generator import ScenarioGenerator, generate_scenario
from opengoalrl.utils.config_loader import load_config
from opengoalrl.utils.logger import get_logger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a procedural scenario spec")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "configs" / "generated_corner.yaml"
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Write spec JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    env_cfg = config.get("environment", {})
    spec_data = env_cfg.get("scenario_spec", env_cfg)
    spec = generate_scenario(spec_data, seed=args.seed)

    logger = get_logger()
    logger.info("Generated scenario: %s", spec.name)
    logger.info("GRF scenario: %s", spec.grf_scenario)
    logger.info("Spec: %s", json.dumps(spec.to_dict(), indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(spec.to_dict(), indent=2))
        logger.info("Saved to %s", out)
    else:
        out_path = Path("models") / f"scenario_{spec.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(spec.to_dict(), indent=2))
        logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
