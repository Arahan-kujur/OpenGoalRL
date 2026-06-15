"""Automatic curriculum discovery CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from opengoalrl.curriculum.auto import AutoCurriculum
from opengoalrl.utils.logger import get_logger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Discover curriculum stages automatically")
    parser.add_argument("--target", type=str, default="corner_kick")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML path (default: opengoalrl/configs/auto_<target>.yaml)",
    )
    args = parser.parse_args(argv)

    logger = get_logger()
    curriculum = AutoCurriculum(
        target=args.target,
        budget=args.budget,
        seed=args.seed,
    )
    curriculum.discover()

    out = args.output or str(
        Path(__file__).resolve().parent.parent / "configs" / f"auto_{args.target}.yaml"
    )
    path = curriculum.save(out)
    logger.info("Discovered %d stages for target %s", len(curriculum.stages), args.target)
    for i, stage in enumerate(curriculum.stages, 1):
        logger.info("  %d. %s (%d timesteps)", i, stage["scenario"], stage["timesteps"])
    logger.info("Saved curriculum to %s", path)


if __name__ == "__main__":
    main()
