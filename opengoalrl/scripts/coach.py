"""LLM / offline coaching CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opengoalrl.coach.coach import Coach, CoachConfig
from opengoalrl.utils.logger import get_logger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate coaching from episode JSON")
    parser.add_argument("episode_json", type=str, help="Episode or diagnosis JSON file")
    parser.add_argument("--provider", type=str, default="offline")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    coach = Coach(CoachConfig(provider=args.provider))
    output = coach.from_episode_file(args.episode_json)
    explanation = output["markdown"]

    logger = get_logger()
    print(explanation)

    out = Path(args.output) if args.output else Path(args.episode_json).with_suffix(".coach.md")
    out.write_text(explanation)
    json_out = out.with_suffix(".coach.json")
    json_out.write_text(json.dumps(output, indent=2))
    logger.info("Coaching saved to %s and %s", out, json_out)


if __name__ == "__main__":
    main()
