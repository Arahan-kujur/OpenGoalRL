"""Download pretrained checkpoints from GitHub Releases into ``models/zoo/``.

Lets users run ``evaluate`` / ``diagnose`` / ``report`` / ``coach`` without
training first. Standard-library only (urllib) so it stays GRF-free.

Examples::

    opengoalrl-download-models --scenario empty_goal_close
    opengoalrl-download-models --all
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "Arahan-kujur/OpenGoalRL"
DEFAULT_RELEASE_TAG = "models-v0.2.0"

# scenario -> release asset filename (SB3 model archive).
ZOO = {
    "empty_goal_close": "empty_goal_close.zip",
    "empty_goal": "empty_goal.zip",
    "run_to_score": "run_to_score.zip",
    "pass_and_shoot": "pass_and_shoot.zip",
    "corner_kick": "corner_kick.zip",
}


def asset_url(release_tag: str, asset: str, repo: str = REPO) -> str:
    return f"https://github.com/{repo}/releases/download/{release_tag}/{asset}"


def download_one(
    scenario: str,
    dest_root: Path,
    release_tag: str,
    *,
    repo: str = REPO,
    extract: bool = True,
) -> Path:
    """Download (and optionally unzip) one scenario checkpoint."""
    if scenario not in ZOO:
        raise KeyError(
            f"Unknown scenario {scenario!r}. Known: {', '.join(sorted(ZOO))}"
        )

    asset = ZOO[scenario]
    url = asset_url(release_tag, asset, repo=repo)
    dest_dir = dest_root / scenario
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / asset

    print(f"Downloading {scenario} <- {url}")
    urllib.request.urlretrieve(url, archive_path)  # noqa: S310 (trusted GitHub URL)

    if extract and zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        print(f"  extracted to {dest_dir}")
    else:
        print(f"  saved to {archive_path}")
    return dest_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download OpenGoalRL pretrained checkpoints")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        choices=sorted(ZOO),
        help="Scenario checkpoint to download",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download every checkpoint in the zoo",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="models/zoo",
        help="Destination root directory (default: models/zoo)",
    )
    parser.add_argument(
        "--release-tag",
        type=str,
        default=DEFAULT_RELEASE_TAG,
        help=f"GitHub release tag to pull from (default: {DEFAULT_RELEASE_TAG})",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep the .zip archive instead of extracting it",
    )
    args = parser.parse_args(argv)

    dest_root = Path(args.dest)
    scenarios = sorted(ZOO) if args.all else [args.scenario]

    failures: list[str] = []
    for scenario in scenarios:
        try:
            download_one(
                scenario,
                dest_root,
                args.release_tag,
                extract=not args.no_extract,
            )
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"  failed: {scenario}: {exc}", file=sys.stderr)
            failures.append(scenario)

    if failures:
        print(
            f"\n{len(failures)} download(s) failed: {', '.join(failures)}.\n"
            "Checkpoints are published on GitHub Releases; if they are not yet "
            "available, train locally instead (see the README quickstart).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"\nDone. Checkpoints in {dest_root}/")


if __name__ == "__main__":
    main()
