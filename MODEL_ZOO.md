# OpenGoalRL Model Zoo

Pretrained PPO checkpoints so you can run `evaluate`, `diagnose`, `report`, and
`coach` **without training first**.

Checkpoints are hosted as assets on
[GitHub Releases](https://github.com/Arahan-kujur/OpenGoalRL/releases) — they are
**not committed to the repository** to keep it lightweight.

## Download

```bash
# Download a single scenario checkpoint into models/zoo/<scenario>/
opengoalrl-download-models --scenario empty_goal_close

# Download everything in the zoo
opengoalrl-download-models --all
```

Then use it directly:

```bash
opengoalrl-eval \
    --config opengoalrl/configs/empty_goal_close.yaml \
    --model models/zoo/empty_goal_close/ppo_opengoalrl
```

## Available checkpoints

> Scoring rate is the percentage of evaluation episodes in which the agent
> scored at least one goal, over the seeds in the benchmark harness. Numbers are
> filled in by a maintainer once the harness is run on Linux/GPU (see
> [`benchmarks/README.md`](benchmarks/README.md)).

| Scenario           | Config                                       | Timesteps | Scoring rate | Checkpoint                |
|--------------------|----------------------------------------------|-----------|--------------|---------------------------|
| `empty_goal_close` | `opengoalrl/configs/empty_goal_close.yaml`   | 50K       | _TBD_        | `empty_goal_close.zip`    |
| `empty_goal`       | `opengoalrl/configs/empty_goal.yaml`         | 100K      | _TBD_        | `empty_goal.zip`          |
| `run_to_score`     | `opengoalrl/configs/run_to_score.yaml`       | 150K      | _TBD_        | `run_to_score.zip`        |
| `pass_and_shoot`   | `opengoalrl/configs/pass_and_shoot.yaml`     | 150K      | _TBD_        | `pass_and_shoot.zip`      |
| `corner_kick`      | `opengoalrl/configs/corner.yaml`             | 200K      | _TBD_        | `corner_kick.zip`         |

## Publishing checkpoints (maintainers)

1. Train with the benchmark configs and verify scoring rate via the harness.
2. Create a GitHub Release tagged like `models-v0.2.0`.
3. Attach each `<scenario>.zip` (an SB3 model archive) as a release asset.
4. Update the `--release-tag` default in
   `opengoalrl/scripts/download_models.py` if the tag changes, and fill in the
   scoring-rate column above.
