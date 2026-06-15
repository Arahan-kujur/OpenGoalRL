# OpenGoalRL

[![CI](https://github.com/Arahan-kujur/OpenGoalRL/actions/workflows/ci.yml/badge.svg)](https://github.com/Arahan-kujur/OpenGoalRL/actions/workflows/ci.yml)
[![Docs](https://github.com/Arahan-kujur/OpenGoalRL/actions/workflows/docs.yml/badge.svg)](https://Arahan-kujur.github.io/OpenGoalRL/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arahan-kujur/OpenGoalRL/blob/main/notebooks/quickstart.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Scenario-based reinforcement learning toolkit for [Google Research Football](https://github.com/google-research/football).

OpenGoalRL decomposes football into structured, repeatable scenarios -- corner kicks, penalties, open-goal finishes, passing drills -- and pairs them with composable reward shaping so agents learn meaningful behaviour faster. Instead of hoping an agent figures out football from scratch in a full 11v11 match, you isolate sub-problems, shape rewards precisely, and progress through a curriculum of increasing difficulty.

## Features

- **7 Gymnasium-compatible environments** spanning trivial (empty goal) to hard (corner kick)
- **Composable reward system** -- mix and weight reward components per scenario via YAML
- **Sequential curriculum learning** -- train on easy scenarios first, transfer to harder ones
- **Reward ablation framework** -- compare shaped vs sparse vs dense-only rewards
- **Parallel environments** via `SubprocVecEnv` for faster training
- **PPO training** via stable-baselines3, fully configurable through YAML
- **Built-in experiment pipeline** -- train, evaluate, baseline, ablation, and plotting
- **Procedural scenario generator** -- structured `scenario_spec` YAML alongside fixed academy scenarios
- **Tactical metrics** -- football-native measurements (xG proxy, progression, possession) separate from rewards
- **Failure diagnostics** -- rule-based classifiers over evaluated trajectories
- **Research reports** -- Markdown/HTML aggregation from CSVs, configs, and diagnosis JSON
- **Skill-graph curricula** -- graph-based prerequisites alongside linear stage lists
- **Auto curriculum discovery** -- probe-based stage selection for a target scenario
- **LLM coach (offline-first)** -- structured coaching from diagnostics and trajectory summaries
- **Multi-agent training path** -- separate 3v3 (and extensible) parameter-sharing PPO via `opengoalrl-train-ma`
- **GRF-free test suite** -- unit tests run without installing the football engine

## Scenarios

| Scenario | GRF Academy Level | Difficulty | Description |
|----------|-------------------|------------|-------------|
| `empty_goal_close` | `academy_empty_goal_close` | Trivial | Ball right next to unguarded goal |
| `empty_goal` | `academy_empty_goal` | Easy | Run from midfield to open goal |
| `run_to_score` | `academy_run_to_score` | Medium | Dribble past a line of defenders |
| `pass_and_shoot` | `academy_pass_and_shoot_with_keeper` | Medium-Hard | Two attackers vs keeper + defender |
| `three_vs_one` | `academy_3_vs_1_with_keeper` | Hard | 3v1 overload with keeper |
| `corner_kick` | `academy_corner` | Hard | Full corner-kick set piece |
| `penalty` | `academy_single_goal_versus_lazy` | Variable | Penalty-kick proxy (lazy defenders) |

## Installation

**Python 3.10+** on **Linux** or **WSL2** (Windows).

### System dependencies (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y git cmake build-essential libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
    libboost-all-dev libfontconfig1-dev
```

### Python package

```bash
git clone https://github.com/Arahan-kujur/OpenGoalRL.git
cd OpenGoalRL
pip install -e ".[dev]"
```

## Quick Start

### Train on an easy scenario (scores goals!)

```bash
python -m opengoalrl.scripts.train --config opengoalrl/configs/empty_goal_close.yaml
```

### Train on corner kicks

```bash
python -m opengoalrl.scripts.train --config opengoalrl/configs/corner.yaml
```

### Evaluate + baseline + plot

```bash
python -m opengoalrl.scripts.evaluate --config opengoalrl/configs/corner.yaml
python -m opengoalrl.scripts.baseline --config opengoalrl/configs/corner.yaml --episodes 50
python -m opengoalrl.scripts.plot_results
```

### Run tests (no GRF required)

```bash
pytest opengoalrl/tests/ -v -k "not test_env"
```

## Reproduce Our Benchmarks

Three ways to reproduce a number from this README, fastest first:

- **Colab (zero setup):** open the [quickstart notebook](https://colab.research.google.com/github/Arahan-kujur/OpenGoalRL/blob/main/notebooks/quickstart.ipynb) and run all cells.
- **Docker (reproducible Linux env):** `make docker-build && make docker-train` (bundles all GRF system deps).
- **Native benchmark harness:**

```bash
python benchmarks/run_benchmarks.py --seeds 3 && python benchmarks/aggregate.py
```

This writes `benchmarks/results/summary.csv` with mean scoring rate and 95%
confidence intervals across seeds. See [`benchmarks/README.md`](benchmarks/README.md)
for the schema and the [docs site](https://Arahan-kujur.github.io/OpenGoalRL/reproduce/)
for the full under-10-minutes path.

Skip training entirely with a pretrained checkpoint:

```bash
opengoalrl-download-models --scenario empty_goal_close
```

See the [Model Zoo](MODEL_ZOO.md) for the full checkpoint table.

## Documentation

Full documentation (MkDocs Material) is published at
**<https://Arahan-kujur.github.io/OpenGoalRL/>**. Build it locally with:

```bash
pip install -e ".[docs]"
mkdocs build --strict   # or: mkdocs serve
```

## Research Roadmap Features

### Procedural scenarios

```bash
opengoalrl-generate-scenario --config opengoalrl/configs/generated_corner.yaml
```

Use `scenario_spec` in YAML instead of a fixed `scenario` name, or keep existing configs unchanged.

### Tactical evaluation

```bash
opengoalrl-eval --config opengoalrl/configs/corner.yaml --metrics tactical
```

### Failure diagnosis

```bash
opengoalrl-diagnose models/curriculum/stage_5_corner_kick --config opengoalrl/configs/corner.yaml
```

### Research reports

```bash
opengoalrl-report models/curriculum/ --format markdown
```

### Skill-graph curriculum

Use `opengoalrl/configs/skill_graph.yaml` with `curriculum.skill_graph.nodes` instead of a linear `stages` list.

### Auto curriculum discovery

```bash
opengoalrl-auto-curriculum --target corner_kick --budget 20
```

### Offline coaching

```bash
opengoalrl-coach experiment/episode_042.json
```

### Multi-agent training

```bash
opengoalrl-train-ma --config opengoalrl/configs/ma_3v3.yaml
```

## Curriculum Learning

Train through scenarios of increasing difficulty, transferring the learned policy at each stage:

```bash
python -m opengoalrl.scripts.curriculum_train --config opengoalrl/configs/curriculum.yaml
```

The default curriculum progresses through 5 stages:

1. **empty_goal_close** (50K steps) -- learn to shoot
2. **empty_goal** (100K steps) -- learn to run and shoot
3. **run_to_score** (150K steps) -- learn to dribble past defenders
4. **pass_and_shoot** (150K steps) -- learn passing coordination
5. **corner_kick** (200K steps) -- apply everything to a hard set piece

Each stage loads the previous stage's trained model and continues training on the new scenario. The policy network transfers between environments because all scenarios share the same observation and action spaces (simple115v2, 19 discrete actions).

## Reward Ablation

Compare different reward configurations on the same scenario:

```bash
python -m opengoalrl.scripts.ablation --config opengoalrl/configs/ablation_empty_goal.yaml
python -m opengoalrl.scripts.plot_results --ablation models/ablation/ablation_metrics.csv
```

Three variants are compared:

| Variant | Rewards | Hypothesis |
|---------|---------|------------|
| **Full shaped** | Goal + Distance + Shot + BallInBox | Dense signal accelerates learning |
| **Scoring only** | Goal only | Sparse but aligned with true objective |
| **Dense only** | Distance + Shot + BallInBox (no goal) | Tests whether shaping alone is sufficient |

## Parallel Training

Speed up training with multiple environment workers:

Add `n_envs: 4` to the `training` section of any config YAML:

```yaml
training:
  n_envs: 4
  total_timesteps: 200000
  # ... other params
```

This uses `SubprocVecEnv` to run 4 GRF instances in parallel subprocesses, roughly 2-3x faster than single-env training.

## Wrapper Stack

Applied in this order (matters for correctness):

```
BaseEnv -> ScenarioWrapper -> RewardWrapper -> ObservationWrapper -> ActionWrapper
```

`RewardWrapper` sits **before** `ObservationWrapper` so reward components receive raw GRF coordinate values, not normalised observations.

## Composable Reward System

Rewards are defined in YAML and combined by weight:

```yaml
rewards:
  - type: "goal"
    weight: 10.0
  - type: "distance_to_goal"
    weight: 1.0
  - type: "shot"
    weight: 2.0
  - type: "ball_in_box"
    weight: 0.5
```

Available components:

| Component | Signal | Type |
|-----------|--------|------|
| `goal` | +1 on goal scored, penalty on concede | Sparse |
| `distance_to_goal` | Continuous [0,1] based on ball proximity | Dense |
| `shot` | +1 when shot action taken | Dense |
| `ball_in_box` | +1 when ball is in penalty area | Dense |

Adding a new reward: subclass `RewardComponent`, implement `compute()`, register in `REWARD_REGISTRY`.

## Project Structure

```
opengoalrl/
|-- envs/                     # 7 Gymnasium-compatible scenario environments
|   |-- base_env.py           # BaseScenarioEnv + GRF API compat layer
|   |-- empty_goal_close.py   # Trivial scoring
|   |-- empty_goal.py         # Easy scoring from midfield
|   |-- run_to_score.py       # Medium -- dribble past defenders
|   |-- pass_and_shoot.py     # Medium-hard -- passing coordination
|   |-- three_vs_one.py       # Hard -- 3v1 decision making
|   |-- corner_kick.py        # Hard -- full corner kick
|   +-- penalty.py            # Penalty-kick proxy
|-- wrappers/                 # Single-responsibility Gymnasium wrappers
|-- rewards/                  # Composable reward components
|-- agents/
|   +-- ppo_agent.py          # SB3 PPO wrapper with save/load
|-- configs/                  # YAML configs for each scenario + curriculum + ablation
|-- scenarios/                # Structured specs + procedural generator
|-- metrics/                  # Tactical metrics (separate from rewards)
|-- diagnostics/              # Rule-based failure classifiers
|-- reports/                  # Experiment report builder
|-- curriculum/               # Skill graphs + auto-discovery
|-- coach/                    # Offline/LLM coaching from diagnostics
|-- scripts/
|   |-- train.py              # Single-scenario training
|   |-- train_ma.py           # Multi-agent parameter-sharing training
|   |-- evaluate.py           # Model evaluation with metrics
|   |-- baseline.py           # Random-action baseline
|   |-- curriculum_train.py   # Sequential curriculum trainer
|   |-- generate_scenario.py  # Procedural scenario preview/export
|   |-- diagnose.py           # Failure diagnosis CLI
|   |-- report.py             # Research report generation
|   |-- auto_curriculum.py    # Automatic curriculum discovery
|   |-- coach.py              # Coaching explanations
|   |-- ablation.py           # Reward ablation experiment
|   +-- plot_results.py       # Learning curves + comparison + ablation plots
|-- utils/
|   |-- config_loader.py      # YAML loading + reward registry
|   |-- config_validation.py  # Config section validation
|   |-- rollout.py            # Shared rollout/eval utilities
|   |-- logger.py             # Structured logging + config snapshots
|   +-- metrics_callback.py   # SB3 callback for CSV metrics
+-- tests/
    |-- test_rewards.py       # Reward component unit tests (no GRF)
    |-- test_wrappers.py      # Wrapper unit tests (no GRF)
    |-- test_config.py        # Config loading tests (no GRF)
    +-- test_env.py           # Integration tests (requires GRF)
```

## Known Limitations

- **GRF engine seeding:** The GRF C++ engine has no seed parameter. NumPy/PyTorch/Python seeds are controlled, but GRF's internal physics are not fully deterministic across runs.
- **Gym deprecation warning:** GRF depends on the unmaintained `gym` package. OpenGoalRL patches the API mismatch at runtime via `_patch_grf_api()`. This is transparent to users but may need updating if gym or gfootball release new versions.
- **Multi-agent maturity:** `opengoalrl-train-ma` provides a separate parameter-sharing path; full PettingZoo/MAPPO integration is not yet included.
- **Procedural GRF limits:** `scenario_spec` maps to the nearest academy scenario; true custom player/ball placement requires deeper GRF integration.

## License

MIT
