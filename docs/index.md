# OpenGoalRL

Scenario-based reinforcement learning toolkit for [Google Research Football](https://github.com/google-research/football).

OpenGoalRL decomposes football into structured, repeatable scenarios — corner
kicks, penalties, open-goal finishes, passing drills — and pairs them with
composable reward shaping so agents learn meaningful behaviour faster. Instead
of hoping an agent figures out football from scratch in a full 11v11 match, you
isolate sub-problems, shape rewards precisely, and progress through a curriculum
of increasing difficulty.

## Features

- **7 Gymnasium-compatible environments** spanning trivial (empty goal) to hard (corner kick)
- **Composable reward system** — mix and weight reward components per scenario via YAML
- **Sequential curriculum learning** — train on easy scenarios first, transfer to harder ones
- **Reward ablation framework** — compare shaped vs sparse vs dense-only rewards
- **Parallel environments** via `SubprocVecEnv` for faster training
- **PPO training** via stable-baselines3, fully configurable through YAML
- **Built-in experiment pipeline** — train, evaluate, baseline, ablation, and plotting
- **Procedural scenario generator** — structured `scenario_spec` YAML alongside fixed academy scenarios
- **Tactical metrics** — football-native measurements (xG proxy, progression, possession)
- **Failure diagnostics** — rule-based classifiers over evaluated trajectories
- **Research reports** — Markdown/HTML aggregation from CSVs, configs, and diagnosis JSON
- **Skill-graph curricula** — graph-based prerequisites alongside linear stage lists
- **Auto curriculum discovery** — probe-based stage selection for a target scenario
- **LLM coach (offline-first)** — structured coaching from diagnostics and trajectory summaries
- **Multi-agent training path** — parameter-sharing PPO via `opengoalrl-train-ma`
- **GRF-free test suite** — unit tests run without installing the football engine

## Where to next

- New here? Start with [Installation](installation.md) then [Quickstart](quickstart.md).
- Want to reproduce a number from the README? See [Reproduce](reproduce.md) — the under-10-minutes path.
- Curious about the design? Browse [Scenarios](scenarios.md) and [Curriculum](curriculum.md).
