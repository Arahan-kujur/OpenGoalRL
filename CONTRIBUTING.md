# Contributing to OpenGoalRL

Thanks for your interest in contributing. This document covers the basics for getting started.

## Setup

```bash
git clone https://github.com/Arahan-kujur/OpenGoalRL.git
cd OpenGoalRL
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

GRF requires system dependencies on Linux/WSL:

```bash
sudo apt install -y cmake build-essential libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
    libboost-all-dev libfontconfig1-dev
```

## Running Tests

Most tests run without GRF installed:

```bash
pytest opengoalrl/tests/ -v -k "not test_env"
```

Full suite (requires GRF):

```bash
pytest opengoalrl/tests/ -v
```

## Adding a New Scenario

1. Create `opengoalrl/envs/your_scenario.py` subclassing `BaseScenarioEnv`
2. Set the GRF `scenario_name` and override termination helpers if needed
3. Add the class to `opengoalrl/envs/__init__.py`
4. Register it in `ENV_REGISTRY` in `opengoalrl/scripts/train.py`
5. Create a matching YAML config in `opengoalrl/configs/`
6. Add a test in `opengoalrl/tests/test_env.py`

## Adding a New Reward Component

1. Create `opengoalrl/rewards/your_reward.py` subclassing `RewardComponent`
2. Implement `compute(obs, action, next_obs, info) -> float`
3. Add the class to `opengoalrl/rewards/__init__.py`
4. Register it in `REWARD_REGISTRY` in `opengoalrl/utils/config_loader.py`
5. Add unit tests in `opengoalrl/tests/test_rewards.py` (no GRF needed)

## Code Style

- Use type hints for all function signatures
- Add docstrings to public classes and functions
- Keep files under 200 lines where possible
- No comments that just narrate what code does -- only explain non-obvious intent

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Ensure all tests pass (`pytest opengoalrl/tests/ -v`)
4. Keep commits focused -- one logical change per commit
5. Open a PR with a clear description of what and why

## Reporting Issues

Open a GitHub issue with:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS, Python version, and GRF version (`pip show gfootball`)
