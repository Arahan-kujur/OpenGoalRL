# Extending OpenGoalRL

## Add a new scenario

1. Create `opengoalrl/envs/your_scenario.py` subclassing `BaseScenarioEnv`.
2. Set the GRF `scenario_name` and override termination helpers if needed.
3. Add the class to `opengoalrl/envs/__init__.py`.
4. Register it in `ENV_REGISTRY` (`opengoalrl/utils/env_factory.py`).
5. Create a matching YAML config in `opengoalrl/configs/`.
6. Add a test in `opengoalrl/tests/test_env.py`.

## Add a new reward component

1. Create `opengoalrl/rewards/your_reward.py` subclassing `RewardComponent`.
2. Implement `compute(obs, action, next_obs, info) -> float`.
3. Add the class to `opengoalrl/rewards/__init__.py`.
4. Register it in `REWARD_REGISTRY` (`opengoalrl/utils/config_loader.py`).
5. Add unit tests in `opengoalrl/tests/test_rewards.py` (no GRF needed).

## Code style

- Type hints for all function signatures.
- Docstrings on public classes and functions.
- Keep files under ~200 lines where possible.
- No comments that just narrate what code does — only explain non-obvious intent.

## Testing

The GRF-free suite is the contract CI enforces:

```bash
pytest opengoalrl/tests/ -v -k "not test_env"
```

New non-GRF logic should be covered here so it stays green without the football
engine. GRF-backed integration tests live in `test_env.py`.
