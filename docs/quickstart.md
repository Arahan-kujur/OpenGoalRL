# Quickstart

## Train on an easy scenario (scores goals!)

```bash
python -m opengoalrl.scripts.train --config opengoalrl/configs/empty_goal_close.yaml
```

## Train on corner kicks

```bash
python -m opengoalrl.scripts.train --config opengoalrl/configs/corner.yaml
```

## Evaluate + baseline + plot

```bash
python -m opengoalrl.scripts.evaluate --config opengoalrl/configs/corner.yaml
python -m opengoalrl.scripts.baseline --config opengoalrl/configs/corner.yaml --episodes 50
python -m opengoalrl.scripts.plot_results
```

## Run tests (no GRF required)

```bash
pytest opengoalrl/tests/ -v -k "not test_env"
```

## Composable rewards

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

| Component          | Signal                                          | Type   |
|--------------------|-------------------------------------------------|--------|
| `goal`             | +1 on goal scored, penalty on concede           | Sparse |
| `distance_to_goal` | Continuous [0,1] based on ball proximity        | Dense  |
| `shot`             | +1 when shot action taken                       | Dense  |
| `ball_in_box`      | +1 when ball is in penalty area                 | Dense  |

Want a no-install path? Jump to [Reproduce](reproduce.md).
