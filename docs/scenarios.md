# Scenarios

OpenGoalRL ships 7 Gymnasium-compatible environments, each mapping to a Google
Research Football academy level.

| Scenario           | GRF Academy Level                      | Difficulty   | Description                          |
|--------------------|----------------------------------------|--------------|--------------------------------------|
| `empty_goal_close` | `academy_empty_goal_close`             | Trivial      | Ball right next to unguarded goal    |
| `empty_goal`       | `academy_empty_goal`                   | Easy         | Run from midfield to open goal       |
| `run_to_score`     | `academy_run_to_score`                 | Medium       | Dribble past a line of defenders     |
| `pass_and_shoot`   | `academy_pass_and_shoot_with_keeper`   | Medium-Hard  | Two attackers vs keeper + defender   |
| `three_vs_one`     | `academy_3_vs_1_with_keeper`           | Hard         | 3v1 overload with keeper             |
| `corner_kick`      | `academy_corner`                       | Hard         | Full corner-kick set piece           |
| `penalty`          | `academy_single_goal_versus_lazy`      | Variable     | Penalty-kick proxy (lazy defenders)  |

## Wrapper stack

Wrappers are applied in this order (it matters for correctness):

```
BaseEnv -> ScenarioWrapper -> RewardWrapper -> ObservationWrapper -> ActionWrapper
```

`RewardWrapper` sits **before** `ObservationWrapper` so reward components receive
raw GRF coordinate values, not normalised observations.

## Procedural scenarios

Use a `scenario_spec` in YAML instead of a fixed `scenario` name:

```bash
opengoalrl-generate-scenario --config opengoalrl/configs/generated_corner.yaml
```

Existing fixed-scenario configs keep working unchanged. `scenario_spec` maps to
the nearest academy scenario; true custom player/ball placement requires deeper
GRF integration.
