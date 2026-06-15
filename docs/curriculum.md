# Curriculum Learning

Train through scenarios of increasing difficulty, transferring the learned
policy at each stage:

```bash
python -m opengoalrl.scripts.curriculum_train --config opengoalrl/configs/curriculum.yaml
```

The default curriculum progresses through 5 stages:

1. **empty_goal_close** (50K steps) — learn to shoot
2. **empty_goal** (100K steps) — learn to run and shoot
3. **run_to_score** (150K steps) — learn to dribble past defenders
4. **pass_and_shoot** (150K steps) — learn passing coordination
5. **corner_kick** (200K steps) — apply everything to a hard set piece

Each stage loads the previous stage's trained model and continues training on
the new scenario. The policy network transfers between environments because all
scenarios share the same observation and action spaces (simple115v2, 19 discrete
actions).

## Skill-graph curricula

For non-linear prerequisites, use a skill graph instead of a linear stage list.
Configure `curriculum.skill_graph.nodes` in
`opengoalrl/configs/skill_graph.yaml`.

## Reward ablation

Compare different reward configurations on the same scenario:

```bash
python -m opengoalrl.scripts.ablation --config opengoalrl/configs/ablation_empty_goal.yaml
python -m opengoalrl.scripts.plot_results --ablation models/ablation/ablation_metrics.csv
```

| Variant          | Rewards                                  | Hypothesis                          |
|------------------|------------------------------------------|-------------------------------------|
| **Full shaped**  | Goal + Distance + Shot + BallInBox       | Dense signal accelerates learning   |
| **Scoring only** | Goal only                                | Sparse but aligned with objective   |
| **Dense only**   | Distance + Shot + BallInBox (no goal)    | Tests whether shaping alone suffices |
