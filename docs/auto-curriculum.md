# Auto Curriculum Discovery

Rather than hand-authoring a stage list, let OpenGoalRL probe candidate
scenarios and select a curriculum toward a target scenario.

```bash
opengoalrl-auto-curriculum --target corner_kick --budget 20
```

- `--target` — the hard scenario you ultimately want to solve.
- `--budget` — how many probe runs the discovery process may spend.

The discovery process estimates which easier scenarios best bootstrap progress
toward the target, then emits an ordered curriculum you can feed into
`curriculum_train`.
