# Failure Diagnostics

Rule-based classifiers inspect evaluated trajectories and explain *why* an agent
is failing — e.g. never shooting, losing possession early, or stalling outside
the box.

```bash
opengoalrl-diagnose models/curriculum/stage_5_corner_kick --config opengoalrl/configs/corner.yaml
```

The diagnostics layer reads per-episode trajectory summaries and emits a
structured diagnosis (JSON) that feeds directly into [reports](reports.md) and
the [coach](coach.md). Because the classifiers are rule-based, they run without
GRF once trajectories have been collected.
