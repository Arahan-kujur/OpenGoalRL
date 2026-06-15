# Tactical Metrics

OpenGoalRL measures football-native quantities separately from the reward
signal, so you can evaluate *how* an agent plays — not just whether it scored.

```bash
opengoalrl-eval --config opengoalrl/configs/corner.yaml --metrics tactical
```

Tactical metrics are kept distinct from rewards on purpose: rewards drive
optimisation, while tactical metrics describe behaviour. Reported quantities
include:

- **Distance advanced** — net progression of the ball toward goal
- **Approx xG** — a proxy expected-goals value from shot locations
- **Box entries** — how often the ball enters the penalty area
- **Possession losses** — turnovers during an episode
- **Pressure proxy** — opponent proximity while in possession
- **Shot timing** — step index of the first shot

These are emitted as extra columns in the evaluation CSV and aggregated for
[reports](reports.md).
