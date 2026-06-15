# Multi-Agent Training

OpenGoalRL includes a separate parameter-sharing PPO path for multi-agent
scenarios such as 3v3.

```bash
opengoalrl-train-ma --config opengoalrl/configs/ma_3v3.yaml
```

All controlled players share a single policy network (parameter sharing), which
keeps training tractable while still learning coordinated behaviour.

!!! note "Maturity"
    This is a dedicated parameter-sharing path. Full PettingZoo / MAPPO
    integration is not yet included — see the project's Known Limitations.
