# Offline Coach

The coach turns diagnostics and trajectory summaries into structured, readable
coaching feedback. It is **offline-first**: it produces useful explanations
without any external LLM, and can optionally use one when configured.

```bash
opengoalrl-coach experiment/episode_042.json
```

Given a diagnosis or episode summary, the coach explains what went wrong and
suggests concrete adjustments (reward weights, curriculum ordering, scenario
choice). Pair it with [diagnostics](diagnostics.md) for the richest feedback.
