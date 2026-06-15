# CLI Reference

Installing the package registers these console entry points.

| Command                        | Module                                  | Purpose                                  |
|--------------------------------|-----------------------------------------|------------------------------------------|
| `opengoalrl-train`             | `opengoalrl.scripts.train`              | Single-scenario PPO training             |
| `opengoalrl-train-ma`          | `opengoalrl.scripts.train_ma`           | Multi-agent parameter-sharing training   |
| `opengoalrl-eval`              | `opengoalrl.scripts.evaluate`           | Evaluate a trained model with metrics    |
| `opengoalrl-baseline`          | `opengoalrl.scripts.baseline`           | Random-action baseline                   |
| `opengoalrl-curriculum`        | `opengoalrl.scripts.curriculum_train`   | Sequential curriculum trainer            |
| `opengoalrl-generate-scenario` | `opengoalrl.scripts.generate_scenario`  | Procedural scenario preview/export       |
| `opengoalrl-diagnose`          | `opengoalrl.scripts.diagnose`           | Failure diagnosis                        |
| `opengoalrl-report`            | `opengoalrl.scripts.report`             | Research report generation               |
| `opengoalrl-auto-curriculum`   | `opengoalrl.scripts.auto_curriculum`    | Automatic curriculum discovery           |
| `opengoalrl-coach`             | `opengoalrl.scripts.coach`              | Offline coaching explanations            |
| `opengoalrl-download-models`   | `opengoalrl.scripts.download_models`    | Fetch pretrained checkpoints             |

Every training/eval command accepts `--config <path>` pointing at a YAML config
in `opengoalrl/configs/`. Each script can also be run as a module, e.g.:

```bash
python -m opengoalrl.scripts.train --config opengoalrl/configs/empty_goal_close.yaml
```
