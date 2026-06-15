# Research Reports

Aggregate training/eval CSVs, configs, and diagnosis JSON into a single
Markdown or HTML report.

```bash
opengoalrl-report models/curriculum/ --format markdown
```

- `--format markdown` — portable, diff-friendly report (default).
- `--format html` — standalone HTML for sharing.

Reports pull together:

- Learning curves and evaluation summaries from CSVs
- The exact config snapshots used for each run
- [Diagnostics](diagnostics.md) output and [tactical metrics](metrics.md)

This makes a run self-documenting — useful for papers, issues, and PRs.
