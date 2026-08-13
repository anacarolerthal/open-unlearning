---
name: continual-unlearning-table
description: Render compact, publication-style Typst tables and PNGs for continual-unlearning evaluations. Use when a user asks to compare unlearning methods across forget requests/tasks, choose metrics or methods, show values immediately after each request or at the final evaluation, include an Original baseline, or bold the best value.
---

# Continual Unlearning Tables

## Purpose

Create a grouped-task table in the style of the bundled template, then compile it to a small PNG and retain the generated `.typ` source. The default timing is the evaluation immediately after each request; use final evaluation only when the user asks for it.

## Workflow

1. Identify the source metrics. Prefer per-stage/per-request logs because immediate evaluation needs the diagonal value (task `t` after request `t`), not the final cumulative value. If only final metrics exist, say so instead of silently relabeling them.
2. Interpret the user's requested methods, tasks/forget sets, and columns. Preserve the requested order. If no columns are named, use the available metrics; if no methods are named, use all available methods.
3. Resolve timing: `immediate` is the default; `final` selects the last-stage evaluation for every task. Include an Original/unmodified row when matching values are available. Mark it as a baseline so it is excluded from best-value comparisons.
4. Normalize the data to the JSON shape below and run `scripts/render_table.py`. Use the supplied asset as the visual template; do not replace it with a generic dataframe/HTML table.
5. Inspect the PNG. Keep labels readable, use compact dimensions, and fix overflow or clipped headers before returning the image and `.typ` source. Do not modify repository files unless the user explicitly asks.

## Input JSON

Pass a JSON file with tasks, metric definitions, and method values. Each method may put timing matrices under `values` or directly under `immediate`/`final`:

```json
{
  "title": "Continual unlearning performance on TOFU",
  "dataset": "forget05",
  "tasks": ["Task 1", "Task 2"],
  "metrics": [
    {"key": "forget_rouge", "label": "FR", "arrow": "↓", "direction": "min", "description": "Forget QA-ROUGE"},
    {"key": "forget_quality", "label": "FQ", "arrow": "↑", "direction": "max", "description": "Forget Quality (KS p-value)"},
    {"key": "model_utility", "label": "MU", "arrow": "↑", "direction": "max", "description": "Model utility"}
  ],
  "methods": [
    {"name": "Original", "baseline": true, "values": {"immediate": [{"forget_rouge": 0.9, "forget_quality": 0.0, "model_utility": 0.637}, {"forget_rouge": 0.8, "forget_quality": 0.0, "model_utility": 0.637}]}},
    {"name": "NPO", "values": {"immediate": [{"forget_rouge": 0.389, "forget_quality": 0.416, "model_utility": 0.595}, {"forget_rouge": 0.394, "forget_quality": 0.096, "model_utility": 0.588}], "final": [{"forget_rouge": 0.389, "forget_quality": 0.045, "model_utility": 0.571}, {"forget_rouge": 0.389, "forget_quality": 0.045, "model_utility": 0.571}]}}
  ]
}
```

`tasks` length must match every selected method's timing matrix. Metric `direction` must be `min`/`max` (aliases `lower`/`higher` are accepted). Set `baseline: true` for Original or other unmodified rows; a method named `Original` is treated as a baseline as well.

## Rendering

```bash
python scripts/render_table.py metrics.json --timing immediate \
  --methods NPO,RoAdBlock --metrics forget_rouge,forget_quality,model_utility \
  --output /tmp/continual_unlearning_table
```

The command writes `/tmp/continual_unlearning_table.typ` and `/tmp/continual_unlearning_table.png`. Omit `--timing` for the default `immediate`; omit selectors to retain all methods/metrics. Best values are bolded independently within each task and metric, considering only non-baseline methods. Ties are all bolded. A baseline is displayed but never wins bolding.

Use the bundled `assets/continual_unlearning_table_typst.typ` as the style reference: serif typography, blue grouped headers, compact insets, task-level column groups, and a short note below the table. The renderer keeps the same visual language while generating arbitrary task/metric counts.
