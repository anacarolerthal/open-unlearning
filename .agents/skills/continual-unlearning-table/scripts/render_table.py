#!/usr/bin/env python3
"""Render a continual-unlearning metric table as Typst and PNG.

The input is intentionally small and explicit: task names, metric definitions,
and one timing matrix per method. This keeps selection and best-value logic
deterministic while leaving artifact-specific parsing to the calling agent.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    {
        "key": "forget_rouge",
        "label": "FR",
        "arrow": "↓",
        "direction": "min",
        "description": "Forget QA-ROUGE",
    },
    {
        "key": "forget_quality",
        "label": "FQ",
        "arrow": "↑",
        "direction": "max",
        "description": "Forget Quality (KS p-value)",
    },
    {
        "key": "model_utility",
        "label": "MU",
        "arrow": "↑",
        "direction": "max",
        "description": "Model utility",
    },
]


def _fail(message: str) -> "NoReturn":
    raise ValueError(message)


def _selector(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or _fail("A selector must contain at least one name")


def _task_name(task: Any) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, dict) and isinstance(task.get("name"), str):
        return task["name"]
    _fail("Each task must be a string or an object with a string 'name'")


def _metric_defs(raw: Any) -> list[dict[str, Any]]:
    raw = DEFAULT_METRICS if raw is None else raw
    if not isinstance(raw, list) or not raw:
        _fail("'metrics' must be a non-empty list")
    result = []
    for metric in raw:
        if not isinstance(metric, dict):
            _fail("Each metric must be an object")
        key = metric.get("key")
        if not isinstance(key, str) or not key:
            _fail("Each metric needs a non-empty string 'key'")
        direction = metric.get("direction", "max")
        direction = {"lower": "min", "higher": "max"}.get(direction, direction)
        if direction not in ("min", "max"):
            _fail(f"Metric {key!r} has invalid direction {direction!r}")
        result.append(
            {
                "key": key,
                "label": str(metric.get("label", key)),
                "arrow": str(metric.get("arrow", "↑" if direction == "max" else "↓")),
                "direction": direction,
                "description": str(metric.get("description", key)),
            }
        )
    return result


def _get_timing(method: dict[str, Any], timing: str) -> Any:
    values = method.get("values", method)
    if isinstance(values, dict):
        values = values.get(timing)
    if not isinstance(values, list):
        _fail(f"Method {method.get('name', '<unnamed>')!r} has no {timing!r} list")
    return values


def _normalise(data: dict[str, Any], timing: str, methods: list[str] | None, metrics: list[str] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        _fail("Input JSON must contain an object")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        _fail("'tasks' must be a non-empty list")
    task_names = [_task_name(task) for task in tasks]
    if len(set(task_names)) != len(task_names):
        _fail("Task names must be unique")

    definitions = _metric_defs(data.get("metrics"))
    definition_by_key = {metric["key"]: metric for metric in definitions}
    selected_metrics = metrics or [metric["key"] for metric in definitions]
    unknown_metrics = [key for key in selected_metrics if key not in definition_by_key]
    if unknown_metrics:
        _fail(f"Unknown metric(s): {', '.join(unknown_metrics)}")
    definitions = [definition_by_key[key] for key in selected_metrics]

    raw_methods = data.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        _fail("'methods' must be a non-empty list")
    method_by_name = {}
    for method in raw_methods:
        if not isinstance(method, dict) or not isinstance(method.get("name"), str):
            _fail("Each method must have a string 'name'")
        name = method["name"]
        if name in method_by_name:
            _fail(f"Duplicate method name: {name}")
        method_by_name[name] = method
    selected_methods = methods or list(method_by_name)
    unknown_methods = [name for name in selected_methods if name not in method_by_name]
    if unknown_methods:
        _fail(f"Unknown method(s): {', '.join(unknown_methods)}")

    normal_methods = []
    for name in selected_methods:
        method = method_by_name[name]
        rows = _get_timing(method, timing)
        if len(rows) != len(task_names):
            _fail(f"Method {name!r} has {len(rows)} rows; expected {len(task_names)}")
        normal_rows = []
        for row_index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                _fail(f"Method {name!r}, task {row_index}: row must be an object")
            normal_row = {}
            for metric in definitions:
                key = metric["key"]
                if key not in row or row[key] is None:
                    _fail(f"Method {name!r}, task {row_index}: missing metric {key!r}")
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    _fail(f"Method {name!r}, task {row_index}: {key!r} is not numeric")
                if not math.isfinite(value):
                    _fail(f"Method {name!r}, task {row_index}: {key!r} is not finite")
                normal_row[key] = value
            normal_rows.append(normal_row)
        normal_methods.append({
            "name": name,
            "baseline": bool(method.get("baseline", name.lower() == "original")),
            "rows": normal_rows,
        })

    if not any(not method["baseline"] for method in normal_methods):
        _fail("At least one non-baseline method is required for best-value bolding")
    return {
        "title": str(data.get("title", "Continual unlearning performance")),
        "dataset": str(data.get("dataset", "")),
        "tasks": task_names,
        "metrics": definitions,
        "methods": normal_methods,
        "timing": timing,
    }


def _typst_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _format_value(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        text = f"{value:.2e}"
    else:
        text = f"{value:.3f}"
        if text.startswith("0"):
            text = text[1:]
        elif text.startswith("-0"):
            text = "-" + text[2:]
    return text


def _best_values(table: dict[str, Any]) -> set[tuple[int, str, str]]:
    best = set()
    methods = table["methods"]
    for task_index in range(len(table["tasks"])):
        for metric in table["metrics"]:
            candidates = [
                (method["name"], method["rows"][task_index][metric["key"]])
                for method in methods
                if not method["baseline"]
            ]
            values = [value for _, value in candidates]
            target = (min if metric["direction"] == "min" else max)(values)
            for method_name, value in candidates:
                if math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-15):
                    best.add((task_index, metric["key"], method_name))
    return best


def _cell(value: str, bold: bool = False) -> str:
    content = f"#text({_typst_string(value)})"
    return f"[#strong[{content}]]" if bold else f"[{content}]"


def _generate_typst(table: dict[str, Any]) -> str:
    tasks = table["tasks"]
    metrics = table["metrics"]
    methods = table["methods"]
    best = _best_values(table)
    n_columns = 1 + len(tasks) * len(metrics)
    width = min(11.5, max(8.5, 1.8 + 0.55 * (n_columns - 1)))
    height = max(1.6, 1.1 + 0.28 * (len(methods) + len(metrics) * 0.25))
    dataset = f" on TOFU #raw(\"{table['dataset']}\")" if table["dataset"] else ""
    timing_phrase = (
        "Each task is evaluated immediately after its corresponding forget request."
        if table["timing"] == "immediate"
        else "Each task is evaluated at the final continual-unlearning stage."
    )
    lower_labels = [metric["label"] for metric in metrics if metric["direction"] == "min"]
    higher_labels = [metric["label"] for metric in metrics if metric["direction"] == "max"]
    direction_note = []
    if lower_labels:
        direction_note.append(f"Lower {', '.join(lower_labels)} is better")
    if higher_labels:
        higher_text = ", ".join(higher_labels[:-1]) + (" and " if len(higher_labels) > 1 else "") + higher_labels[-1]
        direction_note.append(f"higher {higher_text} are better")
    direction_note = "; ".join(direction_note) + "."
    lines = [
        '#let accent = rgb("#2f5578")',
        '#let header-fill = rgb("#eef3f7")',
        '#let rule-gray = rgb("#7f8992")',
        '#let note-gray = rgb("#68727b")',
        f"#set page(width: {width:.2f}in, height: auto, margin: (x: 0.22in, y: 0.20in), fill: white)",
        '#set text(font: "Libertinus Serif", size: 8.6pt, fill: rgb("#17191b"))',
        '#set par(leading: 0.58em)',
        '#let metric(name, arrow) = text(weight: "semibold")[#name#h(2pt)#text(fill: accent, weight: "bold")[#arrow]]',
        "",
        "#block(width: 100%)[",
        '#text(weight: "bold", fill: accent)[Table 1.]',
        "#h(2pt)",
        f'#text(weight: "semibold")[#text({_typst_string(table["title"])})]{dataset}. {timing_phrase} {direction_note}',
        "]",
        "",
        "#v(6pt)",
        "#table(",
        f"  columns: (1.32fr,) + (1fr,) * {n_columns - 1},",
        "  inset: (x: 3.2pt, y: 3.0pt),",
        f"  align: (left,) + (center,) * {n_columns - 1},",
        "  stroke: none,",
        "  table.hline(y: 0, stroke: 0.9pt + accent),",
        '  table.cell(rowspan: 2, fill: header-fill, align: left + horizon)[*Method*],',
    ]
    for task in tasks:
        lines.append(f'  table.cell(colspan: {len(metrics)}, fill: header-fill, align: center)[*#text({_typst_string(task)})*],')
    for task_index in range(len(tasks)):
        start = 1 + task_index * len(metrics)
        end = start + len(metrics)
        lines.append(f"  table.hline(y: 1, start: {start}, end: {end}, stroke: 0.42pt + rule-gray),")
    for _ in tasks:
        for metric in metrics:
            lines.append(f'  table.cell(fill: header-fill)[#metric({_typst_string(metric["label"])}, {_typst_string(metric["arrow"])})],')
    lines += [
        "  table.hline(y: 2, stroke: 0.65pt + rule-gray),",
    ]
    for method in methods:
        method_name = method["name"]
        label = f'#text(weight: "semibold")[#text({_typst_string(method_name)})]'
        lines.append(f"  [{label}],")
        for task_index in range(len(tasks)):
            for metric in metrics:
                value = _format_value(method["rows"][task_index][metric["key"]])
                is_best = (task_index, metric["key"], method_name) in best
                lines.append(f"  {_cell(value, is_best)},")
    lines += [
        f"  table.hline(y: {2 + len(methods)}, stroke: 0.9pt + accent),",
        ")",
        "",
        "#v(4pt)",
        "#align(center)[",
        '#text(size: 7.35pt, fill: note-gray, style: "italic")[',
        "  Notes. ",
        '#text(style: "normal")[',
        "Bold denotes the better value within each task and metric; baseline rows are excluded.",
        "]",
        "]",
        "]",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--timing", choices=("immediate", "final"), default="immediate")
    parser.add_argument("--methods", help="Comma-separated method names, in display order")
    parser.add_argument("--metrics", help="Comma-separated metric keys, in display order")
    parser.add_argument("--output", type=Path, required=True, help="Output prefix, without .typ/.png")
    parser.add_argument("--ppi", type=int, default=150)
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text())
        table = _normalise(data, args.timing, _selector(args.methods), _selector(args.metrics))
        typst_path = args.output.with_suffix(".typ")
        png_path = args.output.with_suffix(".png")
        typst_path.parent.mkdir(parents=True, exist_ok=True)
        typst_path.write_text(_generate_typst(table))
        typst = shutil.which("typst")
        if typst is None:
            _fail("typst is required to render the PNG but was not found on PATH")
        subprocess.run(
            [typst, "compile", "--format", "png", "--ppi", str(args.ppi), str(typst_path), str(png_path)],
            check=True,
        )
    except (OSError, json.JSONDecodeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"render_table.py: {exc}", file=sys.stderr)
        return 2
    print(typst_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
