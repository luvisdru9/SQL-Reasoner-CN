from __future__ import annotations

import statistics
from typing import Any, Iterable


def hard_execution_accuracy(summary: dict[str, Any]) -> float:
    groups = summary.get("by_difficulty", {})
    total = 0
    correct = 0.0
    for difficulty in ("hard", "extra-hard"):
        row = groups.get(difficulty, {})
        count = int(row.get("count", 0))
        total += count
        correct += count * float(row.get("execution_accuracy", 0.0))
    return correct / total if total else 0.0


def quality_key(summary: dict[str, Any]) -> tuple[float, ...]:
    """Validation-only ranking with deterministic, task-relevant tie breakers."""
    return (
        float(summary.get("execution_accuracy", 0.0)),
        hard_execution_accuracy(summary),
        float(summary.get("valid_sql_rate", 0.0)),
        float(summary.get("schema_linking_f1", 0.0)),
        float(summary.get("format_rate", 0.0)),
        float(summary.get("ast_exact_match", 0.0)),
    )


def select_best(candidates: Iterable[str], summaries: dict[str, dict[str, Any]]) -> str:
    names = list(candidates)
    missing = [name for name in names if name not in summaries]
    if missing:
        raise ValueError(f"缺少候选评测结果: {missing}")
    if not names:
        raise ValueError("候选列表为空")
    return max(names, key=lambda name: quality_key(summaries[name]))


def summarize_repeats(
    candidates: Iterable[str], summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    names = list(candidates)
    missing = [name for name in names if name not in summaries]
    if missing:
        raise ValueError(f"缺少重复实验评测结果: {missing}")
    values = [float(summaries[name]["execution_accuracy"]) for name in names]
    return {
        "experiments": names,
        "execution_accuracy": dict(zip(names, values, strict=True)),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }
