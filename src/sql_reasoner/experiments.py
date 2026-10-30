from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl
from .metrics import mcnemar_exact


def get_dotted(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def set_dotted(mapping: dict[str, Any], path: str, value: Any) -> None:
    target = mapping
    parts = path.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"无法设置 {path}: {part} 不是 mapping")
        target = child
    target[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for path, value in overrides.items():
        set_dotted(resolved, path, value)
    return resolved


def validate_profile(
    config: dict[str, Any], profile: dict[str, Any], overrides: dict[str, Any]
) -> None:
    tunable = set(profile.get("tunable", []))
    unknown = sorted(set(overrides) - tunable)
    if unknown:
        raise ValueError(f"实验试图修改未列入 tunable 的参数: {unknown}")
    mismatches = []
    for path, expected in profile.get("locked", {}).items():
        try:
            actual = get_dotted(config, path)
        except KeyError:
            mismatches.append(f"{path}: missing (expected {expected!r})")
            continue
        if actual != expected:
            mismatches.append(f"{path}: {actual!r} != locked {expected!r}")
    if mismatches:
        raise ValueError("资源配置锁校验失败:\n- " + "\n- ".join(mismatches))


def select_experiments(
    experiments: dict[str, dict[str, Any]], names: Iterable[str], groups: Iterable[str]
) -> list[str]:
    requested = list(names)
    group_set = set(groups)
    unknown = sorted(set(requested) - set(experiments))
    if unknown:
        raise ValueError(f"未知实验: {unknown}")
    selected = set(requested)
    for name, spec in experiments.items():
        if group_set.intersection(spec.get("groups", [])):
            selected.add(name)
    if not selected:
        return list(experiments)
    return [name for name in experiments if name in selected]


def expand_dependencies(experiments: dict[str, dict[str, Any]], selected: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"实验依赖存在环: {name}")
        if name not in experiments:
            raise ValueError(f"依赖的实验不存在: {name}")
        visiting.add(name)
        dependency = experiments[name].get("depends_on")
        if dependency:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in selected:
        visit(name)
    return ordered


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paired_execution(prediction_a: str | Path, prediction_b: str | Path) -> dict[str, Any]:
    rows_a = {row["id"]: row for row in read_jsonl(prediction_a)}
    rows_b = {row["id"]: row for row in read_jsonl(prediction_b)}
    if set(rows_a) != set(rows_b):
        raise ValueError("配对检验的预测文件样本 id 不一致")
    ids = list(rows_a)
    correct_a = [bool(rows_a[sample_id]["scores"]["execution"]) for sample_id in ids]
    correct_b = [bool(rows_b[sample_id]["scores"]["execution"]) for sample_id in ids]
    return mcnemar_exact(correct_a, correct_b)
