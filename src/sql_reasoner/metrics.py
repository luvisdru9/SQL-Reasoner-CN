from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from .formats import parse_model_output
from .planning import canonical_sql, extract_sql_features
from .rewards import component_scores


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    overlap = len(predicted & gold)
    precision = safe_div(overlap, len(predicted))
    recall = safe_div(overlap, len(gold))
    return safe_div(2 * precision * recall, precision + recall)


def schema_linking_f1(predicted_sql: str, gold_sql: str) -> float:
    try:
        pred = extract_sql_features(predicted_sql)
        gold = extract_sql_features(gold_sql)
    except Exception:
        return 0.0
    pred_names = {f"t:{x.lower()}" for x in pred["tables"]} | {
        f"c:{x.lower()}" for x in pred["columns"]
    }
    gold_names = {f"t:{x.lower()}" for x in gold["tables"]} | {
        f"c:{x.lower()}" for x in gold["columns"]
    }
    return f1_score(pred_names, gold_names)


def structural_exact_match(predicted_sql: str, gold_sql: str) -> bool:
    """A reproducible AST-normalized EM; not a claim of official CSpider EM."""
    try:
        return canonical_sql(predicted_sql) == canonical_sql(gold_sql)
    except Exception:
        return False


def bootstrap_ci(
    values: Sequence[float], *, confidence: float = 0.95, samples: int = 2_000, seed: int = 42
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    size = len(values)
    for _ in range(samples):
        means.append(sum(values[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[max(0, int(tail * samples))]
    high = means[min(samples - 1, int((1.0 - tail) * samples) - 1)]
    return (low, high)


def mcnemar_exact(left: Sequence[bool], right: Sequence[bool]) -> dict[str, float | int]:
    if len(left) != len(right):
        raise ValueError("两组预测数量必须一致")
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(left_only, right_only)
        lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def classify_error(predicted_text: str, gold_sql: str, scores: dict[str, float]) -> str:
    if not scores["format"]:
        return "format"
    if not scores["safe_sql"]:
        return "syntax_or_unsafe"
    if not scores["schema"]:
        return "schema_linking"
    try:
        predicted_sql = parse_model_output(predicted_text).sql
        pred = extract_sql_features(predicted_sql)
        gold = extract_sql_features(gold_sql)
    except Exception:
        return "parse"
    if bool(pred["joins"]) != bool(gold["joins"]) or set(pred["tables"]) != set(gold["tables"]):
        return "join_or_table"
    if bool(pred["aggregation"]) != bool(gold["aggregation"]):
        return "aggregation"
    if pred["has_subquery"] != gold["has_subquery"]:
        return "nested_query"
    if pred["filters"] != gold["filters"]:
        return "filter_or_value"
    return "semantic_other"


def evaluate_predictions(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detailed = []
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    error_counts: Counter[str] = Counter()
    for record in records:
        text = record.get("prediction", "")
        scores = component_scores(text, record["db_path"], record["gold_sql"])
        try:
            predicted_sql = parse_model_output(text).sql
        except ValueError:
            predicted_sql = ""
        schema_f1 = schema_linking_f1(predicted_sql, record["gold_sql"])
        ast_em = structural_exact_match(predicted_sql, record["gold_sql"])
        error = None
        if not scores["execution"]:
            error = classify_error(text, record["gold_sql"], scores)
            error_counts[error] += 1
        item = {
            **record,
            "scores": scores,
            "schema_linking_f1": schema_f1,
            "ast_exact_match": ast_em,
            "error_type": error,
        }
        detailed.append(item)
        by_difficulty[record.get("difficulty", "unknown")].append(scores["execution"])

    total = len(detailed)
    execution = [item["scores"]["execution"] for item in detailed]
    ci_low, ci_high = bootstrap_ci(execution)
    summary = {
        "count": total,
        "execution_accuracy": safe_div(sum(execution), total),
        "execution_accuracy_ci95": [ci_low, ci_high],
        "ast_exact_match": safe_div(sum(item["ast_exact_match"] for item in detailed), total),
        "valid_sql_rate": safe_div(sum(item["scores"]["safe_sql"] for item in detailed), total),
        "format_rate": safe_div(sum(item["scores"]["format"] for item in detailed), total),
        "schema_linking_f1": safe_div(sum(item["schema_linking_f1"] for item in detailed), total),
        "plan_consistency": safe_div(
            sum(item["scores"]["plan_consistency"] for item in detailed), total
        ),
        "by_difficulty": {
            key: {"count": len(values), "execution_accuracy": safe_div(sum(values), len(values))}
            for key, values in sorted(by_difficulty.items())
        },
        "error_counts": dict(error_counts.most_common()),
        "metric_note": "ast_exact_match 是 sqlglot 规范化 EM，不冒充 CSpider 官方 evaluator 分数。",
    }
    return summary, detailed
