from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from .formats import completion_to_text, parse_model_output
from .planning import extract_sql_features
from .sandbox import SQLExecutionError, UnsafeSQLError, execution_matches, validate_read_only_sql
from .schema import introspect_schema, schema_names


def format_score(text: str) -> float:
    try:
        parse_model_output(text)
        return 1.0
    except ValueError:
        return 0.0


def safe_sql_score(text: str) -> float:
    try:
        validate_read_only_sql(parse_model_output(text).sql)
        return 1.0
    except (ValueError, UnsafeSQLError, SQLExecutionError):
        return 0.0


@lru_cache(maxsize=256)
def _schema_for_path(path: str) -> dict[str, Any]:
    return introspect_schema(path)


def schema_score(text: str, db_path: str) -> float:
    try:
        sql = parse_model_output(text).sql
        features = extract_sql_features(sql)
        tables, columns = schema_names(_schema_for_path(str(db_path)))
    except (ValueError, OSError, RuntimeError):
        return 0.0
    referenced_tables = {name.lower() for name in features["tables"]}
    referenced_columns = {name.lower() for name in features["columns"] if name != "*"}
    return float(referenced_tables <= tables and referenced_columns <= columns)


def plan_consistency_score(text: str) -> float:
    try:
        parsed = parse_model_output(text)
        features = extract_sql_features(parsed.sql)
    except (ValueError, RuntimeError):
        return 0.0
    plan_tables = {str(name).lower() for name in parsed.plan["tables"]}
    sql_tables = {name.lower() for name in features["tables"]}
    table_ok = plan_tables == sql_tables
    join_ok = bool(parsed.plan["joins"]) == bool(features["joins"])
    filter_ok = bool(parsed.plan["filters"]) == bool(features["filters"])
    agg_ok = bool(parsed.plan["aggregation"]) == bool(features["aggregation"])
    return (float(table_ok) + float(join_ok) + float(filter_ok) + float(agg_ok)) / 4.0


def execution_score(text: str, db_path: str, gold_sql: str) -> float:
    try:
        predicted_sql = parse_model_output(text).sql
    except ValueError:
        return 0.0
    return float(execution_matches(db_path, predicted_sql, gold_sql))


def component_scores(text: str, db_path: str, gold_sql: str) -> dict[str, float]:
    return {
        "format": format_score(text),
        "safe_sql": safe_sql_score(text),
        "schema": schema_score(text, db_path),
        "plan_consistency": plan_consistency_score(text),
        "execution": execution_score(text, db_path, gold_sql),
    }


def _reward_adapter(
    scorer: Callable[..., float],
    *field_names: str,
) -> Callable[..., list[float]]:
    def reward(completions: list[Any], **kwargs: Any) -> list[float]:
        values = []
        for index, completion in enumerate(completions):
            args = [completion_to_text(completion)]
            args.extend(kwargs[name][index] for name in field_names)
            values.append(float(scorer(*args)))
        return values

    reward.__name__ = f"{scorer.__name__}_reward"
    return reward


format_reward = _reward_adapter(format_score)
safe_sql_reward = _reward_adapter(safe_sql_score)
schema_reward = _reward_adapter(schema_score, "db_path")
plan_consistency_reward = _reward_adapter(plan_consistency_score)
execution_reward = _reward_adapter(execution_score, "db_path", "gold_sql")


def reward_functions() -> list[Callable[..., list[float]]]:
    return [
        format_reward,
        safe_sql_reward,
        schema_reward,
        plan_consistency_reward,
        execution_reward,
    ]
