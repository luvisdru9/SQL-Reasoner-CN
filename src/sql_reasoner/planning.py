from __future__ import annotations

from typing import Any

try:
    from sqlglot import exp, parse
    from sqlglot.errors import SqlglotError
except ImportError:  # pragma: no cover - actionable error is raised at call time
    exp = None
    parse = None
    SqlglotError = Exception


def _require_sqlglot() -> None:
    if parse is None:
        raise RuntimeError("需要安装 sqlglot：pip install -e .")


def parse_sql(sql: str):
    _require_sqlglot()
    try:
        statements = [statement for statement in parse(sql, read="sqlite") if statement is not None]
    except SqlglotError as exc:
        # Invalid model output is an expected negative prediction, not an
        # evaluator failure. Normalize parser-specific exceptions so reward
        # and sandbox callers can consistently assign score 0.
        raise ValueError(f"SQL 解析失败: {exc}") from exc
    if len(statements) != 1:
        raise ValueError("只允许一条 SQL 语句")
    return statements[0]


def extract_sql_features(sql: str) -> dict[str, Any]:
    tree = parse_sql(sql)
    tables = sorted({node.name for node in tree.find_all(exp.Table) if node.name})
    columns = sorted({node.name for node in tree.find_all(exp.Column) if node.name})
    joins = [node.sql(dialect="sqlite") for node in tree.find_all(exp.Join)]
    filters = []
    where = tree.find(exp.Where)
    if where is not None:
        filters.append(where.this.sql(dialect="sqlite"))
    having = tree.find(exp.Having)
    if having is not None:
        filters.append(having.this.sql(dialect="sqlite"))
    aggregation = [
        node.sql(dialect="sqlite") for node in tree.walk() if isinstance(node, exp.AggFunc)
    ]
    group = tree.find(exp.Group)
    order = tree.find(exp.Order)
    return {
        "tables": tables,
        "columns": columns,
        "joins": joins,
        "filters": filters,
        "aggregation": aggregation,
        "group_by": group.sql(dialect="sqlite") if group else None,
        "order_by": order.sql(dialect="sqlite") if order else None,
        "has_subquery": any(True for _ in tree.find_all(exp.Subquery)),
    }


def plan_from_sql(sql: str) -> dict[str, list[str]]:
    features = extract_sql_features(sql)
    return {
        "tables": features["tables"],
        "joins": features["joins"],
        "filters": features["filters"],
        "aggregation": features["aggregation"],
    }


def estimate_difficulty(sql: str) -> str:
    features = extract_sql_features(sql)
    score = 0
    score += len(features["joins"])
    score += 2 if features["has_subquery"] else 0
    score += 1 if features["group_by"] else 0
    score += 1 if features["order_by"] else 0
    score += 1 if len(features["aggregation"]) > 1 else 0
    if score <= 0:
        return "easy"
    if score <= 2:
        return "medium"
    if score <= 4:
        return "hard"
    return "extra-hard"


def canonical_sql(sql: str) -> str:
    return parse_sql(sql).sql(dialect="sqlite", pretty=False, normalize=True)
