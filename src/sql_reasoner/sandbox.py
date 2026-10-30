from __future__ import annotations

import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from .planning import parse_sql

try:
    from sqlglot import exp
except ImportError:  # pragma: no cover
    exp = None


class UnsafeSQLError(ValueError):
    pass


class SQLExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    elapsed_ms: float
    ordered: bool
    truncated: bool = False


_DENIED_ACTION_NAMES = (
    "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TEMP_INDEX",
    "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_TRIGGER",
    "SQLITE_CREATE_TEMP_VIEW",
    "SQLITE_CREATE_TRIGGER",
    "SQLITE_CREATE_VIEW",
    "SQLITE_DELETE",
    "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_TEMP_TRIGGER",
    "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_TRIGGER",
    "SQLITE_DROP_VIEW",
    "SQLITE_INSERT",
    "SQLITE_ALTER_TABLE",
    "SQLITE_REINDEX",
    "SQLITE_ANALYZE",
    "SQLITE_ATTACH",
    "SQLITE_DETACH",
    "SQLITE_TRANSACTION",
    "SQLITE_SAVEPOINT",
    "SQLITE_UPDATE",
    "SQLITE_PRAGMA",
)
_DENIED_ACTIONS = {
    getattr(sqlite3, name) for name in _DENIED_ACTION_NAMES if hasattr(sqlite3, name)
}


def validate_read_only_sql(sql: str) -> None:
    if not sql or not sql.strip():
        raise UnsafeSQLError("SQL 不能为空")
    tree = parse_sql(sql)
    if exp is None:  # pragma: no cover
        raise RuntimeError("sqlglot 未安装")
    if not isinstance(tree, exp.Query):
        raise UnsafeSQLError("只允许 SELECT/WITH 查询")
    forbidden_types = tuple(
        cls
        for cls in (
            getattr(exp, "Insert", None),
            getattr(exp, "Update", None),
            getattr(exp, "Delete", None),
            getattr(exp, "Create", None),
            getattr(exp, "Drop", None),
            getattr(exp, "Alter", None),
            getattr(exp, "Command", None),
            getattr(exp, "Transaction", None),
        )
        if cls is not None
    )
    if forbidden_types and any(isinstance(node, forbidden_types) for node in tree.walk()):
        raise UnsafeSQLError("SQL 包含禁止的写入或管理操作")


def _authorizer(action: int, _arg1: str, _arg2: str, _db: str, _trigger: str) -> int:
    if action in _DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def execute_read_only(
    db_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float = 2.0,
    max_rows: int = 10_000,
) -> QueryResult:
    validate_read_only_sql(sql)
    path = Path(db_path).resolve(strict=True)
    deadline = time.monotonic() + timeout_seconds
    uri = f"file:{path.as_posix()}?mode=ro"
    started = time.perf_counter()
    try:
        with sqlite3.connect(uri, uri=True, timeout=timeout_seconds) as conn:
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(_authorizer)
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1_000)
            cursor = conn.execute(sql)
            columns = tuple(item[0] for item in (cursor.description or ()))
            rows = cursor.fetchmany(max_rows + 1)
    except (sqlite3.Error, OSError) as exc:
        raise SQLExecutionError(str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1_000
    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]
    tree = parse_sql(sql)
    ordered = tree.find(exp.Order) is not None
    return QueryResult(
        columns=columns,
        rows=tuple(tuple(row) for row in rows),
        elapsed_ms=elapsed_ms,
        ordered=ordered,
        truncated=truncated,
    )


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("number", Decimal(int(value)))
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ("float", repr(value))
        try:
            return ("number", Decimal(str(value)).normalize())
        except InvalidOperation:
            return ("text", str(value))
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    return ("text", str(value))


def normalize_rows(rows: Iterable[Iterable[Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(_normalize_scalar(value) for value in row) for row in rows)


def results_equal(predicted: QueryResult, gold: QueryResult) -> bool:
    if predicted.truncated or gold.truncated:
        return False
    pred_rows = normalize_rows(predicted.rows)
    gold_rows = normalize_rows(gold.rows)
    if gold.ordered:
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


def execution_matches(
    db_path: str | Path,
    predicted_sql: str,
    gold_sql: str,
    *,
    timeout_seconds: float = 2.0,
) -> bool:
    try:
        predicted = execute_read_only(db_path, predicted_sql, timeout_seconds=timeout_seconds)
        gold = execute_read_only(db_path, gold_sql, timeout_seconds=timeout_seconds)
    except (UnsafeSQLError, SQLExecutionError, ValueError):
        return False
    return results_equal(predicted, gold)
