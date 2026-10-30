from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .constants import PLAN_KEYS

_OUTPUT_RE = re.compile(
    r"^\s*<plan>\s*(?P<plan>\{.*?\})\s*</plan>\s*"
    r"<sql>\s*(?P<sql>.*?)\s*</sql>\s*$",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class ParsedOutput:
    plan: dict[str, Any]
    sql: str


def parse_model_output(text: str) -> ParsedOutput:
    match = _OUTPUT_RE.match(text or "")
    if not match:
        raise ValueError("输出必须仅包含 <plan>...</plan><sql>...</sql>")
    try:
        plan = json.loads(match.group("plan"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"plan 不是合法 JSON: {exc}") from exc
    if not isinstance(plan, dict):
        raise ValueError("plan 必须是 JSON object")
    missing = [key for key in PLAN_KEYS if key not in plan]
    if missing:
        raise ValueError(f"plan 缺少字段: {', '.join(missing)}")
    for key in PLAN_KEYS:
        if not isinstance(plan[key], list):
            raise ValueError(f"plan.{key} 必须是数组")
    sql = match.group("sql").strip()
    if not sql:
        raise ValueError("sql 不能为空")
    return ParsedOutput(plan=plan, sql=sql)


def format_model_output(plan: dict[str, Any], sql: str) -> str:
    ordered = {key: plan.get(key, []) for key in PLAN_KEYS}
    plan_json = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    return f"<plan>\n{plan_json}\n</plan>\n<sql>\n{sql.strip()}\n</sql>"


def completion_to_text(completion: Any) -> str:
    """Normalize Transformers/TRL completion shapes to a plain string."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")
