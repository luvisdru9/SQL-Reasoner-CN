from __future__ import annotations

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = """你是一个只读数据库分析助手。根据问题和数据库结构制定简短、可核验的查询规划，并生成一条 SQLite SELECT 查询。
必须严格使用以下格式，不要输出额外内容：
<plan>
{"tables":[],"joins":[],"filters":[],"aggregation":[]}
</plan>
<sql>
SELECT ...
</sql>
禁止生成修改数据、修改结构、ATTACH、PRAGMA 或多语句 SQL。"""

PLAN_KEYS = ("tables", "joins", "filters", "aggregation")

REWARD_WEIGHTS = {
    "format": 0.05,
    "safe_sql": 0.10,
    "schema": 0.15,
    "plan_consistency": 0.10,
    "execution": 1.00,
}
