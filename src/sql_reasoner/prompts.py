from __future__ import annotations

from typing import Any

from .constants import SYSTEM_PROMPT


def build_user_prompt(question: str, schema_text: str) -> str:
    return f"数据库结构：\n{schema_text}\n\n用户问题：\n{question.strip()}"


def build_chat_prompt(question: str, schema_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, schema_text)},
    ]


def messages_for_sft(question: str, schema_text: str, assistant_output: str) -> dict[str, Any]:
    return {
        "prompt": build_chat_prompt(question, schema_text),
        "completion": [{"role": "assistant", "content": assistant_output}],
    }
