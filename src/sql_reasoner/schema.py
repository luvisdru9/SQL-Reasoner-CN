from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def introspect_schema(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path).resolve()
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        result: dict[str, Any] = {"tables": {}}
        for table in tables:
            escaped = table.replace('"', '""')
            columns = []
            for cid, name, dtype, notnull, default, pk in conn.execute(
                f'PRAGMA table_info("{escaped}")'
            ):
                columns.append(
                    {
                        "name": name,
                        "type": dtype or "TEXT",
                        "not_null": bool(notnull),
                        "primary_key": bool(pk),
                    }
                )
            foreign_keys = []
            for row in conn.execute(f'PRAGMA foreign_key_list("{escaped}")'):
                foreign_keys.append({"from": row[3], "to_table": row[2], "to_column": row[4]})
            result["tables"][table] = {
                "columns": columns,
                "foreign_keys": foreign_keys,
            }
        return result


def serialize_schema(schema: dict[str, Any]) -> str:
    lines = []
    for table_name, table in sorted(schema["tables"].items()):
        columns = []
        for column in table["columns"]:
            flags = " PRIMARY KEY" if column.get("primary_key") else ""
            columns.append(f"{column['name']} {column.get('type', 'TEXT')}{flags}")
        lines.append(f"TABLE {table_name} ({', '.join(columns)})")
        for fk in table.get("foreign_keys", []):
            lines.append(
                f"FOREIGN KEY {table_name}.{fk['from']} -> {fk['to_table']}.{fk['to_column']}"
            )
    return "\n".join(lines)


def schema_names(schema: dict[str, Any]) -> tuple[set[str], set[str]]:
    tables = {name.lower() for name in schema.get("tables", {})}
    columns = {
        col["name"].lower()
        for table in schema.get("tables", {}).values()
        for col in table.get("columns", [])
    }
    return tables, columns
