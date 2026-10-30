from __future__ import annotations

import argparse
import json
from pathlib import Path

from .formats import parse_model_output
from .io import read_jsonl
from .modeling import ModelRunner
from .sandbox import SQLExecutionError, UnsafeSQLError, execute_read_only


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL-Reasoner-CN Gradio demo")
    parser.add_argument("--benchmark", default="data/benchmarks/retail_cn/test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--sft-adapter", default="checkpoints/sft")
    parser.add_argument("--grpo-adapter", default="checkpoints/grpo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("请安装 Demo 依赖：pip install -e '.[demo,train]'") from exc

    rows = read_jsonl(args.benchmark)
    by_id = {row["id"]: row for row in rows}
    adapters = {
        "Base": None,
        "SFT": args.sft_adapter,
        "GRPO": args.grpo_adapter,
    }
    cache: dict[str, ModelRunner] = {}

    def select_case(sample_id: str):
        row = by_id[sample_id]
        return row["question"], row["schema"]

    def run(sample_id: str, question: str, checkpoint: str):
        row = by_id[sample_id]
        adapter = adapters[checkpoint]
        if adapter and not Path(adapter).exists():
            return "", "", [], f"adapter 不存在：{adapter}"
        if checkpoint not in cache:
            cache.clear()
            cache[checkpoint] = ModelRunner(args.model, adapter_path=adapter)
        generation = cache[checkpoint].generate(question, row["schema"])
        if generation.parsed is None:
            return generation.text, "", [], f"格式解析失败；耗时 {generation.elapsed_ms:.0f} ms"
        parsed = parse_model_output(generation.text)
        try:
            result = execute_read_only(row["db_path"], parsed.sql)
            table = [list(result.columns), *[list(item) for item in result.rows[:100]]]
            status = f"执行成功；模型耗时 {generation.elapsed_ms:.0f} ms；SQL耗时 {result.elapsed_ms:.1f} ms"
        except (UnsafeSQLError, SQLExecutionError) as exc:
            table = []
            status = f"SQL 拒绝或失败：{exc}"
        return json.dumps(parsed.plan, ensure_ascii=False, indent=2), parsed.sql, table, status

    first = rows[0]
    with gr.Blocks(title="SQL-Reasoner-CN") as demo:
        gr.Markdown("# SQL-Reasoner-CN\n只读中文数据库分析助手；支持 Base/SFT/GRPO 对比。")
        with gr.Row():
            sample_id = gr.Dropdown(list(by_id), value=first["id"], label="预置案例")
            checkpoint = gr.Radio(list(adapters), value="SFT", label="Checkpoint")
        question = gr.Textbox(value=first["question"], label="中文问题")
        schema = gr.Code(
            value=first["schema"], language=None, label="数据库结构", interactive=False
        )
        run_button = gr.Button("生成并安全执行", variant="primary")
        plan = gr.Code(label="结构化规划", language="json")
        sql = gr.Code(label="SQL", language="sql")
        result = gr.Dataframe(label="查询结果", interactive=False)
        status = gr.Textbox(label="状态", interactive=False)
        sample_id.change(select_case, sample_id, [question, schema])
        run_button.click(run, [sample_id, question, checkpoint], [plan, sql, result, status])
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
