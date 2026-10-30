#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Small fixed C-Eval regression check")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["computer_network", "operating_system", "college_programming"],
    )
    parser.add_argument("--limit-per-subject", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    results = []
    for subject in args.subjects:
        dataset = load_dataset("ceval/ceval-exam", subject, split="val")
        for row in list(dataset)[: args.limit_per_subject]:
            prompt = (
                "请只回答正确选项的字母 A、B、C 或 D。\n"
                f"问题：{row['question']}\nA. {row['A']}\nB. {row['B']}\n"
                f"C. {row['C']}\nD. {row['D']}\n答案："
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            text = tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            match = re.search(r"[ABCD]", text.upper())
            prediction = match.group(0) if match else ""
            results.append(
                {
                    "subject": subject,
                    "prediction": prediction,
                    "answer": row["answer"],
                    "correct": prediction == row["answer"],
                }
            )
    summary = {
        "count": len(results),
        "accuracy": sum(row["correct"] for row in results) / max(len(results), 1),
        "subjects": args.subjects,
        "results": results,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({key: value for key, value in summary.items() if key != "results"})


if __name__ == "__main__":
    main()
