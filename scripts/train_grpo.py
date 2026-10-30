#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from sql_reasoner.constants import REWARD_WEIGHTS
from sql_reasoner.io import load_yaml, read_jsonl, write_json
from sql_reasoner.rewards import reward_functions


def checkpoint_missing_files(path: Path) -> list[str]:
    required = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "_SUCCESS",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    model_files = ["adapter_model.safetensors", "adapter_model.bin"]
    if not any((path / name).is_file() for name in model_files):
        missing.append("adapter weights")
    return missing


def resolve_resume_checkpoint(output_dir: str, requested: str | None) -> str | None:
    if requested is None or requested.lower() in {"", "none", "false"}:
        return None
    if requested.lower() != "auto":
        candidate = Path(requested)
        missing = checkpoint_missing_files(candidate)
        if missing:
            raise ValueError(f"checkpoint 不完整: {candidate}; 缺少 {missing}")
        return str(candidate)
    candidates = sorted(
        (
            path
            for path in Path(output_dir).glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    for candidate in candidates:
        missing = checkpoint_missing_files(candidate)
        if not missing:
            return str(candidate)
        print(f"跳过不完整 checkpoint: {candidate}; 缺少 {missing}")
    return None


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO with selected DAPO improvements")
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run exactly 20 optimization steps")
    parser.add_argument("--max-steps", type=int, help="Override config max_steps after ETA guard")
    parser.add_argument("--output-dir", help="Override output_dir from the YAML config")
    parser.add_argument("--sft-adapter", help="Override the SFT adapter path")
    parser.add_argument("--data", help="Override the GRPO training JSONL")
    parser.add_argument("--resume", help="Resume from 'auto', 'none', or a checkpoint directory")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import GRPOConfig, GRPOTrainer

    class DurableCheckpointCallback(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            checkpoint = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            for saved_file in checkpoint.rglob("*"):
                if saved_file.is_file():
                    fsync_file(saved_file)
            marker = checkpoint / "_SUCCESS"
            with marker.open("w", encoding="utf-8") as handle:
                handle.write(f"global_step={state.global_step}\n")
                handle.flush()
                os.fsync(handle.fileno())
            directory_fd = os.open(checkpoint, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return control

    set_seed(cfg["seed"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    sft_adapter = args.sft_adapter or cfg["sft_adapter"]
    data_path = args.data or cfg["data"]["train"]
    output_dir = args.output_dir or (
        f"{cfg['output_dir']}-smoke" if args.smoke else cfg["output_dir"]
    )
    requested_resume = args.resume
    if requested_resume is None:
        requested_resume = "none" if args.smoke else cfg["training"].get("resume_from_checkpoint")
    resume_checkpoint = resolve_resume_checkpoint(output_dir, requested_resume)
    if resume_checkpoint:
        print(f"从完整 checkpoint 恢复训练: {resume_checkpoint}")
    else:
        print("未找到恢复 checkpoint，将从头训练")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_in_4bit = bool(cfg["training"].get("load_in_4bit", False))
    if load_in_4bit and cfg["vllm"]["enabled"]:
        raise ValueError(
            "当前 TRL/vLLM server 权重同步不能发送 bitsandbytes 的压缩 Params4bit；"
            "请为 GRPO 设置 training.load_in_4bit=false。SFT adapter 仍可来自 QLoRA。"
        )
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": local_rank},
        "use_cache": False,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **model_kwargs)
    if load_in_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
    rows = read_jsonl(data_path)
    if not rows:
        raise ValueError(f"GRPO 训练数据为空: {data_path}")
    dataset = Dataset.from_list(rows)
    max_steps = args.max_steps or (20 if args.smoke else cfg["training"]["max_steps"])
    algorithm = cfg.get("algorithm", {})
    reward_weights = cfg.get("rewards", REWARD_WEIGHTS)
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=cfg["training"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type="cosine",
        max_prompt_length=cfg["training"]["max_prompt_length"],
        max_completion_length=cfg["training"]["max_completion_length"],
        num_generations=cfg["generation"]["num_generations"],
        generation_batch_size=cfg["generation"].get("generation_batch_size"),
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        loss_type=algorithm.get("loss_type", "dapo"),
        epsilon=algorithm.get("epsilon", 0.2),
        epsilon_high=algorithm.get("epsilon_high", 0.28),
        mask_truncated_completions=algorithm.get("mask_truncated_completions", True),
        beta=algorithm.get("beta", 0.0),
        scale_rewards=algorithm.get("scale_rewards", "group"),
        importance_sampling_level=algorithm.get("importance_sampling_level", "token"),
        reward_weights=[
            reward_weights["format"],
            reward_weights["safe_sql"],
            reward_weights["schema"],
            reward_weights["plan_consistency"],
            reward_weights["execution"],
        ],
        use_vllm=cfg["vllm"]["enabled"],
        vllm_mode=cfg["vllm"]["mode"],
        vllm_server_host=cfg["vllm"]["host"],
        vllm_server_port=cfg["vllm"]["port"],
        vllm_server_timeout=cfg["vllm"]["timeout"],
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=1 if args.smoke else 5,
        log_completions=True,
        num_completions_to_print=2,
        save_steps=max_steps if args.smoke else cfg["training"].get("save_steps", 10),
        save_total_limit=1 if args.smoke else cfg["training"].get("save_total_limit", 3),
        report_to="none" if args.smoke else cfg["training"].get("report_to", "none"),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=False,
        skip_memory_metrics=False,
        # GRPO's collator intentionally returns a list of raw examples because
        # prompts are generated and tokenized inside GRPOTrainer. Transformers'
        # generic tokens/s pre-scan assumes every batch is an input_ids dict and
        # crashes before the first step (transformers 4.57.x + TRL 0.23.x).
        # GRPOTrainer still logs its own num_tokens/completion length metrics.
        include_tokens_per_second=False,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions(),
        args=grpo_args,
        train_dataset=dataset,
        callbacks=[DurableCheckpointCallback()],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        write_json(
            Path(output_dir) / "vram.json",
            {
                "device": torch.cuda.get_device_name(device),
                "total_bytes": torch.cuda.get_device_properties(device).total_memory,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "wall_seconds": wall_seconds,
                "optimizer_steps": trainer.state.global_step,
                "train_examples": len(rows),
                "sft_adapter": sft_adapter,
                "data": data_path,
                "load_in_4bit": load_in_4bit,
                "per_device_batch_size": cfg["training"]["per_device_batch_size"],
                "gradient_accumulation_steps": cfg["training"][
                    "gradient_accumulation_steps"
                ],
                "generation_batch_size": cfg["generation"].get("generation_batch_size"),
                "num_generations": cfg["generation"]["num_generations"],
                "max_prompt_length": cfg["training"]["max_prompt_length"],
                "max_completion_length": cfg["training"]["max_completion_length"],
            },
        )


if __name__ == "__main__":
    main()
