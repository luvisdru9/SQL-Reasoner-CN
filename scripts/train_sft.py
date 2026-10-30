#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sql_reasoner.io import load_yaml, read_jsonl, write_json


def longest_rows(rows: list[dict], limit: int) -> list[dict]:
    """Select deterministic, long examples for a memory-oriented smoke test."""
    return sorted(
        rows,
        key=lambda row: len(json.dumps(row.get("prompt", []), ensure_ascii=False))
        + len(json.dumps(row.get("completion", []), ensure_ascii=False)),
        reverse=True,
    )[:limit]


def checkpoint_missing_files(path: Path, world_size: int) -> list[str]:
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "_SUCCESS"]
    missing = [name for name in required if not (path / name).is_file()]
    model_files = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    if not any((path / name).is_file() for name in model_files):
        missing.append("model/adapter weights")
    rng_files = (
        [f"rng_state_{rank}.pth" for rank in range(world_size)]
        if world_size > 1
        else ["rng_state.pth"]
    )
    missing.extend(name for name in rng_files if not (path / name).is_file())
    return missing


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resolve_resume_checkpoint(output_dir: str, requested: str | None) -> str | None:
    if requested is None or requested.lower() in {"", "none", "false"}:
        return None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if requested.lower() != "auto":
        candidate = Path(requested)
        missing = checkpoint_missing_files(candidate, world_size)
        if missing:
            raise ValueError(f"checkpoint 不完整: {candidate}; 缺少 {missing}")
        return str(candidate)

    root = Path(output_dir)
    candidates = sorted(
        (
            path
            for path in root.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    for candidate in candidates:
        missing = checkpoint_missing_files(candidate, world_size)
        if not missing:
            return str(candidate)
        print(f"跳过不完整 checkpoint: {candidate}; 缺少 {missing}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for SQL-Reasoner-CN")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short two-GPU training smoke test without overwriting the full checkpoint",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=10,
        help="Optimizer steps used by --smoke (default: 10)",
    )
    parser.add_argument(
        "--resume",
        help="Resume from 'auto', 'none', or an explicit checkpoint directory",
    )
    parser.add_argument("--output-dir", help="Override output_dir from the YAML config")
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        help="Override per-device train batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Override gradient accumulation steps",
    )
    args = parser.parse_args()
    if args.smoke_steps <= 0:
        parser.error("--smoke-steps must be positive")
    if args.per_device_batch_size is not None and args.per_device_batch_size <= 0:
        parser.error("--per-device-batch-size must be positive")
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    cfg = load_yaml(args.config)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    class DurableCheckpointCallback(TrainerCallback):
        """Mark a checkpoint resumable only after every DDP rank finished saving."""

        def on_save(self, training_args, state, control, **kwargs):
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            if distributed:
                torch.distributed.barrier()
            if training_args.should_save:
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
            if distributed:
                torch.distributed.barrier()
            return control

    set_seed(cfg["seed"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    output_dir = args.output_dir or (
        f"{cfg['output_dir']}-smoke" if args.smoke else cfg["output_dir"]
    )
    per_device_batch_size = (
        args.per_device_batch_size
        if args.per_device_batch_size is not None
        else cfg["training"]["per_device_batch_size"]
    )
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else cfg["training"]["gradient_accumulation_steps"]
    )
    requested_resume = args.resume
    if requested_resume is None:
        requested_resume = cfg["training"].get("resume_from_checkpoint")
    resume_checkpoint = resolve_resume_checkpoint(output_dir, requested_resume)
    if local_rank == 0:
        if resume_checkpoint:
            print(f"从完整 checkpoint 恢复训练: {resume_checkpoint}")
        else:
            print("未找到恢复 checkpoint，将从头训练")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        use_cache=False,
    )
    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_rows = read_jsonl(cfg["data"]["train"])
    validation_rows = read_jsonl(cfg["data"]["validation"])
    if args.smoke:
        # Long examples are deliberately used here: a random tiny subset can
        # substantially understate activation memory at max_length=2048.
        train_rows = longest_rows(train_rows, 64)
        validation_rows = longest_rows(validation_rows, 16)
    train = Dataset.from_list(train_rows)
    validation = Dataset.from_list(validation_rows)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["epochs"],
        max_steps=args.smoke_steps if args.smoke else -1,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(cfg["training"]["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["training"]["warmup_ratio"],
        max_length=cfg["training"]["max_length"],
        packing=cfg["training"]["packing"],
        completion_only_loss=True,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if args.smoke else 10,
        eval_strategy="no" if args.smoke else "steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=args.smoke_steps if args.smoke else cfg["training"].get("save_steps", 25),
        save_total_limit=2 if args.smoke else cfg["training"].get("save_total_limit", 3),
        report_to="none" if args.smoke else cfg["training"].get("report_to", "none"),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        skip_memory_metrics=False,
        include_tokens_per_second=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
        peft_config=peft_config,
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
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        write_json(
            Path(output_dir) / f"vram_rank{local_rank}.json",
            {
                "local_rank": local_rank,
                "device": torch.cuda.get_device_name(device),
                "total_bytes": props.total_memory,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "wall_seconds": wall_seconds,
                "optimizer_steps": trainer.state.global_step,
                "train_examples": len(train_rows),
                "max_length": cfg["training"]["max_length"],
                "per_device_batch_size": per_device_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "effective_batch_size": per_device_batch_size
                * gradient_accumulation_steps
                * int(os.environ.get("WORLD_SIZE", "1")),
            },
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
