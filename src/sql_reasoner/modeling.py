from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .formats import ParsedOutput, parse_model_output
from .prompts import build_chat_prompt


@dataclass
class Generation:
    text: str
    parsed: ParsedOutput | None
    elapsed_ms: float
    generated_tokens: int


class ModelRunner:
    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depends on GPU environment
            raise RuntimeError("请安装训练依赖：pip install -e '.[train]'") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(
        self,
        question: str,
        schema: str,
        *,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 0.95,
        seed: int = 42,
    ) -> Generation:
        self.torch.manual_seed(seed)
        messages = build_chat_prompt(question, schema)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            # Qwen's bundled generation config sets top_k=20. Disable it so
            # sampling follows this project's documented temperature/top-p policy.
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        else:
            # Explicitly unset sampling-only values bundled with the model to
            # keep greedy evaluation deterministic and warning-free.
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        completion_ids = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        try:
            parsed = parse_model_output(text)
        except ValueError:
            parsed = None
        return Generation(
            text=text,
            parsed=parsed,
            elapsed_ms=elapsed_ms,
            generated_tokens=int(completion_ids.shape[0]),
        )
