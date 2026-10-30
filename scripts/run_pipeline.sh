#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/CSpider" >&2
  exit 2
fi

CSPIDER_ROOT="$1"
python scripts/capture_environment.py
python scripts/prepare_cspider.py \
  --train-json "$CSPIDER_ROOT/train.json" \
  --dev-json "$CSPIDER_ROOT/dev.json" \
  --database-dir "$CSPIDER_ROOT/database"
python scripts/create_retail_benchmark.py
accelerate launch --multi_gpu --num_processes 2 scripts/train_sft.py --config configs/sft.yaml
python scripts/select_grpo_data.py
SMOKE_STARTED=$(date +%s)
SMOKE=1 bash scripts/run_grpo.sh
SMOKE_ELAPSED=$(( $(date +%s) - SMOKE_STARTED ))
PROJECTED_SECONDS=$(( SMOKE_ELAPSED * 10 ))
if (( PROJECTED_SECONDS > 36000 )); then
  GRPO_STEPS=80
else
  GRPO_STEPS=200
fi
echo "GRPO smoke=${SMOKE_ELAPSED}s projected_200_steps=${PROJECTED_SECONDS}s selected_steps=${GRPO_STEPS}"
MAX_STEPS="$GRPO_STEPS" bash scripts/run_grpo.sh
python scripts/evaluate.py --dataset data/processed/cspider/official_dev.jsonl --output-dir outputs/base --model Qwen/Qwen3-4B-Instruct-2507
python scripts/evaluate.py --dataset data/processed/cspider/official_dev.jsonl --output-dir outputs/sft --model Qwen/Qwen3-4B-Instruct-2507 --adapter checkpoints/sft
python scripts/evaluate.py --dataset data/processed/cspider/official_dev.jsonl --output-dir outputs/grpo --model Qwen/Qwen3-4B-Instruct-2507 --adapter checkpoints/grpo
python scripts/compare_results.py --base outputs/base/predictions.jsonl --sft outputs/sft/predictions.jsonl --grpo outputs/grpo/predictions.jsonl
python scripts/sample_errors.py --predictions outputs/grpo/predictions.jsonl --output outputs/error_analysis.md --limit 30
python scripts/evaluate.py --dataset data/benchmarks/retail_cn/test.jsonl --output-dir outputs/private/base
python scripts/evaluate.py --dataset data/benchmarks/retail_cn/test.jsonl --output-dir outputs/private/sft --adapter checkpoints/sft
python scripts/evaluate.py --dataset data/benchmarks/retail_cn/test.jsonl --output-dir outputs/private/grpo --adapter checkpoints/grpo
python scripts/compare_results.py --base outputs/private/base/predictions.jsonl --sft outputs/private/sft/predictions.jsonl --grpo outputs/private/grpo/predictions.jsonl --output outputs/private/comparison.json
python scripts/evaluate.py --dataset data/benchmarks/retail_cn/robustness.jsonl --output-dir outputs/robustness/base
python scripts/evaluate.py --dataset data/benchmarks/retail_cn/robustness.jsonl --output-dir outputs/robustness/sft --adapter checkpoints/sft
python scripts/evaluate.py --dataset data/benchmarks/retail_cn/robustness.jsonl --output-dir outputs/robustness/grpo --adapter checkpoints/grpo
python scripts/compare_results.py --base outputs/robustness/base/predictions.jsonl --sft outputs/robustness/sft/predictions.jsonl --grpo outputs/robustness/grpo/predictions.jsonl --output outputs/robustness/comparison.json
python scripts/evaluate_ceval.py --output outputs/ceval-base.json
python scripts/evaluate_ceval.py --adapter checkpoints/sft --output outputs/ceval-sft.json
python scripts/evaluate_ceval.py --adapter checkpoints/grpo --output outputs/ceval-grpo.json

