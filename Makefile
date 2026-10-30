PYTHON ?= python3.10
CSPIDER_DIR ?= data/raw/CSpider

.PHONY: install test retail prepare sft sft-smoke select-grpo select-grpo-smoke rl-smoke grpo grpo-smoke evaluate evaluate-ceval calibrate-resources stability experiments-list experiments-dry-run campaign-plan campaign demo pipeline

install:
	$(PYTHON) -m pip install -e '.[train,demo,eval,dev]'

test:
	$(PYTHON) -m unittest discover -s tests -v

retail:
	$(PYTHON) scripts/create_retail_benchmark.py

prepare: retail
	$(PYTHON) scripts/prepare_cspider.py --train-json $(CSPIDER_DIR)/train.json --dev-json $(CSPIDER_DIR)/dev.json --database-dir $(CSPIDER_DIR)/database

sft:
	accelerate launch --multi_gpu --num_processes 2 scripts/train_sft.py --config configs/sft.yaml --resume auto

sft-smoke:
	accelerate launch --multi_gpu --num_processes 2 scripts/train_sft.py --config configs/sft.yaml --smoke --resume none

select-grpo:
	$(PYTHON) scripts/select_grpo_data.py

select-grpo-smoke:
	PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 $(PYTHON) scripts/select_grpo_data.py --adapter checkpoints/sft-smoke --output data/processed/cspider/grpo_smoke.jsonl --stats-output data/processed/cspider/grpo_smoke.stats.json --limit 4 --max-scan 32 --max-new-tokens 192 --log-every 1 --min-variance-positive 1

rl-smoke: select-grpo-smoke
	SMOKE=1 MAX_STEPS=2 SFT_ADAPTER=checkpoints/sft-smoke DATA_PATH=data/processed/cspider/grpo_smoke.jsonl OUTPUT_DIR=checkpoints/grpo-smoke RESUME=none VLLM_GPU_MEMORY_UTILIZATION=0.72 bash scripts/run_grpo.sh

grpo-smoke:
	SMOKE=1 bash scripts/run_grpo.sh

grpo:
	bash scripts/run_grpo.sh

evaluate:
	$(PYTHON) scripts/evaluate.py --dataset data/processed/cspider/official_dev.jsonl --output-dir outputs/base
	$(PYTHON) scripts/evaluate.py --dataset data/processed/cspider/official_dev.jsonl --output-dir outputs/sft --adapter checkpoints/sft
	$(PYTHON) scripts/evaluate.py --dataset data/processed/cspider/official_dev.jsonl --output-dir outputs/grpo --adapter checkpoints/grpo
	$(PYTHON) scripts/compare_results.py --base outputs/base/predictions.jsonl --sft outputs/sft/predictions.jsonl --grpo outputs/grpo/predictions.jsonl
	$(PYTHON) scripts/sample_errors.py --predictions outputs/grpo/predictions.jsonl --output outputs/error_analysis.md --limit 30
	$(PYTHON) scripts/evaluate.py --dataset data/benchmarks/retail_cn/test.jsonl --output-dir outputs/private/base
	$(PYTHON) scripts/evaluate.py --dataset data/benchmarks/retail_cn/test.jsonl --output-dir outputs/private/sft --adapter checkpoints/sft
	$(PYTHON) scripts/evaluate.py --dataset data/benchmarks/retail_cn/test.jsonl --output-dir outputs/private/grpo --adapter checkpoints/grpo
	$(PYTHON) scripts/compare_results.py --base outputs/private/base/predictions.jsonl --sft outputs/private/sft/predictions.jsonl --grpo outputs/private/grpo/predictions.jsonl --output outputs/private/comparison.json
	$(PYTHON) scripts/evaluate.py --dataset data/benchmarks/retail_cn/robustness.jsonl --output-dir outputs/robustness/base
	$(PYTHON) scripts/evaluate.py --dataset data/benchmarks/retail_cn/robustness.jsonl --output-dir outputs/robustness/sft --adapter checkpoints/sft
	$(PYTHON) scripts/evaluate.py --dataset data/benchmarks/retail_cn/robustness.jsonl --output-dir outputs/robustness/grpo --adapter checkpoints/grpo
	$(PYTHON) scripts/compare_results.py --base outputs/robustness/base/predictions.jsonl --sft outputs/robustness/sft/predictions.jsonl --grpo outputs/robustness/grpo/predictions.jsonl --output outputs/robustness/comparison.json

evaluate-ceval:
	$(PYTHON) scripts/evaluate_ceval.py --output outputs/ceval-base.json
	$(PYTHON) scripts/evaluate_ceval.py --adapter checkpoints/sft --output outputs/ceval-sft.json
	$(PYTHON) scripts/evaluate_ceval.py --adapter checkpoints/grpo --output outputs/ceval-grpo.json

experiments-list:
	$(PYTHON) scripts/run_experiments.py --action list

experiments-dry-run:
	$(PYTHON) scripts/run_experiments.py --action all --group core --dry-run

campaign-plan:
	$(PYTHON) scripts/run_campaign.py --preset full --dry-run

campaign:
	$(PYTHON) scripts/run_campaign.py --preset full

calibrate-resources:
	$(PYTHON) scripts/calibrate_resources.py --stage all

stability:
	$(PYTHON) scripts/run_stability_tests.py --stage all --steps 50

demo: retail
	$(PYTHON) -m sql_reasoner.app

pipeline:
	bash scripts/run_pipeline.sh $(CSPIDER_DIR)

