# SQL-Reasoner-CN

面向双 RTX 3090 Ti 的中文 Text-to-SQL 后训练项目：用 QLoRA 做结构化冷启动，再用可执行 SQL 奖励进行 GRPO，并采用 DAPO 的 token-level loss、Clip-Higher 和截断样本屏蔽。

项目不会把“跑通训练”当成效果结论。Base、SFT、GRPO 使用相同 prompt、解码和测试集，逐样本预测、失败类型和显著性检验均会落盘；若 RL 没有提升，报告保留负结果。

## 已实现内容

- CSpider 按 `db_id` 隔离的 train/validation/dev 数据转换，自动记录拒绝样本。
- `sqlglot` AST 自动生成 `<plan>`，不依赖外部教师 API。
- 50 条中文零售私有测试题和 30 条鲁棒性改写，可重复生成。
- 只读 SQLite 沙箱：单语句检查、AST 检查、SQLite authorizer、query-only、超时和最大结果行数。
- 五段 GRPO 奖励、一次性奖励方差筛选、QLoRA-SFT、GRPO/DAPO-loss 训练入口。
- Execution Accuracy、AST EM、Schema F1、难度分层、bootstrap CI、McNemar 和规则错误分类。
- Base/SFT/GRPO 对比脚本、C-Eval 回归检查、Gradio 只读 Demo。

当前仓库不包含模型权重、CSpider 原始数据或伪造实验结果。实际分数只能在完成 GPU 训练后写入。

## 环境

建议使用新的 Python 3.10/3.11 conda 环境，不使用系统 Python：

```bash
conda create -n sql-reasoner python=3.10 -y
conda activate sql-reasoner
pip install -e '.[train,demo,eval,dev]'
```

依赖固定在 [setup.cfg](setup.cfg)，构建和检查配置位于 [pyproject.toml](pyproject.toml)。其中 `trl==0.24.0` 与 `vllm==0.10.2` 配套；不要只升级其中一个。若 PyPI 的 PyTorch wheel 与本机 CUDA 不匹配，先按 PyTorch 官方方式安装 CUDA 对应版本，再安装其余依赖并核对 `pip check`。

运行预检：

```bash
python -m unittest discover -s tests -v
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

## 数据准备

从 CSpider 官方页面下载数据，并整理为：

```text
data/raw/CSpider/
├── train.json
├── dev.json
├── tables.json
└── database/<db_id>/<db_id>.sqlite
```

执行：

```bash
make prepare CSPIDER_DIR=data/raw/CSpider
```

输出位于 `data/processed/cspider/`。`stats.json` 保存数据库级切分和过滤统计；`rejected.jsonl` 保留每条拒绝原因。私有零售基准由固定随机种子生成到 `data/benchmarks/retail_cn/`。

## 训练

### 1. QLoRA-SFT

先用 64 条较长样本运行 10 个 optimizer steps，验证双卡 DDP、checkpoint 保存和显存余量：

```bash
make sft-smoke
```

冒烟输出写入 `checkpoints/sft-smoke/`，其中 `vram_rank0.json` 和
`vram_rank1.json` 分别记录两张卡的峰值 allocated/reserved 显存；该结果不用于效果评测。
确认冒烟测试通过后再运行正式训练：

```bash
make sft
```

正式 SFT 每 25 个 optimizer steps 保存一次完整 checkpoint，并保留最近 3 份。
`make sft` 默认扫描 `checkpoints/sft/checkpoint-*`，跳过因断电或重启而写入不完整的目录，
从最近一个同时包含 adapter、optimizer、scheduler、trainer state 和两张卡 RNG state 的
checkpoint 恢复。系统重启后重新激活同一 conda 环境并再次执行 `make sft` 即可；
恢复会延续 global step、学习率调度、随机数状态和 dataloader 进度，而不是重新开始。

默认使用两卡 DDP，每卡 batch 1、梯度累积 8，所以有效 batch 为 16。训练的是 `<plan>` 与 `<sql>` completion，配置见 [configs/sft.yaml](configs/sft.yaml)。

### 2. 构造 GRPO 子集

```bash
make select-grpo
```

脚本从中高难度样本为每题采样 4 次，优先保留奖励有方差的问题，最多 500 条。这是离线难度筛选，不称作完整 DAPO Dynamic Sampling。

在正式 SFT 之前，可以使用 `sft-smoke` adapter 验证完整 RL 工程链路：

```bash
HF_HUB_OFFLINE=1 make rl-smoke
```

该命令最多扫描 32 条中高难度样本，选择 4 条小数据，然后让 GPU 1 运行 vLLM、
GPU 0 执行 2 个 GRPO optimizer steps。输出分别位于
`data/processed/cspider/grpo_smoke.jsonl`、`checkpoints/grpo-smoke/`；只用于验证
采样、奖励、权重同步、反向传播和保存，不用于效果结论。

### 3. GRPO

先做 20-step 冒烟测试：

```bash
make grpo-smoke
```

记录总耗时并估算 200 steps。若预计超过 10 小时，将 `configs/grpo.yaml` 的 `max_steps` 固定改为 80 并在报告中记录，不更换模型或测试集。正式训练：

```bash
make grpo
```

GPU 1 运行 TRL vLLM server，GPU 0 以 BF16 基座继续训练 SFT LoRA。SFT 仍使用
4-bit QLoRA；GRPO 不使用 4-bit 基座，因为 TRL server 模式需要把合并后的全尺寸权重
同步给 vLLM。GRPO 固定 micro batch 2、梯度累积 2，一次生成 4 个候选。若出现
NCCL 错误，先确认两个进程的 `CUDA_VISIBLE_DEVICES` 没有重叠；若 vLLM OOM，再建立并
记录新的资源配置，不要只对某一个消融实验临时降低显存利用率。

## 资源锁定的消融与调参

统一入口是 [scripts/run_experiments.py](scripts/run_experiments.py)，实验清单位于
[configs/experiments.yaml](configs/experiments.yaml)。每个阶段先固定经过显存冒烟测试的
resource profile，再只允许实验覆盖 `tunable` 白名单中的参数。默认资源配置为：

- SFT：双卡 DDP、每卡 micro batch 1、梯度累积 8、长度 2048、4-bit QLoRA；
- RL：GPU 0 BF16 LoRA 训练、GPU 1 vLLM、micro batch 2、梯度累积 2、4 个候选、
  prompt/completion 上限 1664/384、vLLM 利用率 0.72。

在冻结 profile 前，用相同有效 batch 独立校准 micro batch。校准脚本会测试 SFT 的
`(batch, accumulation)=(1,8)/(2,4)/(4,2)` 和 RL 的 `(1,4)/(2,2)/(4,1)`，只在
峰值 reserved 显存不超过 90% 的成功候选中选择实测吞吐最高者：

```bash
python scripts/calibrate_resources.py --stage all --dry-run
make calibrate-resources
```

结果写入 `outputs/resource_calibration/summary.json`。脚本不会自动改实验矩阵；检查温度、
功耗和日志后，再把选中的配置一次性写入 resource profile。这样资源搜索与模型效果搜索
不会互相污染。短 smoke 不能证明长期温度稳定，正式训练前还应让推荐配置连续运行至少
50 steps，并保留约 10% 显存余量。

选定 profile 后，一条命令顺序运行 SFT 与 RL 各 50-step 长稳定性测试：

```bash
make stability
```

脚本会每 2 秒采样两张 GPU 的温度、功耗、显存与利用率，分别保存训练日志和 telemetry
CSV，并在 `outputs/stability/summary.json` 与 `summary.md` 中汇总。默认验收上限为 83°C
和 90% 显存；单阶段失败不会阻止另一阶段采集诊断数据，但整体命令最终返回非零状态。
已通过的同 step 测试会自动跳过，使用 `--force` 才会重跑。

查看实验或验证将要执行的命令，不会启动训练：

```bash
make experiments-list
make experiments-dry-run
```

核心消融按顺序比较 Base、SFT、GRPO loss、DAPO token-level loss、Clip-Higher 和截断
样本屏蔽：

```bash
python scripts/run_experiments.py --action all --group core
```

学习率和采样温度采用单因素实验，避免直接运行昂贵的笛卡尔积：

```bash
python scripts/run_experiments.py --action all --group sft_lr
python scripts/run_experiments.py --action all --group rl_lr
python scripts/run_experiments.py --action all --group rl_temperature
```

普通调参默认评测 `validation.jsonl`。选定最终候选后，才显式运行官方 dev：

```bash
python scripts/run_experiments.py \
  --action eval \
  --eval-split final \
  --experiment base \
  --experiment sft_default \
  --experiment rl_dapo_components
```

这里的 `rl_dapo_components` 只表示采用 token-level DAPO loss、Clip-Higher 和截断样本
屏蔽，不表示复现完整 DAPO Dynamic Sampling。

每个实验会保存解析后的完整配置、训练日志、checkpoint、逐样本预测和完成标记；配置没
变化时会自动跳过已完成步骤。汇总表和配对 McNemar 结果写到
`outputs/experiments/sql_reasoner_cn_v1/`。`--action` 不是 `list` 时必须显式指定
`--group` 或 `--experiment`，防止误启动全部训练。

完整实验不需要手工逐阶段选择。实验设计见
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)，先 dry-run 检查，再一键启动：

```bash
make campaign-plan
make campaign
```

编排脚本会按 validation 自动选择最佳 SFT、算法组件、RL 超参数和奖励方案，然后用两个额外
随机种子复验，最后才评测官方 dev、私有集和鲁棒性集。中断后重复 `make campaign` 会跳过
配置指纹一致的完成项；若配置指纹改变则拒绝误接旧 checkpoint。最终报告写入
`outputs/experiments/sql_reasoner_cn_v1/campaign/final_report.md`。

## 评测

统一评测：

```bash
make evaluate
```

每个 checkpoint 生成 `summary.json` 和 `predictions.jsonl`，随后产生 `outputs/comparison.json` 和 Markdown 对比表。`ast_exact_match` 是本项目的 sqlglot 规范化指标，不冒充官方 CSpider EM。需要官方 EM 时显式传入原仓库 evaluator：

```bash
python scripts/evaluate.py \
  --dataset data/processed/cspider/official_dev.jsonl \
  --adapter checkpoints/sft \
  --output-dir outputs/sft \
  --official-evaluator /path/to/chisp/evaluation.py \
  --official-python python2 \
  --official-gold /path/to/dev_gold.sql \
  --official-db-dir data/raw/CSpider/database \
  --official-tables data/raw/CSpider/tables.json
```

CSpider 原仓库的 evaluator 是 Python 2 代码；若使用已经迁移到 Python 3 的兼容版本，将 `--official-python` 改为对应解释器。

C-Eval 仅作为灾难性遗忘的轻量监控，不参与模型选择：

```bash
python scripts/evaluate_ceval.py --adapter checkpoints/sft --output outputs/ceval-sft.json
```

## Demo

```bash
make demo
```

页面只允许选择预置数据库，不能上传任意数据库；SQL 仍通过只读沙箱执行。Base/SFT/GRPO adapter 按需加载，同一时刻只缓存一个 checkpoint，以适应 24GB 显存。

## 一条命令流水线

完成依赖和 CSpider 下载后：

```bash
make pipeline CSPIDER_DIR=/absolute/path/to/CSpider
```

该命令依次执行数据准备、私有集生成、SFT、GRPO 数据筛选、GRPO 和三组评测。建议首次运行时仍按阶段执行，以便先验证 20-step GRPO 的 ETA。

## 公平性和安全边界

- validation 只来自训练集中的独立数据库；官方 dev 不参与调参。
- 私有零售集和鲁棒性集不参与 SFT/GRPO。
- 模型生成的 SQL 只能是单条 `SELECT/WITH`，SQLite authorizer 会再次拒绝写操作。
- 执行奖励以 gold SQL 的结果为目标；无 `ORDER BY` 比较行多重集合，有 `ORDER BY` 比较有序结果。
- 结果超过 10,000 行、执行超时或任一查询报错时，不授予执行正确奖励。

