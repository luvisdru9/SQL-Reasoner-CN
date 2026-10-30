# SQL-Reasoner-CN 实验方案

## 目标与选择原则

实验回答三个问题：SFT 是否优于 Base；可验证奖励 RL 是否在 SFT 上继续提升；提升来自
哪一项算法或奖励设计。所有模型选择只看按数据库隔离的 validation，官方 CSpider dev、
私有零售集和鲁棒性集只在配置冻结后评测一次。

主选择指标是 Execution Accuracy。完全相同时，依次使用 hard/extra-hard 的加权执行准确率、
Valid SQL Rate、Schema Linking F1、格式合规率和 AST EM 打破平局。官方 dev 不参与任何
超参数选择。

## 固定资源控制变量

先用 `make stability` 验证资源配置，正式效果实验中不改变下列参数：

- SFT：双卡 DDP、每卡 batch 1、梯度累积 8、最大长度 2048、LoRA r=32、4-bit QLoRA；
- RL：GPU 0 BF16 LoRA 训练、GPU 1 vLLM rollout、batch 2、梯度累积 2、每题 4 个候选、
  prompt/completion 上限 1664/384、最多 200 steps、vLLM 显存利用率 0.72；
- 模型、数据切分、prompt、解码评测参数、随机种子和评测脚本在同类对比中保持一致。

这些配置写在 `configs/experiments.yaml` 的 resource profile 中，实验只能修改 tunable 白名单，
避免把“显存配置变化”混入算法效果比较。

## 分阶段矩阵

| 阶段 | 候选 | 只改变什么 | 回答的问题 |
|---|---|---|---|
| Base | 原始 Qwen3-4B | 无训练 | 零样本基线 |
| SFT 学习率 | 1e-4、2e-4、3e-4 | SFT learning rate | 冷启动是否有效，哪一档更合适 |
| RL 算法增量消融 | GRPO；+ DAPO token loss；+ Clip-Higher；+ 截断屏蔽 | 单项累计加入 | 各 DAPO 子技术是否贡献提升 |
| RL 超参 | LR 5e-6/1e-5/2e-5；温度 0.6/0.8/1.0 | 围绕算法胜者做单因素搜索 | 更新幅度和探索强度是否合适 |
| 奖励消融 | 完整塑形、仅执行奖励、去掉 plan 一致性 | 奖励分量 | 中间奖励是否缓解稀疏性，plan 奖励是否有价值 |
| 随机种子复验 | 42、7、123 | 仅 RL seed | 提升是否只是单次随机波动 |

这不是全参数笛卡尔积：每阶段先在 validation 上冻结胜者，再进入下一阶段。随机种子复验
只报告均值、标准差和范围，不从三个 seed 中挑最好 checkpoint。

## 最终评测与判定

冻结 Base、最佳 SFT、最佳 RL 后，统一评测：

- CSpider 官方 dev：Execution Accuracy、AST EM、Valid SQL、格式率、Schema F1、难度分层；
- 私有零售集：降低公开数据污染对结论的影响；
- 中文鲁棒性集：同义改写、口语省略和字段别名；
- bootstrap 95% CI、相邻 checkpoint 的配对 McNemar 检验、30 条失败案例；
- 峰值显存、吞吐、P50 延迟与 adapter 大小。

只有最佳 RL 在 validation 上的 Execution Accuracy 严格高于最佳 SFT，Demo 才默认使用 RL；
否则默认使用 SFT，并把 RL 作为负结果保留。C-Eval 是可选的灾难性遗忘检查，不参与选模。

## 一键运行

先检查将执行的命令：

```bash
make campaign-plan
```

完整实验与最终评测：

```bash
make campaign
```

中断后重复同一命令会根据配置指纹与完成标记跳过已完成阶段，并从合法 checkpoint 恢复。
结果入口为 `outputs/experiments/sql_reasoner_cn_v1/campaign/final_report.md`。若只需算法主链，
可运行 `python scripts/run_campaign.py --preset core`；若要额外执行 C-Eval，加入
`--include-ceval`。

