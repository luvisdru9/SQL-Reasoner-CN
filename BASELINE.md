# 实验基线记录

本文件只记录真实运行结果。当前环境无法连接 NVIDIA 驱动，尚未运行 Qwen3-4B 的 Base/SFT/GRPO 实验，因此不填入推测分数。

## 固定实验矩阵

| Checkpoint | CSpider Dev EX | AST EM | Valid SQL | Schema F1 | Hard+ EX | 私有集 EX | 鲁棒集 EX | C-Eval | P50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| QLoRA-SFT | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| SFT+GRPO | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

## 已验证的非模型部分

- 单元测试覆盖格式解析、只读拒绝、多语句拒绝、结果顺序语义、奖励满分路径和统计函数。
- 固定种子零售数据生成器产生 50 条测试题与 30 条鲁棒性改写。
- 用 gold completion 回放 50 条零售题时，Execution Accuracy、AST EM、格式率、Valid SQL 和 Schema F1 均为 1.0；该结果仅用于验证 evaluator，不是模型效果。

## 每次实验必须记录

- Git commit、Python/CUDA/驱动及依赖版本；
- 两张 GPU 的型号、峰值显存和卡间拓扑；
- 数据统计、随机种子、训练 steps、wall time 和 tokens/s；
- 完整配置、逐样本预测、bootstrap CI 与 McNemar 结果；
- 是否使用官方 CSpider evaluator，禁止把 AST EM 标记为官方 EM。

