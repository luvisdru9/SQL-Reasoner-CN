# Model Card Template

训练结束后，为 SFT 和 GRPO adapter 分别复制并填写本模板。

## Base model

`Qwen/Qwen3-4B-Instruct-2507`，许可遵循上游模型卡。

## Training

- 方法：4-bit QLoRA-SFT / QLoRA-GRPO；
- 数据版本与数量：待真实运行后填写；
- GPU、wall time、峰值显存：待填写；
- 完整配置：`configs/sft.yaml` 或 `configs/grpo.yaml`；
- 随机种子：42。

## Evaluation

必须链接对应的 `summary.json`、`predictions.jsonl` 和 Base/SFT/GRPO 对比文件。不得只记录最佳案例。若 RL 未显著优于 SFT，需要在此明确说明。

## Intended use

研究和面试演示用途的中文 SQLite Text-to-SQL adapter。生成 SQL 必须经人工检查或只读沙箱执行。

## Limitations

- 主要训练数据的 schema 名多为英文；
- 只针对 SQLite 验证，未覆盖生产数据库方言和权限系统；
- 执行等价不总等于语义等价，空结果等场景可能产生偶然匹配；
- 结构化 plan 是可核验查询规划，不应解释为模型内部真实思维过程。

