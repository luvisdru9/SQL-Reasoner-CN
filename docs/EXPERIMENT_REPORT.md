# Experiment Report

> 训练完成后由 `outputs/` 的真实结果填写，禁止预先填写目标数值。

## Environment

- Commit:
- GPU / topology:
- Driver / CUDA / PyTorch:
- Wall time and peak VRAM:

## Data

- Train / validation / official dev counts:
- Rejected samples and reasons:
- GRPO variance-positive ratio:

## Results

粘贴 `outputs/comparison.md`，再补充 CSpider 官方 EM、私有集、鲁棒性集和 C-Eval。

## Error analysis

从 `predictions.jsonl` 按 error_type 抽取至少 30 条，分别分析 schema、Join、聚合、子查询和值映射错误。

## Conclusion

回答三个问题：SFT 是否有效；RL 是否在 hard/extra-hard 上增益且整体无明显退化；增益是否通过配对检验。结论与指标不一致时以指标为准。

