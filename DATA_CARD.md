# Data Card

## CSpider

主训练与公开评测数据。下载、许可和引用以 CSpider 官方发布为准，仓库不重新分发原始数据。预处理按 `db_id` 保证训练和内部验证数据库互斥；官方 dev 单独保留。转换失败样本不会静默丢弃，而是写入 `rejected.jsonl`。

## Retail-CN Private Benchmark

由 `scripts/create_retail_benchmark.py` 以固定种子创建，包含 7 张表、50 个查询问题和 30 个鲁棒性改写。覆盖聚合、Join、日期、Top-K、HAVING、CTE、相关子查询和反连接。它用于检查公开基准污染及中文表达鲁棒性，不用于训练或模型选择。

这是程序生成的小型诊断集，不代表真实商业流量，也不能据此宣称生产可用。

## 隐私与安全

数据均为公开基准或合成记录，不包含真实个人信息。数据库执行层仅开放只读查询，并设置 authorizer、超时及返回行数上限。

