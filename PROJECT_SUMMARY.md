# SQL-Reasoner-CN：面试项目一页说明

## 问题

通用模型能生成看似合理但字段错误、Join 错误或不可执行的 SQL。中文业务问题还要跨语言关联英文表字段。项目目标是在两张 24GB 消费级显卡上，验证参数高效 SFT 与可验证奖励强化学习能否改善这一问题。

## 方法

使用 Qwen3-4B-Instruct-2507。第一阶段用 4-bit QLoRA 学习结构化 `<plan>` 和 SQL；规划由 gold SQL AST 离线生成，不调用教师 API。第二阶段在中高难度样本上进行 GRPO，以格式、安全、schema、plan-SQL 一致性和执行结果组成奖励，并采用 DAPO token-level loss、0.20/0.28 非对称 clipping 和截断样本屏蔽。

## 为什么不是简单微调展示

- CSpider 按数据库隔离，官方 dev 不参与调参；另有训练后新建的零售测试库。
- Base、SFT、GRPO 的 prompt 和解码固定，保存逐样本输出。
- 除总体执行准确率外，按难度评测并报告 CI、配对检验、错误类型、显存和延迟。
- RL 失败也是结果：若 hard 子集没有提高，分析奖励稀疏、同组候选同质化和训练预算，不隐藏失败样本。

## 面试演示顺序

1. 在同一个问题上切换 Base/SFT/GRPO，观察规划、SQL 和执行结果。
2. 展示一条写 SQL 或多语句被 AST 与 SQLite authorizer 双重拒绝。
3. 展示三阶段统一指标、hard/extra-hard 分层和 McNemar 检验。
4. 展示一个 RL 改对案例和一个奖励投机/仍失败案例。
5. 解释为何双 3090 Ti 选择 4B QLoRA、单卡训练加单卡 rollout，而不是 14B 全参训练。

