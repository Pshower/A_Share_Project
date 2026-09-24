# 项目文档索引

更新日期：2026-09-25

建议先读 [项目 Pipeline](../pipeline.md)，再看 [实施进度与验收记录](implementation_plan.md)。当前阶段四已完成框架，无训练验收不等于模型训练或策略效果验证。

## 文档分工

| 文档 | 负责内容 | 何时阅读 |
| --- | --- | --- |
| [pipeline.md](../pipeline.md) | 当前总体架构、数据流、关键边界与未来候选方向 | 了解项目在做什么 |
| [implementation_plan.md](implementation_plan.md) | 唯一的阶段进度总表、任务勾选、验收证据和下一步 | 查看已完成与未完成事项 |
| [research_workflow.md](research_workflow.md) | 数据产物、预处理和基线回测运行方式 | 重建本地数据或复现基线 |
| [trading_environment.md](trading_environment.md) | 环境输入、动作、账户、成交、事件及约束契约 | 接入策略或修改记账逻辑 |
| [ppo_implementation_plan.md](ppo_implementation_plan.md) | PPO 智能体输入与行为、动作映射、共享网络、扩池和训练设计 | 理解 PPO 为什么这样实现 |
| [ppo_training_framework.md](ppo_training_framework.md) | 实际依赖、检查/未来训练命令、产物和操作限制 | 运行与排查 PPO 框架 |

## 统一维护规则

- 总体阶段只使用实施进度中的一至五，不再混用早期模型演进或平台化里程碑编号。
- `[x]` 只表示对应条目已有验收证据；“已实现、待训练验收”不表示已经运行训练，“待数据”不表示接口未实现。
- 阶段状态、测试数量和验收报告索引集中维护在实施进度。设计文档解释契约，运行文档解释命令，不重复维护另一份任务清单。
- 配置值以 `configs/research.json`、`configs/ppo.json` 为准，文档中的值为当前快照。改变配置或行为时同步更新对应文档及验收记录。
- 原始数据、清洗产物、初始化模型和回测报告保留在本地，不纳入 Git。文档中的 `reports/runs/...` 是本地验收证据路径，其他机器须自行按运行说明复现。
- 历史验收数字标记所属阶段；不能把阶段二、三当时的测试数量误当成当前总数。

## 运行入口

所有命令从仓库根目录、Conda `Graduate` 执行。依赖安装见 [研究流程](research_workflow.md) 和 [PPO 运行说明](ppo_training_framework.md)。

```powershell
# 只运行离线测试，默认排除联网测试
python -B -m pytest -q -p no:cacheprovider

# 复现验证集基线，不训练 PPO
python -B -m src.backtest.run --config configs/research.json

# 检查 PPO 接口、保存加载与参数不变，不训练
python -B -m src.training.train --config configs/ppo.json --check-only
```

训练必须另行明确启动，当前没有已训练 PPO 模型或最终测试集绩效。无论模型是否训练，当前 `hfq_research` 结果都不能等同于现实交易收益。
