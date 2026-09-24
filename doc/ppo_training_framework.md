# PPO 训练框架：实现与运行说明

更新日期：2026-09-25

## 当前状态

阶段四的框架代码和无训练离线验收已完成，尚未启动 PPO 参数更新、短训练或正式训练，未评估真实数据测试集。不能据此判断策略收益或泛化效果。

规划文档已先行提交：`66dbe26`，`docs: specify PPO agent and training framework plan`。本文负责实际运行方式；智能体行为与网络原理见 [PPO 设计](ppo_implementation_plan.md)，任务状态和验收证据统一见 [实施进度](implementation_plan.md)，其余文档见 [文档索引](README.md)。

## 已实现的调用链

```text
ResearchData：校验本地产物哈希、scaler、日期边界与特征顺序
    |
训练日期切片 -> MarketDataProvider -> TradingEnv
    |                                  |
    |                          LatentActionWrapper
    |                                  |
    |                           Monitor / DummyVecEnv
    |                                  |
    +-------------------------- SB3 PPO + SharedStockPolicy
                                       |
                      共享股票编码器、评分头、现金头和价值头
                                       |
                              N+1 维潜在连续动作
                                       |
                           掩码 + softmax + 单票上限
                                       |
                              N 维股票目标权重
                                       |
                      已有 broker：买入/卖出、费用、T+1、整手
```

网络输入为标准化相对特征、持仓/现金比例、应收比例和有效/买卖掩码。当前数据的输入形状为 `[298, 1, 20]`；股票数由数据或 `stock_codes` 配置决定，不硬编码进网络。

动作是目标权重，不是直接买卖标签。目标高于当前持仓时尝试买入，低于当前持仓时尝试卖出，目标为零时尝试清仓；无法成交的订单保留拒绝原因。首版没有独立的“不下单”动作。

PPO 的 rollout 保留原始潜在动作和对应 log probability；环境只执行映射后的权重，不把权重当成高斯动作计算概率。验证、推理和训练包装器共用同一映射。

## 环境与依赖

实际验收环境：Conda `Graduate`，Python 3.11，PyTorch 2.7.1，Stable-Baselines3 2.7.1，Gymnasium 1.2.3。默认 CPU、单进程、单 PyTorch 线程。

新增依赖清单为 `requirements-ppo.txt`，复用现有数据和回测依赖。在其他环境复现时，从仓库根目录执行：

```powershell
conda activate Graduate
python -m pip install -r requirements-ppo.txt
```

安装依赖会访问软件包源；本次没有下载行情，也没有执行联网测试。

环境检查 `python -m pip check` 发现现有 `shap 0.42.1` 要求 `slicer==0.0.7`，而环境已安装 `slicer 0.0.8`。本次安装未修改这两个包，PPO 测试未使用 SHAP，未擅自调整这一独立依赖冲突。

## 只检查，不训练

默认模式就是检查模式，也可显式指定：

```powershell
python -B -m src.training.train --config configs/ppo.json --check-only
python -B -m pytest -q -p no:cacheprovider
```

检查过程校验现有数据、初始化模型、在训练日期内执行一次无梯度推理及环境步进，并验证模型保存加载一致性、参数不变和 `num_timesteps=0`。不调用 `learn()`，不做验证集选模或真实测试集评估。

每次检查生成 `check.json` 和 `initialized_model/`。报告记录观测/动作维度、参数量、训练区间、保存加载一致性和参数是否改变；当前本地报告路径和实测结果见 [实施进度的最新验收快照](implementation_plan.md#最新验收快照)。

初始化模型仅用于连通性检查，独立评估 CLI 会拒绝把它当成已训练的候选模型。

## 未来如何启动训练

只有同时提供 `--train` 和正整数 `--total-timesteps` 才会进入 PPO 更新。单独提供步数、遗漏训练预算、负数预算及同时提供检查/训练开关都会报错。

下列命令只是未来短训练的示例，本次没有执行：

```powershell
python -B -m src.training.train --config configs/ppo.json --train --total-timesteps 512
```

SB3 按完整 rollout 执行，实际步数可能向上补足至 `n_steps` 的整数倍。运行目录保存请求预算及这一说明，示例预算不代表足够的模型训练量。

当前入口从新模型开始训练，没有提供续训 CLI。模型文件包含框架保存的优化器状态，但尚未验收账户、随机状态和 rollout 的精确恢复，不承诺逐步一致续训。

## 验证与产物

- 训练环境仅包含训练截止日及以前的数据；验证环境只到验证截止日，并保留此前历史预热。
- 每 10 个 rollout 更新完成后，在独立验证账户做确定性评估；最后一次更新完成后补做评估。
- 按预先约定的验证 Sharpe 选模，同分比较累计收益；无定义 Sharpe 不参与最优模型选择。
- 选出的模型与现金、买入持有、定期等权、定期动量及每日等权/动量对照使用同一账户审计与指标计算。无合法选模结果时，报告明确使用最终模型且未选出最优模型。
- 不启动测试集，也不自动获取指数。PPO 报告目前不包含外部指数比较。

每次运行独立目录，已有目录拒绝覆盖。未来训练输出包含：

```text
config.json / research_config.json / budget.json
logs/progress.csv
validation_<step>/metrics.json、daily.csv、positions.parquet、orders.csv、report.md
best_<step>/model.zip、metadata.json
selection.json
final_model/model.zip、metadata.json
selected_comparison/ 或 final_comparison_no_valid_selection/
```

各策略的每日、持仓、订单文件分别位于报告目录下的策略子目录。元数据记录数据/scaler 哈希、股票和特征顺序、网络与映射版本、价格/成本口径、依赖、源码哈希、Git 状态及模型来源。

未来可对已训练且契约匹配的模型单独运行验证：

```powershell
python -B -m src.training.evaluate --config configs/ppo.json --model reports/runs/ppo/RUN/best_STEP --output reports/runs/ppo/validation_review
```

`RUN`、`STEP` 为未来实际产物标识，当前没有对应的已训练模型。

## 扩大股票池

网络共享股票编码器、评分头和股票探索方差，现金头及价值头输入为固定维度的池化上下文。扩池不增加股票专属参数。

同一训练实例仍固定观测和动作空间。已有 `PPOAgent.transfer_to(env, config, new_contract)` 接口：创建匹配新股票数的实例、严格迁移共享参数、重置优化器与 rollout，并记录来源。普通 `load()` 不接受股票池契约变化。

迁移必须沿用相同特征、窗口、scaler、价格和执行口径。当前数据仅有 298 只股票，代码验收用小型合成股票池验证迁移，没有生成 500 只股票的数据，也没有验证扩池后的收益。跨股票留出实验仍需另建训练股票专用 scaler；共享参数不等于泛化有效。

## 验收与未完成项

最新测试总数、本地检查和基线回归结果统一维护在 [实施进度](implementation_plan.md)。保留原有 Gymnasium 无限特征空间边界提示，实际观测检查有限值。`pip check` 的独立 SHAP 依赖冲突如前述，不能宣称整个 Graduate 环境没有依赖冲突。

离线测试在 PPO 用例中禁止 `learn()`、`backward()` 和 Adam 参数更新。合成行情的 rollout 仅用于检查潜在动作概率与终止/bootstrap 接口，不执行 PPO 优化，不作为策略训练结果。

覆盖动作约束与买卖执行、冻结持仓、股票置换、参数量不依赖股票数、保存加载和显式迁移、未来数据隔离、元数据边界防护、原始潜在动作概率一致性、截断终止观测与 bootstrap、检查 CLI 防误训练，以及共享基线回测。

后续仍需明确启动并验收：短训练的实际梯度更新与数值稳定性、正式训练、模型选择结果、冻结配置后的样本外测试及跨股票泛化。现有 HFQ 研究报价、固定股票名单、交易状态和公司行为数据缺口继续保留，不能用框架验收替代策略有效性验证。
