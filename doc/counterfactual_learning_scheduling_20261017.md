# 开发对照的并行执行补充

日期：2026-10-17。此补充只改变操作调度，不改变已冻结的统计方法、数据、模型、超参数、步数、样本角色或准入条件。原实验见[预注册](ppo_counterfactual_learning_protocol_20261017.md)，生成和训练的13份输入保持原字节。

原worker会顺序生成teacher、拟合Ridge，再执行两方法。PPO-only并不依赖teacher标签，因此提前在独立进程调用**同一冻结runner的 `trial` 函数**，只执行已登记的 `screen_2023_ppo_only_42` / `screen_2024_ppo_only_42`，每个仍为32,768步、相同配置和两个固定快照。没有额外方法、种子、预算、模型或标签；不提前根据该对照结果修改预热方案。

当前机器只读检查为16个逻辑处理器、约32GB物理内存、约8GB可用；两对照进程各使用原CPU单线程设置。它们与原两teacher worker并行，不中断、替换或重启当前采集。

两个入口写入原登记的唯一case目录。原worker到达PPO-only时若该case已完整封口，则按既有哈希和配置守卫只读复用，不增加一次RL更新。如果尚未封口，原runner会明确拒绝partial，不静默重训或续接；只有取得早期对照进程实际exit0及完成清单后，才允许显式重新进入原worker，它只读复用已完成teacher/fit/control并开始尚未执行的预热臂。此类调度重入若发生须记录退出回执和额外成本，不因观察超时重启任何活进程。

新增 `tools/run_counterfactual_control.py` 是调度入口，先后核对原冻结身份，仅允许上述两开发对照；拒绝2025/2026或其他seed/budget。所有方法和八个开发快照完整封口前不做筛选，确认仍只能执行唯一已选方法/预算的冻结矩阵，不能按提前看到的对照结果换方案。

```powershell
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.run_counterfactual_control --fold 2023
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.run_counterfactual_control --fold 2024
```

本文件记录操作安排，不预写进程已成功、实际训练步数或收益；实际退出与复用成本在最终报告列明。

## 对照完成后复用空闲槽

某fold的提前PPO-only取得实际exit0及完整case封口后，可将这个CPU槽交给该fold已登记的 `inner_holdout` teacher片段。`tools/run_counterfactual_holdout.py` 先验证控制case已完整32K，再调用同一个冻结 `teacher_segment`，使用同一个原目录、480状态、两遍重演、固定scaler/行为流及48,000 Agent+同量shadow预算。它不拟合模型或执行PPO，不新增标签或使用保护期。

```powershell
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.run_counterfactual_holdout --fold 2023
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.run_counterfactual_holdout --fold 2024
```

这一并行片段若比原worker更晚完成，原worker会在已封口inner-train之后、进入正在执行的inner-holdout前明确拒绝partial并退出，尚未执行该片段的任何步。保持其实际退出回执；不得因此重启仍活着的holdout进程。待独立holdout实际exit0并封口后，再显式重入原worker，它只读复用两个完整teacher、拟合唯一Ridge、只读复用控制case、执行唯一预热PPO。若片段已由原worker启动而未封口，新入口同样拒绝，不重复采集。整个调度始终最多四个正式计算进程，case和片段各一个实际执行者；最终按全部唯一封口及失败证据核对成本，而非把重入次数算成独立经历。
