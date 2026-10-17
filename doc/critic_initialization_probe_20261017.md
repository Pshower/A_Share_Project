# Critic 初始化与梯度探针工程准备

日期：2026-10-17。状态：**独立工具及合成验收完成，未开展新的金融研究**。当前因果动作价值预热研究继续按[原协议](ppo_counterfactual_learning_protocol_20261017.md)执行；本工具没有被该研究或两个提前调度入口导入，不改变其初始化、梯度、数据、预算或筛选。

## 动机与边界

[已完成 A 对照诊断](reports/ppo_counterfactual_learning_20261017_control_diagnostic.md)发现早期 bootstrap 的尺度值得受控检验，但没有证明它导致退化。当前 B 和整轮尚未结束，不能提前换方法。本文件只是准备检验假设所需的工程证据，不是新一轮预注册、收益结论或正式训练启动许可。

## 工具

实现：[critic_initialization_probe.py](../tools/critic_initialization_probe.py)。

- `initialize_critic_head(model, gain)`：只接受普通 12,805 参数 CalendarTimingPolicy 的 fresh PPO。默认构造后将最终 `value_net` 的 weight/bias 一次性乘 `1`、`.01` 或 `0`，不重新随机初始化、不持续缩放 forward。检查 RL/update 时钟为0、buffer/Adam为空、没有梯度及重复初始化标记；保存全部8个参数 tensor 的前后哈希。
- 非价值输出层的 Actor 和 Critic 隐藏层不变；缩放不消费随机数，也不新增 forward/backward。fresh 检查是原生状态守卫，不是对任意手工篡改对象的密码学出生证明。
- `ClipGradientProbe(model)`：在单进程、单训练线程的 context 中临时观察原 `torch.nn.utils.clip_grad_norm_`。按原参数顺序和原参数调用原实现一次，并原样返回其结果；不复制 PPO 的训练算法。
- 每次真实裁剪记录 Actor/Critic/global L2 范数、原返回范数、原 norm_type/max_norm/foreach、系数和调用序号。不同 norm_type 仍透传；分组观测量始终明确为 L2，不混淆成原函数使用的范数。
- 每个 `model.num_timesteps` rollout 边界仅保留首次实际裁剪的完整前后梯度；其余保留标量。裁剪调用计数不冒充 Adam 更新数，后续 runner 仍须从原生 optimizer 取得实际步数。
- 异常退出恢复原钩子；无关参数的裁剪仅透传并单独计数，不冒充本策略证据。钩子是进程全局的，不支持同进程并发训练线程；正式接入还需检查无 foreign call、完整保存证据及输入封口。

## 合成验收

运行：

```powershell
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -m pytest tests/test_critic_initialization_probe.py tests/test_timing_policy.py -q
```

root 最终 session25703、退出回执 `f0835c`、exit0，**58 passed / 24.20s**。首轮54项也通过，但不与最终58项累加。作者端测试未取得可复核的终态回执，故不作为最终验收来源。

合成环境无金融文件。四条各1,024步的真实 SB3 训练分别为普通 gain1、带探针 gain1、带探针 gain.01、带探针 gain0：

1. 普通与探针 gain1 的最终模型、Adam、Python/NumPy/Torch RNG、实际采集记录和原8-key rollout 均逐项相同。
2. 三种 gain 的初始 Actor 和 Critic 隐藏层相同；首512次采集的输入、采样动作、原始奖励、logP和done相同，价值预测/GAE可不同。这不是要求 timeout 修正后的 buffer reward 也相同。
3. gain0 首次真实裁剪前 Critic 隐藏层梯度为0，后续出现非零梯度且隐藏参数更新；8个参数均有实际 Adam 状态。零输出并不意味着 Critic 永久不训练。
4. 拒绝非 fresh/重复/未登记 gain；原裁剪的非有限错误、非L2及位置参数保持透传，context 正常和异常退出均恢复钩子。

这些是 CPU 合成工程证据，不是新增市场经历、金融学习增量、GPU验证或真实成交验收。

## 后续条件

本轮完整封口后，若仍需检验此假设，可另登记唯一因素 `{1, .01, 0}` 的同 Actor/数据/预算对照。不能同时改奖励、GAE、动作、网络容量或全局 clipping；不能将诊断中的片段内累计 M 当作缺失未来的替代回报目标。

真正启动前仍需新 runner、冻结输入/预算、默认臂精确复现、完整梯度与 Adam 归档、开发筛选及多种子时间确认协议。原 M20 强基线、正 PPO 增量和最终风险门槛不降低；bootstrap 更小或解释方差改变本身不构成策略成功。
