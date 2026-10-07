# 市场摘要与Critic梯度归因预注册

日期：2026-10-07。状态：设计预注册，尚未运行本轮真实诊断、训练或回测。工程验收后再冻结机器协议及全部实际源码哈希。本文不修改上一轮协议、80份已冻结源码、数据或结果，不把方案视为完成证据。

## 研究问题

上一轮2025历史折中，市场/账户分离编码的long配置平均净ROI为+2.27838%，自身初始化增量约+1.43435个百分点，通过当轮研究门槛；固定展示seed42的单模型净ROI为-1.51906%。配置平均不等于代表模型收益，研究候选也不等于可实盘默认。

分离结构同时改变市场摘要、网络图和初始化函数，不能从上一轮结果断言“纯市场摘要就是改善原因”。本轮固定同一个125,063参数的Split结构，分别改变市场摘要路由、Critic损失是否反传到共享编码器，做2×2对照，检验两项机制及交互作用。

此次只比较预定机制，不新增先验、特征、数据或超参数搜索。不在看到收益后减少cell、改变预算或追加只对赢家有利的实验。

## 数据边界

- 只复用 `reports/runs/ppo_exploration_20261006/folds/2023/research.json` 与 `folds/2024/research.json` 及其已派生数据，来源为现有新浪300股快照、原20列特征。每折scaler已仅用该折3月31日以前的有效训练行拟合，不重拟合、不下载。
- 2023折训练截止2023-03-31，研究验证为2023年4月至6月；2024折训练截止2024-03-31，研究验证为2024年4月至6月。验证初态使用截止日以前最后一个真实报价日收盘，不能为周末截止日虚构行情。每折实际日期及交易步数单独报告。
- 加载provider前物理截断到相应train/val边界，训练环境不能看见验证价格或特征；2026-07-01及以后绩效继续锁住，不训练、不评估、不选模。
- 两个历史折之前已有信号诊断，且固定2026股票名单本身带有选择/存活偏差，不能称真盲测。2025旧研究只作背景，不重跑、不与其他年份的ROI伪装成配对增益。
- 仍使用HFQ研究价格与既有模拟账本，真实未复权成交、公司行为和历史交易状态缺失的限制不变，不能因工程交易规则通过就宣称实盘准确。

## 同构2×2设计

新策略类拟名 `AttributionTradePolicy`，固定 `architecture="split"`，仅增加 `market_context` 与 `critic_features` 两个显式设置。

| cell | `market_context` | `critic_features` | 干预 |
| --- | --- | --- | --- |
| pure_shared | pure | shared | 纯行情市场摘要；value loss允许反传共享编码器 |
| pure_detached | pure | detached | 纯行情摘要；仅value head前的context断开梯度 |
| fused_shared | fused | shared | 混合行情/账户市场摘要；value loss允许反传共享编码器 |
| fused_detached | fused | detached | 混合摘要；仅value head前的context断开梯度 |

对每股设 `m=market_encoder(stock_features)`、`a=account_encoder(account_fields)`、`z=tanh(m+a)`，共同数据有效mask为valid：

```text
pure market  = sum(valid * m) / max(sum(valid), 1)
fused market = sum(valid * z) / max(sum(valid), 1)
held         = sum(current_weight * z)
context      = [market, held, cash, receivables, allocation_context]
actor        = stock heads([z, context])
shared value = value_head(context)
detached value = value_head(context.detach())
```

个股融合z、held摘要、方向/size头、value head、现金剩余语义、合法动作支持与执行器全部不变。无有效股票时market为有限零摘要；无效但仍持有的资产保留在held中。

`detach`不等于增加独立Critic编码器，不减少模型参数，不关闭Critic训练，也不改变奖励、GAE、PPO概率、执行规则或Actor反向传播。value head照常学习；共享编码器仍由Actor损失更新。Critic输入会随Actor编码器学习而变化，因此它仍有表示漂移，不是固定特征Critic。

value head与Actor仍由同一个PPO优化器更新，并共同参与global max_grad_norm=.5。即使Critic到编码器已detach，Critic head梯度仍可通过全局裁剪影响Actor有效步长，不能把该轴解释为移除Actor/Critic全部耦合或独立优化器实验。

## 初始化与可比性

固定L1、F20、hidden256/128、125,063参数，同构造顺序、同名参数及同seed随机初始化。每个fold/seed下四cell保存仅含同序 `state_dict` 的同basename `policy.pth`，逐tensor字节与文件SHA均须严格相同；初始优化器为空。不能保存含variant属性的整个Policy对象后要求其pickle相同，也不能在hash失败后悄悄放宽为“近似同参数”。

- Critic轴：同route下shared/detached的初始Actor、Critic前向函数完全相同，区别只在value loss的梯度传播。
- Route轴：即使参数字节相同，pure/fused的初始前向函数通常不同，初始动作、value、交易轨迹和收益都可能不同。必须记录这种差异，不能声称四cell“初始策略相同”。
- 同seed的训练起点、初始账户mode/逐episode随机流、K条件及共同5日数据有效mask独立控制。所有cell使用相同报价和执行约束。
- 配对约束是相同重置初态和外生日期/mode/K流。策略分歧后账户、现金、T+1锁定、legal_directions、lot_bounds和成交结果内生变化，不要求每步交易支持或后续持仓相同。

机器协议须绑定每折数据/scaler/research配置、初始参数证明、变体字段、执行配置与新增源码；旧80份源码的字节身份继续逐项核验。源码字节一致不等于Git提交号相同，模型元数据和预登记Git身份分别如实记录。

## 固定训练预算

| 项目 | 预定设置 |
| --- | --- |
| 训练组合 | 4cell × 2fold × seeds42/123/2026，共24个独立训练run |
| 每run | 4,096实际环境训练步；2,048步更新后保存共享前缀 |
| 总实际RL步 | 24 × 4,096 = 98,304 |
| 快照 | 48份；快照步数和147,456不是实际训练量，2,048前缀共49,152步不额外计费 |
| Adam上限 | 无KL提前停止时每run160次、共3,840次；2,048快照80次是前缀，实际更新另行核验 |
| PPO | n_steps512，batch128，epochs5，LR1e-5，gamma.99，GAE.95，clip.2，targetKL.03，ent0，vf.5，gradclip.5 |
| 初始化 | actor末层gain.1、bias0、initial_log_std=-.7；无特征/BUY/HOLD先验，不做size退火 |
| 经历 | mixed初始账户，256步episode，recent_every4；每run16个实际episode |
| 随机流 | 日期seed XOR0x510A，mode seed XOR0xA110，账户SeedSequence(seed,0xACCD,episode)，K seed XOR0x7A43 |
| 条件/评估 | 训练K10/20/40；本轮验证仅主K20，不重新跑辅助K |
| 机器 | conda AShareGPU，CUDA，torch_threads1；不在途中切换设备或解释为硬件对比 |

账户仍为100万元，commission.0003/min5、卖税.0005、slip.001、volume_fraction.01、lot100、单票上限.05、单步BUY/REDUCE名义额上限.05、max_commission_ratio.001。低额/余股例外仅限实际完整EXIT，现金为股票交易及费用后的剩余，不新增现金动作。

所有24个run完整执行。本轮没有额外PPO pilot，不按中间ROI选跑、扩预算、继承旧模型或提前淘汰。只允许因数据泄漏、非法概率支持、NaN、参数/缓存身份或账本不变量失败停止，并封存失败与修订记录。

## 控制组与评估

每个fold/seed/route设置自身未训练初始化控制，shared与detached仅在已验证初始函数完全相同后复用同route控制；两个route不得共享收益作为同一初始化。共12组route/seed/fold初始化控制，各有cash/equal/concentrated/random四账户。

复用时明确缓存的canonical variant、初始参数SHA、route、fold、seed和数据/执行身份及函数等价证明，不用忽略不同contract字段的方式强行加载模型。每份训练快照都与本route、本seed、本fold的初始化比较。

每折另有真实HOLD、同经济执行机制的每日动量和20日动量规则，使用相同四账户、K20、共同5日mask及排序/请求规则。HOLD不主动再平衡，初始股数保持不变，不能把旧目标权重执行器混作新同机制对照。20日动量作为预先指定的候选门槛，并非保证它在所有年份都优于每日动量；两者与HOLD均完整报告。

48份训练快照各评估四账户，共192条训练模型验证路径；初始化控制48条、两fold规则24条，共264条主K20路径。各折交易步数以其实际报价日历为准，不预填统一60步。不按账户或单seed挑最好模型，不新增保留期回测。

## 梯度与Critic诊断

只在2,048和4,096快照对应的rollout上做pre_update/post_update：pre在 `_on_rollout_end`，post在随后PPO更新后的 `record_update` 或 `training_end`、保存快照之前。24个run各4批，共96批诊断，不是每个512步rollout都检查。

每批固定该rollout原顺序前128个样本，完整保存obs/actions/old_log_prob/advantages/returns。post必须复用同一批和同一旧目标，不能重算另一份return后把差异解释为更新效果。Actor损失使用现SB3的相同advantage归一化、clipped PPO surrogate；Critic为value MSE，记录原始及vf=.5加权梯度。

对共同的6个market/account encoder参数张量，分别用 `autograd.grad` 求Actor及Critic梯度，记录norm、点积、cosine/夹角、advantage和value target尺度，以及value head梯度。None按零梯度表示，零norm夹角记null，不能除零或伪造0度。

不得更改已有 `parameter.grad`、优化器状态、参数、rollout内容或RNG，不调用额外optimizer step，不把梯度检查算作新增RL经历。global grad clip前的各loss梯度与真实合并/裁剪后的梯度明确分开；Adam状态和自适应缩放使梯度方向不等于实际参数变化或未来收益方向。

固定前128样本是诊断batch，不保证等于PPO实际shuffle后的某个更新batch。已有parameter.grad可能是上一次优化更新的残留，不能与本诊断梯度混为同一次更新；如报告诊断batch的假想裁剪因子，须明确是假想而非实际optimizer记录。

detached中value loss到共享编码器的梯度为零是结构定义，不是实验发现。Critic head应保留梯度路径；但某批梯度可能数值为零，不能据此自动宣布网络失效。shared版本可能因无持仓等状态出现部分编码器零梯度，应按实际batch解释。

同seed/cell后续轨迹和目标会分歧，因此跨cell梯度大小/夹角不是相同状态、相同目标的完全因果比较。Critic拟合改善也不保证策略收益提高，须同时报告净收益、成本和行为。

## 分析与候选门槛

主分析按同fold/seed/预算配对，先平均四账户，再计算：route效应（fused减pure）、Critic梯度效应（detached减shared）及interaction：

```text
interaction = (fused_detached - fused_shared)
              - (pure_detached - pure_shared)
```

同时分别列出两route上的Critic效应与两Critic设置上的route效应，不用一个总体均值遮住交互。跨fold先对每个seed平均两个fold结果，再对三个seed计算均值和样本SD；不能把六个fold/seed当六次独立随机训练。每折结果和最坏账户必须并列披露，跨fold均值不能隐藏某折失效。

将训练后差异拆成初始化差异与学习增量差异：

```text
trained_A - trained_B
  = (initialized_A - initialized_B)
    + [(trained_A - initialized_A) - (trained_B - initialized_B)]
```

这是算术分解，不是完整因果证明：route初始化函数差异仍可能影响探索、状态访问、学习难度和收益，sameθ不能排除这些路径。不同年份不能与旧2025结果伪装成配对收益提升。

候选单位为cell及预算（2,048或4,096）。必须在两个fold各自同时满足：

1. 该折三seed四账户共12条路径的最坏回撤不超过10%。
2. 先平均四账户后的自身初始化增益，三seed均值大于0且至少2/3 seed增益大于0。
3. 该折平均净ROI超过同机制20日动量规则。

任一fold不满足则不合格。合格者按跨fold逐seed平均净ROI、更小全部账户最坏回撤、更低成本、更短预算、固定cell名称排序；无合格者记null。代表seed固定42，两个fold的该seed个体表现单独列出，不将配置平均收益赋给代表模型。

即使通过，也仅是历史部署研究候选，不升级工作台/实盘默认，不声称全球最优。不使用辅助K、旧2025收益、梯度夹角或某个幸运seed替代上述门槛。

## 验收与产物

新产物使用独立 `reports/runs/ppo_attribution_20261007`，拒绝覆盖旧研究。源代码只新增隔离模块；原80份源码与旧协议、数据/模型结果保持不变。

工程验收需覆盖同构参数/归档SHA、critic轴前向等价、route功能差异、股票置换及无有效数据、detach梯度与Actor梯度保留、Critic head训练路径、诊断无副作用、原始完整混合动作logP、保存加载、共同5日mask及未来数据扰动、三RNG隔离和自然终点/TimeLimit处理。

机器协议冻结后保存原生训练动作/旧概率/精确账本、episode覆盖、快照与共享前缀、实际Adam次数、四账户评估及拒单原因、96批梯度证据、初始化等价和控制复用身份、收益/成本/现金/换手/行为、Critic拟合、完整参数/源码/data/scaler/model哈希及执行时间。

最终产出详细报告和摘要，明确七方向探索的后续归因假设、两fold完整结果、初始化与学习分解及剩余限制。独立审计不额外训练或重跑保留期；如有未执行、失败或协议偏离，明确标记，不以计划代替完成记录。
