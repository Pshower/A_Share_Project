# 调仓控制器多步收益归因研究记录

登记日期：2026-10-15。状态：已提交、正式prepare实际成功退出，2023/2024两个独立筛选worker已启动，筛选运行中。prepare登记时正式金融步为0；当前运行尚未全部封口，不从中途日志预填实际步数或收益，也不将运行中的步骤仍写为0。尚无本轮完整结果、固定选择、确认准入或胜出结论。

## 正式注册身份

- 预注册提交：`4791c4fcfbc7d541aa1b5069505137cc3bc4386c`。
- [机器协议](../../reports/runs/ppo_timing_credit_20261015/protocol.json)文件SHA256：`a30c7f5f257d0773f2126abfd6425189ca81d5331fa633da3819321bfa431e70`。
- 新runner SHA：`97c8d5cddc8e09ed9ee5f41a16ff23ebb6503004baaa5a33e79e4bf30bde61cd`；冻结人类协议SHA：`7966dd9f4cd55b955b6a87485be1c2fe7b3c3906ee1bec6ab380788b30db9afe`。
- 复用margin helper SHA：`a4015c75bfbe1be41cc24d37f94961f787d6dca4dabbc648b38067251f7a5b50`，作为实际代码依赖额外登记，不以96份src库存替代。
- 登记作者测试SHA：`75345481ec0f0e12fc22ed293a0337cfc8412bd62414cd7eddee70fab4502034`；登记独立guard测试SHA：`862baad045c4676aa3d9f753106c539b0885d2a92c2b1626b36ca04a84176210`。机器训练输入库存为这两份测试；另11项独立GAE审计wrapper属于验收/审计证据，不能擅称机器协议登记了第三测试输入。
- 原96份src与前轮一致。前轮margin协议SHA `41f673dac7f59bc6973f67cc6fc4acd63253170878700796acd8192a81f8a92b`，最终审计硬钉 `1e8b7363582b452f6f1b42a79049cdd94f067b21a244abf8b8734e4a390453df`。
- 四个lambda0.95控制的复现来源为最早timing研究，协议文件SHA `f65101d627c86d420658ac23efea3a27e954b00ea93ff543ca165c92f5c2c995`，不是把上一批margin数据冒作本批相对奖励控制。

正式目录为 `reports/runs/ppo_timing_credit_20261015`，与两个前轮目录隔离。模型契约、provenance、完成marker和缓存绑定新工具、helper及本轮协议。注册Git元数据如实为dirty=true，冻结以机器SHA库存为准；并发用户Markdown、编辑器配置及PDF变更均保留，不作为本轮研究输入或随意回退。

登记独立核验实际exit0；前轮3,244份签署文件真实纯读缺失0、SHA差异0。此读取没有新增金融步骤或模型前向，不能当成本轮收益验收。协议文件原始字节SHA与模型中的规范化JSON身份不混用。

## 问题与边界

前轮margin对照表明：降低日历初始化强度确实增加探索与交易行为变化，但主要学成少调仓、高现金，没有跨开发折增量，未进入确认。本轮不把初始化、GAE或单日节费认定为唯一失败原因。

详见[冻结前协议](../ppo_timing_credit_protocol_20261015.md)。只比较奖励口径与GAE trace长度，不换股票选择、数量、输入、Actor、资本、费用或交易执行，不将较长trace称为未来信息或收益保证。

## 2×2设计

| 奖励口径 | GAE lambda=0.95 | GAE lambda=1.0 |
| --- | --- | --- |
| log_net | 原绝对奖励控制 | 较长归因 |
| relative_log | 原相对奖励控制 | 相对奖励与较长归因 |

gamma始终0.995，calendar_margin始终2。lambda是0.95/1.0，不是0.99。各cell独立训练自己的网络，同fold/seed的初始Actor及Critic逐tensor一致；输入仍129维、线性categorical Actor和64/64 Critic，共12,805参数。初始确定性策略精确复现未修改的20日M20，日历权重只初始化一次，之后全部可训练，不是永久先验或BC。

Actor仍只输出HOLD/REBALANCE两个macro token；股票排序、rank-dependent数量、冻结/T1/整手/资金/费用由旧规则与Broker执行，不宣称新的神经选股或直接预测股票仓位。净ROI与初始化增量单独评价，不把规则自身收益算作学习成功。

log_net使用Agent净NAV日log收益；relative_log减去独立20日基线账户同日净log收益。两book从完全相同实际Portfolio开始，之后各自执行、扣费和推进。两方法有相同观察和基线算力，不将相对reward冒充净ROI，也不二次扣费用。

GAE的TD误差权重为(0.995×0.95)^k或0.995^k，无限序列近似尺度18.3/200步不是预测期限。实际受240步episode、512步rollout、自然终止和TimeLimit bootstrap限制，优势不跨reset。当前PPO本来就学习多步目标；lambda0.95不只看18天，lambda1也不是完整240步无bootstrap真实回报。更长trace可能增加方差，不预判它更好。

## 数据与共同设置

复用新浪300股、20项特征、各训练截止前scaler和原48种日期/账户/phase起始分层，不下载、重拟合scaler或新增标签输入。训练provider在各折训练截止前物理裁剪，评价provider在Q2末前裁剪；2026-07-01以后保护期不观察或评价。HFQ经济代理和固定2026股票池的限制仍在。

共同设置为标准SB3 PPO、LR3e-4、n_steps512、batch128、epochs5、clip.2、targetKL.03、ent0、vf.5、max_grad_norm.5；CPU单线程、最多两个worker。资本、K20、Broker、费用与交易门槛沿用原协议，不为新方法放宽。

## 计划预算与准入

2023/2024各四cell、seed42，每trial8,192日步：计划八trial、65,536 Agent RL步；独立基线同量推进另计。只评价登记预算终点，不挑早期checkpoint。

lambda0.95的四个reward/fold控制必须本轮重新实际训练。初始化/最终policy tensor、完整Agent/基线native数组与原8-key PPO buffer冷读对比最早同条件封存实验；旧模型和结果不能充本轮样本，比较本身不新增NN或环境步。

全八trial封口后按最差折四账户平均净ROI差、两折平均差、最坏账户回撤、费用加滑点排序；完全平局优先lambda0.95，再优先log_net。选中cell只有两折各至少+1bp，且至少一条同路径有真实确定性macro和经济账本变化，才授权确认。未准入则screen_failed、确认0，不改选第二名。

若准入，2025/2026固定cell各seed42/123/2026从头训练16,384步，计划六trial、98,304 RL步。本批最高十四trial、163,840 RL步；未准入只执行八trial、65,536步。以上均是预注册预算，不将计划当已执行；当前筛选运行中，实际计数待完整封口后核对。

最终同时要求每确认折正基线增量、至少2/3seed胜基线、跨折平均优势至少+0.5pp、自身初始化增量为正、最坏账户回撤≤10%且相对基线恶化≤2pp。已曝光Q2只能称本轮选择未使用的时间确认，不是从未曝光盲测；不自动部署或改变默认模型。

## 工程证据

新增独立 `tools/research_timing_credit.py`，原96份src、前轮margin的3,244份签署产物及原margin工具/53项测试保持不变。新runner、复用helper、两测试、协议和控制来源身份均已由正式prepare显式登记，具体SHA和Git见上。

正式机器协议确认人类协议及runner的SHA与工程准入读取一致。后续只更新本人类结果记录，不回改冻结协议、工具、测试或已签产物。

主runner及独立guard联合73项实际通过，41.78秒，包含33项作者测试与40项独立测试；完整GAE递推独立wrapper11项另测。root最终联合84项实际通过，28.68秒，进程退出码0。此前联合运行83项通过、1项失败，原因是人工fixture缺少 `timing.reward_basis`；补齐样例后完整重测84项通过，正式配置未改。各次有重叠，不相加，合成NN、人工行情和toy Adam不计本轮正式金融经历。

独立guard覆盖完整八cell/固定预算、同Actor/Critic初始化、1bp不向上取整、无fallback、直接确认和报告入口、全stage缓存、workspace-relative签章schema、完整工具/测试库存、模型provenance及四个fresh控制路径。Python/NumPy Bool不能伪装lambda1，包括模型和bootstrap sidecar。

新增bootstrap sidecar保留真实SB3已有tail value/dones、原始Vec reward32/native reward64及现有TimeLimit terminal value；原8-key PPO buffer不变，不增加预测、市场步骤、RNG调用或修改目标。自然终止、TimeLimit中间与尾段的合成测试核实只发生原SB3必需的预测；独立wrapper据存储数据重算GAE/returns，旧value目标诊断不冒充拟合后Critic或真实未来ROI。

## 正式进度

提交与prepare完成，两个fold worker已实际启动，每fold依次运行四cell。当前不读取未封口性能，不按中途收益调整配置、设备或挑选checkpoint；仅在八trial全部封口后更新收益、选择和条件确认，失败同样完整保存。

后续提供完整四账户结果、实际Agent/基线/评价计数、Adam、覆盖/尾段、控制复现、模型索引、摘要和独立审计。本报告不修改README、pipeline、并发用户文档、冻结协议或机器叶文件；运行中状态不等于完成65,536步或通过收益门槛。
