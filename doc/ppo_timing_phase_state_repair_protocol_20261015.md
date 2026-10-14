# 调仓预算实验状态哈希工程修复附录

日期：2026-10-15。状态：新修复driver与独立验收完成前草案；尚未执行本附录下的正式训练。该附录只修工程状态编码，不改原冻结[架构与预算协议](ppo_timing_phase_budget_protocol_20261015.md)，也不依据未封口收益更改研究方法或选模规则。验收实际通过、提交与逐文件身份登记后，才允许prepare新目录。

## 已观察事故

原目录为 `reports/runs/ppo_timing_phase_budget_20261015`。2023 worker session50482实际exit1，输出chunk984001：linear已达到32,768个Agent RL步与1,280次Adam，最终capture在 `_json_state` 到严格 `json_digest` 的路径因nonfinite标量失败。未保存final_model、未生成该run的completed marker，因此不能当作已封口正式样本。

其8,192步short模型、64个原PPO buffer、64个实际bootstrap sidecar及Agent/基线各32,768步训练trace完整存在。是否逐文件健康由事故证据manifest与后续只读核验披露，不将文件存在冒充完整科学审计通过。原2024 worker session90201的linear已completed并保存32K终点，随后进入due训练且仍活跃；不据此称原四run/八快照矩阵成功，也不读取未封口ROI判断方法好坏。

核心缺口是状态哈希标量编码：Broker拒单记录可以合法含 `price=NaN`、`filled=0`，随后进入history。原codec已用dtype/shape/字节SHA表示NumPy数组，但非有限scalar未编码，严格JSON摘要拒绝其原样进入。该工程异常不说明模型参数、模型输入、实际成交或奖励含有可接受的非有限数值。

根代理已用只读清单固定 `reports/runs/ppo_timing_phase_budget_20261015/engineering_failure_2023.json`，登记实际退出、输出引用、318份原文件SHA与已付预算。该manifest实际SHA为 `8b2c1948e89094ab22267c8dff96e2caef27a8fd2cd89815b5e70e8f3eb39b2b`；不包括仍在写入的2024运行。它是失败证据清单，不是完整科学结果签章。原已存在trace、模型、buffer和日志不补写、不伪造completed，不补保存旧32K模型作为本轮修复样本。

## 唯一允许的工程改动

新增明确隔离的driver `tools/research_timing_phase_budget_repaired.py`，作者测试 `tests/test_timing_phase_state_repair.py` 与独立测试 `tests/test_timing_phase_state_repair_review.py`。原96份src、原runner、policy/agent、原测试、原冻结协议和已签父输入字节保持不变。

修复只在状态摘要中为Python/NumPy非有限float scalar增加显式类型及NaN/+Inf/-Inf原始bit编码。有限标量及原数组摘要语义保持；编码不是填补行情、将NaN改成0、修改订单、改变模型输入或放宽财务计算。模型参数、概率、Box、奖励及实际财务数值的既有finite检查继续严格执行，不放宽全局 `json_digest`。

新driver只能通过显式CLI context安装工程codec与增强冻结guard，内存引用原runner功能；进入context前保存原函数，所有正常返回与异常退出均恢复原函数。不得静默改写oldbytes、在import时永久安装patch，或在普通旧入口隐藏改变研究语义。增强guard必须先验证原core身份，再验证repair扩展，直接API与缓存读取在创建provider或运行模型前也须接受两层核验。

新机器协议以 `engineering_repair` 显式绑定新driver、两份测试、本附录、原审计/分析/封存工具及其测试的逐文件SHA，以及原失败manifest实际SHA、原机器协议文件SHA。原机器协议文件SHA为 `f8fba8ec1616f5f23dfc15ba18153ccacce3bc108c772f0d019a7f1a10e717c3`，原runner SHA为 `9c54115cff7a88e4b0c668fff073d7aaaa3c3336c6e40ae5cc80b42c3db7826a`，父Credit终审硬钉为 `fd3449e3d0fcb2df87fa9cd3d87274fa3d93290bb77a3b8d111c5c04751e8955`。core的原96源码、component/helper/test库存和数据契约仍完整冻结。

协议顶层 `tool_sha256` 使用新driver真实SHA，使普通旧入口拒绝该修复协议。仅显式context中的增强guard构造只读core视图，恢复原runner SHA进行旧身份校验，再验证完整新协议及repair扩展；该视图不写文件、不传入模型。实际配置和模型契约始终绑定完整新协议SHA与新driver身份，旧core和新driver的身份分别披露。最终签章还须覆盖失败manifest及其中318份保留文件。

失败manifest不存在、SHA未登记、core或repair任一输入变化、输出与原目录重叠时必须拒绝prepare/训练/缓存复用。正式目录明确为 `reports/runs/ppo_timing_phase_state_repair_20261015`，不覆盖原目录，不借原物理run作新样本。

## 新执行预算与控制

修复目录从头进行2023/2024各linear与due_phase_residual，seed42，四次32,768步物理训练。新增131,072 Agent RL步与同量独立训练基线步，八个8K/32K快照；快照clock之和163,840不重复计为RL。原失败/继续尝试已消耗的时钟、RL、基线和Adam单独计入工程成本，不能抹掉，也不计为这四个修复run的经历。

两fresh linear的8K已更新前缀仍须与Credit对应fold/relative_log/lambda1/seed42控制逐张量、Agent/基线原生数组及原8-key PPO buffer冷读精确复现。新源码身份、状态codec与数值复现分开登记；不复用旧模型、旧梯度或旧经历，不进行gradient重放。

Actor、129特征、Discrete(2)动作、初始映射权重、价格源、HFQ研究口径、M20股票decoder、费用、奖励、gamma.995、GAE lambda1、margin2和全部PPO参数不变。8K仍在Adam完成后隔离保存，金融评价延后对应32K结束；全部四run/八快照封口后才固定选择，不看短收益早停。

原准入和确认门槛不变：每开发折至少+1bp，并至少一条同路径同时有greedy macro及经济账本变化；按原排序完全平局先linear再短预算，无fallback。准入后六确认run从头使用选中架构及同选中8K/32K预算；修复正式研究最高总RL为180,224或327,680，未准入仅131,072。此前工程成本另记，不假称最高预算已完成。多seed/跨折+0.5pp/自身初始化/风险最终gate与原协议完全一致。

不新增下载、标签、股票、观察、奖励或动作；训练/评价provider仍先裁到原fold train_end/val_end，2026年7月以后保护期不访问。没有本附录候选，不晋升默认模型、不自动部署。

## 验收与保留证据

提交和prepare前须用合成/伪归档验收合法拒单history的NaN/Inf标量编码、类型/bit区分、数组dtype/shape/字节摘要、有限值不变和mutation摘要变化；验证context成功与异常退出的函数恢复，以及core/repair库存、失败manifest和不同输出root的早拒。保留模型finite负例，不能以状态codec允许非有限模型参数。

复验快照隔离、全部8/9张量Adam覆盖、真实权重变化、实际clock和两个fresh Credit控制要求。合成测试、函数恢复及保存加载验收不计正式金融经历，不预写通过数或收益。

原2024 worker正常结束后，全部原尝试的真实成本与完成/失败状态分别记录，不混入修复样本或虚称原final_audit已生成。新研究结束后另行报告实际预算、模型/账户/资源、数值复现、固定选择、条件确认和唯一冷审/签章；失败同样如实保留。原4,676父签输入及任何新签文件封口后不再修改。
