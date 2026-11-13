# 仓库清理：删掉 2.34 GB 可删的重复与过期数据

日期：2026-11-14。范围：全仓（`data/` 9.93 GB、`reports/runs/` 11.94 GB、`web/` 0.21 GB）。
本文记录**这次删了什么、凭什么、留了什么、以及一条我自己说错的判断**。

## 1. 结论

| 项 | 删除前 | 删除后 |
| --- | ---: | ---: |
| `data/` | 9.93 GB | **7.59 GB** |
| `reports/runs/` | 11.94 GB | 11.94 GB（未动） |
| 可回收合计 | — | **2.34 GB**（14 个目录） |

`reports/runs/` 的体积 11.94 GB 与 [本地产物目录](../reports/README.md) 记录的 2026-11-07 清理后数字**逐位一致**，
说明那套三层保留策略仍在生效、没有新的泄漏；810 个 `model.zip`（2.93 GB）与 `orders.jsonl.gz`（3.44 GB）属 T1 证据（只有重训才能重建，
且先验归因那轮就是靠权重回答"策略学到了什么"），**本次一个都没删**。

## 2. 删了什么

| # | 对象 | 体积 | 依据 |
| --- | --- | ---: | --- |
| 1 | `data/clean/research_v2/` | 356.1 MiB | 与 `research_v3` **逐文件同哈希**，只有 `manifest.json` 的版本名不同（差值即复审里那句"重复 373 MB"）。`configs/research.json` 的 `clean_dir` 指向 v3，全仓没有一处按路径读 v2；`preprocess.py` 里出现的 `research_v2` 只是默认 `version=` 字符串 |
| 2 | `panel_scale` `panel_4850` `panel_4150` `panel_3400` `panel_3000` `panel_2000` `panel_1500` `panel_1000` `panel_partial`（9 个） | 465.2 MiB | 覆盖率扫描的中间档位，**全仓 0 处引用**（`panel_full` 被 18 处引用，保留） |
| 3 | `forward_pool_20261110/11/12` | 1,397.5 MiB | 前向面板每天全量重下、旧的只是被覆盖的输入；决策本身记在 `reports/shadow/e1_shadow.jsonl`（book + score），不需要面板复算 |
| 4 | `data/online/history/.h_cb7d263ffefd4d8b/` | 13.8 MiB | 2026-11-07 就标记为**未发布的废弃抓取**，139 个文件从未被引用 |

删除前逐条校验：路径必须直接位于 `data/clean/` 或 `data/online/history/` 之下、名字必须在白名单内、
且**不得是** `research_v3` / `panel_full` / `forward_pool_20261113` / `e2_dynamic1800` / `feature_top1800` / `margin_pool`
任何一个保留项；任何一条不符即中止。校验通过后才执行。

另删 3 个空目录：`data/history_data`、`reports/benchmarks`、`src/data/raw`（代码里没有任何地方引用它们）。

## 3. 留了什么，为什么

| 对象 | 体积 | 理由 |
| --- | ---: | --- |
| `reports/runs/`（810 个 `model.zip` 2.93 GB、`orders.jsonl*` 3.44 GB） | 11.94 GB | T1 证据，符合现有保留策略 |
| `e2_dynamic1800` / `feature_top1800` / `margin_pool` | 1.44 GB | E2、IC 与组合研究的输入，结论可复现依赖它们 |
| `agent_improvement_20260927_walk` / `sina_hs300_study_20260927` | 0.91 GB | 9 月 PPO 线的数据，被 `prior_attribution.py` 等工具按路径读取 |
| `panel_full` / `panel_4550` / `research_multiseed_20260926` / `test000001…` | 0.16 GB | 有引用（`panel_4550` 在 CAPM 报告里作为一个覆盖率档位被点名） |
| `data/clean/` 根目录 2026-07-01 旧构建（298 个 `*_features.parquet` + `panel_data/train/val/test`） | 0.61 GB | **2026-07 你已决定暂不删除**（`training_data_improvement_plan.md` §250）。口径确实早就不用了，但这是你的决定，本次不动 |
| `web/node_modules` / `data/online/plans/` / `.workbench/jobs/` | 0.32 GB | 可重建、且不属于研究证据；本次未选入删除范围 |
| `data/online/history/` 其余部分 | 2.74 GB | 不可变抓取快照，是面板的**唯一重建来源**（面板只是缓存） |

## 4. 代码改动：前向面板的滚动保留其实早就存在

我在检查报告里说"每个交易日产出 466 MiB、一年约 110 GB、需要做滚动保留"——**这句是错的**。
`tools/forward_panel.py::prune()` 从建成那天起就在 `main()` 末尾执行，默认 `--keep 5`，
所以稳态一直是 5 × 466 MiB ≈ 2.3 GiB，并没有无限增长；`tests/test_forward_panel.py` 里也早有它自己的测试。
当时的判断错在只看了目录数量、没读 `main()` 的最后一行。

真正需要改的是两件相邻的事：

1. **`shadow_run.DATASET` 还硬编码着 `forward_pool_20261110`** —— 正是本次要删的那个目录。硬编码的路径在第一次
   轮换后就会指向不存在的面板，那时工具会报 `stale`，而原因与数据无关。已改为**跟随磁盘上最新的
   `forward_pool_*`**（`latest_panel()`），`--dataset` 仍然覆盖它。新增 4 项测试：默认取最新、无面板时返回 `None`、
   轮换后仍指向幸存者、当天默认路径必须真实存在。
2. **`keep` 从 5 降到 3** —— 稳态约 1.4 GB：录制器要读的最新一块，加两天历史用于"某个决策看起来不对"时对照。

## 5. 文档侧

- 三份仓库文档**不是重复**，各有分工，均未被改字节：[功能总览](repository_overview_20261108.md)（按研究线分组的工具说明）、
  [机器清点](repository_inventory_20261108.md)（逐工具行数/测试/文档映射）、[仓库复审](repository_review_20261110.md)（四件"已有但没在用"的东西）。
- 复审里那条"至少 7 份 PPO 协议状态停在尚未运行"**按本仓约定不算缺陷**：预登记 Markdown 保持运行前原字节，
  完成情况记在 [文档索引](README.md) 第 132–146 行（每一轮都有协议 → 摘要 → 详细报告 → 结论）。
  这条约定写在 `doc/reports/ppo_exploration_20261006.md` 末段，因此**没有去改任何协议头部**。
- 复审四件事的现状：E2 已跑完并**失败**（[结果](e2_result_20261111.md)）；四条信息源全部测完（龙虎榜、股东户数、
  业绩报表均失败，融资融券 IC 过、组合不过）；国债腿数据源已定位（`data/online/index_long/000012.csv`，产品 511010）；
  `research_v2` 重复面板已在本轮删除。

## 6. 复现与核对

```powershell
# 体积
foreach ($d in @('data','reports','web')) { "{0,9:N1} MB {1}" -f (((Get-ChildItem $d -Recurse -File -Force | Measure-Object -Sum Length).Sum)/1MB), $d }
# 保留项必须还在
Get-ChildItem data\clean -Directory | Select-Object Name
# 录制器的默认面板
python -B -c "from tools import shadow_run; print(shadow_run.latest_panel())"
```

单位：本文的 MiB / GB 与 [本地产物目录](../reports/README.md) 一致，**表格里的删除量按磁盘实际占用（MiB）记录，
合计 2.34 GB 为十进制**；两者差 4.9%，不要混用。
