# 归档：已结束的研究线

日期：2026-11-08
分支：`cleanup/archive-closed-lines`

这 38 个脚本属于三条已经停止的研究线。**保留而不是删除**，因为报告里的负面结论需要能被
重跑验证——删掉就把"可复现的结论"变成了"不可复现的断言"。

| 线 | 数量 | 结论 |
| --- | ---: | --- |
| 按时点干预（timing） | 23 | 多次训练未动先验 |
| 反事实学习（counterfactual） | 14 | 未产生可认证效果 |
| 市场摘要（market summary） | 4 | 同上 |
| 合计 | **38** | |

（其中 `analyze_credit_counterfactual` 是本次一并归档的：它属于反事实线，且 import 了两个已归档工具。）

## 目录变化

```
tools/         43 个（在用）
tools/archive/ 38 个（已结束）
tests/        104 个文件（用：1,715 passed）
tests/archive/ 44 个文件（983 passed / 22 failed / 12 errors）
```

## 怎么用

```powershell
python -B -m tools.archive.timing_counterfactual
```

导入路径是 `tools.archive.<name>`，不再是 `tools.<name>`。

## 测试

归档测试默认不收集，避免污染回归门。要跑：

```powershell
$env:ARCHIVED_TESTS = "1"
python -B -m pytest tests/archive -q -p no:cacheprovider
Remove-Item Env:\ARCHIVED_TESTS
```

**状态：1,017 个测试里 983 通过、22 失败、12 错误（约 96.7% 保留）。**

剩下的 34 个不是新坏掉的逻辑，而是**引用了脚本的物理位置**：它们在 `tmp_path` 里
重建 `tools/<name>.py` 的目录结构再断言哈希，位置一变就对不上。机械改写能覆盖
`from tools import X`、`tools.X`、`tools/X.py` 三种形态，覆盖不到所有拼路径的写法。

**归档不是维护。** 这些脚本依赖的 `src` 接口以后会变，测试也不会再跟着修。

## 文档里的旧路径已失效

历史报告中的 `tools/<name>.py` 现在位于 `tools/archive/<name>.py`。我没有去改 78 篇文档——
那会改写历史记录，而历史记录恰恰应该是"当时是这样跑的"。

## 回滚

```powershell
git checkout main
```
