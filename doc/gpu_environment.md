# GPU 环境配置

日期：2026-09-26。目标环境：`AShareGPU`，与原 `Graduate` 分离。

## 硬件与版本

- GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4 GB 显存。
- 驱动：566.36；`nvidia-smi` 显示 CUDA 12.7，表示驱动支持能力，不代表 PyTorch 必须安装 cu127。
- 目标：Python 3.11、PyTorch 2.7.1+cu126、SB3 2.7.1。项目其余核心依赖通过 constraints 与既有版本对齐。
- 当前 PATH 没有 `nvcc`；使用预编译 PyTorch 不要求另装完整 Toolkit。未改驱动、系统 CUDA 或 PATH。

PyTorch 官方提供该版本的 [CUDA 12.6 构建](https://pytorch.org/get-started/previous-versions/#v2-7-1)，驱动兼容原则见 [NVIDIA 文档](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。本配置暂不要求升级驱动，最终以实际 GPU 运算验收为准。未来更换更高 CUDA 构建或编译扩展时，应重新检查驱动要求，不直接沿用这一结论。

## 安装与运行

在仓库根目录执行：

```powershell
conda create -n AShareGPU python=3.11 pip -y
conda activate AShareGPU
python -m pip install -r requirements-cuda.txt
python -m pip install -r requirements-web.txt -c requirements-gpu-constraints.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -B -m src.web.serve --port 8765
```

环境已存在时不重复创建。`requirements-cuda.txt` 必须单独安装，避免将所有应用依赖都交给 PyTorch 专用索引解析。安装工作台依赖时的约束防止把 GPU torch 换回其他构建。

工作台子任务继承服务进程的 Python 解释器，因此启动服务前须激活正确环境。页面动态显示环境名和 CUDA 可用性，不再写死 Graduate / CPU。

新建训练任务可选 auto / cuda / cpu，auto 优先 CUDA。显式请求 cuda 但环境不可用会报错，不静默退回 CPU。任务保存解析后的设备，模型 metadata 额外记录实际设备与 CUDA 构建版本。历史无设备字段的工作台任务仍按原 CPU 语义处理。

`configs/ppo.json` 的新任务默认 auto；`configs/ppo_lightweight.json` 保留 cpu，保证旧轻量协议不被静默更改。历史模型和配置不重写，旧模型默认仍沿用其保存的设备配置。

CPU 与 GPU 之间不承诺逐位相同的数值结果。需要严格复现旧 CPU 实验时继续使用原环境和 CPU 配置；同设备的保存加载一致性仍需测试。页面环境标记表示硬件可用性，具体任务设备以保存的任务配置及模型 `actual_device` 为准。

## 资源与边界

4 GB 显存先保持现有 128 / 64 网络、batch_size=64、单研究任务槽，不新增并发 GPU 训练。OOM 时任务失败并保留日志，不自动改 batch 或悄悄切换 CPU。行情下载、Pandas 处理及交易环境仍在 CPU；神经网络使用 GPU 不意味着整个流程都在 GPU，也不保证小型 PPO 比 CPU 更快。

本次仅进行环境、张量运算和 PPO 无训练检查，不启动正式训练，不重新下载股票数据。原 Graduate 保留 CPU 版 torch；其中原有 shap / slicer 冲突不迁移到新环境，也不在本次修改旧环境中处理。

代码或依赖变动可能导致旧冻结记录的源码 / 环境核验不通过，这是预期审计行为，不重写旧冻结记录冒充复现。

## 验收记录

- 新环境：Python 3.11.16、torch 2.7.1+cu126；`torch.version.cuda=12.6`，`cuda.is_available()=True`，设备计算能力 8.6。
- 官方 wheel SHA-256 已核对：`f3af23387ac106b5b01dbef0eb021883e0c00ff4073477b7ce1cbade5ef5038d`。大包缓存在被忽略的 `.workbench/wheels/`，未纳入 Git。
- `pip check`：无依赖冲突。真实 GPU 矩阵前向、反向及同步检查通过，结果和梯度均有限。
- 新环境离线测试：97 passed，13 项联网测试排除；3 个非失败警告分别为 Starlette/AnyIO 弃用提示和已有 Gymnasium 无限观察边界提示。
- 原 Graduate 回归：96 passed，1 项 CUDA 硬件测试跳过，13 项联网测试排除；再次确认旧 torch 仍为 2.7.1+cpu。
- 前端生产构建通过；2 项浏览器测试通过，覆盖 CUDA 配置检查和桌面 / 手机行情图表。ECharts 公共包仍有原有的 500 kB 体积提示。
- GPU 工作台检查任务：`j_538b660e1c684476`，32 股、40,901 参数、`actual_device=cuda`、1 步环境检查、保存加载一致、参数未更新、未训练、未评估测试集。产物：`reports/runs/workbench/j_538b660e1c684476/check.json`。
- 工作台已改用 `D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe`，地址保持 `http://127.0.0.1:8765`。截图：`web/test-results/gpu-device-desktop.png`。

结论：当前 566.36 驱动通过本项目实际 CUDA 验证，本次无需升级驱动。不代表已经验证 GPU 长时间训练性能或收益；本次没有进行正式训练。
