# 仓库内 opencode 走 Clash 代理

日期：2026-11-10。范围：本仓库的本地工具链，不涉及研究结论。

## 结论

- **现象成立**：本仓库里的 opencode 默认走直连，不经过 Clash。
- **原因**：opencode 是 Bun 编译的二进制，只读 `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `NO_PROXY` 环境变量，**不读 Windows 系统代理**（`HKCU\...\Internet Settings`）。本机 Clash 只开了系统代理，于是只有浏览器、.NET 这类认 WinINET 的程序走代理，opencode、git、node、python 全部直连。
- **处理**：`opencode.cmd`（仓库根）+ `tools/opencode_proxy.ps1`。只对子进程注入变量，并注入 `--standalone`，让真正发请求的 server 进程继承。

## 证据

| 检查 | 结果 |
| --- | --- |
| 进程 / 用户 / 系统注册表 / git config / opencode 配置里的代理 | **全部为空** |
| Windows 系统代理 | `127.0.0.1:7897`，`ProxyEnable=1`，Clash 正在监听 |
| `node fetch` 直连 npm registry | 200，2.96 s（直连本身可用） |
| 删掉 `models.json` 后跑 `opencode models` | **0 次外连**，4 s 返回（模型目录不联网，测不出代理） |
| 真实模型请求 + `HTTPS_PROXY` 指向本地空监听 | 进程建立 **2 条 TCP 连接** 到该监听 → 认环境变量 |
| 真实模型请求 + 走 `127.0.0.1:7897` | **5.5 s 返回 `OK`** |
| 真实模型请求 + `HTTPS_PROXY` 指向关闭端口 | **93 s 后失败**（流量确实被送去代理，不是直连） |

## 为什么必须加 `--standalone`

模型请求由 opencode 的 **server 进程**发出，不是 CLI 前端。不加 `--standalone` 时，CLI 把请求交给**已经在跑的后台 service**，而那个 service 是在没有代理变量的环境里启动的，所以只在前端设置变量没有任何效果。

`--standalone` 让私有 server 成为本脚本的子进程，直接继承变量。代价是每次启动多 1–2 秒，会话仍写同一个数据库。

## 用法

```powershell
opencode.cmd                          # 代理启动 TUI（PowerShell 里写 .\opencode.cmd）
opencode.cmd run "问题"               # 代理跑一次性任务
opencode.cmd -Check                   # 打印解析到的代理并测试端口
opencode.cmd -Direct                  # 不注入任何变量，等同直连
opencode.cmd -Proxy http://127.0.0.1:7890   # 手动指定
```

`opencode.cmd` 在仓库根，cmd 会优先命中当前目录，所以在仓库目录里直接敲 `opencode` 就会走包装；PowerShell 不搜当前目录，需要 `.\opencode.cmd`。

`.ps1` 因为默认执行策略（`Restricted`/`AllSigned`）会被拒绝加载，所以入口做成 `.cmd`，由它用 `-ExecutionPolicy Bypass` 调用脚本。

## 作用域与限制

- 只影响该子进程树：**不写用户环境变量、不改注册表、不重启**已经运行的后台 service。
- python / akshare 仍然直连（系统代理只对 WinINET 程序生效），数据下载不受影响。
- 直接敲全局 `opencode`（不经包装）仍然直连。
- 脚本会先测代理端口；不通时给出警告，因为此时 opencode 会在第一个请求上长时间挂起（实测 93 s 才返回错误）。
