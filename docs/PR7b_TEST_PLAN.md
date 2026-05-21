# PR#7b 测试计划 — 部署侧（systemd 单元全套 + bootstrap_nv8.sh）

> 阶段 1（架构）+ 阶段 2（测试用例设计），按 rules `00-core.mdc §开发工作流`。
>
> 与 PR#7a 一同从原始 PR#7 拆出；PR#7a 已合并（v9 残余清理），本 PR 只加部署文件、不动业务代码。

## 1 · 哲学：systemd-only，no docker compose

v10 的进程拓扑是"很多个 Python 长循环 + cron 风格 oneshot"，没有需要互相 docker 网络隔离的进程；待测模型的 vLLM/SGLang/Transformers 容器由 `orchestrator/stages_py.py` 直接通过 `docker-py` 拉起，**不属于 compose 范畴**。所以本 PR 不引入 `docker-compose.yml`；用 systemd 单元 + 一个 bootstrap 脚本搞定。

heyi_engine 自己是**用户自管的 production 实例**（INV-1：评估管道不得碰它），它的 compose 不属于本仓库。

## 2 · 进程清单

| 角色 | 入口 | systemd 单元 | 类型 |
|------|------|--------------|------|
| 编排主循环 | `python -m orchestrator loop` | `heyi-eval-orchestrator.service` | simple, Restart=always |
| 模型发现轮询 | `python -m discover loop` | `heyi-eval-discover.service` | simple, Restart=always |
| 只读管理面板 | `python -m panel` (HTTP :8090) | `heyi-eval-panel.service` | simple, Restart=always |
| 数据快照 | `python -m backup` | `heyi-eval-backup.service` + `.timer` | **已存在（PR#6）**，本 PR 不改 |
| 通知出箱→微信 | `python -m notify_sync` | `heyi-eval-notify-sync.service` | simple, Restart=always |

`notify_sync` 模块本仓库**还没实现**（只有 outbox 写文件这一半）。本 PR **不实现** `notify_sync` 业务代码——只占位写 service unit + 注释 TODO，留给后续单独 PR。本 PR 的 systemd 单元如果该 service 启动失败，不应阻断其他服务。

## 3 · 文件清单（新增）

```
deploy/
├── systemd/
│   ├── heyi-eval-orchestrator.service     [新增]
│   ├── heyi-eval-discover.service         [新增]
│   ├── heyi-eval-panel.service            [新增]
│   ├── heyi-eval-notify-sync.service      [新增, 占位; 启动后会立即 exit 0]
│   ├── heyi-eval-backup.service           [已存在, PR#6]
│   └── heyi-eval-backup.timer             [已存在, PR#6]
├── env.example                             [新增] /etc/heyi-eval-v10/env 模板
└── README.md                               [新增] 部署文档
scripts/
├── bootstrap_nv8.sh                        [新增] 一键部署脚本（nv8-only）
└── mac-mirror.sh                           [已存在, PR#6]
tests/
└── test_systemd_units.py                   [新增] 静态 lint：单元文件语法 + 必含字段
```

## 4 · 关键设计决策

### 4.1 用户与目录

- 所有服务以非 root 用户 `ai`/`ai` 跑。
- `WorkingDirectory=/home/ai/heyi-eval-v10`（git 仓库 checkout）。
- `EnvironmentFile=-/etc/heyi-eval-v10/env`（前缀 `-` 表示文件不存在不报错），内容是 `HEYI_EVAL_DATA=`、`HEYI_EVAL_BACKUPS=`、`HEYI_ENGINE_URL=`、`HEYI_ENGINE_API_KEY=` 等覆盖项。
- 数据目录默认值不在 service unit 里硬编码，让 `orchestrator/config.py` 的默认值（`~/heyi-eval-data`）说了算。

### 4.2 失败隔离

- `Restart=on-failure`，`RestartSec=10s`，`StartLimitBurst=5`，`StartLimitIntervalSec=300s`：5 分钟内重启 5 次以上才彻底放弃。
- 各 service **互不 `Requires=`**：主循环不依赖 panel 启动；发现循环不依赖 notify-sync。原因：单点故障不能拖垮整条管道。
- 仅 `After=network-online.target heyi-engine 监听端口`（用 `systemctl is-active` 模糊判断不可靠，所以**不**用 systemd 依赖来等 heyi_engine；orchestrator/main 已经有 `_engine_preflight_gate` 在跑循环里自己等）。

### 4.3 资源限制

- `Nice=10` + `IOSchedulingClass=best-effort` + `IOSchedulingPriority=7`：所有评估侧服务都让出 GPU/IO 给 heyi_engine 生产实例（INV-1）。
- `MemoryHigh=8G` 软上限 + `MemoryMax=16G` 硬上限：防止 orchestrator/discover Python 进程内存泄漏拖垮 NV8 host。
- `TasksMax=512`：每个 service 子进程数硬上限；防止 fork bomb。

### 4.4 bootstrap_nv8.sh 行为

幂等脚本，按顺序：

1. **前置检查**：
   - `uname -a` 含 `nv8` 或 hostname=`heyi-sh-nv8`（防止误跑在 mac/dev）；可用 `--force` 跳过
   - `docker --version` 存在
   - `nvidia-smi` 返回 0
   - 用户 `ai` 存在；当前用户必须是 `ai` 或 sudo 切到 ai
2. **目录创建**：
   - `/home/ai/heyi-eval-v10`（git clone 或 git pull）
   - `/home/ai/heyi-eval-data` (chmod 700, owner ai:ai)
   - `/home/ai/heyi-eval-backups` (同上)
   - `/etc/heyi-eval-v10/env`（如不存在，复制 `deploy/env.example`）
3. **Python 环境**：
   - `python3.11 -m venv /home/ai/heyi-eval-v10/.venv`（如不存在）
   - `pip install -e .` 在仓库根
4. **systemd 安装**：
   - `cp deploy/systemd/*.{service,timer} /etc/systemd/system/`
   - `systemctl daemon-reload`
   - `systemctl enable --now heyi-eval-orchestrator.service heyi-eval-discover.service heyi-eval-panel.service heyi-eval-backup.timer`
5. **健康检查**：
   - 等 10s
   - `systemctl is-active heyi-eval-orchestrator.service` 必须是 `active`
   - `curl -sf http://127.0.0.1:8090/api/health` 必须 200
   - 不通过则 `journalctl -u heyi-eval-orchestrator.service -n 50` 打到 stderr 并 exit 1
6. **总结**：打印 panel URL、systemd unit 状态、备份时间表。

脚本要求：
- `set -euo pipefail`
- 全部走 `#!/usr/bin/env bash`，过 shellcheck SC2086 等
- 每个步骤一个 `step()` 函数封装 + emoji-free 单行 log

### 4.5 静态测试范围

`tests/test_systemd_units.py` 用 stdlib `configparser` 解析 service/timer 文件并断言：

- **U-1**：每个文件能被 configparser 解析（语法对）。
- **U-2**：每个 service 都有 `[Unit]`/`[Service]`/`[Install]` 三段；timer 有 `[Unit]`/`[Timer]`/`[Install]`。
- **U-3**：每个 service 必有 `User=ai`、`Group=ai`、`WorkingDirectory=/home/ai/heyi-eval-v10`、`EnvironmentFile=` 行（前缀 `-`，即可选）。
- **U-4**：每个 service 必有 `ExecStart=`，且路径以 `/home/ai/heyi-eval-v10/.venv/bin/python` 开头（强制走 venv，不能用系统 python）。
- **U-5**：每个 service 必有 `[Install] WantedBy=multi-user.target`。
- **U-6**：禁止任何 service 含 `User=root`（INV：所有评估侧服务非 root）。
- **U-7**：所有 service 都不应 `Requires=`/`Wants=` 其他 heyi-eval-* 服务（互不绑死）。
- **U-8**：每个 service 都有 `Restart=` 字段（值在 `{on-failure, no, always}` 中之一）。
- **U-9**：所有 service 都有 `Nice=` 且 ≥ 0（让出优先级给 heyi_engine，INV-1）。
- **U-10**：bootstrap_nv8.sh 文件存在且首行是 `#!/usr/bin/env bash`，含 `set -euo pipefail`。

`tests/test_bootstrap_script.py`（可与上面合并到一个文件）：

- **B-1**：脚本能被 bash `-n` 静态语法检查通过（用 `subprocess.run(["bash", "-n", "scripts/bootstrap_nv8.sh"])`）。
- **B-2**：脚本 grep 含 `--force` 选项（可绕过 hostname 检查）。
- **B-3**：脚本不应硬编码 `/var/lib/heyi-eval` 等 v9 路径（INV-11 静态守卫已经覆盖，但这里多一道断言，文件名直接 grep）。

## 5 · 验收条件

1. `pytest --no-cov` 全绿；新增 U-1 ~ U-10、B-1 ~ B-3 全过。
2. `ruff check .` 0 错。
3. `mypy` 错误数 ≤ main（CI 阈值不变）。
4. `bash -n scripts/bootstrap_nv8.sh` 0 错（CI 中加 shellcheck 是 PR#7b+1 的事，本 PR 只用 `bash -n`）。
5. **本机干跑（不真改 systemd）**：用 `DRY_RUN=1 bash scripts/bootstrap_nv8.sh --force` 应能打印将要执行的命令并 exit 0；不真的写 /etc。
6. 文档 `deploy/README.md` 含部署/卸载/回滚步骤。

## 6 · 风险与回退

- **风险**：systemd 单元定义错（比如 `User=` 漏写）导致 service 以 root 跑，可能写坏 /home/ai 权限。
  - **缓解**：U-3 + U-6 在 CI 时静态拦截。
- **风险**：bootstrap 脚本误跑在 mac 上 `rm -rf /home/ai/...`。
  - **缓解**：B-2 强制 `--force` 才能绕过 hostname=`nv8` 检查；脚本删目录前必须先白名单匹配 `/home/ai/heyi-eval-*` 前缀。
- **回退**：单一 commit，`git revert`；运行环境上 `systemctl disable --now heyi-eval-{orchestrator,discover,panel,backup}.{service,timer}` 即可彻底回到无 systemd 状态。

## 7 · 不在本 PR 的范围

- `notify_sync` 模块的业务实现（占位 service 启动后 exit 0；写一个 TODO issue）。
- shellcheck CI job（属 CI 演进，单独 PR）。
- nv8 真机部署演练（属 PR#8 e2e 范围）。
- `docs/USAGE.md` / `docs/OVERVIEW.md`（属用户文档 PR；和部署文件分开评审）。
