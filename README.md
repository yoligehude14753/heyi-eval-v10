# heyi-eval-v10

自动化模型评测 pipeline，在 nv8-6000 (`100.127.173.85` / `heyi-sh-nv8`) 上常驻运行。

## 与 v9 的关系

v9 (`heyi-eval-v9`) 在 2026-05-21 出现 cc-agent 数据丢失事件（sqlite + discover/curated 被 CLEANUP 阶段的 Claude bash 工具删除）。post-mortem 显示根因是 cc-agent 同时拥有：

1. docker socket 访问权
2. `/heyi-eval-data:/workspace` 全目录读写挂载
3. Claude 的 `Bash` 工具未禁用

v10 是基于 incident 教训的**架构级重写**，不是补丁。详见 [docs/PLAN.md](docs/PLAN.md)。

## 核心架构变化

| 维度 | v9 | v10 |
|---|---|---|
| cc-agent 涉及的阶段 | 4 个 (DEPLOY, CAPABILITY, SHOWCASE, CLEANUP) | 1 个 (SHOWCASE only) |
| cc-agent 是否有 shell | 是 (bash) | **否** |
| cc-agent 是否有 docker socket | 是 (经 socket-proxy) | **否** |
| cc-agent mount 范围 | 整个 `heyi-eval-data` rw | 仅 `runs/<run_id>/showcase/` rw + 3 个 metadata 文件 ro |
| LLM 接入方式 | CCR + 写死 model name | `heyi_engine` client 自动探测 `:10814/v1/models` |
| docker 命名空间 | `e8-*` | `e9-*`（新前缀，与 v9 隔离） |
| 数据备份 | 无 | 30min rsync + 7d 保留 + Mac 每夜镜像 |

## 信任域

- **PROD** (heyi-engine: xrouter / minimax / glm-51 / kimi-k26) — 评测流水线**禁止触碰**
- **ORCH** (Python orchestrator + heyi_engine client) — 拥有 docker socket 和 data dir 全权
- **CC** (cc-agent showcase only) — 仅能 Read/Write 当前 run 的 showcase 子目录，无 shell 无 docker
- **EPHEMERAL** (`e9-*` 前缀容器) — orchestrator 创建/销毁，cc-agent 只能 HTTP 访问
- **DATA** (`/home/ai/heyi-eval-data/`) — orchestrator 全权，cc-agent 部分 ro

## 目录结构（开发完成后）

```
heyi-eval-v10/
├── discover/              # HF Hub 模型发现 + enqueue policy（v9 迁移）
├── curator/               # 模型卡解读 + LLM enrich（v9 迁移，client 切换）
├── heyi_engine/           # NEW: LLM client + auto model discovery + health
├── orchestrator/
│   ├── state_machine.py   # v9 迁移
│   ├── store.py           # v9 迁移
│   ├── validator.py       # v9 迁移
│   ├── notify.py          # v9 迁移
│   ├── config.py          # 重写（v10 invariants）
│   ├── stages_py.py       # NEW: DEPLOY/READY_WAIT/CAPABILITY/CLEANUP Python 实现
│   └── main.py            # 重写（无 CCR 依赖 + heyi_engine preflight）
├── cc-agent/              # 重写：showcase only, no shell
├── panel/                 # v9 迁移 + 备份卡片
├── backup/                # NEW: 30min rsync + retain logic
├── deploy/
│   ├── compose.yml        # 重写（无 CCR 无 socket-proxy）
│   ├── systemd/           # discover/enqueue/orchestrator/panel/backup units
│   └── bootstrap_nv8.sh   # 重写
├── sops/known_quirks.md   # v9 迁移 + Q-022/Q-023 (incident SOPs)
├── tests/                 # 全部 pytest，覆盖率 ≥ 80%
└── docs/
    ├── PLAN.md            # 此 v10 架构总图
    ├── OVERVIEW.md        # v9 迁移
    └── USAGE.md           # v9 迁移 + v10 差异说明
```

## 开发流程

按 `~/.ai-hub/rules/00-core.mdc § 开发工作流` 强制四阶段：

1. **架构设计** — [docs/PLAN.md](docs/PLAN.md)（已确认）
2. **测试用例设计** — 每个 PR 前先写测试清单
3. **编码实现** — feature branch + AI Reviewer + squash merge
4. **业务目标验收** — E2E run + INV 校验

PR 序列见 [docs/PLAN.md § 7 PR 序列](docs/PLAN.md#7--pr-序列每个-pr--400-行按依赖顺序)。

## 运维入口（开发完成后）

- 管理面板：`http://100.127.173.85:8090`（Tailscale 内）
- WeChat 反馈：经 `zero` 项目 `ai4wechat` + `alld` `POST /notify`
- 故障 SOP：[sops/known_quirks.md](sops/known_quirks.md)

## License

私有项目，未开源。
