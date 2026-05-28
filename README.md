# heyi-eval-v10

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yoligehude14753/heyi-eval-v10/actions/workflows/ci.yml/badge.svg)](https://github.com/yoligehude14753/heyi-eval-v10/actions/workflows/ci.yml)

**自动化评测 pipeline**，在 nv8-6000 / heyi 节点上常驻运行。当前支持 **三条独立车道**：

| 车道 | 评测对象 | 上游来源 | Agent | 状态 |
|---|---|---|---|---|
| `model_lane` | Hugging Face 模型 | HF Hub 日频发现 | vLLM/transformers + LLM judge | 生产 |
| `project_lane` | GitHub 开源项目 | `duanyytop/agents-radar` | MiniMax-M2.7 + Claude Code CLI | 生产 |
| `skill_lane` | Claude / Cursor Skills | 本地 `~/.claude/skills`、`~/.cursor/skills-cursor` | MiniMax-M2.7 + Claude Code CLI | 生产 |

> 想直接看现状架构：
> - **Model lane（11 阶段流水线 + cc-agent 沙箱）** → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> - **Project + Skill lane（agent_driver / m2b-claude-code / Yunwu API）** → [`docs/ARCHITECTURE_LANES.md`](docs/ARCHITECTURE_LANES.md)
> - **红线 / 不变量**：[`docs/INVARIANTS.md`](docs/INVARIANTS.md) + [`docs/INVARIANTS_LANES.md`](docs/INVARIANTS_LANES.md)
> - **运维 / 操作**：[`docs/USAGE.md`](docs/USAGE.md)、[`docs/RUNBOOK_NV8.md`](docs/RUNBOOK_NV8.md)
> - **变更日志**：[`CHANGELOG.md`](CHANGELOG.md)

## 最近一次真机批跑成绩（v0.2.0, heyi 节点, 2026-05-26）

| 车道 | 总数 | PASS | PARTIAL | FAIL | PARSE_ERROR | TIMEOUT | OVERSIZE |
|---|---|---|---|---|---|---|---|
| `skill_lane` | 39 | **24** | **11** | 1 | 3 | — | — |
| `project_lane` | 19 | **4** | **1** | 5 | 6 | 2 | 1 |

- skill_lane 结构化产出率 **89.7%**（PASS + PARTIAL / 总数）
- project_lane 结构化产出率 **52.6%**
- 总耗时 **2 小时 4 分钟**跑完全量 58 个评测，全程无人工干预

## 与 v9 的关系

v9 (`heyi-eval-v9`) 在 2026-05-21 出现 cc-agent 数据丢失事件（sqlite + discover/curated 被 CLEANUP 阶段的 Claude bash 工具删除）。根因：cc-agent 同时拥有 docker socket 访问权 + `/heyi-eval-data:/workspace` 全目录读写挂载 + Claude `Bash` 工具未禁用。

**v10 是基于 incident 教训的架构级重写**，不是补丁。详见 [`docs/PLAN.md`](docs/PLAN.md)。

| 维度 | v9 | v10 |
|---|---|---|
| cc-agent 涉及阶段 | 4 个 (DEPLOY, CAPABILITY, SHOWCASE, CLEANUP) | 1 个 (SHOWCASE only) |
| cc-agent 是否有 shell | 是 (bash) | **否** |
| cc-agent 是否有 docker socket | 是 (socket-proxy) | **否** |
| cc-agent mount 范围 | 整个 `heyi-eval-data` rw | 仅 `runs/<run_id>/showcase/` rw + 3 个 metadata 文件 ro |
| LLM 接入方式 | CCR + 写死 model name | `heyi_engine` client 自动探测 `:10814/v1/models` |
| 数据备份 | 无 | 30min rsync + 7d 保留 + Mac 每夜镜像 |

## 信任域

- **PROD** — 由 `OrchestratorConfig.prod_engine_container` 指定的产线 vLLM 容器（默认 `minimax`，GPU 0-3，`:10814`）。评测流水线**只可 HTTP 访问**，不得 docker-控制（INV-1/INV-12/INV-13）
- **ORCH** — Python orchestrator + heyi_engine client + curator + discover + panel。拥有 docker socket 与 `/home/ai/heyi-eval-data/` 全权
- **EVAL** — `e9-*` 前缀临时容器（默认 GPU 5-7，`:18200`）。orchestrator 创建/销毁，沙箱 cc_agent 只能 HTTP 访问
- **AGENT-SANDBOX** — `model_lane` 的 cc_agent showcase 子进程；`project_lane` / `skill_lane` 的 m2b-claude-code 容器。沙箱 user `heyi-eval-agent`，仅 `runs/<run_id>/` rw + 3 个 metadata 文件 ro；无 shell（model_lane）/ 受限 shell（project + skill lane）（INV-16..INV-22 + INV-P*/S*）
- **DATA** — `/home/ai/heyi-eval-data/`（orchestrator 全权，沙箱部分 ro）+ 备份 `/home/ai/heyi-eval-backups/`（INV-9：在数据根之外）

完整边界与守护机制见 [`docs/ARCHITECTURE.md §1`](docs/ARCHITECTURE.md#1--信任域) 与 [`docs/INVARIANTS.md`](docs/INVARIANTS.md)。

## 目录结构（当前实际）

```
heyi-eval-v10/
├── orchestrator/          # 11 阶段状态机 + 阶段执行 + 故障自愈 + 队列消费
│   ├── state_machine.py   # STAGES_IN_ORDER（权威阶段定义）
│   ├── stages.py / stages_py.py  # 阶段分发；DEPLOY/READY_WAIT/CLEANUP 用 docker SDK
│   ├── capability.py      # 13 类评测 + 评分器注册
│   ├── perf_bench.py      # PERF_BENCH（TTFT / TPS / VRAM）
│   ├── llm_judge.py       # INV-14 跨域 VLM 评分
│   ├── deploy_repair.py   # PR#33 规则式自愈
│   ├── cache_evictor.py   # LRU 缓存逐出
│   ├── project_lane.py    # ★ v0.2.0 — project lane 状态机
│   ├── skill_lane.py      # ★ v0.2.0 — skill lane 状态机
│   └── main.py            # CLI：model / project / skill 三种 subcommand
├── agent_driver/          # ★ v0.2.0 — M2.7 + Claude Code CLI 桥接
│   ├── ccr_bridge.py      # ccr 配置 + 注入到 m2b 容器
│   ├── pool_manager.py    # 容器池 + 复用 + lifecycle
│   ├── exec_runner.py     # 任务编排 + budget watchdog
│   ├── report_extractor.py# 围栏 JSON 抽取与校验
│   ├── budget_guard.py    # token + wall-clock 预算
│   └── schema.py          # RunReport 单一数据契约
├── discover/              # HF Hub 日频发现 + radar_ingest + skill_local_scan + enqueue policy
├── curator/               # README → 结构化 JSON（PROD LLM）
├── heyi_engine/           # 产线 LLM 客户端 + 自动 /v1/models 发现 + Yunwu provider
├── transformers_runner/   # ASR/TTS/diffusers/VLM 容器入口（受 INV-15 保护）
├── panel/                 # 只读 HTTP 面板 :8090 / :8888（含手动 enqueue + lane 视图）
├── cc_agent/              # 仅 model lane SHOWCASE 计划/打分；无 shell 无 docker
├── backup/                # 30min rsync + 7d 保留
├── deploy/                # systemd 9 unit + agent-sandbox ACL + env.example
├── scripts/               # bootstrap_nv8.sh / verify_24h_timer.sh / drill_project_lane.py 等
├── sops/                  # 阶段产物 JSON Schema + 已知 quirk 库
├── tools/                 # 一次性运维 CLI（evict_eval_cache.py 等）
├── tests/                 # pytest（~96 agent_driver + ~955 model lane；含 INV 守护 + e2e）
└── docs/
    ├── ARCHITECTURE.md       # model lane 现状
    ├── ARCHITECTURE_LANES.md # ★ project + skill lane 拓扑
    ├── INVARIANTS.md         # 23 条红线
    ├── INVARIANTS_LANES.md   # ★ lane 红线
    ├── TEST_PLAN_LANES.md    # ★ lane 测试计划
    ├── USAGE.md              # 操作手册
    ├── RUNBOOK_NV8.md        # NV8 真机演练
    ├── PLAN.md               # 启动期设计（历史）
    └── _archive/             # PR#1..#36 历史测试计划/报告快照
```

> 部署形态：**仅 systemd**（无 docker-compose）。评测容器 `e9-*` 由 orchestrator 在 DEPLOY 阶段动态 spawn；`m2b-claude-code` 容器常驻，agent_driver 通过容器池复用。

## Model lane 11 阶段流水线

权威定义：`orchestrator/state_machine.py::STAGES_IN_ORDER`

```
DISCOVER → CURATE → METADATA → ENGINE_SELECT → STAGE_MODEL          ← run-level checkpoint
                                                    ↓
        DEPLOY → READY_WAIT → CAPABILITY → PERF_BENCH → SHOWCASE → CLEANUP   ← stage-level
```

每阶段输入 / 输出 / 失败语义见 [`docs/ARCHITECTURE.md §3`](docs/ARCHITECTURE.md#3--11-阶段流水线)。

## Project + Skill lane 流程

```
project: radar_ingest → enqueue → preflight (oversize/no-readme gate)
                                ↓
                          m2b-claude-code + ccr → Yunwu MiniMax-M2.7
                                ↓
                          run agent → extract fenced JSON → judge → store

skill:   skill_local_scan → enqueue → local_gate (clone-attempt detector)
                                ↓
                          (same agent pipeline as project)
                                ↓
                          store with verdict / demos
```

完整拓扑与守护见 [`docs/ARCHITECTURE_LANES.md`](docs/ARCHITECTURE_LANES.md)。

## Quick Start（开发者本机 / 沙箱机器）

```bash
# 1. 安装
git clone https://github.com/yoligehude14753/heyi-eval-v10.git
cd heyi-eval-v10
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. 配置（复制 env 模板，填 Yunwu API key 等）
cp deploy/env.example /etc/heyi-eval-v10/env
# 编辑该文件，最少配置：YUNWU_GENERAL_KEY、HEYI_EVAL_DATA

# 3. 启动 m2b-claude-code 容器（project/skill lane 必须）
docker compose -f deploy/m2b-claude-code/docker-compose.yml up -d

# 4. 摄取候选
python -m discover.radar_ingest          # GitHub agents-radar → project_candidates.jsonl
python -m discover.skill_local_scan      # 本机 skills → skill_candidates.jsonl

# 5. 跑评测
python -m orchestrator skill enqueue claude-user/mermaid
python -m orchestrator skill run --container m2b-claude-code --limit 5

python -m orchestrator project enqueue anthropics/claude-code
python -m orchestrator project run --container m2b-claude-code --limit 5

# 6. 查看结果
python -m orchestrator skill status
python -m orchestrator project status
python -m panel.server                   # http://127.0.0.1:8090
```

## 开发流程

按 `~/.ai-hub/rules/00-core.mdc § 开发工作流` 强制四阶段：

1. **架构设计** — [`docs/PLAN.md`](docs/PLAN.md)（已确认）
2. **测试用例设计** — 每个 PR 前先写测试清单
3. **编码实现** — feature branch + AI Reviewer + squash merge
4. **业务目标验收** — E2E run + INV 校验

PR 序列见 [`docs/PLAN.md § 7 PR 序列`](docs/PLAN.md#7--pr-序列每个-pr--400-行按依赖顺序)，主干变更见 [`CHANGELOG.md`](CHANGELOG.md)。

## 运维入口

- 管理面板：`http://<NV8_HOST_IP>:8090`（model lane）/ `:8888`（lane API：`/api/skill/runs`、`/api/project/runs`）
- 产线 LLM 端点：`http://127.0.0.1:10814/v1`（heyi_engine）
- 评测 LLM 端点：`http://127.0.0.1:18200/v1`（仅 DEPLOY 后存在）
- 手动 enqueue：`python -m discover.main enqueue --limit 5`，或面板 `POST /api/enqueue`
- 缓存逐出：`python tools/evict_eval_cache.py --quota-gb 200 --apply`
- 24h 计时器自检：`bash scripts/verify_24h_timer.sh`
- 已知 quirk：[`sops/known_quirks.md`](sops/known_quirks.md)

## 贡献

PR 欢迎。请遵守 Conventional Commits（`feat:` / `fix:` / `docs:` / `chore:` / `feat!:` / `BREAKING CHANGE:`），release-please 会自动生成下一版本号与 CHANGELOG 条目。

## License

MIT — 见 [LICENSE](LICENSE)。

> 部署位置 / 主机 IP 在文档与脚本里以占位符出现（`<NV8_HOST_IP>` / `<NV8_TAILNET_IP>` / `<NV8_HOSTNAME>`）。本仓库不预设具体的部署网络拓扑，复用时按你自己的环境替换或通过环境变量覆盖（见 `deploy/env.example`）。
