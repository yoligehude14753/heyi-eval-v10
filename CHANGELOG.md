# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规约与
[Semantic Versioning](https://semver.org/lang/zh-CN/)；后续版本由
[release-please](https://github.com/googleapis/release-please) 根据 Conventional
Commits 自动维护。

## [Unreleased]

_（main 当前与最近一次 tag 对齐；新提交先落 `feat:` / `fix:` 即可。）_

## [0.2.0] — 2026-05-26

### 重大新增

- **三车道评测扩展（Three-Lane Evaluation）** — `feat(eval-lane)` (#19, #22, #23)
  在原有的 `model_lane`（HF 模型 11 阶段流水线）之外新增两条独立车道：
  - `project_lane`：从 [`duanyytop/agents-radar`](https://github.com/duanyytop/agents-radar) 摄取 GitHub 项目候选，在 m2b-claude-code 沙箱里由 MiniMax-M2.7（Yunwu API）通过 Claude Code CLI 完整跑一遍 deploy + smoke + verdict。
  - `skill_lane`：从本地 `~/.claude/skills/` 与 `~/.cursor/skills-cursor/` 镜像 SKILL.md，由同一 agent 演练触发场景 / 应答 / 缺口分析 / 打分。
  - 物理隔离的 store + schema（INV-L0、INV-P1..P8、INV-S1..S3）；agent 与本机文件系统、宿主 docker socket 完全隔绝。
- **agent_driver bridge** — `feat(eval-lane-m1)` (#22)
  新增 `agent_driver/`：`ccr_bridge`（CCR 配置 + 注入容器）、`pool_manager`（容器池 + 复用）、`exec_runner`（任务编排 + 报告抽取）、`report_extractor`（围栏 JSON 校验）、`budget_guard`（token + wall-clock 预算）、`schema`（`RunReport` 单一数据契约）。
- **Yunwu API 全量切入** — `feat(eval-T70)` (#18)
  curator / showcase / deploy_repair / judge 四个 LLM 消费点全部切到 Yunwu（MiniMax-M2.7 / GLM-5.1 / K2.6 可选），不再依赖本机 GPU 跑 LLM；保留 `heyi-glm` fallback。
- **oversize 候选预过滤** — `feat(enqueue)` (#17, #16)
  candidate enqueue 阶段直接拒绝超参模型（>= 70B 或匹配已知 oversize 模式），不进入 STAGE_MODEL 浪费磁盘 / GPU。

### 修复

- `fix(agent_driver)` (#28) — wall-clock 静默挂死：上游 429 storm 让 claude stdout 整窗口为空，watchdog 现在主动 `pkill -TERM` 容器内 claude 进程让 docker exec 流退出，run 落地为 EXCEEDED_WALL_CLOCK。
- `fix(agent_driver)` (#27) — schema 接受 `step.status='partial'`：M2.7 常把"半成功"的 step 标 partial，旧 enum 只允许 ok/fail/skip 让整个 run 被打成 report_parse_error，丢掉真实信号。
- `fix(orchestrator)` (#26) — `project run` / `skill run` 自动探测 m2b 容器的 workspace mount，替代写死 `/tmp/heyi-*-ws`；环境变量 `HEYI_*_WORKSPACE` 可覆盖。
- `fix(stager)` (#25) — `huggingface_hub` 0.23+ 行为变化：`snapshot_download` 在某些情况下把权重落到 hub cache 而非 `local_dir`；stager 跟随真实返回路径并补 symlink，DEPLOY 不再因空目录失败。
- `fix(e2e)` (#24) — `HEYI_EVAL_E2E_FORCE=1` 现在被 conftest 正确识别；`prod_container_snapshot` fixture 从 function scope 提到 module scope。

### 真机验证（heyi 节点）

| 车道 | 总数 | PASS | PARTIAL | FAIL | PARSE_ERROR | TIMEOUT | OVERSIZE |
|---|---|---|---|---|---|---|---|
| skill_lane | 39 | **24** | **11** | 1 | 3 | — | — |
| project_lane | 19 | **4** | **1** | 5 | 6 | 2 | 1 |

- skill_lane 有效产出率 **89.7%**（PASS+PARTIAL / 总数）
- project_lane 有效产出率 **52.6%**（结构化结果 / 总数）
- 总耗时 2 小时 4 分钟跑完全量 58 个评测

### 文档

- 新增 [`docs/ARCHITECTURE_LANES.md`](docs/ARCHITECTURE_LANES.md) — 三车道运行时拓扑（PR #19）
- 新增 [`docs/INVARIANTS_LANES.md`](docs/INVARIANTS_LANES.md) — lane 红线（INV-L0 / INV-P*/ INV-S*）
- 新增 [`docs/TEST_PLAN_LANES.md`](docs/TEST_PLAN_LANES.md) — 三车道测试计划

## [0.1.0] — 2026-05-21..2026-05-22

> v10 主干由 PR#1..#12 一次性建立；本节按主题归并所有早期 PR。

### 架构重写

v9 (`heyi-eval-v9`) 在 2026-05-21 出现 cc-agent 数据丢失事件（sqlite + discover/curated 被 CLEANUP 阶段的 Claude bash 工具删除）。v10 是基于 incident 教训的架构级重写：

- cc-agent 涉及阶段从 4 个收敛到 1 个（只剩 SHOWCASE）
- cc-agent 移除 shell / docker socket / 全目录挂载
- 数据落到 `/home/ai/heyi-eval-data/`，备份独立到 `/home/ai/heyi-eval-backups/`
- 信任域明确化：PROD / ORCH / EVAL / AGENT-SANDBOX / DATA

详见 [`docs/PLAN.md`](docs/PLAN.md) 与 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

### 主要 PR

- `chore(pr1)` (#1) — 从 v9 平移静态模块（cp + path-adjust only）
- `feat(heyi-engine)` (#2) — 产线 LLM 客户端 + 自动 `/v1/models` 发现
- `feat(pr3)` (#3) — Python DEPLOY / READY_WAIT / CLEANUP via docker SDK
- `feat(pr4)` (#4) — Python CAPABILITY 阶段 + 13 类评分器
- `feat(pr5)` (#5) — 受限的 in-process SHOWCASE runner（无 shell / 无 docker）
- `feat(pr6)` (#6) — 数据保护：30min rsync + 7d 保留 + Mac 每夜镜像
- `chore(pr7a)` (#7) — 清理 v9 CCR / cc-agent docker spawn 残留
- `feat(pr7b)` (#8) — systemd units + `bootstrap_nv8.sh`
- `test(pr8)` (#9) — E2E pipeline harness + INV-1/4/12/13 静态守护 + nv8 runbook
- `refactor(pr10)` (#10) — 产线 LLM 与评测 LLM 概念拆分
- `feat(pr11)` (#11) — GPU 隔离 + 资源不足时 graceful-skip
- `docs(pr12)` (#12) — L3 runbook for graceful-skip drill + bootstrap python fallback
- `feat(pr14)` (#14) — agent sandbox bridge + M2.7 API + oversize gate + 真机 e2e

### 23 条红线（INV-1..23）

完整列表见 [`docs/INVARIANTS.md`](docs/INVARIANTS.md)。关键例子：

- **INV-1**：评测 GPU 必须空载才能 DEPLOY
- **INV-9**：备份目录在数据根之外
- **INV-12 / INV-13**：评测流水线不得通过 docker 控制 PROD 容器
- **INV-14**：跨域评分必须经 LLM judge
- **INV-16..22**：AGENT-SANDBOX 沙箱（用户 / ACL / sudoers / socket-proxy / mount 范围）

---

[Unreleased]: https://github.com/yoligehude14753/heyi-eval-v10/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yoligehude14753/heyi-eval-v10/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yoligehude14753/heyi-eval-v10/releases/tag/v0.1.0
