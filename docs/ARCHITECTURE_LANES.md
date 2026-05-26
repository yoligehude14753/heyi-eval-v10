# heyi-eval-v10 三 lane 评测扩展架构

> **本文档定位**：v10 从「单源 HF 模型评测」演化为「三 lane 多源评测平台」的架构设计。
>
> 配套文档：
> · 现有 v10 运行时架构见 [`ARCHITECTURE.md`](ARCHITECTURE.md)
> · 现有 INV-1..23 见 [`INVARIANTS.md`](INVARIANTS.md)；本扩展新增的 INV-P / INV-S 见 [`INVARIANTS_LANES.md`](INVARIANTS_LANES.md)
> · 测试用例清单见 [`TEST_PLAN_LANES.md`](TEST_PLAN_LANES.md)
> · 操作手册扩充见 `RUNBOOK_NV8.md` 的 §16+ 章节（M2/M3 落地时补）
>
> **本文档是设计提案，不是运行时真相**。落地前必须经用户在 PR 中确认。

---

## 0 · 一句话概述

把 v10 从「针对 HuggingFace 模型」的单一评测管线，扩展成三条平行 lane：

| Lane | 评测目标 | 数据源 | 执行体 | 状态 |
|--|--|--|--|--|
| `model_lane` | HF 模型能力 | HuggingFace Hub | vLLM/SGLang/transformers_runner | **现有** 11-stage，行为不变 |
| `project_lane` | GitHub 火热项目实际可用性 | `duanyytop/agents-radar` daily manifest | M2.7 (yunwu) + Claude Code (m2b-claude-code 沙箱) | **新增** |
| `skill_lane` | SKILL.md 在 M2.7 上的真实有用性 | `~/.claude/skills/` + `~/.cursor/skills-cursor/` + agents-radar 抓的 anthropic/skills trending | M2.7 (yunwu) + Claude Code | **新增** |

三 lane 共享同一 `queue.jsonl`、`runs/` 落盘、`panel/` 展示、`yunwu` LLM 调用基础设施。新增统一执行器 `agent_driver/` 模块（M2.7 ↔ Claude Code 桥 + 沙箱池 + 标准化报告抽取 + 预算守护）。

---

## 1 · 设计目标与非目标

### 目标

1. **新增评测维度**：在「模型清单 × modality」之外，引入「火热项目 × 实测可用性」「skill × 实测产物」两条评测线
2. **复用基础设施**：queue / runs / panel / yunwu provider / oversize gate 思路 全部沿用
3. **沙箱安全**：M2.7 自主部署 GitHub 项目，必须有强隔离 + 资源配额 + 互污染防护
4. **标准化输出**：agent 自由发挥但报告字段固定，panel 能机械渲染
5. **可独立 disable**：env `HEYI_EVAL_LANES=model` 仅跑现有 lane，新增 lane 不影响线上

### 非目标

1. **不评测「项目质量」抽象指标**：只评测「在我们沙箱里 M2.7 能不能用起来」，不取代 GitHub stars / 用户评价
2. **不取代 model_lane**：模型本身评测仍走 HF 模型管线，project_lane 不重复测模型能力
3. **不做 GPU 项目**：项目执行默认 CPU-only；需要 GPU 的项目（如 fine-tuning 框架）标 `needs_gpu` 跳过
4. **不做 Web 自动化**：仅命令行 / Python / Node CLI 类项目；浏览器自动化 / GUI demo 留给后续

---

## 2 · 三 lane 全景图

```mermaid
flowchart TB
    subgraph sources [数据源]
        hf[HuggingFace Hub API]
        radar[agents-radar manifest.json daily]
        localSkills[~/.claude/skills + ~/.cursor/skills-cursor]
    end

    subgraph discover [discover/ daemons]
        d1[main.py 现有]
        d2[radar_ingest.py 新]
        d3[skill_local_scan.py 新]
    end

    subgraph queueBlock [queue.jsonl with lane field]
        q[lane: model | project | skill]
    end

    subgraph lanes [orchestrator/lanes/]
        m[model_lane 11-stage 现有]
        p[project_lane 5-stage FETCH PLAN AGENT_EXEC JUDGE REPORT]
        s[skill_lane 4-stage LOAD AGENT_EXEC JUDGE REPORT]
    end

    subgraph driver [agent_driver/ 新模块]
        ccr[ccr_bridge yunwu provider]
        pool[pool_manager m2b 容器池 1-3 个]
        run[exec_runner stream stdout + watchdog]
        rep[report_extractor JSON 围栏 schema]
        bg[budget_guard 50k token 15 min]
    end

    subgraph storage [data/]
        rm[runs/model/ 现有]
        rp[runs/project/ 新]
        rs[runs/skill/ 新]
        rc[radar_cache/ daily snapshots]
    end

    subgraph panelui [panel/server.py]
        u1[/candidates 现]
        u2[/projects 新]
        u3[/skills 新]
    end

    hf --> d1 --> q
    radar --> d2 --> q
    radar --> rc
    localSkills --> d3 --> q
    q --> m
    q --> p
    q --> s
    p --> driver
    s --> driver
    driver --> rp
    driver --> rs
    m --> rm
    rm --> u1
    rp --> u2
    rs --> u3
```

---

## 3 · Lane 抽象（最小重构）

### 3.1 `LaneSpec` dataclass

新文件 `orchestrator/lanes/__init__.py`：

```python
@dataclass(frozen=True)
class LaneSpec:
    name: Literal["model", "project", "skill"]
    target_kind: str
    stages: tuple[StageFn, ...]
    oversize_gate: Callable[[QueueRow], OversizeDecision] | None
    enqueue_dedup_window_h: int = 24
    runs_subdir: str  # e.g. "model", "project", "skill"
```

三 lane 各自一个 module 提供 `LANE_SPEC` 常量，`orchestrator/lanes/__init__.py` 注册到 `REGISTRY: dict[str, LaneSpec]`。

### 3.2 model_lane 重组（行为不变）

把现有 [`orchestrator/state_machine.py`](orchestrator/state_machine.py) 的 11-stage chain 搬到 `orchestrator/lanes/model_lane.py` 包成 `MODEL_LANE_SPEC`。

**硬约束**：必须有 `tests/test_lane_abstraction_fitness.py` 验证 model_lane 重组前后的 stage 顺序 / oversize gate / dedup 行为完全等价（参考 `tests/test_pr10_concept_split.py` 写法）。

### 3.3 main loop 路由

[`orchestrator/main.py`](orchestrator/main.py) 现有按 queue 行串行调度，新加一行：

```python
spec = LANE_REGISTRY[queue_row.get("lane", "model")]
spec.run(queue_row, cfg, store)
```

`lane` 字段在 queue 行缺失时默认 `model`，保持向后兼容（PR#17 oversize pre-filter 写入的旧 row 不需要 reformat）。

---

## 4 · `agent_driver/` 模块详解（M1 重点）

### 4.1 `ccr_bridge.py` — M2.7 ↔ Claude Code

**现状**：[`_tmp/m2b-claude-code/ccr-config.json`](../_tmp/m2b-claude-code/ccr-config.json) 已配 claude-code-router (ccr) + `strip-thinking` transformer，当前 provider 是 `heyi-glm` → GLM-5.1。

**改造**：在 ccr Providers 数组里**新增**一个 `yunwu` provider，不删 `heyi-glm`（保留 fallback）：

```jsonc
{
  "Providers": [
    { "name": "heyi-glm", "api_base_url": "http://10.10.11.198:10817/v1/chat/completions",
      "models": ["GLM-5.1"], "transformer": { "use": ["maxtoken", "reasoning", "strip-thinking"] } },
    { "name": "yunwu-m27", "api_base_url": "https://yunwu.ai/v1/chat/completions",
      "models": ["MiniMax-M2.7"], "api_key": "<from $YUNWU_GENERAL_KEY>",
      "transformer": { "use": ["maxtoken", "strip-thinking"] } }
  ],
  "Router": {
    "default": "yunwu-m27,MiniMax-M2.7",
    "background": "yunwu-m27,MiniMax-M2.7",
    "think": "yunwu-m27,MiniMax-M2.7"
  }
}
```

`agent_driver/ccr_bridge.py` 负责：
- 读 yunwu key（同 PR#18 的 `_resolve_engine_endpoint` 优先级链）
- 生成 ccr-config.json
- 写入 m2b 容器的 `/home/agent/.claude-code-router/config.json`
- 重启 ccr 进程

**模型可切**：env `HEYI_EVAL_AGENT_MODEL=yunwu-m27,MiniMax-M2.7` / `yunwu-k26,Kimi-K2.6` / `heyi-glm,GLM-5.1`，落到 `Router.default`。

### 4.2 `pool_manager.py` — 长驻容器池

**现状**：m2b-claude-code 已经是 docker-compose 长驻容器（`_tmp/m2b-claude-code/docker-compose.yml`）。

**改造**：
- 在 heyi (10.10.11.198) 上**长驻 1-3 个 m2b 容器**（默认 1，env `HEYI_EVAL_AGENT_POOL_SIZE` 可调）
- 每个容器独立 `workspace/` mount，容器之间无共享
- `pool_manager.acquire(run_id)` → 选一个空闲容器，在它 workspace 下创建 `$run_id/` 子目录，返回 handle
- `pool_manager.release(run_id)` → run 完 **整 `$run_id/` 子目录 rm -rf**，回到空闲池
- 容器健康检查：每 5 min 探活，宕了 `docker-compose up -d` 拉起
- 周期重启：每 50 runs 或 24h 整体重启容器，防 venv / pip cache 长期污染

### 4.3 `exec_runner.py` — 喂任务 + 抓输出

```python
def run_agent(
    run_id: str, task_prompt: str,
    extra_files: dict[str, str] | None = None,   # 注入 workspace 的预置文件
    timeout_s: float = 900.0,
    token_budget: int = 50000,
) -> AgentRunResult:
    ...
```

- 用 docker exec 把 `task_prompt` 写到 `workspace/$run_id/TASK.md`
- 启动容器内 `claude --no-interactive --workspace /home/agent/workspace/$run_id` 进程
- stream stdout/stderr 到 `runs/<lane>/<target_id>/<run_id>/agent.log`
- `budget_guard` 异步并发监控 ccr 的 token usage（ccr 自带 `LOG=true` 日志）
- watchdog 计时；超时 → `SIGTERM` ccr 进程，标 `outcome=timeout`

### 4.4 `report_extractor.py` — 强 schema 输出契约

agent **必须**在 stdout 最后一段输出标记围栏：

```
<<<HEYI_RUN_REPORT_JSON>>>
{"schema_version":"1.0", "lane":"project", ... }
<<<END>>>
```

`report_extractor`：
- 在 agent.log 找最后一对围栏标记
- 提取中间内容 → `json.loads` → 用 `jsonschema` 验证
- 不合规 → `outcome=report_parse_error`，原始内容存到 `runs/.../bad_report.txt`
- 合规 → 写到 `runs/.../report.json`

**禁止「自由发挥」式总结**：agent 可以在围栏外写过程描述（进 agent.log），但 panel 只读围栏内 JSON。

### 4.5 `budget_guard.py` — 预算闸门

- token 上限：默认 50k per run（input + output），可调
- 墙钟上限：默认 15 min per run
- 命中任一上限 → 立刻中断 + `outcome=budget_exceeded`
- 部分 step 已完成时，保留中间产物（agent.log + 已生成的 workspace 文件）；下次**不自动重试**（同 model_lane 的 fail-don't-retry 策略）

---

## 5 · 数据源摄入

### 5.1 agents-radar 摄入策略（**待你 PR 中确认的设计调整**）

**用户原选**：fork agents-radar 并本地 run。

**Plan 后发现**：agents-radar 是 TypeScript / pnpm 项目，[`package.json`](https://github.com/duanyytop/agents-radar/blob/master/package.json) + [`tsconfig.json`](https://github.com/duanyytop/agents-radar/blob/master/tsconfig.json) + Node-only fetcher。在 Python v10 内无法直接 import 它的 fetcher。

**Plan 推荐替代方案**（运维量更小）：

| 项 | fork + self-host (你原选) | manifest.json 订阅 (plan 推荐) |
|--|--|--|
| 数据完整性 | 完整 | 完整（[`manifest.json`](https://github.com/duanyytop/agents-radar/blob/master/manifest.json) 32KB 已结构化） |
| 运维负担 | 需装 Node + pnpm + 跑 GitHub Actions workflow / cron | 一次 HTTPS GET，零依赖 |
| 数据新鲜度 | 我们自己控（可任意频率） | 每天 08:00 CST 上游推（已足够） |
| 上游 schema 漂移防御 | fork 锁版本即可 | 需 `radar_schema_v1.json` 锁版本 + 漂移硬拒绝 |
| 完全停更风险 | 我们自己跑不停 | 上游停更 → fallback 到 `radar_cache/<yesterday>.json`，并切到本地 GitHub Trending HTML 兜底 fetcher |

**本 plan 默认采用 manifest.json 订阅 + schema 锁版本**。若你坚持 fork+self-host，请在 PR review 中说明，我改回。

### 5.2 skill 数据源

- **本地源**（`skill_local_scan.py`）：扫 `~/.claude/skills/` (20+) + `~/.cursor/skills-cursor/` (14)，每个目录的 SKILL.md 作为一个 skill 候选
- **远程源**（`radar_ingest.py` 复用）：从 agents-radar manifest 中 `claude_code_skills` 字段抓 anthropic/skills 仓库 trending skills
- **去重**：以 skill name + skill content hash 为唯一键；本地已装的同名优先（避免远程 fetch）

### 5.3 Queue 行 schema（向后兼容扩展）

```json
{
  "lane": "model | project | skill",
  "target_id": "string",
  "source": "hf | agents-radar | local-scan",
  "enqueued_at": "ISO8601",
  "metadata": {
    "// for project_lane": {
      "repo_url": "https://github.com/owner/repo",
      "stars": 12345,
      "primary_language": "Python",
      "topics": ["llm", "agent"],
      "needs_gpu": false
    },
    "// for skill_lane": {
      "skill_id": "mermaid",
      "skill_dir": "/Users/yoligehude/.claude/skills/mermaid",
      "description": "...from SKILL.md frontmatter..."
    },
    "// for model_lane (unchanged)": {
      "hf_id": "...",
      "tp_size_hint": 2
    }
  }
}
```

`lane` 字段缺失时默认 `model`，PR#17 留下的旧 row 不需要 reformat。

---

## 6 · 标准化报告 JSON Schema (v1.0)

```json
{
  "schema_version": "1.0",
  "lane": "project | skill",
  "target_id": "owner/repo or skill-id",
  "outcome": "pass | partial | fail | report_parse_error | budget_exceeded | timeout | sandbox_dead",
  "steps": [
    {
      "name": "string",
      "status": "ok | fail | skip",
      "duration_s": "number",
      "note": "string (optional, <= 500 chars)",
      "stdout_tail": "string (optional, last 2KB)",
      "artifacts": ["relative path under workspace/$run_id"]
    }
  ],
  "verdict": {
    "deploys": "boolean",
    "quickstart_works": "boolean",
    "core_features_demonstrated": ["string"],
    "blockers": ["string"]
  },
  "self_assessment_zh": "string (<= 1KB, 中文总结)",
  "follow_ups": ["string"]
}
```

约束（强 schema 由 [`jsonschema`](https://pypi.org/project/jsonschema/) 验证）：

- `outcome ∈ pass/partial/fail` 必须 由 agent 自己判，其余三态由系统在 agent 失控时强制赋值
- `self_assessment_zh` 必须中文（panel 直接显示，便于扫描）
- `steps[*].artifacts` 必须是 workspace 下相对路径，禁止绝对路径或 `..` 穿越
- `core_features_demonstrated` 至少 1 项才能 `outcome=pass`，否则强制降级 `partial`

---

## 7 · 沙箱安全约束（基于「long_lived_pool」决策）

| 约束 | 实现 |
|--|--|
| host filesystem 隔离 | m2b 容器只 mount `workspace/` 和 `logs/`，不挂任何其他 host 路径 |
| docker-in-docker 禁止 | 容器内**不挂 docker socket**（[`_tmp/m2b-claude-code/docker-compose.yml`](../_tmp/m2b-claude-code/docker-compose.yml) 现状已合规） |
| 网络白名单 | 容器可访问外网（git clone、pip、hf）；不能反向访问 heyi 主机的 :10814 / docker socket / SSH（compose `extra_hosts` 不暴露） |
| 资源配额 | 默认 8 cpu / 32 GB mem / 100 GB disk；docker-compose `deploy.resources.limits` 落地 |
| run 隔离 | 每 run 独占 `workspace/$run_id/`，run 完整目录 `rm -rf`；agent 严禁 `pip install --user` 或改容器全局 `~` 配置 |
| 互污染防御 | agent 在 `workspace/$run_id/` 下 `python -m venv` 起本地 venv，不污染容器 Python；周期重启容器（每 50 runs 或 24h） |
| GPU | 项目执行**不分配 GPU**；`metadata.needs_gpu=true` 的 repo 在 FETCH 阶段标 `needs_gpu` 拒绝（同 INV-23 oversize 思路） |

---

## 8 · Lane Stage 契约

### 8.1 project_lane (5 stages)

| Stage | 输入 | 输出 | 失败处理 |
|--|--|--|--|
| `FETCH` | `repo_url` from queue metadata | `workspace/$run_id/repo/` git cloned | git fail → `clone_failed`；repo > 5GB → `oversize` 拒绝；缺 README → `no_readme` |
| `PLAN` | repo + README | M2.7 给出测试计划 JSON（要部署哪些组件、跑哪个 demo、验证哪些核心特性） | M2.7 yunwu 429 → 退避 3 次后 `judge_unavailable` |
| `AGENT_EXEC` | 测试计划 + 沙箱 | agent.log + 围栏 JSON 报告 | budget_guard 中断 → `budget_exceeded`；容器宕 → `sandbox_dead` |
| `JUDGE` | report.json + agent.log | LLM judge 对 verdict 二次评分 + 把 blockers 翻译成中文 | judge 429 → 不阻断主流程，标 `judge_unavailable` |
| `REPORT` | 全部 | `runs/project/<owner>__<repo>/<run_id>/` 落盘 + panel 渲染 | - |

### 8.2 skill_lane (4 stages)

| Stage | 输入 | 输出 |
|--|--|--|
| `LOAD` | skill_id + skill_dir | 容器内 `~/.claude/skills/$skill_id/SKILL.md` 创建（容器内复用 Claude Code 原生 skill 加载） |
| `AGENT_EXEC` | "按 SKILL.md description 给一个代表性使用案例并执行" prompt | agent.log + 围栏 JSON |
| `JUDGE` | report.json | LLM judge 评分 |
| `REPORT` | 全部 | `runs/skill/<skill_id>/<run_id>/` 落盘 |

### 8.3 dedup / retry 策略（沿用 model_lane 哲学）

- 同 `(lane, target_id)` 24h 内已 `pass` → enqueue 时跳过
- 同 `(lane, target_id)` 24h 内已 `fail` → **不自动重试**（避免雪崩，需人工或 manual force）
- skill_lane 由于 skill 内容会随上游更新，dedup key 加 skill content sha256，内容变了视为新 target

---

## 9 · Panel 改造

[`panel/server.py`](panel/server.py) 现有 `/candidates` `/runs` `/results` tab。新增：

| Tab | 列 | 复用 |
|--|--|--|
| `/projects` | repo / outcome / deploys / quickstart_works / core_features count / 最近 run 时间 / self_assessment_zh 前 100 字 | PR#64 table toolkit (sort/filter/search) |
| `/skills` | skill_id / outcome / 最近 run 时间 / self_assessment_zh 前 100 字 / 跳转 SKILL.md 原文 | 同上 |
| `/runs/<run_id>` 详情页 | 加 `lane` 标签 + 围栏 JSON 渲染 + agent.log tail | 复用现有 run 详情页 |

---

## 10 · 关键 invariants

完整列表见 [`INVARIANTS_LANES.md`](INVARIANTS_LANES.md)。核心 5 条预览：

- **INV-P1**：project_lane oversize gate — repo size > 5GB 或缺 README → FETCH 拒绝，不进 AGENT_EXEC
- **INV-P2**：同 `(lane, target_id)` 24h 内已 pass → enqueue 跳过；24h 内已 fail → 不自动重试
- **INV-P3**：agent 末尾必须输出 schema-valid 围栏 JSON，否则 outcome 强制 `report_parse_error`，禁止「自由发挥」式总结
- **INV-P4**：budget_guard 中断必须 graceful — agent.log 完整、容器健康
- **INV-S1**：skill_lane 禁止 clone 任意 GitHub repo（防 SKILL.md 内嵌 `clone xxx` 把 skill 当 project 用），要测 repo 走 project_lane

---

## 11 · 分期里程碑

| M | 目标 | 估时 | 主要交付 |
|--|--|--|--|
| **前置** | PR#18 yunwu cutover merge | 0.5 天 | curator/showcase/deploy_repair 走 yunwu 的基础设施 |
| **M1** | `agent_driver/` 基础设施 | 3-4 天 | ccr_bridge + pool_manager + exec_runner + report_extractor + budget_guard，1 个 fake task 跑通 |
| **M2** | `project_lane` 端到端 | 3-4 天 | radar_ingest + project_lane 5-stage + panel `/projects`，1 个 trending repo 跑通 |
| **M3** | `skill_lane` 端到端 | 2-3 天 | skill_local_scan + skill_lane 4-stage + panel `/skills`，1 个本地 skill 跑通 |
| **M4** | lane 抽象 fitness test | 穿插于 M1-M3 | model_lane 重组前后 stage 顺序 / oversize gate / dedup 行为完全等价 |

每个 M 一个 PR，独立 reviewable，可独立 disable（env `HEYI_EVAL_LANES`）。

---

## 12 · 风险与回滚

| 风险 | 概率 | 影响 | 缓解 |
|--|--|--|--|
| M2.7 + Claude Code 桥（ccr）对 yunwu 不兼容 | 中 | M1 阻塞 | M1 D1 做 ccr 联通验证，不通 fallback 到 GLM-5.1 继续 M1 |
| agent 自主测试不可预测 → 报告失败率高 | 高 | project_lane 数据质量差 | report_extractor 严 schema + JUDGE 阶段二次 LLM 评判 + INV-P3 强制围栏 |
| 长驻容器互污染 | 中 | run 之间结果互扰 | INV-P 系列 + workspace/$run_id 独立 + 周期重启容器（每 50 runs 或 24h） |
| Yunwu 429 高频 | 中 | run 大量 `judge_unavailable` | budget_guard 退避策略 + 同 repo 当天最多 retry 1 次 |
| agents-radar 停更 / schema 变 | 低 | 项目源断流 | schema_v1 锁 + radar_cache fallback + 切到本地 GitHub Trending HTML 兜底 fetcher |
| 旧 queue.jsonl 没 lane 字段 | 低 | model_lane 误进 project/skill 路径 | 默认 `lane=model`，main loop 路由前显式断言 |

**回滚**：每 M 一个 PR，可独立 revert。env `HEYI_EVAL_LANES=model` 仅跑现有 lane，新增 lane 完全 disable，0 风险线上。

---

## 13 · 待你 PR 中确认的设计调整点

1. **agents-radar 摄入**：plan 推荐 manifest.json 订阅；你最初选 fork+self-host。本文档 §5.1 详述权衡，请定。
2. **report schema rigor**：plan 推荐 strict JSON 围栏 + parse_error 兜底。可选项：自由 markdown + 二次 LLM 提取（实现复杂、数据质量差）。
3. **吞吐 / 预算**：默认 5 projects/day + 3 skills/day，每 run 50k token + 15 分钟墙钟。可调，写到 env。
4. **lane 抽象**：plan 默认穿插实现（M1-M3 各 PR 内做局部抽象，M4 写 fitness test 保护）。可选项：先单独开 PR 把 model_lane 重组完，再做新 lane。
5. **agent 模型选**：plan 默认 M2.7（与 judge 一致）。可选项：K2.6 / GLM-5.1，写到 env。
6. **`agent_driver/` 是否独立 systemd unit**：plan 默认作为 orchestrator 子进程；可选项：独立 unit 便于运维。

确认后按 M1 → M2 → M3 顺序开 PR。
