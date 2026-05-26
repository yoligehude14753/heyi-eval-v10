# INVARIANTS_LANES — project_lane / skill_lane 不变量

> 本文档定位：[`ARCHITECTURE_LANES.md`](ARCHITECTURE_LANES.md) 引入的两条新评测 lane 的红线列表。
> 编号约定：**INV-P**\* 表示 project_lane 专属，**INV-S**\* 表示 skill_lane 专属，**INV-L**\* 表示三 lane 共享。
> 现有 INV-1..INV-23（model_lane）见 [`INVARIANTS.md`](INVARIANTS.md)；本文档**仅追加**新 lane 的不变量，不修改已有编号。

## 设计原则

1. **沿用 model_lane 的不变量哲学**：每条 INV 一句话 + 静态守护（test）+ 运行时守护（runtime assert / gate）
2. **绝对不复用 INV-1..23 的编号**，避免历史回溯混乱
3. **三 lane 共享的约束放 INV-L\***，避免每 lane 重复

---

## 总览表

| ID | 类别 | lane | 一句话 | 静态守护 | 运行时守护 |
|--|--|--|--|--|--|
| **INV-L1** | 路由 | 全部 | queue 行 `lane` 字段缺失 → 默认 `model`，但 main loop 路由前显式 `assert lane in REGISTRY` | `tests/test_lane_routing.py` | `orchestrator/main.py` 路由前断言 |
| **INV-L2** | 数据 | 全部 | `runs/<lane>/<target_id>/<run_id>/` 路径**不可**互相穿越；写入路径必须 `Path.resolve()` 后在 `data_root/runs/<lane>` 子树内 | `tests/test_lane_storage_isolation.py` | `lanes/__init__.py` 路径解析 helper |
| **INV-L3** | 抽象 | model | model_lane 重组（11-stage 搬到 `lanes/model_lane.py`）前后行为完全等价（stage 顺序、oversize gate、dedup 窗口、event 序列） | `tests/test_lane_abstraction_fitness.py`（fitness function） | — |
| **INV-L4** | 资源 | 全部 | env `HEYI_EVAL_LANES` 受控开关：未列出的 lane main loop 拒绝调度（用于线上紧急 disable 新 lane） | `tests/test_lane_disable_switch.py` | `orchestrator/main.py` 调度前查 env |
| **INV-P1** | oversize | project | project_lane oversize gate — repo `size_kb`（GitHub API 返回） > 5 GB 或 README 缺失 → FETCH 拒绝，标 `oversize` 或 `no_readme`，不进 AGENT_EXEC | `tests/test_project_lane_oversize.py` | `lanes/project_lane.py::_oversize_gate` |
| **INV-P2** | 去重 | project | 同 `(lane=project, target_id=owner/repo)` 24h 内已 `pass` → enqueue 跳过；24h 内已 `fail` → **不自动重试**，需 manual force | `tests/test_project_lane_dedup.py` | `discover/radar_ingest.py::_dedup_filter` |
| **INV-P3** | schema | project + skill | agent stdout 末尾**必须**有 `<<<HEYI_RUN_REPORT_JSON>>>...<<<END>>>` 围栏 + schema 合规；否则 outcome 强制 `report_parse_error`，禁止 panel 渲染「自由总结」 | `tests/test_report_extractor.py` | `agent_driver/report_extractor.py::extract_or_fail` |
| **INV-P4** | 预算 | project + skill | budget_guard 中断（超 token / 超时）必须 graceful：agent.log 完整落盘 + 容器健康 + workspace `$run_id/` 子目录可清理 | `tests/test_budget_guard.py` + 沙箱 drill | `agent_driver/budget_guard.py::interrupt` |
| **INV-P5** | GPU 隔离 | project | project_lane 执行**不分配 GPU**；queue metadata `needs_gpu=true` 的 repo 在 FETCH 阶段标 `needs_gpu` 拒绝，绝不与 model_lane 抢 GPU 资源 | `tests/test_project_lane_no_gpu.py` | `lanes/project_lane.py::_needs_gpu_gate` |
| **INV-P6** | 沙箱·FS | project + skill | agent 在容器内仅能写 `workspace/$run_id/` 子目录；禁止 `pip install --user`、禁止改 `/home/agent/.claude/` 全局配置、禁止挂载 docker socket | `tests/test_agent_sandbox_acl_static.py` | docker-compose volume mount 限定 + 容器内 ACL |
| **INV-P7** | 沙箱·NET | project + skill | agent 可访问外网（git / pip / hf）；不可反向访问 heyi 主机 `:10814` / `:2375` (docker daemon) / `:22` (ssh)；compose `extra_hosts` 不暴露 host.docker.internal 到 host 服务 | `tests/test_agent_sandbox_network_static.py` | docker-compose `extra_hosts` + iptables drill |
| **INV-P8** | 重启 | project + skill | 长驻 m2b 容器每 50 runs **或** 24h（取先到）整体 `docker-compose down && up -d`，并清空 workspace 顶级残留；避免长期 pip cache / venv 污染 | `tests/test_pool_recycle.py` | `agent_driver/pool_manager.py::maybe_recycle` |
| **INV-S1** | 隔离 | skill | skill_lane 禁止 clone 任意 GitHub repo（防 SKILL.md 内嵌 `git clone xxx` 把 skill 当 project 用）；要测 repo 必须走 project_lane | `tests/test_skill_lane_no_clone.py`（静态扫 agent.log） | `agent_driver/exec_runner.py::skill_mode_guard` |
| **INV-S2** | dedup | skill | skill dedup key = `(skill_id, sha256(SKILL.md))`；skill 内容变了视为新 target，允许重新跑 | `tests/test_skill_lane_dedup.py` | `discover/skill_local_scan.py::_compute_dedup_key` |
| **INV-S3** | 数据完整性 | skill | skill 必须有 SKILL.md frontmatter + 至少一段 description；缺则 enqueue 时拒绝（标 `incomplete_skill`） | `tests/test_skill_lane_quality_gate.py` | `discover/skill_local_scan.py::_validate_skill` |

---

## 详细说明

### INV-L1 路由完整性

queue 行可能来自多个生产者：
- `discover/main.py`（HF model lane，PR#17 以前的行 lane 字段缺失）
- `discover/radar_ingest.py`（project lane）
- `discover/skill_local_scan.py`（skill lane）
- 手工写入

**约束**：`orchestrator/main.py` 主循环每次 pop queue 行后，必须显式：

```python
lane_name = row.get("lane", "model")
assert lane_name in LANE_REGISTRY, f"unknown lane: {lane_name!r} in row {row.get('target_id')}"
spec = LANE_REGISTRY[lane_name]
```

**静态守护**：`tests/test_lane_routing.py`
- 单测：未知 lane 字段触发 assert
- 单测：缺失 lane 字段路由到 model
- 集成：三 lane 混合 queue 各自路由到对应 LaneSpec

### INV-L2 路径隔离

每 lane 的 `runs/` 子目录 严格不可互穿。`agent_driver` 写入 artifact 时：

```python
def _safe_run_path(data_root: Path, lane: str, target_id: str, run_id: str) -> Path:
    expected = (data_root / "runs" / lane / _sanitize(target_id) / run_id).resolve()
    if not str(expected).startswith(str((data_root / "runs" / lane).resolve())):
        raise ValueError("path escape attempt")
    return expected
```

**静态守护**：`tests/test_lane_storage_isolation.py`
- 单测：恶意 `target_id="../skill/x"` 必须抛 ValueError
- 单测：符号链接 target_id 同样抛错

### INV-L3 model_lane 回归保护（fitness function）

model_lane 11-stage 从 `state_machine.py` 搬到 `lanes/model_lane.py` 时**只是文件重组**，行为完全等价。

**fitness function**（`tests/test_lane_abstraction_fitness.py`）：

```python
def test_model_lane_stage_order_unchanged():
    """重组后 model_lane stage 顺序必须与历史 11-stage 完全相同。"""
    from orchestrator.lanes.model_lane import MODEL_LANE_SPEC
    expected = [
        "DISCOVER", "CURATE", "METADATA", "ENGINE_SELECT", "STAGE_MODEL",
        "DEPLOY", "READY_WAIT", "CAPABILITY", "PERF_BENCH", "SHOWCASE", "CLEANUP",
    ]
    assert [s.name for s in MODEL_LANE_SPEC.stages] == expected
```

参考写法：现有 `tests/test_pr10_concept_split.py` 已经在做类似的 stage-order 保护。

### INV-L4 紧急 disable 开关

env `HEYI_EVAL_LANES=model,project`（逗号分隔）控制 main loop 调度哪些 lane。
- 未设置 → 三 lane 全开（默认）
- 仅 `model` → 紧急回滚，仅跑现有 model 评测，新 lane 静默跳过（不报错，避免 noise）
- `model,skill` → project 暂停（比如 agents-radar 上游 schema 漂移期间）

**runtime 守护**：`orchestrator/main.py`

```python
enabled = set((os.environ.get("HEYI_EVAL_LANES", "model,project,skill")).split(","))
if lane_name not in enabled:
    log.info("lane %s disabled by env, skipping %s", lane_name, target_id)
    continue
```

---

### INV-P1 project oversize gate

repo size 上限：5 GB（解压后估算，GitHub API `size` 字段是 KB）。

```python
def _oversize_gate(row: QueueRow) -> OversizeDecision:
    meta = row.get("metadata", {})
    size_kb = int(meta.get("repo_size_kb", 0))
    if size_kb > 5 * 1024 * 1024:  # 5 GB
        return OversizeDecision(rejected=True, reason="oversize", detail=f"{size_kb} kb")
    if not meta.get("has_readme", True):
        return OversizeDecision(rejected=True, reason="no_readme", detail=None)
    return OversizeDecision(rejected=False)
```

灵感来自 PR#16/PR#17 的 INV-23 oversize gate（model_lane 用 `_extract_b` 估 param count + tp_size），思路一致：**早期拒绝**，不让显然不可行的 candidate 浪费 token 预算。

### INV-P5 GPU 隔离

project_lane 不分配 GPU 是硬约束，理由：
- 与 model_lane 共享 GPU 池会引起资源争抢、影响 model 评测吞吐
- 大部分 GitHub trending 项目（agent / CLI / RAG）不需要 GPU，要 GPU 的（fine-tuning / 大模型 inference）该走 model_lane 评测
- 沙箱容器 GPU 直通本身风险高，宁可不开

**runtime 实现**：m2b-claude-code docker-compose 不声明 `runtime: nvidia`，agent 容器内 `nvidia-smi` 不可用，agent 自然写不出依赖 GPU 的测试方案。

### INV-P3 强 schema 报告（最重要的执行契约）

**为什么强**：agent 自由发挥意味着输出格式不固定。如果允许 markdown 自由总结：
- panel 没法机械渲染
- 二次 LLM 提取成本高、错误率高
- 不同 run 之间无法可靠比较

**实现**：[`agent_driver/report_extractor.py`](../agent_driver/report_extractor.py) 用 `<<<HEYI_RUN_REPORT_JSON>>>` 和 `<<<END>>>` 围栏（参考 v10 现有 `cc_agent/showcase_runner.py` 的输出契约）+ `jsonschema.validate`。

**单测必覆盖**：
- 围栏缺失 → `report_parse_error`
- 围栏内 JSON 无效 → `report_parse_error`
- JSON 有效但缺必填字段 → `report_parse_error`
- 字段类型错（e.g., `verdict.deploys` 是字符串而非 bool）→ `report_parse_error`
- 合规 → 解析到 `RunReport` dataclass

---

### INV-S1 skill 不可 clone

防御场景：恶意 SKILL.md 内嵌 `git clone https://github.com/big/repo.git && do_evil`，借 skill_lane 绕过 project_lane oversize gate。

**实现层**：
1. `exec_runner.py::skill_mode_guard`：skill_lane 启动 agent 时，预置 prompt 显式约束「禁止 `git clone`、`docker pull`、`pip install <large-package>`」
2. 容器内静态扫 agent.log：发现 `git clone` 指令立即中断（agent 即使无视约束也无法实际下载，因为 INV-P7 网络白名单未禁 GitHub，靠静态扫兜底）
3. 单测：构造恶意 SKILL.md 触发 guard

### INV-S2 skill dedup with content hash

skill 内容会因为上游推 update 而变化（Anthropic skills repo 每周 push）。

**dedup key**：`f"{skill_id}:{sha256(SKILL.md前 16KB)}"`

skill_id 不变但内容变了 → 视为新 target，允许重跑（这是 skill_lane 与 project_lane 不同的核心点）。

---

## 守护点回顾

每条 INV 至少有两道防线：

| 防线 | 工具 |
|--|--|
| 静态测试 | `tests/test_*.py` 不依赖外部网络 / docker，CI 必须通过 |
| 运行时 assert | lane 各阶段开始 / 结束时显式校验；违反 = stage 失败 + outcome 标对应 error_kind |
| （可选）真机 drill | `drills/` 下脚本，仿照 v10 现有 `drills/attack_delete_store.sh` 等，部署到 NV8/heyi 时跑一遍 |

---

## 与现有 INV-1..INV-23 的关系

| 现有 INV | 与新 lane 的关系 |
|--|--|
| INV-1..15 (PROD/ORCH/EVAL 隔离) | model_lane 专属，新 lane 不触发（不操作 prod_engine / e9-\* 容器） |
| INV-16..22 (cc_agent 沙箱) | **理念复用**：新 lane 的 m2b 容器复用同样的沙箱哲学，但实现是新的 docker-compose 配置 |
| INV-23 (oversize gate) | **思路复用**：project_lane INV-P1 / INV-P5 是同一思路在新 lane 的应用 |

---

## 待补充

以下 INV 在 M2 / M3 落地时根据真机经验补：

- INV-P9 agents-radar 上游 schema 漂移检测的硬拒绝边界（M2）
- INV-S4 skill 在容器内加载到 `~/.claude/skills/` 后的清理时序（M3，避免 skill A 残留污染 skill B 的 run）
- INV-L5 panel 渲染时的 cross-lane XSS 防护（M3，因为 skill description 来自外部 / 不可信）
