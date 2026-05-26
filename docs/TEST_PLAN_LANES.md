# TEST_PLAN_LANES — project / skill lane 测试用例清单

> 本文档定位：[`ARCHITECTURE_LANES.md`](ARCHITECTURE_LANES.md) 三 lane 扩展的**功能完整性清单**，按 alld rule `19-quality-detail.mdc` 的「业务目标三问」组织。
>
> **每条用例都要在 M1 / M2 / M3 落地时被对应 PR 的 `tests/` 实装**。本文档不是测试代码，是测试合同。

## 业务目标三问

按 alld rule，功能完成的标准是用户能达成业务目标，不是接口返回 200。新 lane 的业务目标三问：

### 主路径可用

| 问题 | 验收方式 |
|--|--|
| M2.7 能在沙箱里跑通 1 个 GitHub trending repo 并输出 schema-valid 报告？ | `tests/e2e/test_project_lane_happy.py`（mocked yunwu + 真 m2b 容器） + 真机 drill |
| M2.7 能加载 1 个本地 skill 并跑出代表性产物？ | `tests/e2e/test_skill_lane_happy.py` + 真机 drill |
| panel `/projects` `/skills` 能渲染最新的 run 状态 + outcome？ | `tests/test_panel_projects.py` / `tests/test_panel_skills.py` |

### 失败路径有反馈

| 失败类型 | outcome | 用户能看到的中文 reason |
|--|--|--|
| repo 太大 | `oversize` | 仓库体积 > 5 GB，超出 project_lane 测试范围 |
| repo 无 README | `no_readme` | 仓库无 README，无法生成测试计划 |
| repo 需要 GPU | `needs_gpu` | 项目需要 GPU 资源，请走 model_lane 评测 |
| 装依赖失败 | `agent_failed_step` | 安装 X 失败：（agent 给的中文原因） |
| quickstart 跑不起来 | `partial` / `fail` | 部署成功但 demo 失败：（agent 给的中文原因） |
| agent 超 token / 超时 | `budget_exceeded` / `timeout` | Agent 超出 50k token / 15 分钟预算 |
| agent 不按 schema 输出 | `report_parse_error` | Agent 末尾未输出合规 JSON 围栏 |
| 容器宕掉 | `sandbox_dead` | m2b 容器异常，已自动拉起；本 run 失败 |
| yunwu 429 | `judge_unavailable` | 远端 LLM 服务 429，请稍后重试 |
| agents-radar HTTP 失败 | `data_stale` (lane 不受影响) | 上游 manifest.json 拉取失败，使用 24h 前快照 |

### 状态集完整

| 字段 | 取值约束 |
|--|--|
| `lane` | `model` \| `project` \| `skill` |
| `outcome` | 上表 10 种 + `pass` + `partial` 共 12 种 |
| `verdict.deploys` | bool |
| `verdict.quickstart_works` | bool |
| `verdict.core_features_demonstrated` | string[] (≥1 时才允许 outcome=pass) |
| `verdict.blockers` | string[] |
| `self_assessment_zh` | 非空字符串 ≤ 1KB，必中文 |

---

## Happy Path（每条至少 1 个测试 + 1 个真机 drill）

### H1 — project lane 小 repo 跑通

- **target**: `simonw/llm`（CLI tool，纯 Python，无 GPU，README 完整）
- **input**: queue 行 `{lane: "project", target_id: "simonw/llm", source: "agents-radar", metadata: {repo_url: ..., stars: ..., needs_gpu: false, repo_size_kb: 5000}}`
- **预期 outcome**: `pass`
- **预期 verdict**: `deploys=true, quickstart_works=true, core_features_demonstrated=["cli_chat"], blockers=[]`
- **预期 steps**: `["clone", "install_deps", "smoke_help", "quickstart_demo"]` 全 ok
- **预期落盘**: `runs/project/simonw__llm/<run_id>/{report.json, agent.log, plan.json}`
- **预期 panel**: `/projects` 出现 `simonw/llm` 行，outcome 列 `pass`

测试实装：
- 单测层（`tests/test_project_lane_happy.py`）：mock yunwu + mock docker，断言 stage 顺序 + outcome + 落盘路径
- e2e 层（`tests/e2e/test_project_lane_happy.py`）：真 m2b 容器 + mock yunwu，断言围栏 JSON schema 合规
- 真机 drill（`drills/project_lane_simonw_llm.sh`）：在 heyi 上真 yunwu 真 m2b，跑一遍

### H2 — skill lane 本地 mermaid skill 跑通

- **target**: `~/.claude/skills/mermaid/SKILL.md`
- **input**: queue 行 `{lane: "skill", target_id: "mermaid", source: "local-scan", metadata: {skill_dir: ..., description: "Guide for creating mermaid diagrams..."}}`
- **预期 outcome**: `pass`
- **预期 verdict**: `deploys=true, quickstart_works=true, core_features_demonstrated=["render_flowchart"]`
- **预期 artifact**: `runs/skill/mermaid/<run_id>/workspace/example.mmd` + agent 在 self_assessment_zh 中说明渲染结果

### H3 — 同 repo 24h 内重入队 → dedup 跳过

- **input**: 同一 `simonw/llm` 在 H1 完成后 1 小时再次 enqueue
- **预期**: `discover/radar_ingest.py` 的 dedup 过滤直接跳过，queue 不增加；orchestrator 不会再次起 m2b 跑

### H4 — skill 内容变化触发重跑

- **input**: H2 完成后，用户更新 `~/.claude/skills/mermaid/SKILL.md`（content sha256 变了）
- **预期**: dedup key 不同，允许 enqueue，能再次跑一次

---

## Sad Path（每个错误类型至少 1 例）

### S1 — agents-radar 上游 HTTP 失败 / schema 漂移

- **input**: mock `radar_ingest._fetch_manifest()` 抛 `URLError` 或返回 schema_version=2.0
- **预期**:
  - URLError → fallback 到 `radar_cache/<yesterday>.json`，标 `data_stale`，主流程不中断
  - schema 漂移 → 硬拒绝，不写 queue，告警 `radar_schema_drift`

### S2 — m2b 容器宕掉

- **input**: 跑 H1 流程，AGENT_EXEC 启动后 mock 容器进程 SIGKILL
- **预期**: outcome = `sandbox_dead`；pool_manager 自动 `docker-compose up -d` 拉起；本 run 已落盘 `agent.log`（不完整）+ `report.json` (outcome=sandbox_dead)

### S3 — agent 输出非 JSON 围栏

- **input**: mock agent 输出 markdown 自由总结，无围栏标记
- **预期**: outcome = `report_parse_error`；原始内容存到 `runs/.../bad_report.txt`；panel `/projects` 该行 outcome 列显示「Agent 末尾未输出合规 JSON 围栏」

### S4 — agent 围栏内 JSON 不合 schema

- **input**: mock agent 输出 `<<<HEYI_RUN_REPORT_JSON>>>{"foo": 1}<<<END>>>`
- **预期**: outcome = `report_parse_error`；error_detail 标明 schema 缺失 `lane` / `outcome` 等必填

### S5 — agent 超 token

- **input**: token_budget=100，跑 H1 流程
- **预期**: outcome = `budget_exceeded`；budget_guard 在 100 token 时 SIGTERM；agent.log 完整落盘到中断点；容器健康可复用

### S6 — agent 超时

- **input**: timeout_s=5，跑 H1 流程
- **预期**: outcome = `timeout`；同 S5 graceful 中断

### S7 — repo > 5GB

- **input**: queue 行 metadata `repo_size_kb=6000000`（6 GB）
- **预期**: FETCH 阶段被 INV-P1 oversize gate 拒绝，outcome = `oversize`；不进 PLAN / AGENT_EXEC；不消耗 yunwu token

### S8 — repo 缺 README

- **input**: queue 行 metadata `has_readme=false`
- **预期**: outcome = `no_readme`

### S9 — repo 需要 GPU

- **input**: queue 行 metadata `needs_gpu=true`
- **预期**: outcome = `needs_gpu`；INV-P5 拒绝在 FETCH 阶段

### S10 — yunwu 429

- **input**: mock yunwu chat 第一次返回 429，第二次 200
- **预期**: PLAN 阶段退避 + 重试 1 次 → 成功；如果连续 3 次 429 → outcome = `judge_unavailable`

### S11 — SKILL.md 内嵌 git clone（skill 越权）

- **input**: 构造一个 SKILL.md 在 description 里写 `先 git clone https://github.com/big/repo.git`
- **预期**: INV-S1 守护触发，agent.log 静态扫到 `git clone` → 中断 run + outcome = `skill_clone_attempt`

---

## Edge Cases（边界）

### E1 — 同 repo 24h 内 fail 过

- **input**: H1 跑出 outcome=fail 后 6 小时，同 target 再次出现在 manifest.json
- **预期**: dedup 拒绝，不重试

### E2 — skill 无 examples 章节

- **input**: SKILL.md 只有 description 没有 examples
- **预期**: AGENT_EXEC 阶段 agent 自己现编一个代表性使用案例并执行；只要末尾 schema-valid 即可 pass

### E3 — agents-radar 当天未更新

- **input**: 09:00 拉 manifest.json 仍是昨天的数据（content hash 没变）
- **预期**: outcome 不报错；标 `data_stale`；不重复 enqueue 旧 row

### E4 — 多 lane 混合 queue 调度

- **input**: queue.jsonl 同时含 model / project / skill 三种 row
- **预期**: main loop 按 enqueue 时间顺序消费，每条 row 路由到对应 LaneSpec；INV-L1 断言每条都识别成功

### E5 — env `HEYI_EVAL_LANES=model` 紧急 disable

- **input**: 设置 env 后重启 orchestrator；queue 中 mixed lane
- **预期**: model row 正常处理；project / skill row 静默跳过（log info，不报错）

### E6 — 同时 3 个 run 抢 pool（pool_size=1）

- **input**: 3 个 row 同时进 AGENT_EXEC，pool_size=1
- **预期**: pool_manager.acquire 阻塞 2 个，等第 1 个 release；不允许并发跑

### E7 — agent_driver 写入路径越权

- **input**: 构造恶意 target_id `../skill/../etc/passwd`
- **预期**: INV-L2 路径校验抛 ValueError，stage 失败

### E8 — manifest.json 单条 repo 重复出现（agents-radar 自身去重不完整）

- **input**: manifest.json 同一 repo 出现在 `trending` 和 `topic_search` 两个段
- **预期**: radar_ingest 内部 dedup 到 (lane, target_id)，queue 只写一次

---

## 落地节奏

| 用例 | 落地 PR | 测试层 |
|--|--|--|
| H1 / S3-S6 / E5 / E7 | M1 PR | 单测 + e2e（mock yunwu，真 m2b 容器） |
| H1 真机 + H3 / S1 / S2 / S7-S10 / E1 / E3 / E4 / E6 / E8 | M2 PR | 单测 + e2e + 真机 drill |
| H2 / H4 / S11 / E2 | M3 PR | 单测 + e2e + 真机 drill |

每 PR 必须：
1. 列出本 PR 覆盖的用例 ID（H/S/E 编号）
2. 未覆盖的用例 ID 在 PR description 显式标 "deferred to MX"
3. CI 必须跑全套未 deferred 的用例

---

## 真机 drill（手动验证）

每个 M 落地后在 NV8 / heyi 上手动跑一遍 drill 脚本（参考现有 `drills/attack_delete_store.sh` 写法）：

| Drill | 内容 |
|--|--|
| `drills/project_lane_e2e.sh` | 拉 simonw/llm，端到端跑通 + 截 panel 截图 |
| `drills/skill_lane_e2e.sh` | 加载本地 mermaid skill，端到端跑通 |
| `drills/project_lane_oversize_reject.sh` | 拉一个 10GB fake repo，验证 FETCH 拒绝 |
| `drills/skill_clone_attempt.sh` | 构造恶意 SKILL.md 验证 INV-S1 |
| `drills/lane_disable_switch.sh` | 设 `HEYI_EVAL_LANES=model` 重启，验证 project/skill 静默跳过 |

drill 输出（截图 / 录屏）作为 PR 描述的 "How Verified" 章节内容。
