# PR#8 测试计划 — e2e 全链路 + 不变量静态守卫 + 真机演练手册

> 阶段 1（架构）+ 阶段 2（测试用例设计），按 rules `00-core.mdc §开发工作流`。

## 1 · 哲学

v10 是给 nv8 真机用的，"end-to-end"理论上意味着拿 Qwen2.5-0.5B 跑一遍完整 DISCOVER → CLEANUP。但本 PR 是在 mac dev workstation 上写的——没有 nvidia-smi，没有 heyi_engine :10814，没有 docker GPU。强行造一个 fake-nv8 集成环境得不偿失。

所以 PR#8 把"e2e"拆成两层：

| 层 | 在哪跑 | 跑什么 | 信心 |
|----|------|------|----|
| **A. e2e 单元化** | mac CI | 给整条 9-stage pipeline 注入 fake docker + fake heyi_engine + fake HF Hub, dispatcher → stages_py → capability → showcase_runner → cleanup 全跑一遍 | 中:验证状态机/checkpoint/dispatcher 不会 wire 错 |
| **B. 真机演练 runbook** | nv8 真机 | `docs/RUNBOOK_NV8.md` 一步步检查清单, 包括 Qwen2.5-0.5B 跑通 + INV-1/INV-4 验证 + 24h timer 验证 | 高: 由人在真机执行, 不在 CI 自动化 |

加上 **不变量静态守卫**——这部分纯 grep + AST, mac 上就能跑, 防止 INV-1/INV-4 这种"产线不可碰"的红线被偷偷违反。

## 2 · 范围

### A 层: e2e 单元测试

**文件**: `tests/test_e2e_pipeline.py`

**思路**: 用 `unittest.mock.patch` 把 stages_py 的 docker-py 调用 + cc_agent.showcase_runner 的 HeyiEngineClient + curator.enricher 的 LLM call 全 stub 成 fake response,然后调 `orchestrator.main.run_pipeline()`,断言:

- run.status = OK
- 9 个 stage 全部 status=OK
- 每个 stage 的 artifacts 列表非空
- state.json 落盘正确

跑两条路径:
- **E-1 happy path**: 全绿走完
- **E-2 capability fail → cleanup still runs**: CAPABILITY 抛 ValidationError, CLEANUP 仍执行 (best-effort 终态保障)

### B 层: 不变量静态守卫

**文件**: `tests/test_inv_production_isolation.py`

| ID | 不变量 | 实现 |
|----|------|----|
| INV-1 | 评估管道不得引用 heyi_engine 生产容器名 | grep 整库 (除 docs)，禁止出现已知产线名: `minimax-m2.7`, `glm-5.1`, `kimi-k2.6`, `voipmonitor`, `xrouter`, `minimax`(独立词), `xrouter`(独立词) — 写到 `tests/data/production_container_names.txt` 单独可改 |
| INV-4 | 评估侧代码不得修改产线 compose / unit 文件 | grep 整库,任何 `.py`/`.sh` 不得写 `/etc/heyi-engine/` 或 `docker-compose -f .* heyi-engine` 模式 |
| INV-12 (新) | docker 操作只走 docker-py SDK, 禁止裸 `subprocess docker ...` | AST 扫描所有 .py: 任何 `subprocess.run` / `subprocess.Popen` / `subprocess.check_*` 的第一个参数 list 若以 `"docker"` 起头则报错。允许列表: `panel/server.py` 的 `docker ps -a` (read-only 状态查询, 不动产线容器) |
| INV-13 (新) | 评估侧不得 `import docker` 在 Python 包外的脚本里 | `scripts/*.sh` 不得直接 `docker run`/`docker exec` 产线相关容器 |

### C 层: backup timer schedule 校验

**文件**: 增到 `tests/test_systemd_units.py`

- `heyi-eval-backup.timer` 的 `OnUnitActiveSec` 必须 ≤ 30min(对应 PR#6 设计的 30min snapshot)
- 没有任何 timer 用 `OnCalendar=*-*-* *:*:*` 这种每秒触发的灾难性配置

### D 层: 真机演练 runbook

**文件**: `docs/RUNBOOK_NV8.md`

固定的 7 步检查清单, 涵盖:

1. preflight: 仓库版本/venv/systemctl 状态/heyi_engine 健康
2. 单 model 烟测: `python -m orchestrator enqueue Qwen/Qwen2.5-0.5B-Instruct` → 等完成 → 检查 9 stage artifact
3. 验证 INV-1: `docker ps` 显示 minimax/glm/kimi 仍在跑, `docker inspect` 没有被 stages_py touch 的痕迹
4. 验证 INV-4: 产线 compose 文件 mtime 与 deploy 前一致
5. 备份验证: 30min 后 `panel` 显示 last_backup_age < 30min
6. 24h timer: 检查 `systemctl list-timers heyi-eval-backup.timer`, 验证 Last/Next 时间合理
7. 拆机: 卸载 systemd 单元, 验证产线无任何残留

## 3 · 验收

1. `pytest --no-cov` 全绿(327 + 新增约 12 个 ≈ 339)
2. `ruff check .` 0 错
3. `mypy` 与 main 持平
4. 覆盖率 ≥ 80%
5. `docs/RUNBOOK_NV8.md` 完整、可执行、含每步的 "失败时怎么办"

## 4 · 不在本 PR

- 真机执行 RUNBOOK_NV8.md(用户独立操作; 本 PR 只写手册)
- notify_sync 业务实现(已在 PR#7b 作为占位 service 落地)
- 用户文档 `docs/USAGE.md`/`docs/OVERVIEW.md`(单独 PR)
- shellcheck CI job(单独 PR)
