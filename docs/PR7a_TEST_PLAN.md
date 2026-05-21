# PR#7a 测试计划 — 清理 v9 残余（CCR / cc-agent docker 生成路径）

> 阶段 1（架构）+ 阶段 2（测试用例设计），按 rules `00-core.mdc §开发工作流` 先确认再编码。
>
> PR#7 原始范围（清理 + compose/systemd/bootstrap）太大，拆为 **PR#7a**（本 PR：纯清理）+ **PR#7b**（部署侧）。

## 1 · 范围

PR#5（restricted SHOWCASE）已经把最后一个走 cc-agent docker 的 stage 改成纯 Python。也就是说 v10 里：

- `_docker_run_cc_agent` / `_execute_cc_stage` / `_CC_STAGES` 已经无人调用；
- `cfg.ccr_url` / `cfg.ccr_apikey` / `cfg.ccr_model` / `cfg.curator_llm_model` 早就被 `cfg.engine_url`/`cfg.engine_api_key` 取代；
- `panel/server.py` 的 `ccr_probe()` 依然在打 v9 的 `:3457`；
- `curator.health.probe_ccr` 依然作为 v9 兼容 shim 存在；
- `curator.enricher.call_ccr_messages` 依然导出。

留着这些"死代码 + 兼容 shim"既增加阅读负担，也会让"再加一个 v9 入口"显得合理。**PR#7a 把它们一次性删掉。**

## 2 · 删除清单（按文件分组）

| 文件 | 删除项 |
| --- | --- |
| `orchestrator/config.py` | 字段 `ccr_url`, `ccr_apikey`, `ccr_model`, `curator_llm_model`, `cc_agent_image`, `cc_agent_max_turns`, `host_docker_bin`, `vllm_container`；属性 `handbook_path`, `tasks_dir`；方法 `task_md` |
| `orchestrator/stages.py` | 函数 `_cc_agent_container_name`, `_docker_run_cc_agent`, `_execute_cc_stage`；常量 `_CC_STAGES`；分发分支里的 `_CC_STAGES` 兜底；`subprocess` import；CURATE 阶段切换到 `cfg.engine_url`/`probe_engine` |
| `orchestrator/main.py` | `LoopState.ccr_unhealthy_since`/`ccr_last_notify` → `engine_*`；`_ccr_preflight_gate` → `_engine_preflight_gate`；`HEYI_CCR_REMIND_INTERVAL` → `HEYI_ENGINE_REMIND_INTERVAL`；调用点同步切到 `cfg.engine_url`/`probe_engine` |
| `curator/enricher.py` | 函数 `call_ccr_messages`；`CuratorConfig.ccr_url/ccr_api_key/ccr_model/ccr_url_legacy_shim`；`__post_init__` 的兼容映射；`from_env` 里 `HEYI_EVAL_CCR_URL` 回退 |
| `curator/health.py` | 类 `CcrHealthReport` 改名 `EngineHealthReport`；删除 `probe_ccr` |
| `panel/server.py` | `CCR_URL`, `CCR_API_KEY`, `CURATOR_MODEL`, `ccr_probe()`，主页 HTML "CCR ok/down" → "engine ok/down"；`health_summary` 中 `docker_status(["heyi-eval-ccr", "heyi-eval-socket-proxy"])` → `docker_status([])` |
| `tests/test_curator_health.py` | `ProbeCcrCompatTests`、对 `CcrHealthReport` 的引用 → `EngineHealthReport` |
| `tests/test_curator_engine_integration.py` | `test_ccr_url_env_used_as_fallback`, `test_v9_ccr_url_param_promoted_to_engine_url` |
| `tests/test_orchestrator_loop_gates.py` | 全文 `_ccr_*` → `_engine_*` |
| `tests/test_orchestrator_python_stages.py` | `_make_cfg` 改字段；`probe_ccr` → `probe_engine`；断言 `ccr-preflight-fail` → `engine-preflight-fail` |
| `tests/test_panel.py` | `ccr` → `engine`；HTML 断言；改 monkeypatch 到 `HeyiEngineClient.health` |
| `tests/test_dispatcher_routes.py` | 删除所有 `patch.object(stages, "_docker_run_cc_agent")`；加 `test_d6_dispatcher_has_no_docker_spawn_helper` |

## 3 · 新增不变量

**INV-11**: v10 代码库里不允许出现任何 v9 残余符号：

```
ccr_url, ccr_apikey, ccr_model, curator_llm_model,
_docker_run_cc_agent, _execute_cc_stage, _CC_STAGES,
cc_agent_image, cc_agent_max_turns, host_docker_bin,
ccr_probe, call_ccr_messages,
HEYI_EVAL_CCR_URL, HEYI_EVAL_CCR_APIKEY, HEYI_EVAL_CCR_MODEL,
HEYI_EVAL_CURATOR_MODEL,
HEYI_EVAL_CC_AGENT_IMAGE, HEYI_EVAL_CC_AGENT_MAX_TURNS,
HEYI_EVAL_HOST_DOCKER_BIN, HEYI_EVAL_VLLM_CONTAINER,
e8-vllm, e8-cc-, CcrHealthReport
```

允许例外：

- `tests/test_no_v9_residue.py` 本身（INV-11 的实现）；
- `tests/test_dispatcher_routes.py`（带 `_docker_run_cc_agent` 等字面量做 `hasattr` 反向断言）；
- `docs/*.md`（历史 PR 文档可以提及历史符号）。

## 4 · 测试用例设计

| ID | 测试 | 验证 |
|----|------|------|
| R-1 | `tests/test_no_v9_residue.py::test_no_v9_residue_for_symbol[<symbol>]` | 在每个被参数化的禁用符号下：扫描 `.py/.yaml/.sh/.timer/.service/.plist`，不在允许列表中的文件不得出现该符号 |
| R-2 | `tests/test_no_v9_residue.py::test_cc_agent_dir_is_python_package_not_docker_buildroot` | 顶层不再有 `cc-agent/`（连字符）目录；`cc_agent/`（下划线）是合法 Python 包并含 `__init__.py` + `showcase_runner.py` |
| R-3 | `tests/test_no_v9_residue.py::test_config_has_engine_fields_not_ccr` | `OrchestratorConfig()` 可被无参构造；含 `engine_url`/`engine_api_key`，不含 `ccr_*`/`cc_agent_*`/`host_docker_bin`/`vllm_container`/`curator_llm_model`/`handbook_path`/`tasks_dir`/`task_md` |
| D-6 | `tests/test_dispatcher_routes.py::test_d6_dispatcher_has_no_docker_spawn_helper` | `hasattr(stages, "_docker_run_cc_agent")` 等四个全部 False |
| 既有测试更新 | `test_curator_health.py`, `test_orchestrator_loop_gates.py`, `test_orchestrator_python_stages.py`, `test_panel.py`, `test_curator_engine_integration.py`, `test_curator.py`, `test_dispatcher_routes.py` | 全部走新符号；不再 import `CcrHealthReport` / `probe_ccr`；事件 title 断言 `engine-down` / `engine-recovered` 等 |

## 5 · 验收条件

1. `pytest --no-cov` 全绿（含 R-1 ~ R-3、D-6）。
2. `ruff check .` 0 错。
3. `mypy orchestrator curator panel orchestrator.notify cc_agent backup heyi_engine` 0 错。
4. CI 通过；覆盖率门槛保持。
5. 手动 grep：除 ALLOWED_PATHS 之外，源码全文不再出现 INV-11 列出的任何符号。
6. PR diff 行数为"纯减"（删除占绝大多数），不引入新的能力或重命名风暴。

## 6 · 风险与回退

- **风险**：删除 `probe_ccr` shim 后任何外部脚本/旧 cron 仍 `from curator.health import probe_ccr` 会立即 ImportError。
  - **缓解**：v10 是新仓库，没有外部依赖人；PR diff 显示所有内部引用已迁移到 `probe_engine`。
- **风险**：`OrchestratorConfig` 字段删除后某些被忽视的脚本传入 `ccr_url=...` 会 `TypeError`。
  - **缓解**：CI 全跑 + `test_config_has_engine_fields_not_ccr` 锁面板；R-1 扫描脚本目录。
- **回退**：单一 commit，`git revert` 即可。

## 7 · 与 PR#7b 的边界

PR#7a 只删、不加。compose / systemd / bootstrap_nv8.sh 等新部署文件、`/var/lib/heyi-eval`/etc. 等路径迁移留给 PR#7b 单独评审。
