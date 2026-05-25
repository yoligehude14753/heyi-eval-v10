# heyi-eval-v10 架构

> 本文档是 **运行时真相**（截至 2026-05-25）。
> · 设计来源 / 历史决策见 [`PLAN.md`](PLAN.md)
> · 红线清单（INV-1..INV-23）见 [`INVARIANTS.md`](INVARIANTS.md)
> · 操作手册见 [`USAGE.md`](USAGE.md) 与 [`RUNBOOK_NV8.md`](RUNBOOK_NV8.md)
> · 早期 PR 测试计划与报告已归档至 [`_archive/`](_archive/)，仅作审计参考，不作为现状描述

---

## 0 · 一句话概述

NV8 (`<NV8_HOST_IP>`，8×6000) 上常驻的自动化 LLM 评测流水线：HuggingFace Hub 发现 → **11 阶段** 评测 → Panel 展示。产线 LLM（MiniMax-M2.7，GPU 0–3）与评测引擎（`e9-*` 容器，GPU 5–7）物理隔离，靠 23 条不变量守护边界。

```
HF Hub ──► discover daemon ──► candidates.jsonl ──► enqueue.timer (15 min)
                                                        │
                                                        ▼
                                                  queue.jsonl
                                                        │
                                                        ▼
                                       orchestrator loop ── 11 stages ──► runs/<run_id>/
                                                        │                   │
                                          heyi_engine :10814 (PROD)         │
                                                        │                   ▼
                                                   e9-* :18200 (EVAL)   panel :8090
```

---

## 1 · 信任域

| 信任域 | 进程 / 组件 | 资源权限 | 守护机制 |
|---|---|---|---|
| **PROD** | 产线 vLLM 容器（默认 `minimax`，可由 `HEYI_EVAL_PROD_ENGINE_CONTAINER` 覆盖），监听 `:10814` | GPU 0–3，外部用户流量 | 评估代码**禁止 docker-控制**（INV-1/INV-12/INV-13），仅作 HTTP 客户端（INV-14） |
| **ORCH** | `orchestrator/`、`discover/`、`curator/`、`heyi_engine/`、`panel/`、`backup/` | docker socket，`/home/ai/heyi-eval-data` 全权 | 不得修改 `/etc/heyi-engine/` 或 systemctl 控制 `heyi-engine.*`（INV-4） |
| **EVAL** | `e9-vllm-*` / `e9-sglang-*` / `e9-tf-*` 临时容器 | GPU 5–7，监听 `:18200` | 名称强制 `e9-*` 前缀（INV-3），CLEANUP 仅删带 `heyi_eval_run` 标签的此前缀容器 |
| **AGENT-SANDBOX** | `cc_agent/` showcase 子进程，沙箱 user `heyi-eval-agent` | 仅 `runs/<run_id>/showcase/` rw + 3 个 metadata 文件 ro；无 shell 无 docker socket | INV-16..INV-22：POSIX ACL + docker-socket-proxy 只读 + sudoers 白名单 + 审计 daemon |
| **DATA** | `/home/ai/heyi-eval-data/`、`/home/ai/heyi-eval-backups/`、`/DATA/Model/_eval-cache/` | ORCH 全权；备份目录在仓库与数据目录之外（INV-9） | `backup/snapshot.py` 30 min 增量 + 7d 保留 |

### v9 incident 教训直接转译

| v9 教训 | v10 约束 |
|---|---|
| cc-agent 拿到 `/heyi-eval-data:/workspace` rw + bash 工具 → 删了 sqlite | cc_agent 只 mount `runs/<run_id>/showcase/`，无 bash |
| cc-agent 有 docker socket → 任意 mount 攻击 | cc_agent 无 docker socket；docker 调度仅由 Python orchestrator |
| CCR 直连 vLLM，model name 写死 | `heyi_engine` 自动 `GET /v1/models` 发现 `served_model_name` |
| sqlite 单点无备份 | 30 min rsync + 7d 保留 + Mac 每夜镜像 |

---

## 2 · 模块依赖图

```
                         heyi_engine ◄────────────┐
                            ▲                     │
                            │ HTTP                │ HTTP
            ┌───────────────┼─────────────────────┤
            │               │                     │
        curator         orchestrator           cc_agent
            │               │                     │
            │       ┌───────┼────────┐            │
            │       ▼       ▼        ▼            │
            │   discover  panel  backup           │
            │       │       │                     │
            │       ▼       ▼                     │
            └──► DATA ◄── runs/ ──── showcase/ ◄──┘
                                       (ro/rw 子树)
                            │
                            ▼
                  transformers_runner (image)
                  e9-tf-* (EVAL container)
```

| 模块 | 入口 | 职责 | 关键依赖 |
|---|---|---|---|
| **`orchestrator/`** | `orchestrator/__main__.py` → `main.py` | 队列消费 → 11 阶段状态机 → 阶段后断言 → 故障自愈 / 沙箱桥接 / 缓存逐出 | 唯一 mutating docker 路径；持有 ORCH 信任域 |
| **`panel/`** | `panel/__main__.py` → `server.py::main()` | 只读 HTTP 面板（:8090），唯一写路径 `POST /api/enqueue` | 读 `store/`, `runs/`，调用 `orchestrator.main.enqueue` |
| **`cc_agent/`** | `cc_agent/showcase_runner.py`, `cc_agent/deploy_repair_agent.py` | SHOWCASE 计划/打分（PROD LLM）；PR#33 deploy 自愈 LLM 代理 | `heyi_engine`、`orchestrator.capability` 中的 HTTP helper |
| **`discover/`** | `discover/main.py` | 日频 HF Hub 扫描（whitelist + trending），追加 `candidates.jsonl` | `huggingface_hub`、`discover/whitelist.yaml` |
| **`curator/`** | `curator/enricher.py`, `curator/health.py` | LLM 解读 README → 结构化 JSON（`capability_tags` / `param_count`） | `heyi_engine`（PROD LLM），输出 `data/curated/` 与 run `_meta/curated.json` |
| **`heyi_engine/`** | `heyi_engine/client.py` | 产线 LLM 单一客户端，自动 `/v1/models` 发现 | 环境：`HEYI_ENGINE_URL`、`HEYI_ENGINE_API_KEY` |
| **`transformers_runner/`** | `server.py`, `detect.py`，镜像 `heyi-eval/transformers-runner:v10` | vLLM 无法承载的模态（ASR/TTS/diffusers/VLM）OpenAI-兼容服务 | 受 INV-15 保护：包内**禁止**引用 PROD 配置 token |
| **`backup/`** | `backup/__main__.py` | rsync `heyi-eval-data` → `heyi-eval-backups`，>7d 自动清理 | `orchestrator.config.OrchestratorConfig` |
| **`deploy/`** | `deploy/systemd/*`, `deploy/env.example`, `deploy/agent-sandbox/` | 9 个 systemd unit + 沙箱 ACL + sudoers，**无 docker-compose** | `scripts/bootstrap_nv8.sh` 一键安装 |
| **`scripts/`** | `bootstrap_nv8.sh`、`build_transformers_runner.sh`、`verify_24h_timer.sh` 等 | 操作员 CLI | — |
| **`sops/`** | `sops/schemas/*.schema.json`、`sops/known_quirks.md` | 阶段产物 JSON Schema + 已知部署 quirk 库（Q-001+） | `sops/invariants.md` 是 v8 历史文档，不再适用 |
| **`tools/`** | `tools/evict_eval_cache.py` | 一次性运维工具（PR#65 LRU 逐出 CLI） | `orchestrator/cache_evictor.py` |
| **`tests/`** | `pytest`（~955 通过） | 静态 INV 守护 + 阶段单测 + e2e + systemd 单元校验 + 沙箱攻击演练 | — |

---

## 3 · 11 阶段流水线

权威定义：`orchestrator/state_machine.py::STAGES_IN_ORDER`（注意：该文件顶部 docstring 仍写 "10-stage"，是 PR#31 之前的过时描述；以 `STAGES_IN_ORDER` 列表为准）。

```
DISCOVER ─ CURATE ─ METADATA ─ ENGINE_SELECT ─ STAGE_MODEL     ← 失败时整 run 重启
                  │
                  ▼
DEPLOY ─ READY_WAIT ─ CAPABILITY ─ PERF_BENCH ─ SHOWCASE ─ CLEANUP  ← 阶段级断点续跑
```

**检查点策略**：前 5 个阶段任一失败 → run-level 重启（DISCOVER 起）；DEPLOY 之后任一失败 → 从最后 OK 阶段继续（保留 60–120s vLLM 启动开销）。CLEANUP 在任何终态都 best-effort 执行。

| # | 阶段 | 作用 | 输入 | 输出（runs/<run_id>/） | 失败语义 |
|---|---|---|---|---|---|
| 1 | DISCOVER | 记录 enqueue 来源 | 队列 job | `_meta/discover.json` | 总是 OK |
| 2 | CURATE | LLM 解读 README → JSON | HF README、可选 `data/curated/` 缓存 | `_meta/curated.json`、`_meta/modelcard.md` | PROD 不健康时退化为空字段，仍 OK |
| 3 | METADATA | 合并 curator + HF Hub API | `_meta/curated.json` | `_meta/metadata.json` | HF 错误 → 部分元数据 |
| 4 | ENGINE_SELECT | 选引擎 + TP；oversize / not-a-model 闸门 | `_meta/metadata.json` | `_meta/engine.json` | `oversize_skip` / `not_a_model_skip` → 优雅跳过（run ABORTED） |
| 5 | STAGE_MODEL | 下载权重到 `/DATA/Model/_eval-cache/<basename>` | `_meta/engine.json` | `_meta/stage_model.json` | oversize 跳过；下载错误硬失败。**逐出**：超出 `cache_quota_bytes`（默认 200GB）时 LRU 删除 orphan→failed→safe |
| 6 | DEPLOY | spawn `e9-{vllm,sglang,tf}-<short>` 容器到 eval GPU 池 | `_meta/engine.json`、缓存权重 | `deploy.json`、`_meta/deploy.json`，可选 `_meta/deploy_repair.json` | 池为空/与 PROD 重叠/tp 过大 → 优雅跳过；docker / image / timeout 错误经 PR#33 LLM 自愈后仍失败则硬失败 |
| 7 | READY_WAIT | 轮询 `/v1/models` + 推理探针（PR#36） | `deploy.json` | `ready.json`、`_meta/ready.json` | 超时 / 容器死亡硬失败 |
| 8 | CAPABILITY | 13 类多模态微评测，按 `capability_tags` 筛选 | `deploy.json`、`_meta/curated.json` | `capability.json` | 单 item 失败不导致阶段失败 |
| 9 | PERF_BENCH | TTFT / TPS / VRAM（仅 text 模型） | `deploy.json`、`_meta/curated.json` | `perf_bench.json` | 无 text 标签 → `applicable: false`（OK 跳过） |
| 10 | SHOWCASE | LLM 计划 ad-hoc prompt + 中文总结 | curated/metadata/modelcard、`deploy.json` | `showcase.json`、可选 `showcase/*.md` | 计划 / 打分错误硬失败 |
| 11 | CLEANUP | 删除带 `heyi_eval_run=<run_id>` 标签的 `e9-*` 容器 | docker | `cleanup.json`、`_meta/cleanup.json` | best-effort；INV-1 守护：**永不**触碰 PROD 容器 |

**优雅跳过 → ABORTED**：`StageResult.extra.aborted=True` → `main.py` 抛 `GracefulSkip` → run 状态 `aborted`，CLEANUP 仍执行，**不计入 failure metric**。

**阶段调度**：`orchestrator/stages.py::execute_stage()`，stub/python/native 三类路由。

---

## 4 · Daemon 拓扑

systemd 安装于 `deploy/systemd/`，bootstrap 走 `scripts/bootstrap_nv8.sh`。**无 docker-compose**（评测容器由 orchestrator 在 DEPLOY 阶段 spawn）。

| Unit | 类型 | 命令 | 频率 |
|---|---|---|---|
| `heyi-eval-discover.service` | long-running | `python -m discover.main loop --mode curated --interval 86400` | 24 h |
| `heyi-eval-enqueue.timer` + `.service` | oneshot | `python -m discover.main enqueue --limit 5 ...` | **15 min** |
| `heyi-eval-orchestrator.service` | long-running | `python -m orchestrator loop` | 连续 |
| `heyi-eval-panel.service` | long-running | `python -m panel`（:8090） | 连续 |
| `heyi-eval-backup.service` + `.timer` | oneshot | `python -m backup` | 30 min |
| `heyi-eval-audit.service` | long-running | INV-21 审计 daemon（root，监听 `/run/heyi-eval-agent-audit.sock`） | 连续 |
| `heyi-eval-notify-sync.service` | long-running | 同步 `notify_outbox.jsonl` 到外部通道 | 连续 |
| `heyi-eval-agent.slice` + `heyi-eval-agent@.service` | template | 沙箱内 cc_agent 进程，受 INV-19 资源预算 | 按 run 启动 |

### 端到端数据流

```
HF Hub
  ▼
discover daemon ─ append ─► discover/candidates.jsonl
                                       │
       enqueue.timer (15m) ─ promote ─►│ ─ append ─► store/queue.jsonl
                                                          │
                                                          ▼
                                          orchestrator loop ─ pop_one()
                                                          │
                                                          ▼ (创建 Run + state.json + sqlite 行)
                                          runs/<run_id>/
                                          ├── state.json
                                          ├── _meta/{curated,metadata,engine,stage_model,deploy,ready,cleanup}.json
                                          ├── deploy.json
                                          ├── capability.json
                                          ├── perf_bench.json
                                          ├── showcase.json
                                          └── cleanup.json
                                                          │
                                                          ▼
                                          curator/{cache} (PROD :10814)
                                          /DATA/Model/_eval-cache/<basename>/
                                          e9-* (EVAL :18200) ── CAPABILITY/SHOWCASE 调用
                                                          │
                                                          ▼
                                          panel (:8090) 读取
                                                          │
                                          backup.timer (30m) ─ rsync ─► /home/ai/heyi-eval-backups/
```

**Intake gate**：`main.py::_engine_preflight_gate()` 在 PROD `:10814` 不健康时**暂停**队列消费（队列保留）。

---

## 5 · 数据布局

```
/home/ai/heyi-eval-data/                       # HEYI_EVAL_DATA
├── discover/
│   ├── candidates.jsonl                       # discover daemon append-only
│   └── cursor.json                            # 扫描游标
├── curated/                                   # 跨 run curator 缓存
│   └── Org__Model.json
├── store/
│   ├── queue.jsonl                            # enqueue 写；orchestrator pop_one
│   ├── runs.sqlite                            # 面板查询索引
│   └── notify_outbox.jsonl                    # 心跳/告警/完成事件
└── runs/
    └── <run_id>/                              # 一次评测一个目录
        ├── state.json                         # 恢复源
        ├── _meta/                             # 各阶段元数据
        ├── deploy.json
        ├── capability.json
        ├── perf_bench.json
        ├── showcase.json
        └── cleanup.json

/DATA/Model/_eval-cache/<basename(hf_id)>/      # HEYI_EVAL_MODEL_CACHE，**不在数据目录内**
                                                # STAGE_MODEL 写；DEPLOY bind-mount；CLEANUP 不删
                                                # PR#65：超 200 GB 时 LRU 逐出

/home/ai/heyi-eval-backups/                    # HEYI_EVAL_BACKUPS（INV-9：在数据根**之外**）
└── snapshot-YYYYMMDD-HHMM/                    # 30 min 增量 + 7d 保留

/etc/heyi-eval-v10/env                         # 配置覆盖（来自 deploy/env.example）
/var/log/heyi-eval-agent/audit.sqlite          # INV-21 沙箱审计（root 写，deny-all ACL）
```

---

## 6 · GPU 拓扑（NV8）

来源：`orchestrator/config.py:176-212` + `docs/INVARIANTS.md`。

| 池 | 默认 GPU | 进程 | 角色 |
|---|---|---|---|
| **PROD** | `(0, 1, 2, 3)` | `minimax` vLLM TP=4，监听 `:10814` | 产线 MiniMax-M2.7；curator / showcase / llm_judge 共用 |
| **ComfyUI（host）** | GPU **4** | `python main.py --port 8188` (~93 GB) | **不属于 eval 池**，是 PR#23 把默认池缩到 `(5,6,7)` 的原因 |
| **EVAL** | `(5, 6, 7)`（len=**3**） | `e9-vllm-*` / `e9-sglang-*` / `e9-tf-*` | 一次一个模型，5–30 min 寿命 |

**TP 选择**（`stages.py::_vllm_args_hint`）：≥65B → TP=4；≥28B → TP=2；其他 → TP=1。

**INV-23 oversize 闸门**：`tp_size > len(eval_gpus)` 时 ENGINE_SELECT 写 `engine=metadata_only` + `oversize=true`，run ABORTED，**不**进入 STAGE_MODEL / DEPLOY（避免浪费 100+ GB 下载）。操作员临时把池扩到 `(4,5,6,7)` 后 TP=4 模型可跑。

**环境覆盖**：`HEYI_EVAL_PROD_ENGINE_GPUS`、`HEYI_EVAL_EVAL_GPUS`、`HEYI_EVAL_PROD_ENGINE_CONTAINER`。

---

## 7 · 引擎路由

决策点：`orchestrator/stages.py::_pick_engine()`（不是独立的 `engine_select.py`）。

| 条件 | 引擎 | 镜像 |
|---|---|---|
| `pipeline_tag` 为 ASR / TTS / diffusion | `transformers` | `heyi-eval/transformers-runner:v10` |
| `text-generation` / `image-text-to-text` / `modality=text` | `vllm` | `vllm/vllm-openai:v0.11.0` |
| `library_name=diffusers` | `transformers` | 同上 |
| `modality=audio`（无 pipeline_tag） | `transformers` | 同上 |
| 默认 | `vllm`（fallback `transformers`） | — |
| oversize / not_a_model | `metadata_only`（**非运行时引擎**，跳过 DEPLOY） | — |

**SGLang**：在 DEPLOY 层支持（`stages_py._ENGINE_IMAGES["sglang"]`），作为 PR#33 故障自愈的备用切换（`swap_engine_sglang`），**不**作为 ENGINE_SELECT 的首选。

**transformers_runner 模态探测**：`transformers_runner/detect.py` 读 `config.json` / `model_index.json`；不支持的模态返回 HTTP 501 → CAPABILITY 把该项打 `gated`。

---

## 8 · 不变量索引（详情见 [INVARIANTS.md](INVARIANTS.md)）

| INV | 类别 | 保护对象 | 静态守护 / 运行时守护 |
|---|---|---|---|
| INV-1, INV-12, INV-13 | 隔离 | 评估代码不得 mutating docker 任何产线容器 | `tests/test_inv_production_isolation.py`；`stages_py.execute_cleanup` 仅删 `e9-*` |
| INV-2 | 隔离 | 产线容器持续运行 | `validator.assert_invariants` 每阶段后 `docker inspect` |
| INV-3 | 隔离 | 评估容器名强制 `e9-*` 前缀 | `stages_py.container_name_for()` 唯一来源 |
| INV-4 | 隔离 | 不得写 `/etc/heyi-engine/`、不得 systemctl 控制 `heyi-engine.*` | bootstrap 路径白名单 |
| INV-5 | 操作 | 不得 host apt/pip；仅 venv 内 | `bootstrap_nv8.sh` |
| INV-9 | 数据 | 备份目录在仓库与数据根之外 | `tests/test_backup_snapshot.py` |
| INV-11 | 卫生 | 禁止 v9 残余符号（ccr_*, e8-*, cc-agent） | `tests/test_no_v9_residue.py` |
| INV-14 | 隔离 | LLM-judge 仅可把 **EVAL 产物字节** 发 PROD VLM，**禁发**测试 prompt | `tests/test_inv14_llm_judge_boundary.py` |
| INV-15 | 隔离 | `transformers_runner/` 不得引用 PROD 配置 token | `tests/test_inv15_transformers_runner_isolation.py` |
| INV-16..19 | 沙箱 | cc_agent 用户 FS ACL / docker socket 只读 / 审计目录 deny-all / 资源预算 | `tests/test_inv16_19_agent_sandbox_static.py` + 真机演练 `drills/` |
| INV-20, INV-22 | 沙箱·sudo | 沙箱 + orchestrator 的 sudoers 双白名单 | `tests/test_pr22b_m3_orch_sudoers_static.py` |
| INV-21 | 沙箱·审计 | audit.sqlite 仅追加，写入走 root daemon + `SO_PEERCRED` | `tests/test_pr22b_audit_daemon.py` + drill |
| INV-23 | 评测 | oversize 模型在 ENGINE_SELECT 闸门 → metadata_only | `tests/test_pr23_m27_api_and_oversize.py` |

> INV-6, INV-7, INV-8, INV-10 已编号但当前文档未定义（演进过程中被合并或废止），新增不变量沿 INV-24 递增。

---

## 9 · 配置与端口

| 项 | 默认值 | 环境变量覆盖 |
|---|---|---|
| 数据根 | `/home/ai/heyi-eval-data` | `HEYI_EVAL_DATA` |
| 备份根 | `/home/ai/heyi-eval-backups` | `HEYI_EVAL_BACKUPS` |
| 模型缓存 | `/DATA/Model/_eval-cache` | `HEYI_EVAL_MODEL_CACHE` |
| 缓存配额 | 200 GB | `HEYI_EVAL_CACHE_QUOTA_GB` |
| 产线 LLM URL | `http://127.0.0.1:10814` | `HEYI_ENGINE_URL` |
| 产线 LLM API Key | 无 | `HEYI_ENGINE_API_KEY` |
| 产线 LLM judge model | `MiniMax-M2.7` | `HEYI_EVAL_JUDGE_MODEL` |
| 产线容器名 | `minimax` | `HEYI_EVAL_PROD_ENGINE_CONTAINER` |
| 产线 GPU | `(0,1,2,3)` | `HEYI_EVAL_PROD_ENGINE_GPUS` |
| 评测 GPU | `(5,6,7)` | `HEYI_EVAL_EVAL_GPUS` |
| 评测 vLLM 端口 | 18200 | `HEYI_EVAL_VLLM_PORT` |
| 评测 HF token | 无 | `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` |
| HF 镜像 | 官方 | `HF_ENDPOINT=https://hf-mirror.com` |
| 面板端口 | 8090 | `HEYI_PANEL_PORT` |
| Curator max tokens | 4096（CURATE 阶段实际用 8192） | `HEYI_EVAL_CURATOR_MAX_TOKENS` |
| DEPLOY / CAPABILITY / SHOWCASE / CLEANUP 超时 | 600 / 900 / 3600 / 300 s | `HEYI_EVAL_*_TIMEOUT` |

完整模板：[`deploy/env.example`](../deploy/env.example)。

---

## 10 · 阶段产物 Schema

JSON Schema 在 [`sops/schemas/`](../sops/schemas/)，受 `tests/` 校验：

| Schema | 校验产物 |
|---|---|
| `manifest.schema.json` | run 顶层清单 |
| `ready.schema.json` | `ready.json`（READY_WAIT 输出） |
| `capability.schema.json` | `capability.json`（13 类评分） |
| `perf_bench.schema.json` | `perf_bench.json` |
| `showcase.schema.json` | `showcase.json` |

新增阶段输出字段时必须同步更新 schema，否则 `tests/test_*_schema.py` 报错。

---

## 11 · 文档与代码索引

### 当前生效文档（运行时真相 / 操作）

| 文档 | 角色 |
|---|---|
| `README.md` | 项目门面，5 分钟读完 |
| `docs/ARCHITECTURE.md`（本文件） | 运行时架构真相 |
| `docs/INVARIANTS.md` | 23 条红线 + 守护机制 |
| `docs/USAGE.md` | 部署后操作：enqueue、E2E 跑通、DR 演练 |
| `docs/RUNBOOK_NV8.md` | NV8 真机演练手册（Qwen smoke、INV 校验、oversize 演练等） |
| `sops/known_quirks.md` | 已知部署 quirk 库（Q-001+） |
| `sops/schemas/*.schema.json` | 阶段产物 schema |
| `transformers_runner/README.md` | 模块级 README |
| `deploy/README.md`、`deploy/agent-sandbox/README.md` | 部署 / 沙箱安装说明 |

### 设计 / 历史快照（不作为现状描述）

| 文档 | 性质 |
|---|---|
| `docs/PLAN.md` | v10 启动期架构计划（仍写 9 阶段；STAGE_MODEL 是 PR#31 后补的）；保留作 design rationale |
| `docs/_archive/PR*_TEST_PLAN.md` × 11 | 每个 PR 的测试矩阵快照；功能已合入主干 |
| `docs/_archive/PR{24,26,29,33,36}_*REPORT.md` | 真机 E2E 报告快照 |
| `sops/_archive_v8_invariants.md` | **v8 历史**（`e8-*`, GPU 4-7），仅作审计参考，**不再适用** |

### 模块入口速查

| 想找 | 看这里 |
|---|---|
| 11 阶段顺序 | `orchestrator/state_machine.py::STAGES_IN_ORDER` |
| 阶段分发 | `orchestrator/stages.py::execute_stage()` |
| DEPLOY / READY_WAIT / CLEANUP Python 实现 | `orchestrator/stages_py.py` |
| CAPABILITY 评分器注册 | `orchestrator/capability.py::_SCORERS` |
| LLM 跨域 judge | `orchestrator/llm_judge.py`（INV-14 边界） |
| 引擎路由 | `orchestrator/stages.py::_pick_engine()` |
| GPU 选择 | `orchestrator/stages_py.py::_select_eval_gpus()` |
| 缓存逐出 | `orchestrator/cache_evictor.py`（CLI：`tools/evict_eval_cache.py`） |
| 自愈策略 | `orchestrator/deploy_repair.py` + `cc_agent/deploy_repair_agent.py` |
| Showcase 计划/打分 | `cc_agent/showcase_runner.py` |
| 产线 LLM 客户端 | `heyi_engine/client.py` |
| Panel HTML/JS | `panel/server.py`（单文件，含 `_PANEL_STYLES` 与 `_TABLE_TOOLKIT_JS`） |
| Discover 主循环 | `discover/main.py::loop` |
| Backup 主循环 | `backup/__main__.py`、`backup/snapshot.py` |

### 已知文档遗留问题

- `orchestrator/state_machine.py` 顶部 docstring 仍写 "10-stage state machine"；以 `STAGES_IN_ORDER` 为准（11）
- `docs/PLAN.md` 描述 9 阶段；以本文件 §3 为准
- `docs/PLAN.md` 提到 `deploy/compose.yml`；**实际不存在**，部署完全走 systemd
- `sops/invariants.md` 是 v8 文档；**不要**作为现状参考，看 `docs/INVARIANTS.md`

---

## 12 · 常见运维快速链接

- 面板：`http://<NV8_HOST_IP>:8090`（Tailnet）
- 产线 LLM：`http://127.0.0.1:10814/v1`
- 评测 LLM（运行时）：`http://127.0.0.1:18200/v1`（仅 DEPLOY 后存在）
- 缓存逐出：`python tools/evict_eval_cache.py --quota-gb 200 --apply`
- 24h 计时器健康检查：`bash scripts/verify_24h_timer.sh`
- 手动 enqueue：`python -m discover.main enqueue --limit 5`，或面板 `POST /api/enqueue`
