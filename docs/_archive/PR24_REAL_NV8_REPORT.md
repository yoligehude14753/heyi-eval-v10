# PR#24 · 真实 nv8 端到端评测报告（PR#22b-M3 + PR#23 落地后）

> 时间：2026-05-23 10:25 ~ 10:32 UTC+8（约 7 分钟窗口）
> 目标：在 PR#22b-M3（agent sandbox 接入）+ PR#23（M2.7 API + eval pool=5,6,7 + INV-23 oversize 闸门）的真实生产配置下，跑一轮端到端评测，验证四件事：
>
> 1. 评估 pipeline 在新配置（eval_gpus=(5,6,7)）下完整跑通一个真模型
> 2. INV-23 oversize 闸门在生产路径上自动拦截 72B+ 模型，不浪费 100 GB+ 下载
> 3. LLM-judge 走 M2.7 API（model="MiniMax-M2.7"）能正常工作
> 4. orchestrator 的 sandboxed-agent 桥接（PR#22b-M3）系统层端到端绿

## 0 · 环境快照

```
hostname:        heyi (nv8, .179)
orchestrator:    heyi-eval-orchestrator.service active
panel:           http://127.0.0.1:8090 active
GPU 占用:
  0/1/2/3 → 89 GB 各 = minimax 容器 (MiniMax-M2.7, TP=4)
  4       → 93 GB    = ComfyUI host 进程 (PR#23 之所以收缩 eval pool 的原因)
  5/6/7   →  3 MiB 各 = 空闲, 评估池
free_disk:       722 GB
M2.7 endpoint:   http://127.0.0.1:10814 (本机), model="MiniMax-M2.7" 已实测 200

/etc/heyi-eval-v10/env:
  HEYI_EVAL_DATA=/home/ai/heyi-eval-data
  HF_ENDPOINT=https://hf-mirror.com
  HEYI_EVAL_PROD_ENGINE_CONTAINER=minimax
  HEYI_EVAL_PROD_ENGINE_GPUS=0,1,2,3
  HEYI_EVAL_EVAL_GPUS=5,6,7              # PR#23
  HEYI_EVAL_JUDGE_MODEL=MiniMax-M2.7     # PR#23
```

## 1 · Run-A：Qwen2.5-0.5B-Instruct 全流程绿（PR#23 默认 pool）

**入队**：
```
$ python -m orchestrator.main enqueue Qwen/Qwen2.5-0.5B-Instruct
enqueued: r-20260523T022535Z-qwen_qwen2.5-0.5b-instruct-4533
```

**实际时序**（orchestrator journal）：

| 阶段           | 启动             | 结束             | 用时    | 备注 |
|---------------|------------------|------------------|--------|------|
| DISCOVER      | 10:25:36         | 10:25:36         | 0.0 s  | cache hit |
| CURATE        | 10:25:36         | 10:25:37         | 1.2 s  | curated.json cache hit |
| METADATA      | 10:25:37         | 10:25:38         | 0.4 s  | hf_info + first-impression |
| ENGINE_SELECT | 10:25:38         | 10:25:38         | 0.0 s  | tp=1 < pool=3 → not oversize |
| DEPLOY        | 10:25:38         | 10:25:39         | 0.9 s  | e9-vllm-* 容器启动 |
| READY_WAIT    | 10:25:39         | 10:26:57         | 78.1 s | vLLM 启动+权重载入+/v1/models 200 |
| CAPABILITY    | 10:26:57         | 10:27:19         | 22.6 s | 25 个 item 跑完 |
| PERF_BENCH    | 10:27:19         | 10:27:22         | 2.3 s  | TTFT / TPS / 并发 |
| SHOWCASE      | 10:27:22         | 10:27:43         | 21.0 s | M2.7 planner + 默认 item |
| CLEANUP       | 10:27:43         | 10:27:43         | 0.5 s  | 容器删除 |
| **总用时**    | —                | —                | **~127 s** | |

**capability.json 摘要**（25 题）：
- 通过 12 / 失败 13 → **pass_rate = 48 %**（0.5B 参数的合理表现）
- 通过的题目集中在简单算数（`60`, `30 + 10 = 40`）+ 短答；
  失败集中在多步推理（如 Janet 鸭蛋 / 火车均速）

**perf_bench.json 摘要**（来自 GPU 5）：
```json
{
  "ttft_ms":       {"n":3, "p50": 8.8, "mean":   9.1},
  "tps_single":    {"n":3, "p50": 484, "mean": 484},
  "concurrent":    {"n":4, "aggregate_tps": 1955,
                   "wall_ms": 524, "total_completion_tokens": 1024},
  "vram_mib":      {"5": 88687, "6": 3, "7": 3, "total": 88693}
}
```

关键点：
- VRAM 只占 GPU 5（88.7 GB），GPU 6/7 各 3 MiB → **tp=1 绑定到 selected_gpus[0]=5**，
  符合 `_select_eval_gpus` 的 `selected_gpus[:tp_size]` 切片策略
- 单路 TPS 484、4 路并发聚合 TPS 1955（线性扩展约 4 倍）
- TTFT 9 ms 量级（vLLM warm 后的本机直连）

## 2 · Run-B：Qwen2.5-72B-Instruct 触发 INV-23 oversize 闸门

**入队**：
```
$ python -m orchestrator.main enqueue Qwen/Qwen2.5-72B-Instruct
enqueued: r-20260523T023126Z-qwen_qwen2.5-72b-instruct-4b58
```

**实际时序**：

| 阶段           | 启动             | 结束             | 用时    | 备注 |
|---------------|------------------|------------------|--------|------|
| DISCOVER      | 10:31:31         | 10:31:31         | 0.0 s  | |
| CURATE        | 10:31:31         | 10:31:31         | 0.2 s  | |
| METADATA      | 10:31:31         | 10:31:31         | 0.3 s  | hf_info: `param_count="72.7B"` |
| ENGINE_SELECT | 10:31:31         | 10:31:31         | 0.0 s  | **SKIPPED (graceful)** |
| (DEPLOY 以下) | —                | —                | —     | **全部 pending（不执行）** |
| CLEANUP       | 10:31:31         | 10:31:31         | 0.0 s  | |
| **总用时**    | —                | —                | **0.6 s** | vs 130 GB 下载 + 多分钟 DEPLOY 浪费 |

**engine.json 实际内容**（关键字段已加粗）：
```json
{
  "stage": "ENGINE_SELECT",
  "hf_id": "Qwen/Qwen2.5-72B-Instruct",
  "engine": "metadata_only",            ← 不是 vllm
  "engine_image": "vllm/vllm-openai:v0.11.0",
  "reason": "text-generation family",
  "fallback_engine": "transformers",
  "vllm_args": {
    "max_model_len": 32768,
    "tensor_parallel_size": 4           ← 由 _vllm_args_hint("72.7B") 正确算出 4
  },
  "eval_pool_size": 3,
  "eval_pool_gpus": [5, 6, 7],
  "oversize": true                      ← INV-23 闸门触发标志
}
```

**state.json 结果**：
```
status:          aborted
failure_reason:  aborted at ENGINE_SELECT: oversize: model needs
                 tensor_parallel_size=4 but eval pool has 3 GPUs ([5, 6, 7]);
                 metadata captured at _meta/metadata.json + _meta/engine.json,
                 no DEPLOY
stages:
  DISCOVER       ok
  CURATE         ok
  METADATA       ok
  ENGINE_SELECT  skipped   ← 不是 FAILED
  DEPLOY         pending   ← 全部跳过
  READY_WAIT     pending
  CAPABILITY     pending
  PERF_BENCH     pending
  SHOWCASE       pending
  CLEANUP        ok        ← best-effort cleanup 仍执行
```

**关键验证**：
- `aborted` 不是 `failed` → failure metric 不增加，Panel 标"主动跳过"
- 全部元数据（metadata.json + engine.json）已沉淀，可供后续 audit
- **0 个** docker 容器被创建，**0 GB** 模型权重被下载
- 闸门触发延迟 < 1 秒（vs 没有闸门时会消耗 130 GB 磁盘 + 10+ 分钟下载）

## 2.5 · 中间小坑（边测边修，已 fix）

跑 Run-B 第一次时（PR#23 初版），engine.json 显示 `tensor_parallel_size: 1`，
oversize 闸门没触发。根因：`_vllm_args_hint` 用的是子串匹配
（`"72b" in "72.7b"`），但 `"72.7b"` 不含 `"72b"` 子串（中间有小数
点），所以 HF 模型卡上典型的小数参数（`"72.7B"`, `"1.5B"`,
`"236.5B"`）一律落进 tp=1 分支，绕过 INV-23。

**Fix**（已落 `orchestrator/stages.py::_vllm_args_hint`）：改用正则
`(\d+(?:\.\d+)?)\s*b\b` 提取首个数字，按阈值分档：

| 参数量          | tp_size |
|----------------|---------|
| ≥ 65 B         |    4    |
| 28 ~ 65 B      |    2    |
| < 28 B / 无    |    1    |

测试守护：`tests/test_pr23_m27_api_and_oversize.py::TestVllmArgsHintTpHeuristic`
共 10 例覆盖 `72.7B / 1.5B / MoE-236.5B-A21B / 405B / 30B / 34.5B /
7B / "" / None / "unknown"`。

修完重跑 Run-B → engine.json 现在显示 `tp=4`、`oversize=true`，闸门
正常拦截。

## 3 · Panel API 实时反映新配置

```
$ curl http://127.0.0.1:8090/api/health
```

关键字段：
```json
{
  "engine":  {"ok": true, "http": 200, "model": "MiniMax-M2.7"},  // PR#23 ✅
  "gpu":     [{"index":0, "mem_used_mb":89045}, … {"index":4, "mem_used_mb":93715},
              {"index":5, "mem_used_mb":3},  {"index":6, "mem_used_mb":3},
              {"index":7, "mem_used_mb":3}],
  "last_heartbeat": {"body":"in_flight=0\ncompleted_today=0\nfailed_today=0\nfree_disk=722GB"}
}
```

orchestrator status 列表（最近 4 行）：
```
r-20260523T023126Z-qwen_qwen2.5-72b-instruct-4b58  ...72B-Instruct  aborted   0.6s
r-20260523T022824Z-qwen_qwen2.5-72b-instruct-7626  ...72B-Instruct  aborted  13.5s   ← PR#23 hotfix 前
r-20260523T022535Z-qwen_qwen2.5-0.5b-instruct-4533 ...0.5B-Instruct ok       127.1s  ← 完整跑完
```

## 4 · PR#22b-M3 sandbox 桥接的独立验证（独立于评测 pipeline）

PR#22b-M3 的桥接已在 §14 RUNBOOK 验证段记录，本次再做了一次确认压测：

```
$ for i in 1 2 3; do
    python -m orchestrator.agent_runner m3pulse-$i-$(date +%s) --mode smoke
  done
m3pulse-1-1779502589  ok=True  dur=1.09 s  audit={begin_id:26, end_exit:0, end_duration_ms:4}
m3pulse-2-1779502590  ok=True  dur=1.08 s  audit={begin_id:28, end_exit:0, end_duration_ms:3}
m3pulse-3-1779502591  ok=True  dur=1.08 s  audit={begin_id:30, end_exit:0, end_duration_ms:3}
```

- audit_id 单调递增（26 → 28 → 30），中间偶数是 end-row，符合 INV-21
  append-only 模型
- outbox（payload.stdout + payload.stderr + run_meta.json）每次都被
  `heyi-eval-agent-harvest` 拷回 `/home/ai/heyi-eval-data/runs/<id>/outbox/`
  ownership ai:ai
- 6 个沙箱 drill (`run_all.sh`) 全绿 → INV-16/17/18/19/20/21 不动

## 5 · 结论 · 北极星指标

| 维度                       | 目标                                         | 实测 | 结果 |
|---------------------------|---------------------------------------------|------|------|
| 评估 pipeline 走通          | 1 个真模型 9 阶段 OK                          | Qwen 0.5B 127 s 完整跑完 | ✅ |
| INV-23 oversize 闸门生效    | 72B+ 模型在 ENGINE_SELECT 拦截，不进 DEPLOY    | Qwen 72B 0.6 s 标 aborted，0 容器 | ✅ |
| M2.7 API 作为 LLM-judge    | 调用 `model="MiniMax-M2.7"` 返回 200 + 内容    | live curl + showcase 阶段实跑 | ✅ |
| Panel 反映新配置           | 显示 engine=M2.7 + GPU 5/6/7 idle + 状态行    | `/api/health` 字段全对    | ✅ |
| sandbox 桥接稳定           | 3x 连续 smoke 全 ok=True                       | 平均 1.08 s, audit 单调   | ✅ |
| 不引入回归                 | 全量 sandbox drill 6/6 PASS                    | run_all.sh 退 0          | ✅ |

## 6 · 已知遗留（不阻塞 PR#24）

| 项                                 | 影响                                  | 计划 |
|-----------------------------------|--------------------------------------|------|
| `_vllm_args_hint` tp 估算粗糙       | 部分稀疏 MoE（如 Qwen3-30B-A3B）按总参数报 tp=2，可能保守 | 后续 PR 改读 modelcard 的实际 num_attention_heads / hidden_size |
| `test_stages_py_deploy::test_s3_model_path_missing` 失败 | 测试期望 `model_missing`，实际返回 `insufficient_gpu`（_graceful_skip 统一吐 insufficient_gpu） | 与 PR#23 无关；下一个 PR 修 _graceful_skip 把 error_kind 参数化 |
| Showcase planner 偶发 `non-JSON; falling back to default item` | M2.7 输出带 `<think>` 块，解析虽容错但走 fallback；item 仍生成 | 后续 PR 在 showcase planner 加 `<think>` 剥离器 |
| Mac 远端跑 `gh pr create` 需 push 才能拿 PR URL | 不影响功能 | 现已 push，PR URL 见 GitHub |

## 7 · 复现命令（操作员视角）

```bash
ssh ai@<NV8_HOSTNAME>
cd ~/heyi-eval-v10

# 一次评估真模型
.venv/bin/python -m orchestrator.main enqueue Qwen/Qwen2.5-0.5B-Instruct
sleep 130                                      # 等 9 阶段跑完
.venv/bin/python -m orchestrator.main status | head

# 触发 INV-23 oversize 闸门
.venv/bin/python -m orchestrator.main enqueue Qwen/Qwen2.5-72B-Instruct
sleep 5                                         # 不到 1 s 就 abort
cat /home/ai/heyi-eval-data/runs/r-*72b*/_meta/engine.json | jq .oversize  # → true

# 沙箱桥接验证
.venv/bin/python -m orchestrator.agent_runner m3demo-$(date +%s) --mode smoke

# 沙箱完整 drill 回归
sudo bash deploy/agent-sandbox/drills/run_all.sh
```
