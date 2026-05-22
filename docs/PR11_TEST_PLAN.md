# PR#11 — GPU 隔离实施 + Graceful Skip 测试计划

> 阶段 1 (架构) + 阶段 2 (测试用例) 设计文档。本 PR 把 PR#10 准备好的 `prod_engine_gpus` / `eval_gpus` 配置接到 `stages_py.execute_deploy`,实施真正的 GPU 隔离与 graceful-skip 路径。

## 0. 为什么要做这个 PR

PR#10 把"产线 GPU 占用"和"评估 GPU 池"做成了配置项,但代码还没用上:

```312:318:orchestrator/stages_py.py
device_requests=[
    docker.types.DeviceRequest(
        count=int(vllm_args.get("tensor_parallel_size", 1) or 1),
        capabilities=[["gpu"]],
    )
] if engine in ("vllm", "sglang") else [],
```

`count` 让 docker **任意挑** N 张 GPU,可能挑到产线的 0-3 → 把产线 LLM 打飞。本 PR 改成 `device_ids=["4","5","6","7"]`(由 `cfg.eval_gpus` 决定),加 deploy 前的 nvidia-smi 探测,占用/超容时 graceful skip 而非硬错。

> ⚠️ 不同时设 `CUDA_VISIBLE_DEVICES` env:docker DeviceRequest.device_ids 已经做了 GPU 隔离 + 重映射 (容器内看到的就是 0..N-1),再叠加 `CUDA_VISIBLE_DEVICES=4,5,6,7` 会让容器内进程找不到这些索引。两者只能用一个,选 device_ids 因为它走的是 NVIDIA Container Runtime 的标准路径。

## 1. 决策回顾 (用户 2026-05-22 拍板)

| | 规约 |
|---|---|
| GPU 池 | 评估**独占** `cfg.eval_gpus`,严禁碰 `cfg.prod_engine_gpus` |
| 并发 | 评估**一次只跑一个**,无并行 (本 PR 不实施 mutex;留给后续 PR) |
| 资源不够 | **graceful skip + `aborted_reason=insufficient_gpu`**,不抛 error |
| 探测 | deploy 前 `nvidia-smi` 看 eval pool 是否空闲 (mem.used < 1 GB/卡) |

## 2. 不变量决定

PR#11 引入或重申以下契约:

1. **DEPLOY spawn 的 vllm/sglang 容器只能用 `cfg.eval_gpus` 里的 GPU**。通过 docker `DeviceRequest(device_ids=[str(i) for i in selected_gpus], capabilities=[["gpu"]])` 强制限定。
2. **TP > `len(eval_gpus)` 是 graceful skip,不是 error**。被测模型太大用不了我的卡,记一下继续下一个,不让 pipeline 死掉。
3. **eval pool 与 prod pool 有交集是 graceful skip**。产线临时切到 K2.6 TP=8 占满 0-7 时,自动跳过当前 run,不抢产线 GPU。
4. **eval pool 实测 mem.used > 1 GiB 是 graceful skip**。运维手动跑了别的东西占了评估卡,我让出来。
5. **graceful skip 的 stage status = SKIPPED,run status = ABORTED**。明显区分于 FAILED(失败要 retry),ABORTED 是"知道当前条件跑不了所以跳过",resume 时不重新跑。
6. **nvidia-smi 不可用 (CalledProcessError / FileNotFoundError) 是 graceful skip**,保守。

## 3. 文件改动清单

| 文件 | 改动 | 类型 |
|---|---|---|
| `orchestrator/stages_py.py` | 新增 `_select_eval_gpus(cfg, tp_size, smi_query)` 函数;`execute_deploy` 在 docker run 前调用,返回 graceful-skip `StageResult` 或继续;`DeviceRequest` 改 `device_ids=[str(i) for i in selected]`;`_nvidia_smi_used_mib()` helper | feat |
| `orchestrator/state_machine.py` | `StageInfo` 加 `mark_skipped(reason)` 方法 (用 `StageStatus.SKIPPED`) | refactor |
| `orchestrator/main.py` | `_execute_stage_real` 识别 `result.error_kind == "insufficient_gpu"` (或 `extra.get("aborted")`),标 stage SKIPPED + run ABORTED,后续 stage 全 skip;不算 FAILED,不计入 `failed_today` | refactor |
| `orchestrator/notify.py` | 新增 `run_aborted(outbox_path, run_id, hf_id, stage, reason)` 函数 | feat |
| `docs/PR11_TEST_PLAN.md` | 本文档 | docs (新增) |
| `tests/test_pr11_gpu_isolation.py` | 新增,详见 §4 | test (新增) |

不动的文件:
- `orchestrator/config.py` (PR#10 已做完)
- `orchestrator/validator.py` (PR#10 已做完)
- `orchestrator/capability.py` / `cc_agent/showcase_runner.py` (不涉及 GPU spawn)
- 任何 e2e 测试 (不能挂)
- INVARIANTS.md (PR#12 一起做)

## 4. 测试用例清单

新增 `tests/test_pr11_gpu_isolation.py`,覆盖 14 个 case。基于业务目标:"产线 GPU 占用变化时,评估管道安全降级,不抢产线卡"。

### A. GPU 选择算法 (`_select_eval_gpus`) — 8 case

测试方式:直接调函数,monkeypatch `_nvidia_smi_used_mib` 返回 fake mem dict。

| ID | 场景 | 输入 | 预期可观察结果 |
|---|---|---|---|
| G1 | Happy: 默认 eval pool 全空 | cfg.eval_gpus=(4,5,6,7), prod=(0,1,2,3), tp=4, smi={i:0 for i in 0-7} | `(selected=[4,5,6,7], reason=None)` |
| G2 | TP < pool: 选前 N | tp=2, eval=(4,5,6,7) | `(selected=[4,5], reason=None)` |
| G3 | TP > pool: graceful skip | tp=8, eval=(4,5,6,7) | `(selected=None, reason 含 "tensor_parallel_size=8" 与 "pool has 4")` |
| G4 | Eval pool 空 | eval=(), tp=1 | `(selected=None, reason 含 "eval pool is empty")` |
| G5 | Eval pool 与 prod 重叠 (K2.6 TP=8 临时态) | prod=(0,1,2,3,4,5,6,7), eval=(4,5,6,7), tp=4 | `(selected=None, reason 含 "overlaps prod_engine_gpus" 与 "[4, 5, 6, 7]")` |
| G6 | smi 显示 eval GPU 被占 | eval=(4,5,6,7), smi={4: 50000, 5: 0, 6: 0, 7: 0} | `(selected=None, reason 含 "GPU 4" 与 "50000")` |
| G7 | smi 调用失败 | smi raises FileNotFoundError | `(selected=None, reason 含 "nvidia-smi")` |
| G8 | smi 调用超时 | smi raises TimeoutError | `(selected=None, reason 含 "nvidia-smi")` |

### B. `execute_deploy` 端到端 — 4 case

测试方式:monkeypatch `_docker_client` 返回 fake docker client (类似 PR#8 e2e),monkeypatch `_nvidia_smi_used_mib`。

| ID | 场景 | 预期 |
|---|---|---|
| D1 | Happy: 默认配置 + GPU 全空 → docker.run 被调一次 | DeviceRequest.device_ids == ["4","5","6","7"];StageResult.ok=True |
| D2 | Happy: TP=2 | DeviceRequest.device_ids == ["4","5"] |
| D3 | Graceful skip on TP overflow | docker.containers.run **从未** 被调用;StageResult.ok=False, error_kind="insufficient_gpu", extra["aborted"]==True, extra["reason"] 含 "tensor_parallel_size" |
| D4 | Graceful skip on prod overlap | docker.containers.run 未被调用;extra["aborted"]==True;extra["reason"] 含 "overlaps" |

### C. State machine integration — 2 case

| ID | 场景 | 预期 |
|---|---|---|
| S1 | mark_skipped 设状态 | StageInfo.status == SKIPPED;error 记 reason;duration_s 计算 |
| S2 | run_pipeline 见到 aborted DEPLOY 不重试 | run.status == ABORTED;DEPLOY.status == SKIPPED;run.failure_reason 含 "insufficient_gpu";后续 stages 全 PENDING (没被推进) |

### D. 回归 (现有测试必须不挂)

| ID | 范围 |
|---|---|
| R1 | `tests/test_stages_py_*.py` 全部 (DEPLOY/READY_WAIT/CLEANUP) |
| R2 | `tests/test_e2e_pipeline.py` (PR#8 e2e harness;需要更新 fake DeviceRequest 验证) |
| R3 | `tests/test_pr10_concept_split.py` (PR#10 配置层) |
| R4 | 全套 378 testcase + 5 e2e skipped |

## 5. 业务目标三问

合并前三问全 "是":

1. **主路径**: 产线维持 M2.7 (GPU 0-3) 稳态时,DEPLOY 正确把容器限制到 GPU 4-7,docker DeviceRequest 用 device_ids?
   → D1 + D2 覆盖
2. **降级路径**: 产线临时切 K2.6 TP=8 占满 0-7 时,DEPLOY 不抢产线 GPU,而是 graceful skip,run 标 ABORTED 不是 FAILED?
   → G5 + D4 + S2 覆盖
3. **状态集**: 太大 / 重叠 / 占用 / nvidia-smi 不可用 四种 graceful-skip reason 都有明确文案告诉 operator 是哪种情况?
   → G3/G4/G5/G6/G7/G8 + D3/D4 覆盖

## 6. 不在 PR#11 范围

- ❌ Eval mutex (一次只跑一个评估的强制锁) — 可后续 PR
- ❌ 把 `nvidia-smi` 替换为 NVML 库 — 当前 subprocess 探测够用
- ❌ 跨多机调度 (cfg.eval_gpus 跨多台机) — N/A
- ❌ INVARIANTS.md / RUNBOOK.md 大改 — PR#12 一起做
- ❌ 改 capability/showcase/cleanup — 不涉及

## 7. 执行步骤

1. 切到分支 `feat/pr11-gpu-isolation` (已切)
2. 写 `tests/test_pr11_gpu_isolation.py` (TDD)
3. 跑测试 → 大量 fail (代码还没改)
4. 改 `orchestrator/stages_py.py` (新 helper + execute_deploy)
5. 改 `orchestrator/state_machine.py` (`mark_skipped`)
6. 改 `orchestrator/main.py` (识别 aborted)
7. 改 `orchestrator/notify.py` (`run_aborted`)
8. 跑测试 → 全绿
9. 全套回归 + ruff + mypy
10. self-review + push + 开 PR + AI Reviewer 首过
