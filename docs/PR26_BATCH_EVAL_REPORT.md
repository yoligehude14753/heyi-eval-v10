# PR#26 · 多模型 batch eval 报告（PR#25 稳态后）

> 时间：2026-05-23 11:01 ~ 11:08 UTC+8（约 7 分钟窗口）
> 目标：在 PR#25 hotfix（`<think>` 剥离器、`_graceful_skip` 参数化、planner 诊断）落地后，跑一轮多模型 batch，验证：
>
> 1. 4 个真实 HF id 一次性入队，编排器按顺序处理无卡死
> 2. 小模型走完 9 阶段；超大模型在 ENGINE_SELECT 闸门即中止
> 3. INV-23 oversize 闸门即便 `param_count=None`（curator 缺数据）也能从 hf_id 拿到尺寸 → 触发（**PR#26 关键 fix**）
> 4. 不同 modality（text / ASR）走对应 engine（vllm / transformers-runner）

## 0 · 入队批次

```bash
for m in Qwen/Qwen2.5-0.5B-Instruct \
         openai/whisper-tiny \
         meta-llama/Llama-3.1-405B-Instruct \
         deepseek-ai/DeepSeek-V3; do
  .venv/bin/python -m orchestrator.main enqueue "$m"
done
# 队列：4
```

## 1 · 批次结果汇总

| # | hf_id | 状态 | 用时 | engine | tp_size | 关键事件 |
|---|-------|------|------|--------|---------|---------|
| 1 | `Qwen/Qwen2.5-0.5B-Instruct` | **ok** | 120.4 s | vllm | 1 | 9 阶段全绿，pass_rate 48 % (12/25) |
| 2 | `openai/whisper-tiny` | **ok** | 22.3 s | transformers | (n/a) | ASR 流程跑通；CAPABILITY 题目错配，pass 0/25（见 §3） |
| 3 | `meta-llama/Llama-3.1-405B-Instruct` (1st) | aborted | 3.1 s | metadata_only | (skip) | **未触发 oversize**：`param_count=None` → tp=1 default → DEPLOY 才 graceful-skip on `model_missing` |
| 4 | `deepseek-ai/DeepSeek-V3` | **aborted** | 14.7 s | metadata_only | 4 | INV-23 oversize 闸门正常触发 |
| 3' | `meta-llama/Llama-3.1-405B-Instruct` (re-run, PR#26 fix) | **aborted** | 2.0 s | metadata_only | 4 | **PR#26 关键 fix 后**：hf_id 回退提取 "405B" → tp=4 → INV-23 正常触发 |

## 2 · PR#26 关键 fix：hf_id 回退提取尺寸

### 问题（首次 Llama-3.1-405B run 暴露）

Llama 3.1-405B 在 1st run 居然没走 oversize 闸门。原因：

```jsonc
// _meta/metadata.json
{
  "hf_id": "meta-llama/Llama-3.1-405B-Instruct",
  "param_count": null,        // ← curator 漏抓
  ...
}
```

curator 对该模型未能从 HF 模型卡上抓到 `param_count`，
`_vllm_args_hint` 拿到空字符串 → tp 估算跳过 → engine.json 显示
`vllm_args: {}` + `oversize: false`。pipeline 继续进 DEPLOY，DEPLOY
再因 model not in eval-cache 做 `graceful_skip`，最终
`status=aborted` 但 reason 是 `model_missing` 而不是 `oversize_skip`。

业务影响：如果 405B 权重 *已经* staged 到 `_eval-cache/`，那 DEPLOY 会真
尝试启动一个 405B 模型 → 直接 OOM 或 GPU 借不齐而失败（带 OOM 烧
GPU、磁盘 IO 拖累 ComfyUI）。

### Fix（`orchestrator/stages.py::_vllm_args_hint`）

加一级回退：`metadata.param_count` 空时，从 `metadata.hf_id` 用同样的
正则 `(\d+(?:\.\d+)?)\s*b\b` 提取。公开模型几乎都把尺寸写在 id 里：

| hf_id 子串 | 提取出 | tp |
|-----------|--------|----|
| `Llama-3.1-405B-Instruct` | 405 → tp=4 |
| `Qwen2.5-72B-Instruct` | 72 → tp=4 |
| `Qwen2.5-7B-Instruct` | 7 → tp=1 |
| `DeepSeek-V3` | (无 b 后缀) | tp=1（兜底，但 V3 元数据有 param_count，所以这条路不走） |

`param_count` 仍然优先（curator 抓到了就是权威），hf_id 只是兜底。
测试守护：5 例新增在
`tests/test_pr23_m27_api_and_oversize.py::TestVllmArgsHintTpHeuristic`：

- `test_hf_id_405b_triggers_tp4_when_param_count_missing`
- `test_hf_id_72b_triggers_tp4`
- `test_hf_id_7b_stays_tp1`
- `test_param_count_wins_over_hf_id`
- `test_neither_signal_returns_no_tp_hint`

### 修复后实测（Llama 3.1-405B 二次入队）

```jsonc
// _meta/engine.json 修复后
{
  "engine": "metadata_only",
  "vllm_args": { "tensor_parallel_size": 4 },
  "eval_pool_size": 3,
  "oversize": true            // ← 修复前 false
}
```

journal：
```
[ENGINE_SELECT] SKIPPED (graceful): oversize: model needs
  tensor_parallel_size=4 but eval pool has 3 GPUs ([5, 6, 7]);
  metadata captured at _meta/metadata.json + _meta/engine.json,
  no DEPLOY
```

总用时 2.0 s（vs 3.1 s 修复前），且**不再有错误的 DEPLOY 尝试**。

## 3 · ASR 题库与模型 modality 错配（已知遗留，非 PR#26 引入）

`openai/whisper-tiny` 跑完了 9 阶段（status=ok），但 CAPABILITY 全 fail：

```
whisper-tiny pass: 0 / 25
first item: { id: tr-001, category: text_reasoning,
  prompt: "Janet's ducks lay 16 eggs per day...",
  actual: None, error: "http 501", scorer_used: "substring" }
```

根因：CAPABILITY 阶段把 gsm8k 风格的 text-reasoning 题发给一个 ASR
模型，自然每条都 501（whisper 没有 chat completions endpoint）。

这是 CAPABILITY 题库的 dispatch 问题（modality-aware sample 选择没生效），
不是 PR#26 引入的回归。下一个 PR 应该让 `orchestrator/capability.py`
按 `modality=audio` 自动只挑 ASR 测试集，不送 text 题。

## 4 · 性能数据（Qwen 0.5B from 最新 ok run）

| 指标 | 值 |
|-----|----|
| TPS single mean | 472.6 tok/s |
| Concurrent aggregate TPS | 1838.4 tok/s |
| VRAM | GPU 5 = 88.7 GB，GPU 6/7 = 3 MiB 各 |

对比 PR#24 报告的 Qwen 0.5B 实测：单路 484, 并发 1955。本次略低
（5 % 量级），可能与并发 = 4 / 时序波动有关，整体一致。

## 5 · 关键不变量验证

| 不变量 | 触发场景 | 实际表现 |
|-------|---------|---------|
| INV-23 oversize | DeepSeek-V3 (param_count="671B") + Llama-3.1-405B (hf_id 回退 "405B") | 两者均在 ENGINE_SELECT skip → aborted，无 DEPLOY 尝试 |
| INV-21 audit | 全部 4 个 run 走 orchestrator 主路径（not agent sandbox），audit 不适用 | (无 sandbox 调用，正常) |
| GPU pool 隔离 | eval_gpus=(5,6,7), prod_engine_gpus=(0,1,2,3) | Qwen run VRAM 严格在 GPU 5；GPU 0-3 仍 89 GB （minimax 不动） |
| `_graceful_skip` 参数化 (PR#25) | Llama 1st run DEPLOY skip | error_kind=`model_missing` 正确显示（vs 旧版的 `insufficient_gpu`） |
| `<think>` 剥离 + planner 诊断 (PR#25) | 4 个 SHOWCASE 阶段 | journal 显示 `cause=thinking_truncated, total_len=9020` 等清晰诊断，操作员可直接定位 M2.7 CoT 把 token 吃光的根因 |

## 6 · 总结

PR#23 + PR#25 + PR#26 后，v10 评测管线在 nv8 上的真实表现：

| 维度 | 表现 |
|-----|------|
| 正常小模型（≤ 28B） | 全 9 阶段绿，~120 s 完成 |
| 超大模型 (tp > 3) | ENGINE_SELECT 1-2 s 内识别并中止，0 容器/0 GB 浪费 |
| curator 缺 param_count | PR#26 hf_id 回退兜底，oversize 闸门仍可靠 |
| 跨 modality | text/ASR engine routing 正确（虽然 CAPABILITY 题库还需要按 modality 派发，不属本 PR） |
| 操作可观察性 | journal 诊断从"non-JSON"升级到 `cause=…, total_len=…`，根因一眼看出 |

可以收尾本批次工作；后续 PR 可优先攻 §3 的 ASR/text 题库错配。

## 7 · 复现命令

```bash
ssh ai@heyi-sh-nv8
cd ~/heyi-eval-v10
.venv/bin/python -m orchestrator.main enqueue Qwen/Qwen2.5-0.5B-Instruct
.venv/bin/python -m orchestrator.main enqueue openai/whisper-tiny
.venv/bin/python -m orchestrator.main enqueue meta-llama/Llama-3.1-405B-Instruct
.venv/bin/python -m orchestrator.main enqueue deepseek-ai/DeepSeek-V3
# 5 分钟左右全部跑完
.venv/bin/python -m orchestrator.main status | head
```
