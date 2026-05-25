# PR#36 honesty experiment report

> 部署位置：nv8 (`<NV8_HOST_IP>`)，分支 `feat/pr22b-agent-runner`，commits `ccf82ae..77bb917`。
> 测试方法：在 PR#33 + PR#35 之上叠加 PR#36，重新跑 PR#33 报告里列出的两类「说谎」案例（`unsloth/Qwen3.6-27B-GGUF` + `MahmoudAshraf/mms-300m-1130-forced-aligner`），看是否把 `status=ok` 变成 honest `failed/repaired`。

## 背景：PR#33+PR#35 报告里识别出的两个 honest failure

来源：`docs/PR33_DEPLOY_REPAIR_EXPERIMENT.md` 结尾段：

1. **Container-healthy-but-inference-broken**：`/v1/models` 200，`/v1/chat/completions` 501 — PR#33 标记 repair 成功，CAPABILITY 产出 0/N pass。
2. **CAPABILITY status=ok 却 0% pass**：whisperkit-coreml + qwen3.6-GGUF 都是 0% pass + 全部 `http 501`，但 panel 显示绿勾。

PR#36 就是冲着这两条来的。

## PR#36a：DEPLOY 层 inference probe

### 设计

在 `_try_once` 的 early-crash-window 轮询结束、且 `/v1/models` 已列出模型之后，新增一次 `POST /v1/chat/completions { messages: [{role:user, content:"hi"}], max_tokens: 1 }` 探针调用：

- 2xx + 非空 `choices[]` → 真活；继续正常 OK 路径
- 5xx / 4xx / 0xx / 空 choices → 当作 `error_kind="inference_broken"` 上抛，进入 PR#33 修复循环
- 把响应体（包含 `NotImplementedError` 等关键词）放进 `logs`，让 classifier 能区分 `inference_not_implemented` vs `inference_other_5xx`

### 三个补充修复

PR#36a 上线前漏了三件事，live 实验暴露后当场补：

1. **strategy 列表 frozen 在 iter-0**（commit `c5ec59f`）：原 PR#33 的修复循环只在初始失败时调一次 `propose_attempts`，后续 attempt 演化出新失败 shape 时不会重新分类。改成 while 循环 + `seen_strategies` 去重 + 每次 attempt 后用实际 outcome 构建新的 `DeployFailure`。`max_iterations=8` 防死循环。

2. **`cc_agent.deploy_repair_agent` 调用了不存在的 `client.chat()`**（同上 commit）：`HeyiEngineClient` 暴露的是 `.call(messages=...)`，不是 `.chat(model=..., messages=..., temperature=..., max_tokens=..., timeout=...)`。PR#33 在 dogfooding 时所有 rule-based 策略都成功了，agent 通道从未被触达，所以这个 AttributeError 是 PR#36 跑 Qwen3.6 时第一次现行。

3. **probe 对 ASR/TTS/image-gen 模型过度激进**（commit `77bb917`）：mms-300m 是 wav2vec ASR 模型，transformers-runner **正确地**在 chat/completions 返回 501（它的接口是 `/v1/audio/transcribe`）。改成读 `<run>/_meta/curated.json::capability_tags`，只对包含 `text/code` tag 或 tag 缺失的模型 fire probe；纯 `audio/asr/tts/image-gen/...` 模型跳过 probe。

### PR#36b：CAPABILITY 层 honesty gate

在 `execute_capability` 末尾，根据 `all_items_flat` 的 actual + error 模式做三道闸：

```
total >= 4
pass_count == 0
errored items (error 非空 AND actual 为空) 占比 >= 80%
errored items 中最常见的 normalized signature 占比 >= 80%
```

四条都满足 → 返回 `ok=False, error_kind="capability_endpoint_broken"`，并在 `capability.json` 里写入 `broken_endpoint: <reason>` 字段。

为了避免误伤「模型其实能跑、就是答不对题」的 case（0.5B 小模型刷不动 GSM8K），区分两种 0% 的失败：
- 真错（model is dumb）：actual 非空（生成出来的答案错了），error 空 → 不触发 gate
- 假装活着（endpoint broken）：actual 空，error 全是相同 HTTP 状态码 → 触发 gate

数字正则化（`re.sub(r"\d{2,}", "<n>", ...)`）把 `http 501/502/503` 折叠成同一个 signature，避免单一类错误因状态码抖动逃过 gate。

## 实验结果（nv8 live）

### 案例 1：`unsloth/Qwen3.6-27B-GGUF`

| 字段 | PR#33+PR#35 期间 | PR#36 之后 |
|---|---|---|
| Final status | `ok` | `failed` |
| pass_rate | 0.0 | n/a（DEPLOY 阶段就失败了） |
| score | 0/50（全是 `http 501`） | DEPLOY 失败，CAPABILITY 不跑 |
| failure_reason | 空 | `container exited within early-crash window; tail logs: ...` |
| strategies_proposed | `['use_gguf_file_path', 'swap_engine_transformers']`（2 个，frozen） | `['use_gguf_file_path', 'swap_engine_transformers', 'swap_vllm_image_latest', 'swap_engine_sglang', 'add_trust_remote_code', 'lower_max_model_len']`（6 个，iterative） |
| attempts | 3（initial + 2 strategies） | 7（initial + 6 strategies） |
| agent_escalation | `AttributeError: 'HeyiEngineClient' object has no attribute 'chat'` | `parse_error`（MiniMax 在 `<think>` 里思考但不输出 JSON），见 §未解决 |

最关键的差异：
- PR#36a iter 2 在 `cls=inference_not_implemented` 上重新提案，吐出了 `[swap_vllm_image_latest, swap_engine_sglang]` —— 这是 PR#33 frozen 列表里**从未出现过**的两个策略。
- iter 2 / iter 3 全跑完后真正穷尽时，run 被标记 `failed`，**不是说谎的 `ok`**。
- attempt 3 (`swap_engine_transformers`) 的 `error_kind` 是 `inference_broken`，证明 PR#36a 的 probe 在 transformers-runner 上真的命中 501 并归类到 `inference_not_implemented`。

### 案例 2：`MahmoudAshraf/mms-300m-1130-forced-aligner`

| 字段 | PR#36 之前 | PR#36 之后 |
|---|---|---|
| Final status | `ok` | `failed` |
| pass_rate | 0.0 | 0.0 |
| score | 0/5 | 0/5 |
| failure_reason | 空 | **`0/5 pass; 5/5 items errored with the same pattern 'http <n>'`** |
| capability.json::broken_endpoint | 不存在该字段 | `0/5 pass; 5/5 items errored with the same pattern 'http <n>'` |
| Panel 显示 | 绿勾 ✓（假阳性） | 红叉 ✗ + 明确原因 |

mms-300m 验证了 PR#36a 的 modality gate（DEPLOY OK、probe 没 false-fire）+ PR#36b 的 honesty gate（CAPABILITY 0/5 → failed）。

### 副产品：`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` 是这一批的真冠军

不在 PR#36 直接目标里，但同期跑过的 gemma 4 26B AWQ-4bit 拿了 **35/45 = 0.778 pass_rate**，是 PR#22 以来第一个跨 13 个 capability 类别全部跑通且整体高分的模型。AWQ-4bit 这条路径目前看是「主流文本模型 + 商用 vLLM 镜像」的最佳组合，单卡能塞 26B 量化版本。

## 单测覆盖

`tests/test_pr36_inference_probe.py`（26 tests，全绿）：

- `InferenceProbeTests`（4）：probe 函数本身的正确性，2xx/empty-choices/501/连接拒绝四档
- `InferenceProbeInTryOnceTests`（8）：包括 modality gate 的 4 个新 case（audio-only skip / text fire / text+audio fire / no curated fire）
- `InferenceClassificationTests`（5）：`inference_not_implemented` vs `inference_other_5xx` 分类 + strategy applicability 跨 engine
- `CapabilityHonestyGateTests`（6）：uniform error → trigger / legit wrong → no trigger / mixed errors / too few items / 任何 pass>0 都豁免 / 生产代码源代码 grep 防 ripout
- `IterativeRepairFlowTests`（4）：propose_attempts 在 4 个连续失败 shape 上分别返回不同策略集

Local + nv8 双向跑通：**825 passed**, 7 skipped, 1 deselected（PR#32 GGUF 测试在我笔记本上磁盘不够，nv8 上能过；故 deselect-on-laptop）。

## 未解决（留给 PR#37）

1. **MiniMax-M2.7 agent escalation 输出纪律差**：Qwen3.6-GGUF 的 3 次 agent attempt 全因 `parse_error: no JSON in response` 失败，模型把全部内容塞进 `<think>` block 然后停止生成。需要：(a) system prompt 加入「禁止 `<think>` 包裹最终输出」约束；(b) 退化解析：若 `<think>` 后没有 JSON，从 `<think>` 内部抓 JSON。

2. **inference_broken classification 内 `swap_engine_transformers` 排在 `swap_vllm_image_latest` 前面**：PR#36a iter 1 默认顺序是 `[use_gguf_file_path, swap_engine_transformers]`，会导致先掉进 transformers-runner 再被迫从那里反弹回 vllm:latest。优化顺序应该是 `gguf_needs_file_path → use_gguf_file_path → swap_vllm_image_latest → swap_engine_sglang → swap_engine_transformers`（transformers 作为最后兜底）。

3. **discover/curate 给 ASR 模型分配了一整套不合身的 capability category（mu/code/tr 等都跑）**：mms-300m 的 cats list 包含 `text_reasoning, code_gen, vision, ...` 全套 13 类，5 个被实际跑过的都是 music_understanding 测试。这是 PR#27 modality dispatch 的盲点 —— `capability_tags=['audio']` 时应只允许 `asr` 类别，其他全 skip。修这个能让 mms-300m 从「endpoint broken」变成「audio model with no music tests applicable → 跳过 CAPABILITY」。

## 提交记录

```
ccf82ae feat(pr36): inference probe at DEPLOY + capability honesty gate
c5ec59f fix(pr36a): iterative repair propose + agent client.call fix
77bb917 fix(pr36a): make inference probe modality-aware
```
