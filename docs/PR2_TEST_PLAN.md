# PR#2 测试用例清单

> rules § 00-core 阶段 2：编码前先写测试用例清单，业务目标视角，**用户可观察结果**，不是"接口被调用 N 次"。

## 业务目标

curator + showcase 调用 LLM 时：
1. **不依赖写死的 model name** —— user 在 :10814 上随时切换模型（M2.7 / Kimi / GLM）pipeline 自动识别，不需要重启
2. **失败可观察** —— :10814 down 时 orchestrator 暂停取任务（队列保留），WeChat 收到 incident，等 :10814 恢复后自动复活
3. **call 接口稳定** —— 业务代码 `client.call(messages, max_tokens=N)` 行为一致，不感知背后哪个模型在跑

## 功能完整性清单

### Happy Path（主路径）

| 编号 | 用例 | 用户视角的可观察结果 |
|---|---|---|
| H1 | `:10814` 上跑 `Kimi-K2.6`，client.health() | `Healthy(model_id='Kimi-K2.6')` |
| H2 | client.discover_model() 首次调用 | 返回 `'Kimi-K2.6'`，缓存 60s |
| H3 | 60s 内重复 discover_model() | 命中缓存，**不发** /v1/models 请求 |
| H4 | client.call(messages, max_tokens=50) | 返回 `CallResult(text='PONG', input_tokens=N, output_tokens=M)` |
| H5 | user 在 :10814 切到 `MiniMax-M2.7`，60s 后再 discover_model() | 返回 `'MiniMax-M2.7'`（自动跟随上游） |
| H6 | curator.enricher 用 client 取代 CCR | 产出 curated.json schema 与 v9 一致 |
| H7 | orchestrator 启动时 preflight gate 用 client.health() | unhealthy 时**不消队列**，每 60s 重试一次，恢复后继续 |

### Sad Path（失败路径）

| 编号 | 用例 | 用户视角的可观察结果 |
|---|---|---|
| S1 | :10814 connection refused | `Unhealthy(detail='connection refused')`，**不 raise** |
| S2 | :10814 返回 HTTP 500 | `Unhealthy(detail='HTTP 500')`，body 截断写入 detail |
| S3 | :10814 返回 HTTP 200 但 body 不是 JSON | `Unhealthy(detail='invalid JSON: ...')` |
| S4 | :10814 返回 `{"data":[]}` 空模型列表 | `Unhealthy(detail='no models available')` |
| S5 | :10814 超时（默认 30s） | `Unhealthy(detail='timeout after 30s')`，**不挂主线程** |
| S6 | client.call() 而 health 未通过 | raise `HeyiEngineError(...)`，含 last health detail |
| S7 | client.call() vllm 返回 4xx 错误 model name | raise `HeyiEngineError`，触发 discover_model() 强制刷新 |
| S8 | curator preflight 失败 | curator stage 进入 degraded mode（HF Hub only），emit incident，**返回 ok=True**（队列保留） |
| S9 | orchestrator preflight gate 第一次失败 | 写 incident `orchestrator-paused-engine-down`，state.engine_unhealthy_since 记录时刻 |
| S10 | orchestrator preflight 连续失败 30 min | 仍仅维护 unhealthy 状态，每 30 min 重发一次 incident（避免刷屏） |
| S11 | orchestrator preflight 恢复 | 写 incident `orchestrator-resumed-engine-recovered`，down 时长记录在 body |

### 边界场景

| 编号 | 用例 | 用户视角的可观察结果 |
|---|---|---|
| E1 | `/v1/models` 返回 `data[0].id` 字段缺失 | `Unhealthy(detail='model entry missing id')` |
| E2 | 模型名含特殊字符（`Kimi-K2.6`，`.` 在）| 正常使用，无 escape 错误 |
| E3 | 多个模型条目 `data=[m1, m2]` | 用 `data[0].id`，warning 日志记 2nd+ 被忽略 |
| E4 | :10814 服务存在但 `/v1/models` 404（部分 vllm 在 `/models`） | fallback 探测 `/models`；都失败才 Unhealthy |
| E5 | concurrent call() during refresh | 两个 call 共用一次刷新，不发起重复 /v1/models |
| E6 | client.refresh_model() 强制刷新（绕 60s TTL） | 立即发起请求，更新缓存 |

## 测试组织

| 测试文件 | 覆盖用例 |
|---|---|
| `tests/test_heyi_engine_client.py` | H1-H5, S1-S7, E1-E6 |
| `tests/test_curator.py`（修改） | H6 + S8（curator 行为不感知后端切换） |
| `tests/test_orchestrator_loop_gates.py`（修改） | H7, S9-S11（rename ccr → engine） |
| `tests/test_curator_health.py`（重写） | probe → client.health() 适配 |

## 覆盖率目标

`heyi_engine/` 模块单测覆盖率 ≥ 95%（新代码，应当满覆盖）；
整体仓库覆盖率 ≥ 80%（rules § 09-cicd）。

## Mock 策略

- 所有 HTTP I/O 通过 `unittest.mock.patch('urllib.request.urlopen')` 注入，**不**起真实 server
- 集成测试（标记 `@pytest.mark.slow`）跑 nv8 :10814 真实服务，仅手动触发，不入 CI

## 业务目标三问（PR#2 收尾时再过一遍）

1. 主路径可用：curator + orchestrator 真实在 nv8 上调通 :10814 上的 Kimi/M2.7
2. 失败路径有反馈：:10814 down 时 wechat 收到 incident
3. 状态集完整：unhealthy / unhealthy-persistent / recovered 三态都覆盖
