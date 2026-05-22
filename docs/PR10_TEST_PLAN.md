# PR#10 — 概念分离 (产线 LLM vs 评估 LLM) 测试计划

> 阶段 1 (架构) + 阶段 2 (测试用例) 设计文档。本 PR 不实现 PR#11 的 GPU 隔离动作;只做**概念边界拆清**和**配置参数化**,让后续 PR 有可拼装的零件。

## 0. 为什么要做这个 PR

v10 当前代码库**混淆了两件事**:

| | 产线 LLM | 评估 LLM |
|---|---|---|
| 谁 | heyi_engine 框架背后那台 vLLM 实例 (常驻,几个月才换一次) | 评估管道在 DEPLOY 阶段临时 spawn 的 `e9-*` 容器 (跑 5-30 分钟即销毁) |
| 模型 | 默认 **MiniMax-M2.7** (用户主稳态);临时态 Kimi-K2.6;未来还会换 | DISCOVER 阶段拉的待测模型 (任意) |
| 引擎 | 固定 vLLM (heyi-engine 决定) | **vLLM / SGLang / Transformers** 三选一 (DEPLOY 阶段决定) |
| GPU | M2.7 TP=4 占 0-3;K2.6 TP=8 占 0-7 (临时) | **必须独占** GPU 4-7,严禁碰 0-3 |
| 并发 | 1 (常驻) | 1 (一次只跑一个,无并行) |
| 资源不够 | n/a | graceful skip + `aborted=insufficient_gpu`,不报错 |

但代码里:

- `orchestrator/validator.py:194` `minimax_gpus: tuple[int, ...] = (0, 1, 2, 3)` 把"产线占哪几张卡"当成默认参数硬编
- `orchestrator/validator.py:204` `docker inspect minimax` 字面量字符串硬编产线容器名
- `orchestrator/config.py` 完全没有"产线容器名"配置项
- `tests/data/production_container_names.txt` 是 repo 常量,产线模型切换时无人会同步更新
- `heyi_engine/client.py` docstring 写死 "minimax / glm-51 / kimi-k26" 三个具体型号作为可能服务对象,把"产线 LLM 是配置化的当前模型"这件事讲糊了
- `docs/INVARIANTS.md` INV-2 描述把多种产线容器名 (`minimax-*`/`xrouter`/`glm-*`/`kimi-*`/`voipmonitor`) 揉成一句话,没区分 "产线 LLM (单一,可切换)" 和 "产线辅助服务 (xrouter 等,稳定)"

PR#10 的范围就是**只把这五处混淆拆开**,让后续 PR#11 能在干净的接口上加 GPU 隔离逻辑。

## 1. 不变量决定 (设计契约)

PR#10 引入或重申以下契约:

1. **产线 LLM 容器名是配置项,不是常量**。默认 `minimax`,通过 `HEYI_EVAL_PROD_ENGINE_CONTAINER` 覆盖。
2. **产线 LLM 占的 GPU 是配置项**。默认 `(0,1,2,3)`,通过 `HEYI_EVAL_PROD_ENGINE_GPUS` 覆盖 (逗号分隔)。
3. **评估侧独占的 GPU 是配置项**。默认 `(4,5,6,7)`,通过 `HEYI_EVAL_EVAL_GPUS` 覆盖。
4. **产线 LLM 与产线辅助服务是两个概念**。INV-2 表述区分:
   - "**产线 LLM 容器**"(由 `prod_engine_container` 指定,默认 minimax) - 受运行时 `assert_invariants` 守护
   - "**产线辅助服务**"(`xrouter`/`voipmonitor` 等) - 仅受静态 INV-1/INV-4 守护,不在 `assert_invariants` 检查范围 (它们的可用性不影响评估 pipeline)
5. **`tests/data/production_container_names.txt` 是静态扫描守卫边界清单,不是运行时配置**。文件含义重述清楚,新增几个真机实际看到的容器名 (`kimi-k26`、`Xinf`、`Open-Webui` 等),把注释更新到位。

## 2. 文件改动清单

| 文件 | 改动 | 类型 |
|---|---|---|
| `orchestrator/config.py` | 新增 `prod_engine_container`/`prod_engine_gpus`/`prod_engine_min_gpu_mib`/`eval_gpus` 字段 + env 覆盖 | feat |
| `orchestrator/validator.py` | `assert_invariants` 签名改成接受 `prod_engine_container`/`prod_engine_gpus`/`expected_prod_gpu_min_mib` 参数;错误消息用变量,不再有 "minimax" 字面量;旧参数名 `minimax_gpus` 保留作 deprecated alias,内部转发 (向后兼容,但发 DeprecationWarning) | refactor |
| `orchestrator/main.py` | 调用 `assert_invariants` 处改为传 `cfg.prod_engine_container` / `cfg.prod_engine_gpus` (具体改动看 main.py 现状,可能只 1-2 处) | refactor |
| `heyi_engine/client.py` | 模块 docstring 去掉 "minimax / glm-51 / kimi-k26" 三型号枚举,改成抽象的 "the production LLM currently loaded on :10814,configured by ops" | docs |
| `tests/data/production_container_names.txt` | 注释更新清楚:这是**静态扫描边界清单**,不是运行时配置;补全真机实际看到的产线容器名 | docs+data |
| `tests/test_inv_production_isolation.py` | 文件头注释里多处 "minimax" 作为产线代表的措辞改成抽象;INV-2 描述区分产线 LLM/辅助服务 | docs |
| `docs/INVARIANTS.md` | 加 "产线 vs 评估两层" 对比表;INV-2 描述参数化 (引用 `OrchestratorConfig.prod_engine_container`) | docs |
| `docs/PR10_TEST_PLAN.md` | 本文档 | docs (新增) |
| `tests/test_pr10_concept_split.py` | 新增,详细见下面 §3 | test (新增) |

不动的文件:
- `orchestrator/stages_py.py` (PR#11 才动)
- `tests/data/production_container_names.txt` 的命名规则 (文件名不改)
- 任何 e2e 测试 (不能挂)
- INVARIANTS.md 的 INV-1/INV-3/INV-9/INV-11/INV-12/INV-13 行 (PR#10 只改 INV-2 + 加表)

## 3. 测试用例清单 (功能完整性 — Happy + Sad + 边界)

新增 `tests/test_pr10_concept_split.py`,覆盖如下 12 个 case。每个 case 都基于业务目标 ("产线/评估概念分离生效,产线模型切换不再让代码失灵") 描述用户可观察结果。

### A. 配置层 (5 case)

| ID | 场景 | 输入 | 预期可观察结果 |
|---|---|---|---|
| C1 | 默认值不变 | 不设任何 env var,`OrchestratorConfig()` 构造 | `cfg.prod_engine_container == "minimax"`; `cfg.prod_engine_gpus == (0,1,2,3)`; `cfg.prod_engine_min_gpu_mib == 80000`; `cfg.eval_gpus == (4,5,6,7)` |
| C2 | 产线切到 K2.6 临时态 | `HEYI_EVAL_PROD_ENGINE_CONTAINER=kimi-k26`; `HEYI_EVAL_PROD_ENGINE_GPUS=0,1,2,3,4,5,6,7` | `cfg.prod_engine_container == "kimi-k26"`; `cfg.prod_engine_gpus == (0,1,2,3,4,5,6,7)` |
| C3 | 评估池配置覆盖 | `HEYI_EVAL_EVAL_GPUS=2,3` | `cfg.eval_gpus == (2,3)` |
| C4 | 异常输入 (非数字) | `HEYI_EVAL_PROD_ENGINE_GPUS="0,abc,3"` | 构造时抛 `ValueError`,错误消息明确指向是哪个 token 解析失败 |
| C5 | 异常输入 (空字符串元素) | `HEYI_EVAL_EVAL_GPUS=",,"` | 解析为 `()` (空 tuple);不抛错 (代表"评估侧无 GPU 可用",PR#11 graceful skip 路径会用到) |

### B. validator 参数化 (5 case)

注: 这些 case 用 `monkeypatch` 替换 `subprocess.run`,不真打 docker/nvidia-smi。

| ID | 场景 | 输入 | 预期可观察结果 |
|---|---|---|---|
| V1 | 默认产线容器 happy path | `assert_invariants()` 不传参;fake subprocess: `docker inspect ... minimax` → `running`; `nvidia-smi` → GPU 0-3 各 89000 MiB | 不抛异常 |
| V2 | 自定义产线容器 happy path | `assert_invariants(prod_engine_container="kimi-k26", prod_engine_gpus=(0,1,2,3,4,5,6,7))`; fake subprocess: `docker inspect ... kimi-k26` → `running`; GPU 0-7 各 89000 MiB | 不抛异常;验证 fake `docker inspect` 实际收到的容器名参数 == `kimi-k26` (不是 `minimax`) |
| V3 | 容器名不匹配 sad path | `assert_invariants(prod_engine_container="kimi-k26")`; fake: `docker inspect ... kimi-k26` → `exited` | 抛 `ValidationError`,**消息含字符串 `kimi-k26`**,**不含字符串 `minimax`** |
| V4 | GPU 内存不足 sad path | `assert_invariants(prod_engine_gpus=(4,5))`; fake GPU 4 mem=1000 MiB (< 80000) | 抛 `ValidationError`,消息含 `GPU 4`,**仅对 (4,5) 检查不对 (0,1,2,3) 检查** |
| V5 | 旧参数名 backward compat | `assert_invariants(minimax_gpus=(0,1,2,3))` (调用旧 API) | 发 `DeprecationWarning`,但功能等同于 `prod_engine_gpus=(0,1,2,3)`;passes |

### C. 文档/数据合规 (2 case,静态扫描)

| ID | 场景 | 输入 | 预期可观察结果 |
|---|---|---|---|
| D1 | client.py docstring 抽象化 | 读取 `heyi_engine/client.py` 文件文本 | 模块 docstring **不含**子串 `"minimax / glm-51 / kimi-k26"` 或 `"minimax/glm-51/kimi-k26"`;**含**子串 `"production LLM"` 或 `"prod_engine_container"` 或类似抽象表达 |
| D2 | INVARIANTS.md 双层化 | 读取 `docs/INVARIANTS.md` | 文件含字符串 `"产线 LLM"` 和 `"评估 LLM"` 两个表头标识;INV-2 行**不含**裸 `"minimax-*"` 作为唯一硬编名,**含**对 `prod_engine_container` 的引用 |

### D. 回归 (现有测试必须继续 pass)

| ID | 范围 |
|---|---|
| R1 | `tests/test_inv_production_isolation.py` 全部 case (INV-1/4/12/13 静态守卫不挂) |
| R2 | `tests/test_no_v9_residue.py` (INV-11 不挂) |
| R3 | `tests/test_e2e_pipeline.py` (e2e harness 不挂) |
| R4 | `tests/test_systemd_units.py`、`tests/test_backup_snapshot.py` 等所有其他测试不挂 |

## 4. 业务目标三问 (合并门禁)

合并 PR#10 前必须三问全 "是":

1. **主路径**: 产线维持 minimax (默认稳态) 时,所有调用 `assert_invariants()` 的 stage gate 行为与 PR#10 前**完全一致**?
   → 由 V1 case + R1-R4 回归覆盖
2. **切换路径**: 产线临时切到 K2.6 时,operator 只需要设两个 env var (`HEYI_EVAL_PROD_ENGINE_CONTAINER=kimi-k26`、`HEYI_EVAL_PROD_ENGINE_GPUS=0,1,2,3,4,5,6,7`),`assert_invariants` 立即切换守护对象,不需要改任何代码?
   → 由 C2 + V2 case 覆盖
3. **状态集**: 容器名错、GPU 不够、参数无效三种 sad path 都有明确错误消息,operator 一看就知道是哪里的配置错?
   → 由 C4 + V3 + V4 case 覆盖

## 5. 不在 PR#10 范围 (留给 PR#11/PR#12)

- ❌ `stages_py.execute_deploy` 显式注入 `CUDA_VISIBLE_DEVICES=4,5,6,7` → PR#11
- ❌ 评估侧 deploy 前 nvidia-smi 探测 4-7 是否被产线临时占用 → PR#11
- ❌ Graceful skip on `aborted_reason=insufficient_gpu` → PR#11
- ❌ RUNBOOK / PLAN.md / ARCHITECTURE.md 全面双层化 → PR#12
- ❌ `bootstrap_nv8.sh` hostname/python3 假设 → PR#12

## 6. 执行步骤 (用户确认后)

1. 切到分支 `feat/pr10-concept-split` (已切)
2. 写 `tests/test_pr10_concept_split.py` (先写测试,TDD)
3. 跑测试 → 全 fail (因为代码还没改)
4. 改 `orchestrator/config.py` + `orchestrator/validator.py` → 让 A/B 段 case 通过
5. 改 `orchestrator/main.py` 调用处适配
6. 改 `heyi_engine/client.py` docstring → D1 通过
7. 改 `docs/INVARIANTS.md` → D2 通过
8. 改 `tests/data/production_container_names.txt` 注释 + 补真机名 → 跑 INV-1/13 静态扫描验证 R1 不挂
9. 改 `tests/test_inv_production_isolation.py` 文件头注释里的 "minimax" 措辞
10. 跑全套测试 + ruff + mypy → 全绿
11. self-review (在 GitHub Files changed 视图过一遍 diff)
12. push + 开 PR + AI Reviewer 首过
