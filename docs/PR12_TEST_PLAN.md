# PR#12 — L2/L3 双层化 + bootstrap 假设清理

> PR#10 把"产线 LLM"和"评估 LLM"在配置层拆开,PR#11 把拆分落到 DEPLOY 路径。PR#12 的工作是把这些**已经发生的代码层变化**反映到运维侧的两层文档(L2 不变量 / L3 操作手册),并顺手清掉 `bootstrap_nv8.sh` 里几处隐性环境假设。

## 0. L2 vs L3 在本仓库的定义

| 层 | 文件 | 受众 | 内容形态 | 修改门槛 |
|---|---|---|---|---|
| L2 (Why) | `docs/INVARIANTS.md`、`docs/ARCHITECTURE.md` | 架构师、reviewer | 不变量编号 + 信任边界 + 设计决策记录 | 每条不变量配静态/运行时守卫 |
| L3 (How) | `docs/RUNBOOK_NV8.md`、`docs/USAGE.md`、`scripts/bootstrap_nv8.sh` | 运维、首次接手者 | 命令清单 + 通过标准 + 失败时怎么办 | 每个步骤可以盲跑(无脑复制粘贴) |

L2 写"为什么不能碰产线 GPU";L3 写"什么命令能看到产线 GPU 当前状态"。两者不能合在一起,否则:
- L3 被原则注释淹没,运维看不到下一步要敲什么
- L2 被命令片段污染,架构 reviewer 看不到核心不变量

## 1. 本 PR 改动清单

| 文件 | 改动 | 类型 |
|---|---|---|
| `scripts/bootstrap_nv8.sh` | 删除 `require_cmd python3.11` 与 `python3.11 -m venv` 硬编码;改为 `find_python_311_plus()`,依次试 `python3.14/13/12/11/3`,首个版本 ≥ 3.11 的胜出;summary 步骤打印实际用的 python | refactor |
| `docs/RUNBOOK_NV8.md` | 新增 §10 "K2.6 临时态 graceful-skip 演练" — PR#13 真机验证的脚本预排;把 hardcoded "GPU 4-7" / "minimax" 改成引用 `cfg.eval_gpus` / `cfg.prod_engine_container` 的形式 | docs |
| `docs/INVARIANTS.md` | INV-1 / INV-2 / INV-3 / INV-4 描述微改:统一引用 PR#10 的配置项名,删除残留的"minimax" 字面量(只在示例段保留) | docs |
| `tests/test_bootstrap_static.py` | 新增:静态扫 `scripts/bootstrap_nv8.sh` 必含 `find_python_311_plus` 函数 + 多版本 fallback;防止未来有人改回 `python3.11` 硬编码 | test (新增) |
| `docs/PR12_TEST_PLAN.md` | 本文档 | docs (新增) |

不动的文件:
- `orchestrator/*`(PR#10/#11 已完成代码侧)
- `tests/conftest.py`(PR#11 刚加完)
- `README.md`、`PLAN.md`、`USAGE.md`(范围超出本 PR)

## 2. 测试用例清单

### A. bootstrap 静态守卫 (`tests/test_bootstrap_static.py`) — 4 case

| ID | 场景 | 断言 |
|---|---|---|
| B1 | `find_python_311_plus` 函数定义存在 | bootstrap 内含此函数名,且函数体引用多个候选 (`python3.11`、`python3.12`、`python3.13`) |
| B2 | 不再硬编码 `python3.11` 单独 require | bootstrap 中 `require_cmd python3.11` 字符串不再出现 |
| B3 | venv 创建使用变量而非硬编码 | `python3.11 -m venv` 已被替换为 `"${PYTHON_BIN}" -m venv` 或类似 |
| B4 | hostname guard 保留 `*nv8*` 子串匹配 + `--force` 兜底 | INV-边界检查:guard 没有被弱化为"无条件运行" |

(都是静态文本扫描,不调 shell,无环境依赖。)

## 3. 不在 PR#12 范围

- ❌ 重写 ARCHITECTURE.md(当前 INVARIANTS.md 已经承载了 L2 核心,ARCHITECTURE 是更大重构,留给后续 PR)
- ❌ 改 v9 残留扫描清单(`INV-11` 已经覆盖 20+ 符号,够用;新增 forbidden symbol 要有具体 incident 触发)
- ❌ 改 systemd unit 文件(PR#7b 已经做完)
- ❌ bootstrap 中 docker/nvidia-smi 探测(可独立 PR,本 PR 仅做 python)
- ❌ 在 mac 上真跑 bootstrap(需要 docker/nvidia-smi,非 PR#12 关注点;PR#13 真机会跑)

## 4. 业务目标三问

1. **主路径**: 运维拿到 PR#12 后,只读 RUNBOOK §10 就能在 NV8 上演练 K2.6 临时态而不破坏产线? → §10 自身覆盖
2. **降级路径**: NV8 上 python 升级到 3.13 后(假设运维换了 SLES 版本),bootstrap 还能跑? → bootstrap fallback 链覆盖
3. **状态集**: bootstrap 找不到任何 ≥ 3.11 的 python 时,会清晰退出 + 提示装哪个版本,不会偷偷用 3.9? → bootstrap exit 1 + 错误消息覆盖

## 5. 执行步骤

1. 切到分支 `feat/pr12-l2-l3-docs`
2. 写 `tests/test_bootstrap_static.py`(TDD)→ 跑测试 fail
3. 改 `scripts/bootstrap_nv8.sh` 加 `find_python_311_plus`
4. 跑测试 → 全绿
5. 写 RUNBOOK §10 + INVARIANTS 微改
6. 全套回归 + ruff + mypy
7. self-review + push + AI Reviewer 首过
