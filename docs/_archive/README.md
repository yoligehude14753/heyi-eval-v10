# docs/_archive — PR 历史快照

本目录归档 v10 早期 PR 的测试计划与真机验收报告。**功能已合入主干**，作为开发过程的审计快照保留，不再作为现状描述。

| 文件 | 性质 |
|---|---|
| `PR2_TEST_PLAN.md` | heyi_engine 客户端测试矩阵 |
| `PR3_TEST_PLAN.md` | DEPLOY / READY_WAIT / CLEANUP Python 化 |
| `PR4_TEST_PLAN.md` | CAPABILITY 阶段 |
| `PR5_TEST_PLAN.md` | SHOWCASE runner 信任边界 |
| `PR6_TEST_PLAN.md` | Backup 层 |
| `PR7a_TEST_PLAN.md` | v9 CCR/cc-agent 清理 |
| `PR7b_TEST_PLAN.md` | systemd + bootstrap |
| `PR8_TEST_PLAN.md` | E2E + 静态 INV 守护 |
| `PR10_TEST_PLAN.md` | Prod / Eval LLM 概念拆分 |
| `PR11_TEST_PLAN.md` | GPU 隔离 + graceful skip |
| `PR12_TEST_PLAN.md` | L2/L3 文档分层 + bootstrap 清理 |
| `PR24_REAL_NV8_REPORT.md` | PR#23 后真机 E2E 报告（2026-05-23） |
| `PR26_BATCH_EVAL_REPORT.md` | 多模型批量 + oversize hf_id 修复 |
| `PR29_CROSS_MODALITY_REPORT.md` | 跨模态评测批次结果 |
| `PR33_DEPLOY_REPAIR_EXPERIMENT.md` | 部署自愈真机实验日志 |
| `PR36_HONESTY_EXPERIMENT.md` | 推理探针 / honesty gate 实验 |

**当前生效文档**回到 [`docs/`](../) 顶层：

- Model lane：`ARCHITECTURE.md` / `INVARIANTS.md` / `USAGE.md` / `RUNBOOK_NV8.md` / `PLAN.md`
- Project + Skill lane（v0.2.0 新增）：`ARCHITECTURE_LANES.md` / `INVARIANTS_LANES.md` / `TEST_PLAN_LANES.md`
- 顶层变更记录：[`CHANGELOG.md`](../../CHANGELOG.md)

历史 lane 真机批跑（2026-05-26 v0.2.0 release sweep）：39 skill + 19 project，**89.7% / 52.6%** 结构化产出率，明细见 `CHANGELOG.md` v0.2.0 节。

