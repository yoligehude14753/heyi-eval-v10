"""agent_driver — M2.7 Claude Code 执行器（M1 落地）.

本模块是 [`docs/ARCHITECTURE_LANES.md`](../docs/ARCHITECTURE_LANES.md) §4 描述的
agent_driver 五件套：

- ``schema``           — RunReport dataclass + JSON Schema v1.0（围栏内 JSON 契约）
- ``report_extractor`` — 从 agent.log 抽取 ``<<<HEYI_RUN_REPORT_JSON>>>...<<<END>>>``
                         围栏并按 schema 验证（INV-P3）
- ``budget_guard``     — token + 墙钟双上限，命中即 graceful 中断（INV-P4）
- ``ccr_bridge``       — 把 yunwu provider 注入 m2b 容器的 ccr-config.json
- ``pool_manager``     — 长驻 m2b 容器池 (1-3 个) 的获取 / 释放 / 健康 / 周期重启
                         （INV-P8 防互污染）
- ``exec_runner``      — 编排上面四件套，把 task_prompt 喂给容器内 ``claude`` CLI

M1 不直接依赖任何 docker 实例：所有真容器交互通过依赖注入的 ``docker_client``
（``docker-py`` SDK）发生，单测一律 mock 之。沙箱真机联调留给 M2 在 heyi
(10.10.11.198) 上做。
"""
from __future__ import annotations

from agent_driver.schema import (
    REPORT_JSON_SCHEMA,
    Outcome,
    RunReport,
    StepStatus,
    Verdict,
)

__all__ = [
    "REPORT_JSON_SCHEMA",
    "Outcome",
    "RunReport",
    "StepStatus",
    "Verdict",
]
