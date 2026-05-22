# heyi-eval-v10 不变量清单

> 这里是 v10 评估管道的**红线列表**。每条不变量由一个或多个组件守护;违反 = 阶段失败 + outbox 告警。
> 编辑前请先读 `docs/ARCHITECTURE.md §信任边界`。历史 v8 文档参见 `sops/invariants.md`(保留作审计参考,不再适用)。

## 总览

| ID | 类别 | 一句话 | 静态守护 | 运行时守护 |
|----|----|------|------|------|
| **INV-1** | 隔离 | 评估代码不得 docker-控制(rm/stop/exec/run/restart)任何产线容器 | `tests/test_inv_production_isolation.py::TestINV1*` | `orchestrator/stages_py.execute_cleanup` 仅删 `e9-` 前缀 + run 标签 |
| **INV-2** | 隔离 | 产线容器(`minimax-*`/`xrouter`/`glm-*`/`kimi-*`/`voipmonitor`)持续运行 | — | `orchestrator/validator.assert_invariants` 每阶段后 `docker inspect minimax` 必须 running |
| **INV-3** | 隔离 | 评估侧产生的所有容器名必须以 `e9-` 前缀 | `tests/test_stages_py_cleanup.py` | `stages_py.container_name_for` 唯一来源 |
| **INV-4** | 隔离 | 评估代码不得写入 `/etc/heyi-engine/`、不得 systemctl 控制 `heyi-engine.*` 服务、不得 `docker-compose -f .*heyi-engine` | `tests/test_inv_production_isolation.py::TestINV4*` | bootstrap_nv8.sh 路径白名单 + systemd unit 仅安装 `heyi-eval-*.service` |
| **INV-5** | 操作 | 评估侧不得在主机 apt/pip 安装。仅可在评估自建 venv (`/home/ai/heyi-eval-v10/.venv`)内 pip | `scripts/bootstrap_nv8.sh` 仅在 venv 内 `pip install` | — |
| **INV-9** | 数据 | 备份目录在仓库外(`/home/ai/heyi-eval-backups/`),仓库内绝不出现 backup snapshot | `tests/test_backup_snapshot.py::test_backup_dir_is_outside_repo` | `backup/snapshot.py` 路径校验 |
| **INV-11** | 卫生 | 任何 v9 残余符号(ccr_*、cc-agent、e8-、HEYI_EVAL_CCR_*)禁止重新出现 | `tests/test_no_v9_residue.py` | — |
| **INV-12** | 卫生 | 任何 mutating docker 操作(rm/stop/exec/run/restart 等)必须走 docker-py SDK,禁止 `subprocess.run(["docker", ...])` mutating verb | `tests/test_inv_production_isolation.py::TestINV12*` | docker-socket-proxy 也阻断,见 INV-2 |
| **INV-13** | 卫生 | `scripts/*.sh` 不得对产线容器名出现 mutating docker verb | `tests/test_inv_production_isolation.py::TestINV13*` | — |

## 为什么这些是红线

**评估管道必须从产线视角"不存在"**。产线 `minimax`/`xrouter`/`glm-*` 在 nv8 上常驻提供 LLM 服务;评估管道每跑一个新模型就在 GPU 4-7 上 spawn 一个 `e9-vllm-xxx` 拉起来,跑完 5-30 分钟后即销毁。任何一次错误把产线容器 `docker rm` 或者占走 GPU 0-3,都会直接打掉产线流量。

我们已经一次性发生过数据丢失(v9 → v10 仓库重建的起因);v10 的全部架构都是围绕"再也不能发生"展开:

1. **物理隔离**: 产线在 GPU 0-3,评估在 GPU 4-7,docker-socket-proxy 仅放行 `^e9-` 前缀容器的写操作(INV-3 + INV-12 配合)
2. **代码隔离**: 仓库源代码不允许任何 mutating 调用提到产线名字(INV-1 静态扫描 + INV-13 shell 扫描)
3. **配置隔离**: 评估系统的 systemd unit 全部叫 `heyi-eval-*.service`,不会与产线 `heyi-engine.service` 名字相撞(INV-4 命名空间)
4. **数据隔离**: 备份目录在仓库外,模型缓存也在仓库外(INV-9),`git clean -fdx` 不会误删任何东西

## 怎么读一条不变量违例

CI 失败时 pytest 输出会带文件:行号 + 那一行的内容(裁剪到 120 字符)。例如:

```
INV-1 violation: production container 'minimax-m2.7' is docker-controlled by evaluation code:
  scripts/cleanup_legacy.sh:42: docker rm -f minimax-m2.7
```

定位 `scripts/cleanup_legacy.sh:42` → 改写为通过 `e9-` 前缀过滤 → 再跑测试 → 绿。

如果是合理的读-only diagnostic(例如 `docker inspect minimax` 用来断言 INV-2),把那个文件加到 `tests/test_inv_production_isolation.py` 的 `INV_DOC_ALLOWLIST` 里,并写明理由。

## 添加新不变量

1. 在本表加一行,给一个 `INV-N` 编号(连续递增,不复用)
2. 在 `tests/test_inv_production_isolation.py` 或独立文件加静态/运行时守护
3. 把 ID 加到 `tests/test_inv_production_isolation.py::TestInvariantsDocExists` 的检查列表
4. PR 描述里写 "新增 INV-N"
