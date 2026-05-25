# heyi-eval-v10 不变量清单

> 这里是 v10 评估管道的**红线列表**。每条不变量由一个或多个组件守护;违反 = 阶段失败 + outbox 告警。
> 编辑前请先读 [`docs/ARCHITECTURE.md §1 信任域`](ARCHITECTURE.md#1--信任域)。历史 v8 文档参见 [`sops/_archive_v8_invariants.md`](../sops/_archive_v8_invariants.md)（保留作审计参考,不再适用）。

## 两个 LLM,一个管道

v10 同一台机器上同时存在**两类**完全不同的 LLM 容器。把它们混为一谈是 v10
早期最严重的概念错误,PR#10 拆开,以后所有不变量描述都按这张表对号入座:

| 维度 | **产线 LLM** | **评估 LLM** |
|----|---|---|
| 谁跑 | heyi_engine 框架背后的 vLLM 实例 (用户运维) | 评估管道在 DEPLOY 阶段临时 spawn 的 `e9-*` 容器 |
| 容器名 | 由 `OrchestratorConfig.prod_engine_container` 指定,默认 `minimax`,由 env `HEYI_EVAL_PROD_ENGINE_CONTAINER` 覆盖 | 由 `stages_py.container_name_for(run)` 生成,必以 `e9-` 前缀 |
| 跑哪个模型 | 默认 MiniMax-M2.7;**每几个月**用户切换一次(Kimi-K2.6 / 未来其他)| DISCOVER 阶段拉的待测模型 (任意) |
| 引擎 | 固定 vLLM | vLLM / SGLang / Transformers (ENGINE_SELECT 阶段决定) |
| GPU | 由 `OrchestratorConfig.prod_engine_gpus` 指定,默认 `(0,1,2,3)` | 由 `OrchestratorConfig.eval_gpus` 指定,默认 `(4,5,6,7)`,严禁碰产线 GPU |
| 寿命 | 常驻;运维 SLA | 5-30 分钟即销毁 |
| 并发 | 1 | 1 (一次只评测一个模型,无并行) |
| 资源不够 | n/a | graceful skip + `aborted=insufficient_gpu`,不抛错 |

下文 INV-1 / INV-2 / INV-3 全部按这张表的语义读。

## 总览

| ID | 类别 | 一句话 | 静态守护 | 运行时守护 |
|----|----|------|------|------|
| **INV-1** | 隔离 | 评估代码不得 docker-控制(rm/stop/exec/run/restart)任何产线容器 | `tests/test_inv_production_isolation.py::TestINV1*` | `orchestrator/stages_py.execute_cleanup` 仅删 `e9-` 前缀 + run 标签 |
| **INV-2** | 隔离 | 产线 LLM 容器(由 `OrchestratorConfig.prod_engine_container` 指定,默认 `minimax`)持续运行 | — | `orchestrator/validator.assert_invariants` 每阶段后 `docker inspect <prod_engine_container>` 必须 running |
| **INV-3** | 隔离 | 评估侧产生的所有容器名必须以 `e9-` 前缀 | `tests/test_stages_py_cleanup.py` | `stages_py.container_name_for` 唯一来源 |
| **INV-4** | 隔离 | 评估代码不得写入 `/etc/heyi-engine/`、不得 systemctl 控制 `heyi-engine.*` 服务、不得 `docker-compose -f .*heyi-engine` | `tests/test_inv_production_isolation.py::TestINV4*` | bootstrap_nv8.sh 路径白名单 + systemd unit 仅安装 `heyi-eval-*.service` |
| **INV-5** | 操作 | 评估侧不得在主机 apt/pip 安装。仅可在评估自建 venv (`/home/ai/heyi-eval-v10/.venv`)内 pip | `scripts/bootstrap_nv8.sh` 仅在 venv 内 `pip install` | — |
| **INV-9** | 数据 | 备份目录在仓库外(`/home/ai/heyi-eval-backups/`),仓库内绝不出现 backup snapshot | `tests/test_backup_snapshot.py::test_backup_dir_is_outside_repo` | `backup/snapshot.py` 路径校验 |
| **INV-11** | 卫生 | 任何 v9 残余符号(ccr_*、cc-agent、e8-、HEYI_EVAL_CCR_*)禁止重新出现 | `tests/test_no_v9_residue.py` | — |
| **INV-12** | 卫生 | 任何 mutating docker 操作(rm/stop/exec/run/restart 等)必须走 docker-py SDK,禁止 `subprocess.run(["docker", ...])` mutating verb | `tests/test_inv_production_isolation.py::TestINV12*` | docker-socket-proxy 也阻断,见 INV-2 |
| **INV-13** | 卫生 | `scripts/*.sh` 不得对产线容器名出现 mutating docker verb | `tests/test_inv_production_isolation.py::TestINV13*` | — |
| **INV-14** | 隔离 | LLM-judge 单向跨域：CAPABILITY 可把 EVAL 产物字节发 PROD VLM 评分,但绝不把测试 prompt 文本 / expected_substring 发 PROD | `tests/test_inv14_llm_judge_boundary.py` | `orchestrator/llm_judge.py` 仅使用 `_JUDGE_PROMPT` 模板 |
| **INV-15** | 隔离 | transformers-runner 镜像源码(`transformers_runner/`)不得引用任何 PROD 配置(`heyi_engine` 容器名 / `OrchestratorConfig` / `prod_engine_*`),只服务 `--model-path` 指向的本地目录 | `tests/test_inv15_transformers_runner_isolation.py`(静态扫描) | Dockerfile `HF_HUB_OFFLINE=1` 阻断意外的 HF Hub 拉取 |
| **INV-16** | 沙箱·FS | 沙箱 agent 用户 `heyi-eval-agent` 对 `store/*`、`runs/`(父目录)、源码树仅有 `r-x`;`/home/ai` 仅有 `--x` (traverse,不可 ls);store/ 任何写/删/截断均必须 EACCES | `tests/test_inv16_19_agent_sandbox_static.py::TestInv16AclScript` | `deploy/agent-sandbox/acl_install.sh` 安装 POSIX ACL + `drills/attack_delete_store.sh` 真机演练 |
| **INV-17** | 沙箱·Docker | 沙箱 agent 通过 `127.0.0.1:2377` 的专属 `heyi-eval-agent-socket-proxy` 访问 Docker API,**只读** (`POST=0 EXEC=0 DELETE=0 ALLOW_START=0 ALLOW_STOP=0 ALLOW_RESTARTS=0 IMAGES=0 VOLUMES=0 NETWORKS=0 BUILD=0 COMMIT=0`);任何写动词必须 403 | `tests/test_inv17_agent_socket_proxy_static.py` | `deploy/agent-sandbox/compose.agent-socket-proxy.yml` + `drills/attack_exec_prod.sh` 真机演练 (含 exec into minimax / stop / create / delete / commit / volumes) |
| **INV-18** | 沙箱·审计 | `/var/log/heyi-eval-agent/` 目录 root:adm 0750,沙箱 agent 通过显式 deny ACL (`u:heyi-eval-agent:---`) 完全无访问 — 不能 ls、不能 cat、不能 rm、不能 truncate、不能 redirect into;`audit.sqlite` 在 PR#22b 由 root 写入(此 PR 仅占位) | `tests/test_inv16_19_agent_sandbox_static.py::TestInv16AclScript::test_audit_dir_acl_is_deny_all` | `deploy/agent-sandbox/acl_install.sh §5` 安装 deny ACL + `drills/attack_evade_audit.sh` 真机演练 |
| **INV-19** | 沙箱·预算 | `heyi-eval-agent@%i.service` 必须以 `User=heyi-eval-agent Slice=heyi-eval-agent.slice` 启动,带 `RuntimeMaxSec ≤ 1800` + `TasksMax ≤ 128` + 空 `CapabilityBoundingSet` + `NoNewPrivileges + ProtectSystem=strict + MemoryDenyWriteExecute + InaccessiblePaths=/var/log/heyi-eval-agent`;slice 自身 `MemoryMax + TasksMax` 总闸 | `tests/test_inv19_systemd_unit_static.py` | `deploy/systemd/heyi-eval-agent.slice` + `heyi-eval-agent@.service` + `drills/attack_resource_budget.sh` (fork-bomb + RuntimeMaxSec watchdog 真机演练) |
| **INV-20** | 沙箱·sudo | 沙箱 agent 用户的 sudoers 白名单只包含 `systemctl restart/status heyi-eval-orchestrator.service`(M1 短暂存在的 audit setuid 白名单已在 M2 移除——`NoNewPrivileges=true` 与 sudo 不兼容,改走 daemon socket);`heyi-engine` / `minimax` / `docker.service` / `visudo` / `passwd` / `su` 等必须落入 `HEYI_EVAL_FORBIDDEN` 别名;同时 agent 用户**不得**属于 docker/sudo/wheel/adm 组,登录 shell 为 `/usr/sbin/nologin` | `tests/test_inv16_19_agent_sandbox_static.py::TestInv19Sudoers` (含 `ALLOWED_NOPASSWD_ALIASES = {ORCH_BOUNCE}` 集合守护) + `TestInv16SetupScriptForbiddenGroups` | `deploy/agent-sandbox/sudoers.d/heyi-eval-agent` (visudo -c 校验) + `setup_agent_user.sh` + `drills/attack_sudo_escalate.sh` 真机演练 |
| **INV-23** | 评测·oversize 闸门 | `ENGINE_SELECT` 阶段**必须**在落 `engine.json` 时同时计算 `tp_size = vllm_args.tensor_parallel_size`,若 `tp_size > len(cfg.eval_gpus)` 则:(a) `engine.json` 字段 `engine="metadata_only"` + `oversize=true` + `eval_pool_size` + `eval_pool_gpus`,**不**写 vllm/transformers engine 名;(b) StageResult 返回 `ok=False error_kind="oversize_skip" extra={aborted:true}` 触发 pipeline 的 `GracefulSkip` 路径(run 标 `ABORTED`,不计入 failure metric);(c) DEPLOY / READY_WAIT / CAPABILITY 全部跳过——元数据已经在 `_meta/metadata.json` + `_meta/engine.json` 沉淀,Panel 仍可显示该模型行,只是 `engine=metadata_only`。**触发点**:nv8 eval pool 默认 `(5,6,7) len=3`(`config.py::eval_gpus`,PR#23 收缩——GPU 4 给 ComfyUI host 进程占用),所以 tp=4 / TP=8 的 70B+/MoE 大模型一律 oversize-skip;若操作员临时把 eval pool 扩到 4 GPU(`HEYI_EVAL_EVAL_GPUS=4,5,6,7`),则 tp=4 又可以跑(测试 `test_70b_passes_when_pool_widened_to_4` 守护该路径)。**禁止**直接在 DEPLOY 里硬拒——会浪费已经下载好的 model snapshot 100+GB 重新跑 CURATE | `tests/test_pr23_m27_api_and_oversize.py::TestOversizeGate` (405B/70B/30B/7B 全谱 + `TestEngineSelectArtifactShape` engine.json 字段守护) | `orchestrator/stages.py::_execute_engine_select_stage` (oversize 计算 + StageResult.aborted) + Panel 的"metadata_only"行渲染 |
| **INV-22** | 沙箱·orchestrator·调度 | orchestrator 的 `ai` 用户**只能**通过 `/etc/sudoers.d/heyi-eval-orchestrator` 中两个白名单别名调用 root:(a) `HEYI_EVAL_AGENT_LIFECYCLE` 限定 `systemctl start/stop/status/show heyi-eval-agent@*.service`(注意 `*` 受 sudoers 引擎限制只能匹配单 `service` 实例名,不能横切 `heyi-engine.service`/`minimax`);(b) `HEYI_EVAL_AGENT_HARVEST` 限定 `/usr/local/sbin/heyi-eval-agent-harvest <run-id>` 单一根帮助脚本——脚本本身复用 run-id 正则、固定 src=`$AGENT_HOME/runs/<id>/outbox`、固定 dst=`$DATA_ROOT/runs/<id>/outbox`,不接受任意路径参数;`HEYI_EVAL_ORCH_FORBIDDEN` 显式拒绝 `su`/`visudo`/`passwd` 与任何 `minimax`/`heyi-engine.service`/`docker.service` 的 stop/restart | `tests/test_pr22b_m3_orch_sudoers_static.py`(NOPASSWD 白名单 = `{LIFECYCLE, HARVEST}` 唯一 + 每条 LIFECYCLE 必须 target `heyi-eval-agent@*.service` + HARVEST 必须只指向根脚本 + bootstrap 必须 0440 安装并 `visudo -c` 校验) + `TestHarvestHelperStatic`(harvest 脚本 `set -euo pipefail` + EUID==0 守护 + run-id 正则与 prepare/run 同步 + 拒绝 rsync 任意路径) | `deploy/sudoers.d/heyi-eval-orchestrator`(`visudo -c` 校验) + `deploy/agent-sandbox/heyi-eval-agent-harvest` + `orchestrator/agent_runner.py::invoke_agent`(走 sudo 路径或 root 内进程路径) + 真机:`sudo bash scripts/bootstrap_nv8.sh` 后跑 `python -m orchestrator.agent_runner <run-id> --mode smoke` 端到端 |
| **INV-21** | 沙箱·审计 | 沙箱 agent 对审计日志(`/var/log/heyi-eval-agent/audit.sqlite`)的所有写入路径都是**追加式**:(a) 唯一写入入口是 root 用户的 audit daemon(`heyi-eval-audit.service`)监听 `/run/heyi-eval-agent-audit.sock`(0660 root:heyi-eval-agent),daemon 在握手层用 `SO_PEERCRED` 二次校验 peer euid;(b) daemon 与 `orchestrator/agent_audit.py` 源码中**禁止出现** `UPDATE` / `DELETE` / `DROP` / `REPLACE` / `TRUNCATE` SQL 语句,且 daemon 只能调用 `init_schema` / `record_command` / `record_result` 三个 helper;(c) DB 文件本身受 INV-18 deny-all ACL 保护,agent 直接 `cat` / `tee` / `truncate` / `rm` 均 EACCES。**注**:M1 setuid wrapper 设计已废弃——`NoNewPrivileges=true` 拒绝 sudo 的 setuid,因此 M1 在 ad-hoc drill 中能过,但在 agent unit 内 100% 失败。M2 daemon socket 是唯一兼容的设计 | `tests/test_pr22b_agent_audit.py::TestInv21AppendOnlyStaticGuard` + `tests/test_pr22b_audit_daemon.py::TestInv21DaemonStaticGuard`(daemon 源码 SQL 词法扫描 + `agent_audit.` 调用白名单) | `deploy/systemd/heyi-eval-audit.service`(daemon,root only) + `orchestrator/agent_audit_daemon.py`(`SO_PEERCRED` 强校验) + `orchestrator/agent_audit.py` 双 phase 行 (`begin`/`end`,不更新 `begin` 行) + `drills/attack_evade_audit_writes.sh` 真机演练(4 socket-协议攻击 + 5 INV-18 文件攻击 + 2 happy path) |

## 为什么这些是红线

**评估管道必须从产线视角"不存在"**。产线 LLM (由 `prod_engine_container` 指定的那一个 vLLM 容器,以及 `xrouter` 等辅助服务) 在 nv8 上常驻提供 LLM 服务;评估管道每跑一个新模型就在 GPU 4-7 上 spawn 一个 `e9-vllm-xxx` 拉起来,跑完 5-30 分钟后即销毁。任何一次错误把产线容器 `docker rm` 或者占走产线 GPU,都会直接打掉产线流量。

我们已经一次性发生过数据丢失(v9 → v10 仓库重建的起因);v10 的全部架构都是围绕"再也不能发生"展开:

1. **物理隔离**: 产线 LLM 默认占 `prod_engine_gpus` (默认 0-3),评估占 `eval_gpus` (默认 4-7);docker-socket-proxy 仅放行 `^e9-` 前缀容器的写操作(INV-3 + INV-12 配合)
2. **代码隔离**: 仓库源代码不允许任何 mutating 调用提到产线名字(INV-1 静态扫描 + INV-13 shell 扫描)
3. **配置隔离**: 评估系统的 systemd unit 全部叫 `heyi-eval-*.service`,不会与产线 `heyi-engine.service` 名字相撞(INV-4 命名空间)
4. **数据隔离**: 备份目录在仓库外,模型缓存也在仓库外(INV-9),`git clean -fdx` 不会误删任何东西

## INV-14 详解：LLM-judge 跨域的窄豁免

PR#15 引入多模态分轨 CAPABILITY 后,出现一个新的合法跨域流量:
**llm_judge 评分器**(`orchestrator/llm_judge.py`)需要把 EVAL 引擎产出的
图像/视频字节发给 PROD 端 heyi_engine 的 VLM 让它描述,然后据此判分。

这看起来违反 PR#4 文档里的"INV-2: CAPABILITY 只调 `deploy.json::base_url`"
表述。INV-14 把这个例外**显式声明**为合法路径,并加守护:

1. **方向**: EVAL → PROD,**单向**;PROD 永不回写 EVAL 上下文。
2. **内容**: 只允许发 EVAL **产出的 artifact 字节** + 固定的
   `_JUDGE_PROMPT` 模板(描述是否含 `expected_description`)。
3. **禁止**: 不得把测试 prompt 原文、`expected_substring`、capability_data
   下任何 JSONL 内容发给 PROD VLM。这条由
   `tests/test_inv14_llm_judge_boundary.py` 静态扫描守护。
4. **作用域**: 仅 `image_gen` / `video_gen` 两个 category 走此路径;
   其余 11 个 category 仍严格遵守 INV-2(只调 eval base_url)。

违反 INV-14 = 数据污染:把测试 prompt 给 PROD 等于用线上模型给评估打分,
完全破坏评估独立性。

## INV-15 详解：transformers-runner 不得引用 PROD 配置

PR#19 引入 `transformers_runner/` 作为 ASR / TTS / image_gen / video_gen
/ music_gen 等非 vLLM 模态的容器入口。该镜像跑在 EVAL 端,但因为它的
源代码就在 v10 仓库里、且 Docker `network_mode=host`,如果不约束就有
两种风险:

1. **配置混淆**:`server.py` 误 import `OrchestratorConfig` 等 PROD 配置,
   把容器名 / GPU 列表 / 凭证带进 image 层。
2. **被动调用 PROD**:`server.py` 偷懒访问 `127.0.0.1:<prod_engine_port>`
   来 "校验自己" — 这会把 EVAL 产出送到 PROD 模型,违反 INV-14 之外
   的另一条静默通道。

INV-15 用静态扫描守住这两条:`transformers_runner/` 包内任何 `.py`
文件,不得出现以下 token:

- `heyi_engine`、`minimax`、`prod_engine_container`、`prod_engine_gpus`
- `OrchestratorConfig`、`orchestrator.config`、`orchestrator.capability`
- `xrouter`、`/etc/heyi-engine`

允许 import: stdlib + `torch` + `transformers` + `diffusers` +
`accelerate` + `safetensors` + `PIL` + `numpy` + `soundfile` + `librosa`。
其余白名单外 import 在 review 阶段单独评估。

## 怎么读一条不变量违例

CI 失败时 pytest 输出会带文件:行号 + 那一行的内容(裁剪到 120 字符)。例如:

```
INV-1 violation: production container 'minimax-m2.7' is docker-controlled by evaluation code:
  scripts/cleanup_legacy.sh:42: docker rm -f minimax-m2.7
```

定位 `scripts/cleanup_legacy.sh:42` → 改写为通过 `e9-` 前缀过滤 → 再跑测试 → 绿。

如果是合理的读-only diagnostic(例如 `docker inspect <prod_engine_container>` 用来断言 INV-2),把那个文件加到 `tests/test_inv_production_isolation.py` 的 `INV_DOC_ALLOWLIST` 里,并写明理由。

## 添加新不变量

1. 在本表加一行,给一个 `INV-N` 编号(连续递增,不复用)
2. 在 `tests/test_inv_production_isolation.py` 或独立文件加静态/运行时守护
3. 把 ID 加到 `tests/test_inv_production_isolation.py::TestInvariantsDocExists` 的检查列表
4. PR 描述里写 "新增 INV-N"
