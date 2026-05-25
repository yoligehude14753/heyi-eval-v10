# PR#6 测试计划 — 数据保护层（30 min rsync + 7 天保留 + panel + mac 镜像）

> 阶段 1（架构）+ 阶段 2（测试用例设计），按 rules `00-core.mdc §开发工作流` 先确认再编码。

## 1 · 范围

模块边界（**新增**）：

```
backup/
├── __init__.py
├── snapshot.py        # 入口：take_snapshot(cfg) → SnapshotResult
├── retention.py       # prune_old_snapshots(backups_root, keep_days=7)
└── __main__.py        # python -m backup → 跑一次 snapshot 并 exit code 反馈
```

修改：
- `orchestrator/config.py` — 增 `backups_root: Path`
- `panel/server.py` — 增 `/api/backup` + 主页备份卡片
- `pyproject.toml` — `backup` 加入 packages + coverage source + mypy
- `.github/workflows/ci.yml` — mypy 目标加 `backup/`

新增 deploy / scripts（**不参与 unit test**，部署阶段 PR#7 真正接入 systemd）：
- `deploy/systemd/heyi-eval-backup.{service,timer}`
- `scripts/mac-mirror.sh`
- `deploy/launchd/com.heyi.eval.mac-mirror.plist`

**显式不做**（推到后续 PR）：
- 把 `backup.timer` 写进 `bootstrap_nv8.sh` → PR#7
- 在生产 nv8 实际跑 24h 验证 → PR#8
- Mac launchd plist 真实安装到 `~/Library/LaunchAgents/` → 文档化即可，由用户手工 `launchctl load`

## 2 · 架构决策

### 2.1 为什么走 rsync subprocess 而不是纯 Python

PLAN.md §4.4 + §INV-6 都明确要求 `rsync -a --delete`。空间优化用 `--link-dest=<前一次成功的 snapshot>` —— 每个 snapshot 是完整树（直接 `cp` 出来就能恢复），未变更的文件靠 hard-link 共享。这是 Time Machine 风格的标配做法；自己写一遍既会出 bug，也享受不到 rsync 的网络/中断恢复语义。

测试不会 fork rsync。`backup.snapshot._run_rsync(...)` 是唯一的子进程 seam，所有单测 patch 这一处。

### 2.2 备份元数据的写入位置（修订 INV-6）

PLAN.md §INV-6 原写"备份成功后写 timestamp 到 `store/last_backup.txt`"。**修订为写到 `<backups_root>/last_backup.txt`**：

- 严格遵守"源目录只读"原则。`store/` 在源目录里，如果备份子系统反过来写源目录，源目录就不是只读了。
- 把"备份元数据"放到"备份目录"，方向自洽：source-of-truth 在 backups_root，panel 跨过去读。
- 数据丢失场景下（源目录删干净），`last_backup.txt` 还在 `backups_root/`，恢复时第一眼就看见最新成功是哪个时间戳。

panel 读 `cfg.backups_root / "last_backup.txt"` 来渲染卡片。

### 2.3 目录布局

```
HEYI_EVAL_BACKUPS/                       # 默认 ~/heyi-eval-backups
├── 20260521_180000/                     # snapshot 时间戳（UTC，YYYYMMDD_HHMMSS）
│   ├── store/runs.sqlite
│   ├── runs/<run_id>/...
│   └── backup_meta.json                 # 这次快照元数据
├── 20260521_183000/
│   └── ...
├── latest -> 20260521_183000/           # symlink（atomic rename）
└── last_backup.txt                      # 最新成功 ts（ISO-8601）
```

`backup_meta.json` 字段：

```json
{
  "snapshot_ts": "20260521_183000",
  "snapshot_ts_iso": "2026-05-21T18:30:00+00:00",
  "src": "/home/ai/heyi-eval-data",
  "rsync_seconds": 12.4,
  "rsync_files_total": 1842,
  "rsync_files_transferred": 23,
  "size_bytes": 1234567890,
  "link_dest_from": "20260521_180000"
}
```

### 2.4 保留策略

`prune_old_snapshots(backups_root, keep_days=7)`：

- 仅删除目录名匹配 `^\d{8}_\d{6}$` 的快照（避免误删 `latest` 软链 / `last_backup.txt` / 用户手工放的东西）。
- 按 mtime 排序，删除 `now - mtime > 7 days` 的所有快照。
- **永远保留最近的至少 1 个 snapshot**（即使全部超龄；不让"7 天没跑评测"的场景把所有备份清光）。
- 删除失败不抛异常（返回 `(removed, failed)` 二元组），单次保留失败不阻塞下次快照。

### 2.5 错误传播

`take_snapshot` 返回 `SnapshotResult(ok: bool, …)`。任何 rsync 非零退出 / 写元数据失败 / 更新 `latest` 失败 → `ok=False`。

- 失败 **不** 写 `last_backup.txt`，**不** 推 latest 软链。
- 失败时已经写到 partial 目录的内容用 `shutil.rmtree(missing_ok=True)` 清掉；rmtree 自身失败也只是 warning，下一次 retention 会清。
- 失败一律走 `orchestrator.notify` 现成的 outbox 通道（事件类型 `backup_failed`），让 mac sync_agent 把告警推到微信。这是 PR#6 唯一对 orchestrator 的依赖。

### 2.6 panel 卡片字段

`/api/backup` 返回：

```json
{
  "last_backup_ts": "2026-05-21T18:30:00+00:00",
  "last_backup_age_s": 423,
  "snapshot_count": 48,
  "total_size_bytes": 5_900_000_000,
  "health": "ok",
  "backups_root": "/home/ai/heyi-eval-backups"
}
```

`health` 三档：
- `ok` — `age_s ≤ 60 min`（一次失败容错：30 min cron + 一次失败重试 = 60 min 内必须看到下一次成功）
- `warn` — `60 min < age_s ≤ 24 h`
- `down` — `age_s > 24 h` **或** `last_backup_ts` 缺失且 `backups_root` 已存在

主面板 grid 加一格：值 = `health` 颜色化时间差；副标 = snapshot 数量 + 总占用。

### 2.7 不变量

- **INV-6（修订版）**：30 min rsync 备份到 `<backups_root>/<ts>/`，保留 7d；成功后写 ISO 时间戳到 `<backups_root>/last_backup.txt`，panel 读取展示。
- **INV-9（新增）**：备份系统永不写源目录任何文件（`data_root` 下任何路径都不可写）。`take_snapshot` 入口对 src/dst 重叠做 hard-check（`backups_root` 必须不在 `data_root` 子树里，反之亦然），违反即拒绝运行并返回 ok=False。
- **INV-10（新增）**：备份永远不依赖 docker / heyi_engine / claude — 它是数据层的事后镜像，与控制平面完全解耦。`backup/` 模块导入图里禁止出现 `cc_agent`/`heyi_engine`/`docker`/`anthropic`。

## 3 · 测试用例清单

### 3.1 `tests/test_backup_snapshot.py`

**Happy path**

| ID | 用例 | 验证点 |
|----|----|----|
| H1 | 首次快照（无 prev snapshot） | rsync 调用无 `--link-dest`；目标目录创建；`backup_meta.json` 写入；`last_backup.txt` 写入；`latest` 软链指向当次目录 |
| H2 | 增量快照（已有上次成功） | rsync 调用带 `--link-dest=<上次目录绝对路径>`；新 snapshot 目录创建；`latest` 原子更新到新目录 |
| H3 | 元数据完整 | `backup_meta.json` 含 `snapshot_ts/snapshot_ts_iso/src/rsync_seconds/size_bytes/link_dest_from` 全字段 |
| H4 | `last_backup.txt` 格式 | ISO-8601 with `+00:00` 时区；可被 `datetime.fromisoformat` 解析 |

**Sad path**

| ID | 用例 | 验证点 |
|----|----|----|
| S1 | rsync 非零退出 | `SnapshotResult.ok=False`；`last_backup.txt` 不被覆盖；`latest` 软链不改动；partial 目录被清；outbox 写 `backup_failed` 事件 |
| S2 | 元数据写入失败（perm 模拟） | snapshot 目录被清；ok=False；outbox 写事件 |
| S3 | `backups_root` 和 `data_root` 重叠 | 立即返回 ok=False，error=`overlap`；rsync 完全不被调用 |
| S4 | rsync 二进制不存在（`FileNotFoundError`）| ok=False, error_kind=`rsync_missing` |

**Edge**

| ID | 用例 | 验证点 |
|----|----|----|
| E1 | 源目录为空（首次 nv8 全新部署） | rsync 还是成功；snapshot 目录创建为空目录；元数据 `size_bytes=0` |
| E2 | 源目录里有 sqlite WAL / SHM | rsync 命令必须用 `-a`（默认含 `--specials`）；具体不验 sqlite WAL 内容（rsync mocked），只验 `--specials/-a` flag 透传 |
| E3 | 备份目录已存在同名 ts（同一分钟内重跑） | 改用秒级精度后不应发生；测试覆盖：如果 `<backups_root>/<ts>` 已存在，认为是上次中断的 partial，先 rmtree 再继续 |
| E4 | `latest` 软链已经指向不存在的快照 | 不阻塞；本次成功后 `latest` 仍能原子切到新目标 |

### 3.2 `tests/test_backup_retention.py`

| ID | 用例 | 验证点 |
|----|----|----|
| R1 | 全部在 7d 内 | 0 个被删 |
| R2 | 部分超龄 | 仅超龄的目录被删，未超龄的全保留 |
| R3 | 全部超龄但只有 1 个 | 0 个被删（最小保留 1） |
| R4 | 全部超龄有 N 个 | 保留最新的 1 个，其它全删 |
| R5 | 非快照目录（`latest`/`last_backup.txt`/手工放的别的） | 完全不动 |
| R6 | rmtree 失败 | 不抛异常；返回 `(removed_count, failed_count)`；其它快照仍尝试删 |
| R7 | `backups_root` 不存在 | 返回 `(0, 0)` 不抛异常 |

### 3.3 `tests/test_panel.py`（追加）

| ID | 用例 | 验证点 |
|----|----|----|
| P1 | `/api/backup` 在 fixture 数据齐全时返回 health=ok | snapshot_count=2, last_backup_age_s 介于 0..3600, health="ok" |
| P2 | `last_backup.txt` 缺失但 `backups_root` 存在 | health="down", last_backup_ts=None |
| P3 | `last_backup.txt` 存在但 24h+ 之前 | health="down" |
| P4 | `last_backup.txt` 存在 ≤60 min | health="ok" |
| P5 | `backups_root` 完全不存在 | 端点返回 200，health="down", snapshot_count=0 |
| P6 | 主页 HTML 包含"备份"卡片标签 | grep 字符串"备份"在 INDEX_HTML 中 |

### 3.4 不变量测试

| ID | 用例 | 验证点 |
|----|----|----|
| INV6 | 成功一次后 `last_backup.txt` 写入 `backups_root/`，**不**写 `data_root/store/` | 断言 `data_root/store/last_backup.txt` 不存在 |
| INV9 | 备份过程中 src 目录里所有文件 mtime / size 不变 | 用 `os.stat` snapshot 比对 |
| INV10 | `backup/` 子模块的 `ast` 导入图不含 `cc_agent` / `heyi_engine` / `docker` / `anthropic` | walk 模块 + 静态分析 |

## 4 · 公共边界（什么可以测、什么不能）

- **不**走真实 rsync：所有 happy/sad/edge 都 monkeypatch `_run_rsync`。
- **不**写真实 docker / heyi_engine：backup 模块没有任何 docker / LLM 依赖（INV-10 强制）。
- panel 端点用 `Handler` 直接 dispatch（沿用现有 `test_panel.py` 风格），不起 HTTP 服务。
- 时间用 `freezegun` 或注入 `now_fn` seam；不 import `time.sleep`。

## 5 · 验收门（PR#6 合并条件）

- [ ] `pytest --cov`：229 → ≥240 测试，整体覆盖率不低于 85.0%（PR#5 基线）
- [ ] `backup/` 模块覆盖率 ≥ 90%
- [ ] `ruff check .` 通过
- [ ] `mypy backup/ orchestrator/ panel/` 通过
- [ ] panel 主页 HTML 加了备份卡片（人工目测 + 一条 grep 单测）
- [ ] PR 描述包含 INV-6 / INV-9 / INV-10 显式声明
- [ ] CI 4 个 job（lint / typecheck / test / check）全绿
