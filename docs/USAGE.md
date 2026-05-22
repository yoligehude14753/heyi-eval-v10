# heyi-eval-v10 · 运维与验证手册

> 部署细节看 `deploy/README.md`。这里只覆盖**部署完成后**的日常操作与三种验证演练。

## 1 · 状态速查

```bash
# 服务状态
systemctl status heyi-eval-orchestrator.service \
                 heyi-eval-discover.service \
                 heyi-eval-panel.service \
                 heyi-eval-backup.service

# 定时器
systemctl list-timers heyi-eval-backup.timer

# 队列 / 历史 / 健康
curl -s http://127.0.0.1:8090/ | head -60   # panel HTML
tail -200 ~/heyi-eval-data/store/notify_outbox.jsonl
```

## 2 · 入队跑一个模型（手动）

```bash
cd /home/ai/heyi-eval-v10
.venv/bin/python -m orchestrator enqueue Qwen/Qwen2.5-0.5B-Instruct
# 由 systemd 起的 loop 进程会自动 pick；也可以同步前台跑一次：
.venv/bin/python -m orchestrator run
```

跑完后产物在 `~/heyi-eval-data/runs/<run_id>/`。模型权重已删除（INV-2），保留：
`state.json` / `curate.json` / `metadata.json` / `engine.json` / `deploy.json` /
`ready.json` / `capability.json` / `showcase.json` / `cleanup.json` / `showcase/*.md`。

## 3 · 验证演练（在 nv8 上跑）

### 3.1 全链路 E2E（PR#8 §E-1~E-5）

> Qwen2.5-0.5B-Instruct ~20min，包含 INV-1 生产容器隔离硬证据。

```bash
cd /home/ai/heyi-eval-v10
HEYI_EVAL_E2E_ALLOW=1 \
  .venv/bin/python -m pytest tests/e2e/ -m e2e -v
```

要求：
- 主机名以 `heyi-sh-nv8` 开头（dev 机想干跑可加 `HEYI_EVAL_E2E_FORCE=1`，但 docker / nvidia-smi 仍必须可用）
- heyi_engine 在 `:10814` 健康（pipeline 的 metadata + showcase 阶段会调它）
- 至少 1 张 GPU 空闲 ≥ 8 GiB

### 3.2 24 小时定时器健康（PR#8 §T-24h-1~5）

systemd 上线 ≥ 24h 后跑：

```bash
/home/ai/heyi-eval-v10/scripts/verify_24h_timer.sh | tee /tmp/verify-24h.json
echo "exit=$?"
```

返回 JSON 报告，非 0 退出码即某项 fail。可串到 cron 或 `notify_outbox`。

### 3.3 静态守卫（任何主机，CI 默认会跑）

```bash
cd /home/ai/heyi-eval-v10
.venv/bin/pytest -m "not e2e and not slow" tests/test_prod_container_safety.py \
                                          tests/test_systemd_units.py \
                                          tests/test_no_v9_residue.py -v
```

P-1..P-9 + B-* + V-* + INV-11，全部必须绿。

## 4 · 灾难恢复要点

1. **要恢复一次 run 的产物**：从 `~/heyi-eval-backups/` 找最新快照，`rsync -a` 回 `~/heyi-eval-data/runs/<run_id>/`。
2. **数据库异常**：`store.sqlite` 在每个快照里都有副本；停 orchestrator，替换文件，重启。
3. **整机崩溃**：在新机上 `git clone heyi-eval-v10` → `bash scripts/bootstrap_nv8.sh --force` → 从 mac 镜像或 nv8 backups 拉回 `~/heyi-eval-data/`。

## 5 · 不变量速记

| ID | 含义 | 守卫位置 |
|---|---|---|
| INV-1 | 评估管道 docker 操作只碰 `e9-*` 容器 | `tests/test_prod_container_safety.py::P-1~P-9` + cleanup 实现 |
| INV-2 | run 产物在 cleanup 后必须保留 | `e4_cleanup_removed_only_e9` + 文档约束 |
| INV-4 | deploy 出来的容器必须打 `heyi_eval_run` 标签 | `P-4` + cleanup label 过滤 |
| INV-9 | rsync 备份不写入数据源 | `backup/` 内 hard-coded 路径白名单 + `b3` 路径测试 |
| INV-11 | 代码里不再含 v9 残余符号 | `tests/test_no_v9_residue.py` |
