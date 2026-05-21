# deploy/

nv8 部署一次性资料 — `systemd` 单元 + `bootstrap_nv8.sh`。

## 哲学

heyi-eval-v10 是"很多个 Python 长循环 + cron 风格 oneshot"，进程之间不需要 docker 网络隔离，待测模型容器由 `orchestrator.stages_py` 用 `docker-py` 拉起。所以这里**没有** `docker-compose.yml`，全部走 systemd。

heyi_engine（生产 LLM）是用户自管的另一个 compose（INV-1：评估管道不得碰它），它的 unit 不在本仓库。

## 文件

| 文件 | 说明 |
|------|------|
| `systemd/heyi-eval-orchestrator.service` | 主循环 `python -m orchestrator loop` |
| `systemd/heyi-eval-discover.service` | HF 模型轮询 `python -m discover loop` |
| `systemd/heyi-eval-panel.service` | 只读管理面板 :8090 |
| `systemd/heyi-eval-backup.{service,timer}` | 30 min rsync 快照（PR#6 已存在） |
| `systemd/heyi-eval-notify-sync.service` | notify_outbox → 微信桥（**目前是占位 exit 0**，等业务实现） |
| `env.example` | `/etc/heyi-eval-v10/env` 的模板 |

## 一键部署

```bash
# 在 nv8 上, 以用户 ai 身份:
cd /home/ai/heyi-eval-v10
git pull --ff-only
bash scripts/bootstrap_nv8.sh
```

干跑预览：

```bash
DRY_RUN=1 bash scripts/bootstrap_nv8.sh
```

非 nv8 测试机上强制跑（不会真 systemctl）：

```bash
DRY_RUN=1 bash scripts/bootstrap_nv8.sh --force
```

## 卸载

```bash
sudo systemctl disable --now \
  heyi-eval-orchestrator.service \
  heyi-eval-discover.service \
  heyi-eval-panel.service \
  heyi-eval-notify-sync.service \
  heyi-eval-backup.timer \
  heyi-eval-backup.service

sudo rm -f /etc/systemd/system/heyi-eval-*.service \
           /etc/systemd/system/heyi-eval-*.timer
sudo systemctl daemon-reload
```

数据保留：`HEYI_EVAL_DATA` 和 `HEYI_EVAL_BACKUPS` 目录**不会**被 bootstrap/卸载自动删除。

## 失败排查

```bash
systemctl status heyi-eval-orchestrator.service
journalctl -u heyi-eval-orchestrator.service -n 80 --no-pager
journalctl -u heyi-eval-discover.service -n 80 --no-pager
journalctl -u heyi-eval-panel.service -n 80 --no-pager

# panel API 健康
curl -sf http://127.0.0.1:8090/api/health | python -m json.tool
```

orchestrator 自己在循环里有 `_engine_preflight_gate`，会把 heyi_engine 不可用的状态写到 `notify_outbox.jsonl`（panel "engine ok/down" 卡片可见）；systemd 不会因此重启 orchestrator。

## 不变量回顾

- INV-1（评估侧从不碰 heyi_engine 生产容器）：每个 service unit 都 `Nice=10+` + `IOSchedulingClass=best-effort`；`heyi-eval-*` units 之间没有 `Requires=`/`Wants=`。
- INV-9（备份永远写出树之外）：`heyi-eval-backup.service` 用 `EnvironmentFile=` 解析 `HEYI_EVAL_BACKUPS`，路径在代码层校验。
- INV-11（不引入 v9 残余）：所有 unit 文件名形如 `heyi-eval-*`，env.example 不含 `HEYI_EVAL_CCR_*` / `cc_agent_*` 等键。
