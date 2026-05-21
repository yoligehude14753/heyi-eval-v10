# RUNBOOK · NV8 真机演练 (PR#8 D 层)

> 这是 PR#8 唯一需要在 nv8-6000 真机上跑的部分。CI/mac 不能验,只能在 `100.127.173.85` ssh 上手动执行。
> 每一步含"通过标准"和"失败时怎么办"。建议在 ≥ 2 个清醒小时的窗口执行,中间不要并行做产线动作。

## 0 · 演练前置

```bash
# 在 nv8 上:
sudo -iu ai
cd /home/ai/heyi-eval-v10
git fetch origin
git log -1 --oneline       # 期望: PR#7b 合并后的 main commit
```

**保留快照**: 在动手前留一个产线 snapshot,以便事后对比 INV-1/INV-2:

```bash
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' \
    | sort > /tmp/preflight_prod_snapshot.txt
nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader > /tmp/preflight_gpu_snapshot.txt
```

---

## 1 · Preflight (5 min)

### 1.1 仓库 & venv

```bash
test -d /home/ai/heyi-eval-v10/.venv || echo "FAIL: venv missing"
/home/ai/heyi-eval-v10/.venv/bin/python -c "import orchestrator, curator, cc_agent, backup; print('imports ok')"
```

通过标准: `imports ok`。
失败时: 重新跑 `scripts/bootstrap_nv8.sh`。

### 1.2 heyi_engine 健康

```bash
curl -fsS http://127.0.0.1:10814/v1/models | jq '.data[0].id'
```

通过标准: 返回一个非空 model id。
失败时: 不要继续。先恢复 heyi_engine,因为评估管道现在直连 heyi_engine 做 metadata + showcase。

### 1.3 systemd 单元已安装但不一定启用

```bash
systemctl --no-pager status \
    heyi-eval-orchestrator heyi-eval-discover heyi-eval-panel \
    heyi-eval-notify-sync heyi-eval-backup.timer 2>&1 | head -50
```

通过标准: 4 个 service 显示 `loaded`(active 或 inactive 都行),timer 显示 `loaded`。
失败时: `sudo systemctl daemon-reload && sudo /home/ai/heyi-eval-v10/scripts/bootstrap_nv8.sh`。

---

## 2 · 单 model 烟测 (15-30 min)

让 Qwen2.5-0.5B-Instruct 跑一遍完整 9 stage,全程**不**启 systemd loop——直接命令行调用,这样可以一次性观察日志。

```bash
sudo systemctl stop heyi-eval-orchestrator 2>/dev/null || true

cd /home/ai/heyi-eval-v10
.venv/bin/python -m orchestrator enqueue Qwen/Qwen2.5-0.5B-Instruct

# 然后人手跑一轮 pipeline (不走 systemd 自动循环):
.venv/bin/python -m orchestrator run --once
```

**期望日志关键节点**:
- `[DISCOVER] OK` → `[CURATE] OK` → `[METADATA] OK` → `[ENGINE_SELECT] OK` → `[DEPLOY] OK` (这步耗时 60-180s,vllm 拉模型 + 启动)
- `[READY_WAIT] OK` (5-30s,/v1/models 返回 200)
- `[CAPABILITY] OK` → `[SHOWCASE] OK` → `[CLEANUP] OK`

**通过标准**: 9 个 stage 全 OK,`runs/<run_id>/cleanup.json` 显示 `removed` 列表非空,`failed` 列表为空。

**失败时**:
- DEPLOY 失败: 检查 `docker pull vllm/vllm-openai` 是否成功,GPU 4-7 有空闲显存
- READY_WAIT 超时: vllm 启动太慢,看 `runs/<run_id>/_meta/deploy.json` 的 `container_name` → `docker logs <name> --tail 200`
- CAPABILITY 0 pass: 这个 0.5B 模型本来就弱,只要 `pass_rate >= 0` 不报 error 即可。如果整个 stage 报 ValidationError → 看 capability.json 字段缺失

---

## 3 · INV-1 / INV-2 实地验证 (5 min)

烟测跑完立刻对比:

```bash
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' \
    | sort > /tmp/postflight_prod_snapshot.txt

diff /tmp/preflight_prod_snapshot.txt /tmp/postflight_prod_snapshot.txt
```

**通过标准**: diff 输出**只有** `e9-vllm-*` 在 preflight 没有、postflight 也没有(因为 CLEANUP 删干净了),或者 `e9-*` 仅在某一个 snapshot 出现(取决于你做 diff 的时机)。产线行(`minimax-*`/`xrouter`/`glm-*`/`kimi-*`)必须**完全一致**——名字、status、image 都不动。

**失败时**: 立即停机查根因。任何产线行变动 = INV-2 违反 = 评估管道严重故障。

GPU:

```bash
nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader > /tmp/postflight_gpu_snapshot.txt

diff /tmp/preflight_gpu_snapshot.txt /tmp/postflight_gpu_snapshot.txt | head -20
```

**通过标准**: GPU 0-3 的 memory.used 与 preflight 差异 < 1 GB(产线常量负载小幅波动);GPU 4-7 烟测后应该 ≈ 0 MB(CLEANUP 释放完了)。

---

## 4 · INV-4 产线文件未触碰 (2 min)

```bash
sudo find /etc/heyi-engine /etc/systemd/system/heyi-engine.* \
    -type f -newer /tmp/preflight_prod_snapshot.txt 2>/dev/null
```

**通过标准**: 输出**空**。如果有任何 file mtime 比 preflight 新 = INV-4 违反,立即 rollback。

---

## 5 · 启动 systemd loop (1 min)

```bash
sudo systemctl start heyi-eval-orchestrator heyi-eval-discover heyi-eval-panel
sudo systemctl start heyi-eval-backup.timer

sudo systemctl status --no-pager \
    heyi-eval-orchestrator heyi-eval-discover heyi-eval-panel heyi-eval-backup.timer
```

**通过标准**: 4 个单元都 `Active: active`,`Main PID` 非零。

打开 panel 验证:

```bash
curl -fsS http://127.0.0.1:8085/api/health | jq
```

应当看到 `engine.ok=true` 和 `containers` 中只有 `e9-` 前缀(或为空)。

---

## 6 · 备份验证 (≥ 30 min 后)

让 backup timer 自然触发至少一次:

```bash
sleep 1800   # 等 30 min
ls -lh /home/ai/heyi-eval-backups/ | tail
```

**通过标准**:
- 目录里出现至少 1 个 `snapshot-YYYYMMDDTHHMMSS/` 子目录
- 通过 panel 验证: `curl -fsS http://127.0.0.1:8085/api/backups | jq` → `last_backup_age_s < 1800`

**失败时**:
- 检查 `journalctl -u heyi-eval-backup.service --since "1 hour ago"`
- 通常是 rsync 权限问题 / 目标盘满

---

## 7 · 24h timer 完整验证 (隔天)

≥ 24 h 之后回来跑:

```bash
cd /home/ai/heyi-eval-v10
scripts/verify_24h_timer.sh | tee /tmp/verify_24h.json
```

**通过标准**: `jq '.overall == "pass"' /tmp/verify_24h.json` 输出 `true`,5 个 check id 全 pass。

**失败时**: 看 `details[].id == "T-24h-N"` 的 status,逐条排查。

---

## 8 · 拆机 (可选)

如果只是演练、要还原现场:

```bash
sudo systemctl stop heyi-eval-orchestrator heyi-eval-discover \
    heyi-eval-panel heyi-eval-notify-sync heyi-eval-backup.timer

sudo systemctl disable heyi-eval-orchestrator heyi-eval-discover \
    heyi-eval-panel heyi-eval-notify-sync heyi-eval-backup.timer

sudo rm -f /etc/systemd/system/heyi-eval-*.service \
           /etc/systemd/system/heyi-eval-*.timer
sudo systemctl daemon-reload
```

**通过标准**: `systemctl list-unit-files 'heyi-eval-*'` 输出空。`docker ps` 与 step 0 的 snapshot 完全一致。

数据保留: `/home/ai/heyi-eval-data/`、`/home/ai/heyi-eval-backups/` 不动,留作历史。

---

## 9 · 演练后核对清单

- [ ] step 2 烟测 9 stage 全 OK
- [ ] step 3 产线容器 snapshot diff 为空
- [ ] step 3 GPU 4-7 释放完成
- [ ] step 4 产线 /etc/ 文件未被触碰
- [ ] step 6 备份目录有 30min 内的 snapshot
- [ ] step 7 verify_24h_timer.sh overall=pass
- [ ] 演练全程不影响产线 LLM 流量(用 panel 或外部探活验证)

任何一项 ✗ → 写一段 incident 记录到 `docs/INCIDENTS.md`,标记 RCA-required,再重跑相关 step。
