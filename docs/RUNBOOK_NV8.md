# RUNBOOK · NV8 真机演练 (PR#8 D 层)

> 这是 PR#8 唯一需要在 nv8-6000 真机上跑的部分。CI/mac 不能验,只能在 `100.127.173.85` ssh 上手动执行。
> 每一步含"通过标准"和"失败时怎么办"。建议在 ≥ 2 个清醒小时的窗口执行,中间不要并行做产线动作。

## 0 · 演练前置

```bash
# 在 nv8 上:
sudo -iu ai
cd /home/ai/heyi-eval-v10
git fetch origin
git log -1 --oneline       # 期望: PR#8 (本 runbook 所属 PR) 合并后的 main commit
                           # 即包含 tests/test_e2e_pipeline.py / tests/test_inv_production_isolation.py /
                           # docs/INVARIANTS.md / 本文件本身
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
.venv/bin/python -m orchestrator run
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

---

## 10 · 产线临时切到 TP=8 时,验证 graceful skip(PR#11 引入)

> 仅在运维真把产线切到 K2.6 TP=8(占满 GPU 0-7)时跑这一节。常态(M2.7 TP=4 占 0-3)不需要做。
> 目的:确认评估管道**自动避让**而不抢卡,run 标 `ABORTED` 而非 `FAILED`。

### 10.1 准备:确认产线确实占满了

```bash
docker inspect minimax 2>/dev/null | jq '.[0].HostConfig.DeviceRequests' || \
    docker inspect kimi-k26 | jq '.[0].HostConfig.DeviceRequests'

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

通过标准:GPU 0-7 全部 memory.used > 50 GiB(产线分摊占用)。
失败时:运维还没切完,等切完再跑;或者根本没切,跳过本节。

### 10.2 告诉评估管道"产线在 TP=8 临时态"

```bash
sudo tee -a /etc/heyi-eval-v10/env > /dev/null <<EOF
# 临时:产线切到 K2.6 TP=8,占满 0-7
HEYI_EVAL_PROD_ENGINE_CONTAINER=kimi-k26
HEYI_EVAL_PROD_ENGINE_GPUS=0,1,2,3,4,5,6,7
EOF

sudo systemctl restart heyi-eval-orchestrator
```

通过标准:`systemctl status heyi-eval-orchestrator` 显示 `Active: active (running)`。
失败时:env 文件语法错或 systemctl 拉不起来 → `journalctl -u heyi-eval-orchestrator -n 50`。

### 10.3 enqueue 一个真模型,观察 graceful skip

```bash
cd /home/ai/heyi-eval-v10
.venv/bin/python -m orchestrator enqueue Qwen/Qwen2.5-0.5B-Instruct
.venv/bin/python -m orchestrator run 2>&1 | tee /tmp/k28_drill.log
```

期望日志关键节点:
- `[DISCOVER] OK` → `[CURATE] OK` → `[METADATA] OK` → `[ENGINE_SELECT] OK`
- `[DEPLOY] starting` → `[DEPLOY] SKIPPED (graceful): eval pool ... overlaps prod_engine_gpus ...`
- `[run] <run_id> aborted at DEPLOY: ...`
- 没有 docker run 调用 (容器列表中无新 `e9-*`)

通过标准:
```bash
grep -E "SKIPPED \(graceful\)|aborted at DEPLOY" /tmp/k28_drill.log
docker ps --format '{{.Names}}' | grep '^e9-' || echo "no e9-* spawned (correct)"
```

应输出 graceful 提示行,且**无** `e9-*` 容器。

### 10.4 outbox 应该写了 run_aborted 事件(不是 run_failed)

```bash
tail -1 /home/ai/heyi-eval-data/notify_outbox.jsonl | jq '{event_type, level, body}'
```

通过标准:
```json
{
  "event_type": "run_aborted",
  "level": "warn",
  "body": "stage=DEPLOY\nreason=eval pool ... overlaps prod_engine_gpus ..."
}
```

`level=warn` 而非 `error`,`event_type=run_aborted` 而非 `run_failed` — 这是 PR#11 的核心契约,防止运维一看到告警就以为产线坏了。

### 10.5 验证产线没受任何影响

```bash
docker inspect "${HEYI_EVAL_PROD_ENGINE_CONTAINER:-kimi-k26}" \
    --format '{{.State.Status}}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

通过标准:产线容器仍 `running`,GPU 占用与 10.1 snapshot 一致(±1 GiB 容差)。

### 10.6 产线切回 M2.7 后,清理 env override 并验恢复

运维把产线切回稳态后:

```bash
sudo sed -i '/^HEYI_EVAL_PROD_ENGINE_CONTAINER=kimi-k26$/d' /etc/heyi-eval-v10/env
sudo sed -i '/^HEYI_EVAL_PROD_ENGINE_GPUS=0,1,2,3,4,5,6,7$/d' /etc/heyi-eval-v10/env
sudo systemctl restart heyi-eval-orchestrator
```

重新 enqueue 同一个模型,这次应该走完整 9 stage(回到 §2 通过标准)。

### 10.7 §10 核对清单

- [ ] 10.3 日志含 `SKIPPED (graceful)` + `aborted at DEPLOY`
- [ ] 10.3 没有任何 `e9-*` 容器被 spawn
- [ ] 10.4 outbox 有 `event_type=run_aborted` `level=warn`
- [ ] 10.5 产线容器和 GPU 占用未变化
- [ ] 10.6 切回稳态后能正常跑 9 stage

---

## 11 · 已知限制(PR#15 多模态架构落地后,2026-05)

### 11.1 (历史) transformers-runner 镜像 → 见 §12

> PR#19 已交付源码 + 构建脚本;首次部署见 §12。本节保留为"何时彻底完工"
> 的 checkbox。下面是 PR#19 实际状态:

`orchestrator/stages_py.py::_ENGINE_IMAGES["transformers"]` 指向
`heyi-eval/transformers-runner:v10`,PR#19 已经把镜像源码落地在
`transformers_runner/`(Dockerfile + 入口 server + 检测器),但仍需
**在 nv8 上首次构建** 才能真正解锁非 vLLM 模态(ASR / TTS / image_gen
/ video_gen / music_gen)。

**构建步骤**(详细落地见 §12):

```bash
cd ~/heyi-eval-v10
bash scripts/build_transformers_runner.sh --smoke
```

**当前 image 已实现的端点**:
- `/v1/chat/completions`(text / vlm)
- `/v1/audio/transcriptions`(asr)
- `/v1/audio/speech`(tts)
- `/v1/images/generations`(image_gen)
- `/v1/videos/generations` → 501 deferred(留给 PR#22 接入 CogVideoX/Mochi)
- `/v1/music/generations` → 501 deferred(留给 PR#22 接入 MusicGen)

**何时彻底解除**:构建成功 + PR#22(video/music 推理接入)合并后。

### 11.2 capability artifact 不自动清理

PR#15 的 tts / image_gen / video_gen / music_gen dispatcher 会把
生成的二进制写到 `runs/<run_id>/_artifacts/*.bin`。CLEANUP 阶段
**不删**这些字节产物(仅删 `e9-*` 容器和模型权重 cache)。
长期会在 `~/heyi-eval-data/runs/*/` 下累积。

**何时解除**:PR#18 面板会展示这些 artifact;到时考虑 14 天后自动归档/删除。

### 11.3 (PR#20) audio category 仅作 plumbing 烟测

PR#20 把 `asr.jsonl` / `music_understanding.jsonl` 从空文件填充到 5+5 个
items,但所有 items 都用合成正弦波 WAV 作 fixture(`a01_tone_440hz...`、
`a04_arpeggio_up...` 等),内容非真实语音/音乐。

为此,这 10 个 items 都通过 **新增的 `scorer_override="non_empty_output"`**
机制把默认 substring 评分换成"模型只要返回非空字符串即视为通过"。
这等于把这一波 audio category 当作**端到端管道烟测**——它能验证:

- DEPLOY 阶段是否拉起了 ASR/音频 LLM 容器
- dispatcher → HTTP → model → response 整条链路是否闭环
- 是否生成了合规的 CategoryRunResult

它**无法**验证模型的 ASR / 音乐理解准确度。真正的精度评估需要在
PR#21+ 接入 CC0 LibriSpeech / MusicCaps 样本后,把 `scorer_override`
去掉,改回 `substring`。

`video_understanding` 仍然 N/A(无 stdlib 生成 MP4 的路径,真 CC0
视频还在 PR#21+ 排期)。

---

## 12 · 构建 transformers-runner:v10 镜像(PR#19)

> 一次性操作,首次部署非 vLLM 模态前必做。重新构建只有在升级
> torch/transformers/diffusers 大版本时才需要(走 ADR)。

### 12.1 前置检查

```bash
# 确认 nvidia container runtime 已注册
docker info 2>/dev/null | grep -i "runtimes" | grep -q nvidia || \
    echo "WARN: nvidia runtime 未启用,需要先 systemctl restart docker"

# 确认 /var/lib/docker 有 ≥ 30 GB(镜像约 13 GB + 缓冲)
df -h /var/lib/docker | tail -1

# 确认 cu124 wheel 镜像源可访问(国内可能要走代理)
curl -sI https://download.pytorch.org/whl/cu124/ | head -1
```

### 12.2 构建

```bash
cd ~/heyi-eval-v10

# 仅构建(约 15-25 min,看网络)
bash scripts/build_transformers_runner.sh

# 构建 + /health 烟测(推荐;耗时多约 30 s)
bash scripts/build_transformers_runner.sh --smoke
```

预期输出末尾:`✓ smoke OK`,且 `/health` 返回
`{"status":"ok","capability":"text",...}`。

### 12.3 单模型联调(用真实的 Whisper-tiny 做最便宜的 ASR 烟测)

```bash
# 取一个 Whisper-tiny(~150 MB),放到本地
MODEL_DIR=$(mktemp -d)
huggingface-cli download openai/whisper-tiny --local-dir "$MODEL_DIR"

# 启动 runner
docker run --rm --name tf-runner-asr-smoke \
    --gpus '"device=4"' \
    -p 18000:8000 \
    -v "$MODEL_DIR:/model:ro" \
    heyi-eval/transformers-runner:v10 &

# 等 15 s 让模型 load
sleep 15

# 探针:检测应该判定为 asr
curl -fsS http://127.0.0.1:18000/health
# → {"capability": "asr", ...}

# 拿一段 fixture 音频试转录
curl -fsS -X POST http://127.0.0.1:18000/v1/audio/transcriptions \
    -F file=@orchestrator/capability_data/fixtures/audio/tone_440hz.wav
# → {"text": "..."}

docker stop tf-runner-asr-smoke
```

### 12.4 加入 orchestrator 全流程

构建并 smoke 通过后,**不需要任何配置改动**——
`orchestrator/stages_py.py::_ENGINE_IMAGES["transformers"]` 已经指向
`heyi-eval/transformers-runner:v10`,下一个 enqueue 进来的非 vLLM 模型
会自动用上。

### 12.5 §12 核对清单

- [ ] `docker images | grep transformers-runner` 显示 `:v10` 标签
- [ ] `--smoke` 烟测通过
- [ ] §12.3 Whisper-tiny 联调返回非空 `text`
- [ ] §11.1 「未就绪」状态可以从文档移除
