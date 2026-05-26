# RUNBOOK · NV8 真机演练 (PR#8 D 层)

> 这是 PR#8 唯一需要在 nv8-6000 真机上跑的部分。CI/mac 不能验,只能在 `<NV8_TAILNET_IP>` ssh 上手动执行。
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
PR#22+ 接入 CC0 LibriSpeech / MusicCaps 样本后,把 `scorer_override`
去掉,改回 `substring`。

### 11.4 (PR#21) video_understanding 用 lavfi 合成视频烟测

PR#21 用 ffmpeg `lavfi` 合成源生成了 3 段小 H.264 MP4(SMPTE 色条 +
红色纯色 + RGB 色彩循环,合计 60 KiB),并填了 5 条 `video_understanding`
items:其中 3 条用 `substring` 评(可断言"red" / "test" 等颜色或测试
模式关键字),2 条用 `scorer_override="non_empty_output"` 评(开放式
"summary"问题,只验证非空回答)。

固件依赖 ffmpeg。**ffmpeg 不在 PATH 时**,`scripts/build_capability_fixtures.py`
会跳过视频生成并打印 WARN;现有已 commit 的视频文件不会被删。nv8 的
`transformers-runner` Dockerfile (`apt-get install ffmpeg`)和 Mac
开发机上的 Homebrew 都自带 ffmpeg,所以默认就能用。

真正的视频理解评估(MVBench / Video-MME 等)需要真实视频语料,排在
PR#22+。`video_gen` category 仍然按原计划走 LLM-judge,无需视频固件。

---

## 12 · 构建 transformers-runner:v10 镜像(PR#19 / PR#19b)

> 一次性操作,首次部署非 vLLM 模态前必做。重新构建只有在升级
> torch/transformers/diffusers 大版本时才需要(走 ADR)。
>
> **PR#19b(2026-05-22)在 nv8 真机验证后将 base 镜像从
> `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` 改为
> `vllm/vllm-openai:v0.21.0`**,见 §12.6「基底镜像选择历程」。

### 12.1 前置检查

```bash
# 确认 nvidia container runtime 已注册
docker info 2>/dev/null | grep -iE "runtimes|nvidia" | head -5
# 期望看到 `nvidia` runtime 或 CDI `nvidia.com/gpu=N`

# 确认 /var/lib/docker 有 ≥ 30 GB(layer 复用后增量 ~3 GB)
df -h /var/lib/docker | tail -1

# 确认 base 镜像已在本地(PROD vLLM 用同一镜像,通常已经 pull 过)
docker images vllm/vllm-openai --format "{{.Tag}}\t{{.Size}}" | head -3
# 期望看到 v0.21.0(或更高);如果没有,先在网络好的窗口
# `docker pull vllm/vllm-openai:v0.21.0`

# 确认 Aliyun PyPI 可访(在容器内 pip install diffusers/librosa 用)
curl -sI https://mirrors.aliyun.com/pypi/simple/ | head -1
```

### 12.2 构建

```bash
cd ~/heyi-eval-v10

# 仅构建(layer 已 cache 时 ~80 s;首次 cold 约 2 min)
bash scripts/build_transformers_runner.sh

# 构建 + /health 烟测(推荐;再多 ~15 s)
bash scripts/build_transformers_runner.sh --smoke
```

预期输出末尾:`✓ build OK`(以及 `✓ smoke OK`),且 `/health` 返回
`{"status":"ok","capability":"text",...}`。

最终镜像大小约 **24 GB**(base 已含 torch/transformers/CUDA,
我们仅 layer 了 diffusers + librosa + soundfile,新增 ~600 MB)。

### 12.3 Whisper-tiny ASR 真机联调(已在 nv8 验证 ✅ 2026-05-22)

```bash
# 取 Whisper-tiny(~150 MB),走 hf-mirror 国内镜像
MODEL_DIR=/tmp/whisper-tiny && mkdir -p "$MODEL_DIR"
for f in config.json generation_config.json model.safetensors \
         preprocessor_config.json tokenizer.json tokenizer_config.json \
         vocab.json normalizer.json added_tokens.json merges.txt \
         special_tokens_map.json; do
    curl -sL -o "$MODEL_DIR/$f" \
         "https://hf-mirror.com/openai/whisper-tiny/resolve/main/$f"
done

# 启动 runner(用 EVAL 池 GPU 4)
docker rm -f tf-runner-asr-smoke 2>/dev/null
docker run -d --name tf-runner-asr-smoke \
    --gpus '"device=4"' \
    -p 18000:8000 \
    -v "$MODEL_DIR:/model:ro" \
    heyi-eval/transformers-runner:v10

# 等 ≤ 5 s,/health 应该返回 capability=asr
for i in $(seq 1 15); do
    body=$(curl -m 2 -fsS http://127.0.0.1:18000/health 2>/dev/null)
    [ -n "$body" ] && { echo "$body"; break; }
    sleep 1
done
# → {"capability": "asr", "framework": "transformers", ...}

# 用我们的 PR#20 合成音频 fixture 真转录
curl -fsS -X POST http://127.0.0.1:18000/v1/audio/transcriptions \
    -F file=@orchestrator/capability_data/fixtures/audio/a04_arpeggio_up_C_3s.wav
# → {"text": " Thank you very much."}   ← 合成音上的幻觉,非空即可

docker stop tf-runner-asr-smoke && docker rm tf-runner-asr-smoke
```

**2026-05-22 nv8 实测结果**:首条请求 ~7s(weights 加载 + JIT),
后续 ~150 ms;GPU 4 显存 ~860 MiB。所有 5 个合成 fixture 都返回
非空文本——`non_empty_output` scorer 全部判 pass,符合 PR#20 设计。

### 12.4 加入 orchestrator 全流程

构建并 smoke 通过后,**不需要任何配置改动**——
`orchestrator/stages_py.py::_ENGINE_IMAGES["transformers"]` 已经指向
`heyi-eval/transformers-runner:v10`,下一个 enqueue 进来的非 vLLM 模型
会自动用上。

### 12.5 §12 核对清单

- [x] `docker images | grep transformers-runner` 显示 `:v10` 标签
- [x] `bash scripts/build_transformers_runner.sh` 报 `✓ build OK`
- [x] §12.3 Whisper-tiny 联调返回非空 `text`(2026-05-22 nv8 ✅)
- [x] §11.1 「未就绪」状态可以从文档移除

### 12.6 基底镜像选择历程(失败 → 成功)

| 尝试 | 基底 | 结果 | 失败原因 |
|---|---|---|---|
| ① | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + pip cu124 wheels | ❌ pull 阶段就失败 | Docker Hub 国内拉镜像 manifest size mismatch;且 cu124 wheel 无 Blackwell(sm_120)kernel |
| ② | `voipmonitor/llm-pytorch-blackwell:nightly`(本地已有) | ❌ 推理失败 | torch 2.10+cu130 在 Blackwell 上 `cublasLtMatmulAlgoGetHeuristic` 对 384×384 方阵 Linear 返回 0 算法(`CUBLAS_STATUS_NOT_INITIALIZED`),Whisper q_proj 必踩 |
| ③ | **`vllm/vllm-openai:v0.21.0`**(本地已有,PROD vLLM 同款) | ✅ 通过 | torch 2.11+cu130 的 cublasLt catalog 完整覆盖 sm_120,Linear/Whisper/SDPA 全部正常 |

**结论**:nv8 这台 Blackwell 机器的非 vLLM 容器统一以
`vllm/vllm-openai:v0.21.0` 作为 base(它自带 torch 2.11+cu130 +
transformers 5.8),我们只 layer 真正缺的 `diffusers / soundfile /
librosa` 三个包。任何后续 base 升级需要重跑 §12.3 + §12.6 矩阵。

## 13 · Agent sandbox 部署(PR#22a + PR#22b-M2)

PR#22a 在 nv8 上落地了一个用来跑 Claude Code agent 的隔离沙箱(详见
`docs/INVARIANTS.md` §INV-16~21 与 `deploy/agent-sandbox/README.md`)。
PR#22b-M2 在沙箱里加了**追加式审计写入通道**:root daemon (`heyi-eval-audit.service`)
监听 `/run/heyi-eval-agent-audit.sock`,agent 通过 unix socket 写入,
`SO_PEERCRED` 强制校验 peer uid + INV-21 静态守护 + INV-18 deny-all
ACL 保护 DB 文件,保证 agent 既不能伪造审计记录也不能擦掉已记录的命令。

> **历史注**:M1 (commit 508b639) 曾经走 setuid 包装脚本路径,但
> `NoNewPrivileges=true` 与 `sudo` 不兼容(sudo 在 no_new_privs 下
> 拒绝 setuid),所以 M1 在 ad-hoc `sudo -u` 测试里能过、在 agent
> unit 内 100% 失败。M2 通过 daemon socket 绕过 setuid,这是唯一
> 与现有 systemd 硬化共存的设计。详见 `INVARIANTS.md` §INV-21。
v9 时代 agent 直接以 `ai` 用户跑——而 `ai` 在 `docker` + `sudo` 组,
导致 `docker exec minimax bash -c 'rm -rf /'` 等事故可以一行命令发起。
v10 的 PR#22a 把 agent 钉死在 `heyi-eval-agent` 这个无 docker / 无
sudo / 无登录的系统账号下,Docker API 走只读 socket-proxy,cgroup +
RuntimeMaxSec 守住资源,五层防御彼此独立。

### 13.1 一次性部署

`scripts/bootstrap_nv8.sh` 第 4b 阶段会跑 §13 全部步骤,首次部署
直接:

```bash
sudo -u ai bash /home/ai/heyi-eval-v10/scripts/bootstrap_nv8.sh
```

如果仅升级 sandbox 部分(没动 orchestrator),可以只手跑:

```bash
cd /home/ai/heyi-eval-v10/deploy/agent-sandbox
sudo bash setup_agent_user.sh
sudo bash acl_install.sh
sudo install -m 0440 sudoers.d/heyi-eval-agent /etc/sudoers.d/heyi-eval-agent
sudo visudo -c -f /etc/sudoers.d/heyi-eval-agent
sudo docker compose -f compose.agent-socket-proxy.yml up -d
sudo install -m 0644 ../systemd/heyi-eval-agent.slice    /etc/systemd/system/
sudo install -m 0644 ../systemd/heyi-eval-agent@.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/heyi-eval-agent@.service \
    /etc/systemd/system/heyi-eval-agent.slice
sudo systemctl daemon-reload
# PR#22b-M2 (audit daemon + agent runner)
sudo install -m 0755 -o root -g root heyi-eval-agent-audit-client.py /usr/local/bin/heyi-eval-agent-audit-client
sudo install -m 0755 -o root -g root heyi-eval-agent-prepare         /usr/local/sbin/heyi-eval-agent-prepare
sudo install -m 0755 -o root -g root heyi-eval-agent-run             /usr/local/bin/heyi-eval-agent-run
sudo install -m 0644 ../systemd/heyi-eval-audit.service              /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now heyi-eval-audit.service
# socket appears at /run/heyi-eval-agent-audit.sock (mode 0660 root:heyi-eval-agent)
```

### 13.2 验证沙箱六层

`drills/run_all.sh` 顺序跑 6 个攻击演练,任一 BREACH 立刻 exit 1
(drill-6 在 audit wrapper 未安装时自动跳过,不影响 PR#22a-only 的部署):

```bash
sudo bash /home/ai/heyi-eval-v10/deploy/agent-sandbox/drills/run_all.sh
```

期望输出尾部(完整 PR#22a+PR#22b-M1 部署):

```
ALL 6 DRILLS PASSED — INV-16/17/18/19/20/21 hold end-to-end
  drill-1 rc=0   # store deletion blocked (filesystem ACL)
  drill-2 rc=0   # docker exec into minimax blocked (socket proxy 403)
  drill-3 rc=0   # sudo escalation blocked (whitelist denies forbidden)
  drill-4 rc=0   # audit log evasion blocked (deny-all ACL)
  drill-5 rc=0   # fork-bomb cgroup-capped + RuntimeMaxSec watchdog
  drill-6 rc=0   # audit write append-only (wrapper EUID + subcommand WL + INV-18 ACL)
```

任何一项 BREACH = 沙箱漏洞,**不要继续 PR#22b-M2 接入 agent runner**,
先回到 `deploy/agent-sandbox/` 排查。

### 13.3 常见排错

| 症状                                            | 根因 / 处理 |
|------------------------------------------------|---|
| `useradd: user 'heyi-eval-agent' exists`(idempotent OK) | setup_agent_user.sh 是幂等的,第二次跑就是 normalise,不报错 |
| drill 1 报 `权限不够` 在 drill 脚本本身          | `/home/ai` 是 `0750 ai:ai` → agent 用户连 traverse 都不能;acl_install.sh §0 加了 `setfacl -m u:heyi-eval-agent:x /home/ai`(只通过,不可 ls)。重跑 acl_install.sh 修复 |
| `docker compose up` 后 proxy 容器 restart-loop  | tecnativa 镜像需要写 `/tmp/haproxy.cfg` + `/run/haproxy.pid`;compose 里已经声明 `tmpfs:/tmp size=8m + /run size=4m`,**不要**给 `/var/run/docker.sock` 加 `:ro`(unix socket 双向,会让 haproxy 永远阻塞);也不要把 `pids_limit` 降到 64 以下(haproxy worker fork 会 EAGAIN) |
| drill 2 显示 `read -> 5xx`                       | proxy 还没 healthy,等 6 秒再跑;或 `docker logs heyi-eval-agent-socket-proxy` 看真实状态 |
| drill 5 在 ssh 远端"无声卡住"超过 30 秒          | 不要用 `systemd-run --wait` 经 ssh+sudo+pipe 调用 — fd 继承让 ssh channel 不关。drill 已经改用 detach + `systemctl is-active` 轮询;若再卡,kill ssh 子进程 + 直接登录 nv8 跑 `bash deploy/agent-sandbox/drills/attack_resource_budget.sh > /tmp/drill5.out 2>&1` |
| drill 4 报 `BREACH cat audit.sqlite succeeded`  | INV-18 被破:看 `getfacl /var/log/heyi-eval-agent`,正确状态是 `user:heyi-eval-agent:---`(default ACL 也要 `---`);重跑 acl_install.sh §5 修复 |

### 13.4 §13 核对清单

- [x] `id heyi-eval-agent` 不含 `docker` / `sudo` / `wheel` / `adm`
- [x] `getfacl /home/ai/heyi-eval-data/store` 显示 `user:heyi-eval-agent:r-x`(不含 `w`)
- [x] `getfacl /var/log/heyi-eval-agent` 显示 `user:heyi-eval-agent:---`
- [x] `curl http://127.0.0.1:2377/_ping` 返回 `OK`,`curl -X POST .../containers/minimax/stop` 返回 `403`
- [x] `systemctl list-unit-files heyi-eval-agent@.service` 显示 `static`
- [x] `bash deploy/agent-sandbox/drills/run_all.sh` 退出 0,6/6 BLOCKED OK
- [x] PR#22b-M2: `heyi-eval-audit.service` active 且 `/run/heyi-eval-agent-audit.sock` 存在且 mode 0660 root:heyi-eval-agent
- [x] PR#22b-M2: drill-6 真机绿(`run_all.sh` 末尾"drill-6 rc=0")
- [x] PR#22b-M2: `systemctl start heyi-eval-agent@m2demo.service` 完整 lifecycle 通过(prepare→audit-begin→smoke→audit-end),`/var/lib/heyi-eval-agent/runs/m2demo/outbox/run_meta.json` 有内容
- [x] PR#22b-M3: orchestrator `ai` 用户可通过 sudoers NOPASSWD 调 `systemctl start heyi-eval-agent@<id>.service` + `heyi-eval-agent-harvest <id>`(见 §14)
- [ ] (推迟到 PR#23)LLM-judge 接 M2.7 API + eval_gpus 默认 (5,6,7) + ENGINE_SELECT oversize gating

## 14 · Orchestrator 接入 sandbox(PR#22b-M3)

PR#22b-M3 在 M2 沙箱基础上加了**两段桥接管线**,让 `ai` 用户跑的
orchestrator 主 loop 可以"无密码、最小权限"地拉起一次沙箱 agent
run,然后把 agent 在 root-only 0750 HOME 里写的 outbox 拿回到
`/home/ai/heyi-eval-data/runs/<id>/outbox/`:

- `Python` 端:`orchestrator/agent_runner.py::invoke_agent(run_id, AgentSpec)`
  - 把 spec.json 写到 `<DATA_ROOT>/runs/<id>/spec.json`(ai 可写)
  - `sudo -n systemctl start heyi-eval-agent@<id>.service` 拉起 agent
  - 轮询 `ActiveState != active|activating|deactivating|reloading`,过 1800+60 s 报 timeout
  - 读 `ExecMainStatus`(unit 退码)+ `agent_audit.query_recent` 查 begin/end 配对
  - `sudo -n /usr/local/sbin/heyi-eval-agent-harvest <id>` 把 outbox 拷出 + chown ai:ai
  - 把以上写成 `<DATA_ROOT>/runs/<id>/agent_summary.json` 供 Panel
- `Shell` 端:`/usr/local/sbin/heyi-eval-agent-harvest <run-id>`
  - 仅接受 run-id(同一份正则:`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)
  - 固定 src=`$AGENT_HOME/runs/<id>/outbox` / 固定 dst=`$DATA_ROOT/runs/<id>/outbox`
  - 不接受任意路径,不调用 rsync,杜绝路径注入

### 14.1 一次性部署(已并入 bootstrap)

```bash
sudo -u ai bash /home/ai/heyi-eval-v10/scripts/bootstrap_nv8.sh
# 第 4b 阶段会:
#   - install -m 0755 heyi-eval-agent-harvest -> /usr/local/sbin/
#   - install -m 0440 sudoers.d/heyi-eval-orchestrator -> /etc/sudoers.d/
#   - visudo -c -f /etc/sudoers.d/heyi-eval-orchestrator
```

仅升级 M3 部分(不动 M1/M2):

```bash
cd /home/ai/heyi-eval-v10
sudo install -m 0755 -o root -g root deploy/agent-sandbox/heyi-eval-agent-harvest /usr/local/sbin/heyi-eval-agent-harvest
sudo install -m 0440 deploy/sudoers.d/heyi-eval-orchestrator /etc/sudoers.d/heyi-eval-orchestrator
sudo visudo -c -f /etc/sudoers.d/heyi-eval-orchestrator
```

### 14.2 端到端验证 smoke

```bash
sudo -u ai bash -lc '
  set -euo pipefail
  cd /home/ai/heyi-eval-v10
  run_id="m3demo-$(date +%s)"
  .venv/bin/python -m orchestrator.agent_runner "$run_id" --mode smoke
  echo "--- summary ---"
  cat /home/ai/heyi-eval-data/runs/$run_id/agent_summary.json
  echo "--- harvested outbox ---"
  ls -la /home/ai/heyi-eval-data/runs/$run_id/outbox/
  cat /home/ai/heyi-eval-data/runs/$run_id/outbox/run_meta.json
'
```

期望:`agent_summary.json::ok == true`,`unit_exit_code == 0`,
outbox 至少有 `run_meta.json` + `payload.stdout`,且 `payload.stdout`
头一行是 `agent-runner smoke run_id=m3demo-...`。

### 14.3 常见排错

| 症状 | 根因 / 处理 |
|---|---|
| `sudo: a password is required` | sudoers 没安装或 visudo 报错;手跑 §14.1 末段 visudo -c |
| `start_failed rc=5 stderr='Unit heyi-eval-agent@xxx.service not found'` | unit 模板没装;`sudo systemctl daemon-reload && systemctl list-unit-files heyi-eval-agent@.service` |
| `outbox_files: []` 且 unit_exit_code=0 | harvest 帮助脚本权限错;`sudo getfacl /var/lib/heyi-eval-agent/runs/<id>/outbox`,正确状态 owner 是 heyi-eval-agent;然后 `sudo -u ai sudo -n /usr/local/sbin/heyi-eval-agent-harvest <id>` 看 stderr |
| `AgentRunnerError(kind=timeout)` | smoke 跑 1800 s+ 不正常,先 `systemctl status heyi-eval-agent@<id>.service` 看是否卡在 `ExecStartPre=` |
| summary `ok==False` 但 `unit_exit_code==0` | audit `end` 行没回写;最常见原因是 audit daemon 崩溃,`journalctl -u heyi-eval-audit.service -n 50` |

### 14.4 §14 核对清单

- [x] `sudo visudo -c -f /etc/sudoers.d/heyi-eval-orchestrator` 退 0
- [x] `sudo -u ai sudo -n /usr/local/sbin/heyi-eval-agent-harvest m3demo-X` 不弹密码(可能空 outbox)
- [x] `sudo -u ai .venv/bin/python -m orchestrator.agent_runner m3demo-Y --mode smoke` ok=True
- [x] `agent_summary.json` 中 `audit.end_exit == 0` 且 `audit.begin_id != null`
- [x] `outbox/run_meta.json` 由 ai 用户可读(ownership ai:ai)

## 15 · 接入 M2.7 API + GPU 池收缩(PR#23)

PR#23 把 v10 评估管线对齐到 2026-05 的 nv8 实际拓扑:

- **生产 LLM = MiniMax-M2.7**(vLLM 容器 `minimax`,TP=4,GPU 0-3,端口
  10814,模型名字符串 `MiniMax-M2.7`)。LLM-judge 全部走这一条管线。
- **GPU 4** 被 ComfyUI host 进程长期占用(`python main.py --port 8188`,
  ~93 GB),评估池**绝不**触碰。
- **评估池** 默认 `(5, 6, 7)` 共 3 张卡(188 GB 余量),操作员可通过
  `HEYI_EVAL_EVAL_GPUS` 临时扩缩。
- **超大模型**(`tensor_parallel_size > 3`,即 70B+/MoE)在
  `ENGINE_SELECT` 阶段被 INV-23 oversize 闸门拦下,只落元数据,不进
  DEPLOY。

### 15.1 LLM-judge 模型名(强制)

历史的 `"model": "auto"` 已废弃——M2.7 vLLM 服务名是字符串
`MiniMax-M2.7`,新代码硬性引用:

- `orchestrator/llm_judge.py` 调用 chat completions 时 body 必填
  `"model": "MiniMax-M2.7"`;
- 改名走 env `HEYI_EVAL_JUDGE_MODEL`(例如临时切到 staging M2.8),
  不要改源码;
- 静态守护 `tests/test_pr23_m27_api_and_oversize.py::TestJudgeModelName::test_no_auto_literal_anywhere` 拦截任何回归把 `"model": "auto"` 提交回来。

三条访问路径见 `rules/42-heyi-m27-api.md`:
本机 `http://127.0.0.1:10814/v1` / Tailscale `http://<NV8_TAILNET_IP>:10814/v1` / 公网 `cat /home/ai/cf-m27-url.txt`。
nv8 上 orchestrator 默认走本机;mac 开发箱跑 e2e 时:

```bash
export HEYI_ENGINE_URL=http://<NV8_TAILNET_IP>:10814
export HEYI_EVAL_JUDGE_MODEL=MiniMax-M2.7
```

#### 15.1.1 切到云雾(yunwu)作为 judge provider — PR#68

当本机 prod_engine 容器繁忙、暂离线、或操作员希望降低本地负载时,可以把
LLM-judge 切到云雾的 OpenAI 兼容 endpoint(同样支持 `MiniMax-M2.7` 这个
model id,经 `curl https://yunwu.ai/v1/models` 验证)。三个环境变量即可:

```bash
export HEYI_EVAL_JUDGE_PROVIDER=yunwu
export YUNWU_BASE_URL=https://yunwu.ai/v1
export YUNWU_GENERAL_KEY=sk-...           # 从 ~/.yoli.env 取
# HEYI_EVAL_JUDGE_MODEL 不必设,默认 MiniMax-M2.7 与云雾 id 完全一致
```

在 nv8 上的常驻配置写到 `/etc/heyi-eval-v10/env`,改完执行
`sudo systemctl restart heyi-eval-orchestrator.service` 让 systemd 重读。
关键不变量:

- INV-14 仍生效 — judge 只接收 EVAL 产物字节,不传测试 prompt。云雾换的
  只是承载 M2.7 的物理 endpoint,不是 trust domain。
- URL 拼接已做幂等处理:`/v1` 既可写也可不写,绝不会出现历史 bug 里那种
  `/v1/v1/chat/completions` 的双拼。回归用例 `TestYunwuJudgeProvider` 守
  着这条边界。
- 兜底链:`YUNWU_GENERAL_KEY` → `YUNWU_KEY_2` → `YUNWU_GPT_KEY`。任一非空
  即用,避免单 key 限流时操作员要改源码。

### 15.2 评估池切换(操作员视角)

```bash
# 常态(PR#23 默认,GPU 4 给 ComfyUI):
unset HEYI_EVAL_EVAL_GPUS

# 临时回到 4 卡池(ComfyUI 已下线):
export HEYI_EVAL_EVAL_GPUS="4,5,6,7"

# 仅 2 卡评测窗口(GPU 6/7 借出):
export HEYI_EVAL_EVAL_GPUS="5"

# 当晚完全没有 GPU 可用(全员上 prod):
export HEYI_EVAL_EVAL_GPUS=""    # PR#11 graceful-skip 全量
```

### 15.3 oversize 闸门(开发者视角)

- 判定时机:`ENGINE_SELECT`(NOT DEPLOY) — 避免浪费 100 GB+ 模型下载。
- 判定公式:`tp_size = vllm_args.tensor_parallel_size`(由
  `_vllm_args_hint(metadata)` 根据 `param_count` 粗估),与
  `len(cfg.eval_gpus)` 直接比较。
- 触发结果:engine.json `engine="metadata_only" oversize=true`,
  pipeline 走 `GracefulSkip` → run 标 `ABORTED`(不计 failure)。
- Panel 仍能拉到 metadata.json + engine.json,该模型以"仅采集"展示。
- 想强行测一把超大模型:操作员临时扩 `HEYI_EVAL_EVAL_GPUS` 到匹配
  tp_size 的 GPU 数即可,**无需改代码**。

### 15.4 §15 核对清单

- [x] `cfg.eval_gpus == (5,6,7)`(无 env 时)
- [x] `cfg.judge_model_name == "MiniMax-M2.7"`(无 env 时)
- [x] `orchestrator/llm_judge.py` 不再含 `"model": "auto"`
- [x] 405B 模型 ENGINE_SELECT 后 `engine.json::engine == "metadata_only"` 且 `oversize == true`
- [x] 7B 模型 ENGINE_SELECT 后正常 `engine == "vllm"` 且 `oversize == false`
