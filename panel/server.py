"""heyi-eval-v9 management panel.

Single-file Python stdlib HTTP server that renders a read-only dashboard
over HEYI_EVAL_DATA. Intended to run on nv8 :8090 (tailnet only).

No external deps. Serves:
  - GET /              dashboard (HTML)
  - GET /api/health    JSON: ccr probe + heartbeat + GPU + container states
  - GET /api/runs      JSON: list of all runs with stage status + duration
  - GET /api/runs/<id> JSON: detail of one run (state + capability + showcase + meta)
  - GET /api/queue     JSON: queue status
  - GET /api/discover  JSON: discover candidates summary
  - GET /api/outbox    JSON: last N notify events

It NEVER writes anything to disk. NEVER runs subprocesses besides
`nvidia-smi` and `docker ps` for liveness signals.
"""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DATA_ROOT = Path(os.environ.get("HEYI_EVAL_DATA", "/home/ai/heyi-eval-data"))
BACKUPS_ROOT = Path(os.environ.get("HEYI_EVAL_BACKUPS", "/home/ai/heyi-eval-backups"))
ENGINE_URL = os.environ.get("HEYI_ENGINE_URL", "http://127.0.0.1:10814")
ENGINE_API_KEY = os.environ.get("HEYI_ENGINE_API_KEY")
LISTEN_HOST = os.environ.get("HEYI_PANEL_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("HEYI_PANEL_PORT", "8090"))


# ---------- data readers (read-only over HEYI_EVAL_DATA) ----------

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        return []
    if limit is not None and len(out) > limit:
        return out[-limit:]
    return out


def list_runs() -> list[dict]:
    """Each run summary: run_id, hf_id, status, started, ended, stages overview,
    plus result highlights (capability score, showcase first_impression, engine)
    so the dashboard can show real test results inline.
    """
    runs_root = DATA_ROOT / "runs"
    if not runs_root.exists():
        return []
    summaries: list[dict] = []
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        state = _read_json(run_dir / "state.json")
        if not state:
            summaries.append({
                "run_id": run_dir.name,
                "hf_id": "?",
                "status": "no-state",
                "stages": {},
                "started_at": None,
                "ended_at": None,
                "duration_s": None,
                "capability_score": None,
                "capability_pass_rate": None,
                "first_impression": None,
                "summary_preview": None,
                "engine": None,
                "engine_image": None,
                "params_b": None,
                "license": None,
                "modality": None,
                "publisher": None,
            })
            continue
        stages = state.get("stages") or {}
        stage_dur = {name: round(s.get("duration_s") or 0.0, 1) for name, s in stages.items()}
        stage_status = {name: s.get("status", "?") for name, s in stages.items()}
        started = state.get("created_at")
        ended = state.get("ended_at")

        cap = _read_json(run_dir / "capability.json") or {}
        show = _read_json(run_dir / "showcase.json") or {}
        perf = _read_json(run_dir / "perf_bench.json") or {}  # PR#14
        eng = _read_json(run_dir / "_meta" / "engine.json") or {}
        meta = _read_json(run_dir / "_meta" / "metadata.json") or {}
        curated = _read_json(run_dir / "_meta" / "curated.json") or {}

        summary_text = show.get("summary") or ""
        publisher = curated.get("publisher") or meta.get("publisher") or {}
        if isinstance(publisher, dict):
            publisher_name = publisher.get("name")
        else:
            publisher_name = str(publisher) if publisher else None

        modalities = meta.get("modality") or curated.get("modalities") or []
        if isinstance(modalities, str):
            modality_str = modalities
        elif isinstance(modalities, list):
            modality_str = ",".join(modalities) if modalities else None
        else:
            modality_str = None

        # PR#18: surface per-category counts so the leaderboard can hint
        # which modalities were actually tested. ``categories`` is the
        # PR#15 dict; older artifacts only have ``results`` and fall
        # back to a single-bucket figure.
        cap_categories = cap.get("categories") or {}
        applicable_cats = sorted(
            name for name, info in cap_categories.items()
            if isinstance(info, dict) and info.get("applicable")
        )

        # PR#18: PERF_BENCH p50 highlights for the leaderboard.
        ttft_p50 = None
        tps_p50 = None
        if isinstance(perf, dict):
            ttft_obj = perf.get("ttft_ms") or {}
            tps_obj = perf.get("tps_single") or {}
            if isinstance(ttft_obj, dict):
                ttft_p50 = ttft_obj.get("p50")
            if isinstance(tps_obj, dict):
                tps_p50 = tps_obj.get("p50")

        summaries.append({
            "run_id": state.get("run_id") or run_dir.name,
            "hf_id": state.get("hf_id"),
            "status": state.get("status", "?"),
            "stages": stage_status,
            "stage_durations": stage_dur,
            "failure_reason": state.get("failure_reason"),
            "started_at": started,
            "ended_at": ended,
            "duration_s": (ended - started) if (started and ended) else None,
            "capability_score": cap.get("score"),
            "capability_pass_rate": cap.get("pass_rate"),
            "capability_categories": applicable_cats,
            "showcase_items": len(show.get("items") or []),
            "first_impression": show.get("model_first_impression"),
            "summary_preview": summary_text[:140] if summary_text else None,
            "ttft_ms_p50": ttft_p50,
            "tps_p50": tps_p50,
            "perf_applicable": (
                perf.get("applicable") if isinstance(perf, dict) else None
            ),
            "engine": eng.get("engine"),
            "engine_image": eng.get("engine_image"),
            "params_b": curated.get("param_count") or meta.get("param_count"),
            "license": meta.get("license") or curated.get("license"),
            "modality": modality_str,
            "publisher": publisher_name,
        })
    return summaries


def results_leaderboard() -> dict:
    """Aggregate view of all runs as a flat result table — only the runs that
    actually produced capability/showcase data (i.e. not failed mid-DEPLOY)."""
    rows = []
    for r in list_runs():
        if r["status"] == "no-state":
            continue
        rows.append({
            "run_id": r["run_id"],
            "hf_id": r["hf_id"],
            "status": r["status"],
            "publisher": r.get("publisher"),
            "modality": r.get("modality"),
            "params": r.get("params_b"),
            "license": r.get("license"),
            "engine": r.get("engine"),
            "capability": r.get("capability_score"),
            "pass_rate": r.get("capability_pass_rate"),
            "categories": r.get("capability_categories") or [],
            "ttft_ms_p50": r.get("ttft_ms_p50"),
            "tps_p50": r.get("tps_p50"),
            "perf_applicable": r.get("perf_applicable"),
            "showcase_items": r.get("showcase_items") or 0,
            "first_impression": r.get("first_impression"),
            "summary": r.get("summary_preview"),
            "duration_s": r.get("duration_s"),
            "ended_at": r.get("ended_at"),
            "failure_reason": r.get("failure_reason"),
        })
    completed = sum(1 for r in rows if r["status"] == "ok")
    failed = sum(1 for r in rows if r["status"] == "failed")
    in_progress = sum(1 for r in rows if r["status"] == "in_progress")
    avg_pass = None
    pass_rates = [r["pass_rate"] for r in rows if isinstance(r.get("pass_rate"), (int, float))]
    if pass_rates:
        avg_pass = round(sum(pass_rates) / len(pass_rates), 3)
    return {
        "total": len(rows),
        "completed_ok": completed,
        "failed": failed,
        "in_progress": in_progress,
        "avg_pass_rate": avg_pass,
        "rows": rows,
    }


def run_detail(run_id: str) -> dict | None:
    run_dir = DATA_ROOT / "runs" / run_id
    if not run_dir.exists():
        return None
    return {
        "run_id": run_id,
        "state": _read_json(run_dir / "state.json"),
        "ready": _read_json(run_dir / "ready.json"),
        "capability": _read_json(run_dir / "capability.json"),
        "perf_bench": _read_json(run_dir / "perf_bench.json"),  # PR#14 / PR#18
        "showcase": _read_json(run_dir / "showcase.json"),
        "curated": _read_json(run_dir / "_meta" / "curated.json"),
        "metadata": _read_json(run_dir / "_meta" / "metadata.json"),
        "engine": _read_json(run_dir / "_meta" / "engine.json"),
    }


def queue_status() -> dict:
    pending = _read_jsonl(DATA_ROOT / "store" / "queue.jsonl")
    return {
        "pending_count": len(pending),
        "pending": pending[:50],
    }


def discover_summary() -> dict:
    path = DATA_ROOT / "discover" / "candidates.jsonl"
    candidates = _read_jsonl(path)
    by_reason: Counter = Counter()
    by_pipeline: Counter = Counter()
    by_org: Counter = Counter()
    for c in candidates:
        by_reason[c.get("reason", "?")] += 1
        by_pipeline[c.get("pipeline_tag") or "(none)"] += 1
        org = (c.get("hf_id") or "").split("/", 1)[0]
        if org:
            by_org[org] += 1
    top = sorted(
        candidates,
        key=lambda c: (c.get("downloads") or 0, c.get("likes") or 0),
        reverse=True,
    )[:20]
    return {
        "total": len(candidates),
        "by_reason": dict(by_reason),
        "by_pipeline": dict(by_pipeline.most_common(15)),
        "by_org_top10": dict(by_org.most_common(10)),
        "top20_by_downloads": [
            {
                "hf_id": c.get("hf_id"),
                "reason": c.get("reason"),
                "pipeline_tag": c.get("pipeline_tag"),
                "downloads": c.get("downloads"),
                "likes": c.get("likes"),
                "last_modified": c.get("last_modified"),
            }
            for c in top
        ],
    }


def outbox_recent(limit: int = 20) -> list[dict]:
    return _read_jsonl(DATA_ROOT / "store" / "notify_outbox.jsonl", limit=limit)


def backup_status() -> dict:
    """Read-only view of the v10 data-protection layer (PR#6, INV-6 / INV-9).

    Delegates to ``backup.read_backup_state`` so panel and the systemd
    backup unit always agree on the same source of truth. Returns a plain
    dict so the JSON serializer doesn't need to know about dataclasses.
    """
    from backup import read_backup_state  # lazy: panel still works if backup/ is being refactored
    state = read_backup_state(BACKUPS_ROOT)
    return {
        "last_backup_ts": state.last_backup_ts,
        "last_backup_age_s": state.last_backup_age_s,
        "snapshot_count": state.snapshot_count,
        "total_size_bytes": state.total_size_bytes,
        "health": state.health,
        "backups_root": state.backups_root,
        "snapshots_tail": state.snapshots[-10:],  # 10 newest is enough for the panel
    }


def engine_probe() -> dict:
    """heyi_engine /v1/models probe — much cheaper than v9's chat-completion
    canary and gives us back the auto-discovered model name as a free side
    effect."""
    from heyi_engine.client import HeyiEngineClient
    client = HeyiEngineClient(base_url=ENGINE_URL, api_key=ENGINE_API_KEY,
                              timeout_s=8.0)
    h = client.health()
    return {
        "ok": h.ok,
        "http": h.http_code,
        "elapsed_s": round(h.elapsed_s, 2),
        "model": h.model_id,
        "detail": (h.detail or "")[:120],
    }


def gpu_status() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=4,
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                idx, used, total, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                rows.append({"index": idx, "mem_used_mb": used, "mem_total_mb": total, "util_pct": util})
            except ValueError:
                continue
    return rows


def docker_status(names: list[str]) -> list[dict]:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}|{{.RunningFor}}"],
            timeout=4,
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    found: dict[str, dict] = {}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            n, s, age = parts[0], parts[1], parts[2]
            found[n] = {"name": n, "status": s, "age": age}
    return [found.get(n, {"name": n, "status": "(absent)", "age": ""}) for n in names]


def health_summary() -> dict:
    outbox = outbox_recent(limit=50)
    last_hb = None
    last_incident = None
    for ev in reversed(outbox):
        et = ev.get("event_type")
        if et == "heartbeat" and last_hb is None:
            last_hb = ev
        elif et == "incident" and last_incident is None:
            last_incident = ev
        if last_hb and last_incident:
            break
    return {
        "engine": engine_probe(),
        "gpu": gpu_status(),
        # v10 has no orchestrator-side containers to watch by default —
        # the heyi_engine prod containers live in user-managed compose
        # and are not the panel's responsibility.
        "containers": docker_status([]),
        "last_heartbeat": last_hb,
        "last_incident": last_incident,
        "now": datetime.now(UTC).isoformat(),
    }


# ---------- HTML rendering (single inline template) ----------

INDEX_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>heyi-eval-v9 panel</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         background: #0c0c10; color: #e7e7ea; margin: 0; padding: 0; }
  header { padding: 16px 24px; background: #14141a; border-bottom: 1px solid #26262e; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .sub { color: #8a8a96; font-size: 12px; margin-top: 4px; }
  main { padding: 18px 24px 80px; max-width: 1400px; margin: 0 auto; }
  section { background: #14141a; border: 1px solid #26262e; border-radius: 8px;
            padding: 14px 18px; margin-bottom: 16px; }
  section h2 { font-size: 14px; margin: 0 0 12px; color: #a9a9b6; font-weight: 600;
               text-transform: uppercase; letter-spacing: 0.04em; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #20202a; }
  th { color: #8a8a96; font-weight: 500; font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.05em; }
  tr:hover td { background: #1a1a22; }
  .ok    { color: #5ad48d; }
  .warn  { color: #e9b870; }
  .err   { color: #ef5f64; }
  .muted { color: #6a6a76; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px;
          background: #20202a; color: #b8b8c4; }
  .pill.ok  { background: #1a3a26; color: #5ad48d; }
  .pill.err { background: #3a1a1f; color: #ef5f64; }
  .pill.run { background: #1a2a3a; color: #6ec0ff; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .stat { padding: 10px 14px; background: #1a1a22; border-radius: 6px; }
  .stat .label { color: #8a8a96; font-size: 11px;
                 text-transform: uppercase; letter-spacing: 0.05em; }
  .stat .value { font-size: 22px; margin-top: 4px; font-weight: 500; }
  a { color: #6ec0ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .stage-row td { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  font-size: 12px; }
  details { margin-top: 8px; }
  summary { cursor: pointer; color: #8a8a96; font-size: 12px; }
  pre { background: #08080c; border: 1px solid #20202a; border-radius: 4px;
        padding: 10px; font-size: 11px; overflow-x: auto; max-height: 360px; }
</style>
</head>
<body>
<header>
  <h1>heyi-eval-v9 · 管理面板</h1>
  <div class="sub" id="updated">loading…</div>
</header>
<main>

  <section>
    <h2>系统健康</h2>
    <div class="grid" id="health-grid"></div>
    <details><summary>详情 JSON</summary><pre id="health-json"></pre></details>
  </section>

  <section>
    <h2>数据备份 (30min rsync · 7d 保留)</h2>
    <div class="grid" id="backup-grid"></div>
    <details><summary>最近 snapshots</summary><pre id="backup-snapshots"></pre></details>
  </section>

  <section>
    <h2>队列（待评测）</h2>
    <div id="queue-summary" class="muted">loading…</div>
    <table id="queue-table"><thead><tr>
      <th>run_id</th><th>hf_id</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section>
    <h2>评测结果总览 <a href="/results" style="font-size:11px;font-weight:normal;margin-left:8px">查看完整结果表 →</a></h2>
    <div class="grid" id="results-grid"></div>
    <table id="results-table"><thead><tr>
      <th>hf_id</th><th>状态</th><th>modality</th><th>engine</th>
      <th>能力得分</th><th>pass rate</th>
      <th>TTFT</th><th>TPS</th>
      <th>首印象</th><th>评价摘要</th><th>耗时</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section>
    <h2>9 阶段状态明细</h2>
    <table id="runs-table"><thead><tr>
      <th>run_id</th><th>hf_id</th><th>状态</th>
      <th>DISCOVER</th><th>CURATE</th><th>METADATA</th><th>ENGINE_SELECT</th>
      <th>STAGE_MODEL</th>
      <th>DEPLOY</th><th>READY_WAIT</th><th>CAPABILITY</th><th>SHOWCASE</th><th>CLEANUP</th>
      <th>总时长</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section>
    <h2>discover 候选概览</h2>
    <div class="grid" id="discover-grid"></div>
    <details><summary>Top 20 by downloads</summary><pre id="discover-top"></pre></details>
  </section>

  <section>
    <h2>近期通知（outbox）</h2>
    <table id="outbox-table"><thead><tr>
      <th>时间</th><th>类型</th><th>级别</th><th>标题</th>
    </tr></thead><tbody></tbody></table>
  </section>

</main>
<script>
async function getJSON(url) { const r = await fetch(url); return await r.json(); }

function pillStatus(s) {
  if (s === 'ok') return '<span class="pill ok">ok</span>';
  if (s === 'failed' || s === 'aborted') return '<span class="pill err">' + s + '</span>';
  if (s === 'running') return '<span class="pill run">running</span>';
  return '<span class="pill muted">' + (s || '-') + '</span>';
}

function formatDuration(s) {
  if (s == null) return '-';
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(1) + 'm';
  return (s/3600).toFixed(1) + 'h';
}

function formatTs(ts) {
  if (!ts) return '-';
  if (typeof ts === 'number') return new Date(ts*1000).toLocaleString('zh-CN');
  return new Date(ts).toLocaleString('zh-CN');
}

const STAGES = ['DISCOVER','CURATE','METADATA','ENGINE_SELECT','STAGE_MODEL','DEPLOY','READY_WAIT','CAPABILITY','SHOWCASE','CLEANUP'];

async function refresh() {
  document.getElementById('updated').textContent = '更新中… ' + new Date().toLocaleString('zh-CN');

  // health
  const h = await getJSON('/api/health');
  const eng = h.engine || {};
  const gpu = h.gpu || [];
  const total_used = gpu.reduce((a,g)=>a+g.mem_used_mb, 0);
  const total_mem  = gpu.reduce((a,g)=>a+g.mem_total_mb, 0);
  const engCls = eng.ok ? 'ok' : 'err';
  const engLabel = eng.ok ? 'engine ok' : 'engine down';
  document.getElementById('health-grid').innerHTML = `
    <div class="stat"><div class="label">${engLabel}</div>
      <div class="value ${engCls}">${eng.elapsed_s != null ? eng.elapsed_s + 's' : '?'}</div>
      <div class="muted" style="font-size:11px">model: ${eng.model || '?'}</div></div>
    <div class="stat"><div class="label">GPU 使用</div>
      <div class="value">${Math.round(total_used/1024)}/${Math.round(total_mem/1024)}G</div>
      <div class="muted" style="font-size:11px">${gpu.length} 块卡</div></div>
    <div class="stat"><div class="label">最近 heartbeat</div>
      <div class="value" style="font-size:14px">${h.last_heartbeat ? formatTs(h.last_heartbeat.ts) : '<span class="muted">无</span>'}</div></div>
    <div class="stat"><div class="label">最近 incident</div>
      <div class="value ${h.last_incident ? 'warn' : 'ok'}" style="font-size:14px">${h.last_incident ? formatTs(h.last_incident.ts) : '<span class="muted">无</span>'}</div></div>
  `;
  document.getElementById('health-json').textContent = JSON.stringify(h, null, 2);

  // backup
  const bk = await getJSON('/api/backup');
  const bkCls = bk.health === 'ok' ? 'ok' : (bk.health === 'warn' ? 'warn' : 'err');
  function fmtAge(s) {
    if (s == null) return '<span class="muted">无记录</span>';
    if (s < 60) return s + 's 前';
    if (s < 3600) return Math.floor(s/60) + 'm 前';
    if (s < 86400) return (s/3600).toFixed(1) + 'h 前';
    return (s/86400).toFixed(1) + 'd 前';
  }
  function fmtGB(b) { return b ? (b / (1024**3)).toFixed(2) + ' GB' : '<span class="muted">0</span>'; }
  document.getElementById('backup-grid').innerHTML = `
    <div class="stat"><div class="label">备份状态</div>
      <div class="value ${bkCls}">${bk.health.toUpperCase()}</div>
      <div class="muted" style="font-size:11px">${bk.backups_root}</div></div>
    <div class="stat"><div class="label">最近成功</div>
      <div class="value" style="font-size:14px">${fmtAge(bk.last_backup_age_s)}</div>
      <div class="muted" style="font-size:11px">${bk.last_backup_ts || ''}</div></div>
    <div class="stat"><div class="label">snapshot 数量</div>
      <div class="value">${bk.snapshot_count}</div>
      <div class="muted" style="font-size:11px">7d 保留窗口</div></div>
    <div class="stat"><div class="label">总占用</div>
      <div class="value">${fmtGB(bk.total_size_bytes)}</div>
      <div class="muted" style="font-size:11px">hard-link 共享</div></div>
  `;
  document.getElementById('backup-snapshots').textContent = JSON.stringify(bk.snapshots_tail, null, 2);

  // queue
  const q = await getJSON('/api/queue');
  document.getElementById('queue-summary').innerHTML = `pending: <strong>${q.pending_count}</strong>`;
  const qbody = document.querySelector('#queue-table tbody');
  qbody.innerHTML = q.pending.map(p => `<tr><td><code>${p.run_id||'-'}</code></td><td>${p.hf_id||'-'}</td></tr>`).join('') ||
    '<tr><td colspan="2" class="muted">空</td></tr>';

  // results summary (real evaluation data)
  const res = await getJSON('/api/results');
  document.getElementById('results-grid').innerHTML = `
    <div class="stat"><div class="label">总评测 runs</div><div class="value">${res.total}</div></div>
    <div class="stat"><div class="label">完成</div><div class="value ok">${res.completed_ok}</div></div>
    <div class="stat"><div class="label">失败 / 进行中</div><div class="value warn">${res.failed} / ${res.in_progress}</div></div>
    <div class="stat"><div class="label">平均 pass rate</div><div class="value">${res.avg_pass_rate != null ? (res.avg_pass_rate*100).toFixed(0)+'%' : '<span class="muted">N/A</span>'}</div></div>
  `;
  const resBody = document.querySelector('#results-table tbody');
  const visibleRows = res.rows.filter(r => r.hf_id !== '?').slice(0, 30);
  resBody.innerHTML = visibleRows.map(r => {
    const pr = (typeof r.pass_rate === 'number') ? `<strong class="${r.pass_rate>=0.8?'ok':r.pass_rate>=0.5?'warn':'err'}">${(r.pass_rate*100).toFixed(0)}%</strong>` : '<span class="muted">-</span>';
    const cap = r.capability ? `<span class="pill ok">${r.capability}</span>` : '<span class="muted">-</span>';
    const fi  = r.first_impression ? `<span class="pill">${r.first_impression}</span>` : '<span class="muted">-</span>';
    const sm  = r.summary ? `<span class="muted" style="font-size:11px">${r.summary}…</span>` : (r.failure_reason ? `<span class="err" style="font-size:11px">✗ ${r.failure_reason}</span>` : '<span class="muted">-</span>');
    const ttft = (typeof r.ttft_ms_p50 === 'number') ? `${r.ttft_ms_p50.toFixed(0)}ms` : (r.perf_applicable === false ? '<span class="muted">N/A</span>' : '<span class="muted">-</span>');
    const tps  = (typeof r.tps_p50 === 'number') ? `${r.tps_p50.toFixed(1)} tok/s` : (r.perf_applicable === false ? '<span class="muted">N/A</span>' : '<span class="muted">-</span>');
    return `<tr>
      <td><a href="/run/${encodeURIComponent(r.run_id)}"><strong>${r.hf_id||'-'}</strong></a>
        <div class="muted" style="font-size:11px">${r.publisher||''}</div></td>
      <td>${pillStatus(r.status)}</td>
      <td>${r.modality||'<span class="muted">-</span>'}</td>
      <td>${r.engine||'<span class="muted">-</span>'}</td>
      <td>${cap}</td>
      <td>${pr}</td>
      <td>${ttft}</td>
      <td>${tps}</td>
      <td>${fi}</td>
      <td style="max-width:380px">${sm}</td>
      <td>${formatDuration(r.duration_s)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="11" class="muted">无评测结果</td></tr>';

  // detailed stage table
  const runs = await getJSON('/api/runs');
  const rbody = document.querySelector('#runs-table tbody');
  rbody.innerHTML = runs.slice(0, 30).map(r => {
    const stages = STAGES.map(s => pillStatus((r.stages||{})[s])).join('</td><td>');
    return `<tr class="stage-row">
      <td><a href="/run/${encodeURIComponent(r.run_id)}"><code>${r.run_id.substring(0,28)}…</code></a></td>
      <td>${r.hf_id || '-'}</td>
      <td>${pillStatus(r.status)}</td>
      <td>${stages}</td>
      <td>${formatDuration(r.duration_s)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="14" class="muted">无 runs</td></tr>';

  // discover
  const d = await getJSON('/api/discover');
  document.getElementById('discover-grid').innerHTML = `
    <div class="stat"><div class="label">候选总数</div><div class="value">${d.total}</div></div>
    <div class="stat"><div class="label">whitelist 命中</div><div class="value ok">${(d.by_reason||{}).whitelist || 0}</div></div>
    <div class="stat"><div class="label">trending 命中</div><div class="value">${(d.by_reason||{}).trending || 0}</div></div>
    <div class="stat"><div class="label">Top org</div><div class="value" style="font-size:14px">${Object.entries(d.by_org_top10||{})[0]?.[0] || '-'}</div></div>
  `;
  document.getElementById('discover-top').textContent = JSON.stringify(d.top20_by_downloads, null, 2);

  // outbox
  const o = await getJSON('/api/outbox');
  const obody = document.querySelector('#outbox-table tbody');
  obody.innerHTML = o.slice().reverse().map(e => {
    const lvl = e.level || 'info';
    const cls = lvl === 'error' ? 'err' : (lvl === 'warn' ? 'warn' : 'muted');
    return `<tr><td>${formatTs(e.ts)}</td><td>${e.event_type||'-'}</td><td class="${cls}">${lvl}</td><td>${e.title||'-'}</td></tr>`;
  }).join('') || '<tr><td colspan="4" class="muted">无</td></tr>';

  document.getElementById('updated').textContent = '已更新 ' + new Date().toLocaleString('zh-CN') + ' · 每 30s 自动刷新';
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


def render_results_page() -> str:
    """Standalone results leaderboard — all-test-results-at-a-glance view."""
    data = results_leaderboard()
    rows_html = []
    for r in data["rows"]:
        if r.get("hf_id") in (None, "?"):
            continue
        hf = html.escape(r.get("hf_id") or "-")
        pub = html.escape(r.get("publisher") or "")
        modality = html.escape(r.get("modality") or "")
        engine = html.escape(r.get("engine") or "")
        license_ = html.escape(r.get("license") or "")
        params = html.escape(r.get("params") or "")
        status = r.get("status") or "?"
        status_cls = {"ok": "ok", "failed": "err", "in_progress": "run"}.get(status, "muted")
        cap = html.escape(r.get("capability") or "-")
        pr = r.get("pass_rate")
        if isinstance(pr, (int, float)):
            pr_pct = f"{pr * 100:.0f}%"
            pr_cls = "ok" if pr >= 0.8 else "warn" if pr >= 0.5 else "err"
        else:
            pr_pct, pr_cls = "-", "muted"
        fi = html.escape(r.get("first_impression") or "")
        summary = html.escape(r.get("summary") or "")
        dur = r.get("duration_s")
        if dur is not None:
            dur_s = f"{dur / 60:.1f}m" if dur >= 60 else f"{dur:.0f}s"
        else:
            dur_s = "-"
        fail = html.escape(r.get("failure_reason") or "")
        showcase_n = r.get("showcase_items") or 0

        # PR#18 perf columns
        ttft_p50 = r.get("ttft_ms_p50")
        tps_p50 = r.get("tps_p50")
        if isinstance(ttft_p50, (int, float)):
            ttft_cell = f"{ttft_p50:.0f}ms"
        elif r.get("perf_applicable") is False:
            ttft_cell = "<span class='muted'>N/A</span>"
        else:
            ttft_cell = "<span class='muted'>-</span>"
        if isinstance(tps_p50, (int, float)):
            tps_cell = f"{tps_p50:.1f} tok/s"
        elif r.get("perf_applicable") is False:
            tps_cell = "<span class='muted'>N/A</span>"
        else:
            tps_cell = "<span class='muted'>-</span>"

        # PR#18 categories pill row (compact)
        cats = r.get("categories") or []
        cats_html = (
            " ".join(
                f"<span class='pill' style='font-size:10px'>{html.escape(c)}</span>"
                for c in cats[:8]
            ) if cats else "<span class='muted'>-</span>"
        )

        rows_html.append(
            f"<tr><td><a href='/run/{html.escape(r.get('run_id', ''))}'><strong>{hf}</strong></a>"
            f"<div class='muted' style='font-size:11px'>{pub}</div></td>"
            f"<td><span class='pill {status_cls}'>{html.escape(status)}</span></td>"
            f"<td>{modality}</td><td>{params}</td><td>{license_}</td>"
            f"<td>{engine}</td>"
            f"<td><span class='pill'>{cap}</span><div style='margin-top:3px'>{cats_html}</div></td>"
            f"<td class='{pr_cls}'><strong>{pr_pct}</strong></td>"
            f"<td>{ttft_cell}</td><td>{tps_cell}</td>"
            f"<td>{showcase_n} 条</td>"
            f"<td><span class='pill'>{fi}</span></td>"
            f"<td style='max-width:380px;font-size:12px;color:#b8b8c4'>{summary}"
            f"{f'<div class=err style=font-size:11px>✗ {fail}</div>' if fail else ''}</td>"
            f"<td>{dur_s}</td></tr>"
        )

    table_body = "".join(rows_html) or '<tr><td colspan="14" class="muted">无评测结果</td></tr>'

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>heyi-eval-v9 · 评测结果</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;background:#0c0c10;color:#e7e7ea;margin:0;padding:0}}
header{{padding:16px 24px;background:#14141a;border-bottom:1px solid #26262e}}
header h1{{margin:0;font-size:18px}} header .sub{{color:#8a8a96;font-size:12px;margin-top:4px}}
main{{padding:18px 24px 60px;max-width:1700px;margin:0 auto}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
.stat{{padding:12px 16px;background:#14141a;border:1px solid #26262e;border-radius:6px}}
.stat .label{{color:#8a8a96;font-size:11px;text-transform:uppercase;letter-spacing:0.05em}}
.stat .value{{font-size:26px;margin-top:6px;font-weight:500}}
section{{background:#14141a;border:1px solid #26262e;border-radius:8px;padding:14px 18px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #20202a;vertical-align:top}}
th{{color:#8a8a96;font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:0.05em}}
tr:hover td{{background:#1a1a22}}
.ok{{color:#5ad48d}} .err{{color:#ef5f64}} .warn{{color:#e9b870}} .muted{{color:#6a6a76}}
.pill{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;background:#20202a;color:#b8b8c4;white-space:nowrap}}
.pill.ok{{background:#1a3a26;color:#5ad48d}}
.pill.err{{background:#3a1a1f;color:#ef5f64}}
.pill.run{{background:#1a2a3a;color:#6ec0ff}}
a{{color:#6ec0ff;text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body>
<header><a href="/">← 返回主面板</a> ·
<h1 style="display:inline">📊 评测结果总览</h1>
<div class="sub">所有跑过的 runs（含失败 / 进行中）；点 hf_id 看完整 capability + showcase 详情</div></header>
<main>
<div class="grid">
  <div class="stat"><div class="label">总评测 runs</div><div class="value">{data['total']}</div></div>
  <div class="stat"><div class="label">完成 OK</div><div class="value ok">{data['completed_ok']}</div></div>
  <div class="stat"><div class="label">失败</div><div class="value err">{data['failed']}</div></div>
  <div class="stat"><div class="label">进行中</div><div class="value warn">{data['in_progress']}</div></div>
</div>
<section><table>
<thead><tr>
  <th>hf_id / publisher</th><th>状态</th><th>modality</th><th>params</th><th>license</th>
  <th>engine</th><th>能力得分 / categories</th><th>pass rate</th>
  <th>TTFT (p50)</th><th>TPS (p50)</th><th>showcase</th>
  <th>首印象</th><th>评价摘要 / 失败原因</th><th>耗时</th>
</tr></thead><tbody>{table_body}</tbody>
</table></section>
</main></body></html>"""


# ── PR#18: CAPABILITY (multimodal) + PERF_BENCH renderers ─────────────────


def _render_item_row(r: dict) -> str:
    """Render a single capability item row. Used by both the per-category
    table and the legacy flat-results fallback."""
    cls = "ok" if r.get("pass") else "err"
    return (
        f"<tr><td>{html.escape(r.get('id',''))}</td>"
        f"<td><code>{html.escape((r.get('prompt') or '')[:80])}</code></td>"
        f"<td class='{cls}'>{'PASS' if r.get('pass') else 'FAIL'}</td>"
        f"<td>{r.get('latency_ms')}ms</td>"
        f"<td><code>{html.escape((r.get('actual') or '')[:80])}</code></td></tr>"
    )


def _render_capability_html(cap: dict) -> str:
    """Render the CAPABILITY section.

    PR#15+: ``cap.categories`` is a dict
        { category_name: {applicable, scorer, score, pass_rate, items, reason?} }
    Each category renders as a collapsible <details> block with its own
    pass-rate banner + items table.

    Legacy artifacts (pre-PR#15) only have ``cap.results``: rendered as a
    single flat table for back-compat.
    """
    if not cap:
        return ""

    overall = (
        f"<p>score: <strong>{html.escape(cap.get('score') or '?')}</strong> · "
        f"pass_rate: {cap.get('pass_rate')}</p>"
    )

    categories = cap.get("categories") or {}
    if categories:
        blocks: list[str] = []
        for cat_name in sorted(categories):
            info = categories[cat_name] or {}
            applicable = bool(info.get("applicable"))
            score = info.get("score") or "0/0"
            pass_rate = info.get("pass_rate")
            scorer = info.get("scorer") or ""
            reason = info.get("reason") or ""

            if not applicable:
                blocks.append(
                    f"<details><summary class='muted'>"
                    f"{html.escape(cat_name)} — "
                    f"<span class='pill'>N/A</span> "
                    f"{html.escape(reason)}</summary></details>"
                )
                continue

            items = info.get("items") or []
            pass_pct = (
                f"{pass_rate*100:.0f}%"
                if isinstance(pass_rate, (int, float)) else "?"
            )
            pass_cls = (
                "ok" if isinstance(pass_rate, (int, float)) and pass_rate >= 0.8
                else "warn" if isinstance(pass_rate, (int, float)) and pass_rate >= 0.5
                else "err"
            )
            rows = "".join(_render_item_row(r) for r in items)
            blocks.append(
                f"<details {'open' if pass_rate not in (1, 1.0) and items else ''}>"
                f"<summary><strong>{html.escape(cat_name)}</strong> "
                f"<span class='pill {pass_cls}'>{score}</span> "
                f"<span class='muted'>({pass_pct} pass · scorer={html.escape(scorer)})</span>"
                f"</summary>"
                "<table><thead><tr><th>id</th><th>prompt</th><th>result</th>"
                "<th>latency</th><th>actual</th></tr></thead>"
                f"<tbody>{rows}</tbody></table></details>"
            )
        return overall + "".join(blocks)

    if cap.get("results"):
        rows = "".join(_render_item_row(r) for r in cap["results"])
        return (
            overall
            + "<table><thead><tr><th>id</th><th>prompt</th><th>result</th>"
            "<th>latency</th><th>actual</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    return ""


def _fmt(v, *, suffix: str = "", places: int = 1) -> str:
    if v is None:
        return "<span class='muted'>—</span>"
    if isinstance(v, (int, float)):
        return f"{v:.{places}f}{suffix}"
    return html.escape(str(v))


def _render_perf_bench_html(perf: dict) -> str:
    """Render the PERF_BENCH section. PR#14 artifact shape (perf_bench.json)."""
    if not perf:
        return ""
    if not perf.get("applicable"):
        reason = perf.get("reason") or ""
        return (
            f"<p><span class='pill'>N/A</span> "
            f"<span class='muted'>{html.escape(reason)}</span></p>"
        )

    ttft = perf.get("ttft_ms") or {}
    tps = perf.get("tps_single") or {}
    conc = perf.get("concurrent") or {}
    vram = perf.get("vram_mib")
    warnings_list = perf.get("warnings") or []

    cards = (
        "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px'>"
        "<div class='stat'><div class='label'>TTFT (p50)</div>"
        f"<div class='value'>{_fmt(ttft.get('p50'), suffix=' ms')}</div></div>"
        "<div class='stat'><div class='label'>TTFT (p95)</div>"
        f"<div class='value'>{_fmt(ttft.get('p95'), suffix=' ms')}</div></div>"
        "<div class='stat'><div class='label'>TPS single (p50)</div>"
        f"<div class='value'>{_fmt(tps.get('p50'), suffix=' tok/s')}</div></div>"
        f"<div class='stat'><div class='label'>TPS concurrent (×{conc.get('n','?')})</div>"
        f"<div class='value'>{_fmt(conc.get('aggregate_tps'), suffix=' tok/s')}</div></div>"
        "<div class='stat'><div class='label'>VRAM total</div>"
        f"<div class='value'>{_fmt(vram.get('total') if isinstance(vram, dict) else None, suffix=' MiB', places=0)}</div></div>"
        "</div>"
    )

    warn_html = ""
    if warnings_list:
        items = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings_list)
        warn_html = f"<div class='muted' style='margin-top:8px;font-size:12px'><strong>warnings</strong><ul>{items}</ul></div>"

    return cards + warn_html


def render_run_detail(run_id: str) -> str:
    detail = run_detail(run_id)
    if detail is None:
        return "<h1>404 run not found</h1>"
    state = detail.get("state") or {}
    cap = detail.get("capability") or {}
    show = detail.get("showcase") or {}
    cur = detail.get("curated") or {}
    meta = detail.get("metadata") or {}
    eng = detail.get("engine") or {}
    ready = detail.get("ready") or {}

    def esc(s):
        return html.escape(json.dumps(s, ensure_ascii=False, indent=2)) if s else ""

    cap_html = _render_capability_html(cap)
    perf_html = _render_perf_bench_html(detail.get("perf_bench") or {})

    show_html = ""
    if show.get("items"):
        cards = []
        for it in show["items"]:
            cards.append(
                "<div style='border:1px solid #26262e;padding:12px;border-radius:6px;margin-bottom:10px'>"
                f"<div class='muted' style='font-size:11px'>{html.escape(it.get('id',''))} · {it.get('tokens_out')}t out · {it.get('latency_ms')}ms</div>"
                f"<div style='margin:4px 0;color:#b8b8c4;font-size:12px'><strong>rationale</strong>: {html.escape(it.get('rationale',''))}</div>"
                f"<details><summary>prompt</summary><pre>{html.escape(it.get('prompt',''))}</pre></details>"
                f"<details open><summary>actual</summary><pre>{html.escape(it.get('actual',''))}</pre></details>"
                f"<div style='margin-top:4px;color:#e9b870;font-size:12px'><strong>comment</strong>: {html.escape(it.get('comment',''))}</div>"
                "</div>"
            )
        first = html.escape(show.get("model_first_impression") or "")
        summary = html.escape(show.get("summary") or "")
        show_html = (
            f"<p><strong>first_impression</strong>: <span class='pill'>{first}</span></p>"
            f"<p style='line-height:1.6'>{summary}</p>"
            + "".join(cards)
        )

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>{html.escape(run_id)}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0c0c10;color:#e7e7ea;margin:0;padding:0}}
header{{padding:16px 24px;background:#14141a;border-bottom:1px solid #26262e}}
main{{padding:18px 24px;max-width:1400px;margin:0 auto}}
section{{background:#14141a;border:1px solid #26262e;border-radius:8px;padding:14px 18px;margin-bottom:16px}}
section h2{{font-size:14px;margin:0 0 12px;color:#a9a9b6;text-transform:uppercase;letter-spacing:0.04em}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #20202a}}
.ok{{color:#5ad48d}} .err{{color:#ef5f64}} .muted{{color:#6a6a76}}
.pill{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;background:#20202a;color:#b8b8c4}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
pre{{background:#08080c;border:1px solid #20202a;border-radius:4px;padding:10px;font-size:11px;overflow-x:auto;max-height:360px}}
a{{color:#6ec0ff;text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body>
<header><a href="/">← 返回</a> · <strong>{html.escape(run_id)}</strong>
<div class='muted' style='font-size:12px'>hf_id: {html.escape(state.get('hf_id') or '?')}</div>
</header><main>
<section><h2>state</h2><pre>{esc(state)}</pre></section>
<section><h2>engine</h2><pre>{esc(eng)}</pre></section>
<section><h2>metadata（HF + curator 合并）</h2><pre>{esc(meta)}</pre></section>
<section><h2>curated（LLM 解读）</h2><pre>{esc(cur)}</pre></section>
<section><h2>ready</h2><pre>{esc(ready)}</pre></section>
<section><h2>capability（多模态分轨）</h2>{cap_html or '<div class="muted">无</div>'}</section>
<section><h2>perf_bench（TTFT / TPS / 并发 / VRAM）</h2>{perf_html or '<div class="muted">无</div>'}</section>
<section><h2>showcase（claude 自主设计的 8 题）</h2>{show_html or '<div class="muted">无</div>'}</section>
</main></body></html>"""


# ---------- HTTP handlers ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html_text, status=200):
        body = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path == "/" or path == "":
                self._html(INDEX_HTML)
            elif path == "/api/health":
                self._json(health_summary())
            elif path == "/api/runs":
                self._json(list_runs())
            elif path == "/api/queue":
                self._json(queue_status())
            elif path == "/api/discover":
                self._json(discover_summary())
            elif path == "/api/outbox":
                self._json(outbox_recent(limit=30))
            elif path == "/api/backup":
                self._json(backup_status())
            elif path == "/api/results":
                self._json(results_leaderboard())
            elif path == "/results":
                self._html(render_results_page())
            elif path.startswith("/api/runs/"):
                run_id = path[len("/api/runs/"):]
                d = run_detail(run_id)
                if d is None:
                    self._json({"error": "not_found"}, status=404)
                else:
                    self._json(d)
            elif path.startswith("/run/"):
                run_id = path[len("/run/"):]
                self._html(render_run_detail(run_id))
            else:
                self._json({"error": "not_found", "path": path}, status=404)
        except Exception as e:
            self._json({"error": str(e), "type": type(e).__name__}, status=500)


def main(argv=None):
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"heyi-eval-v9 panel on http://{LISTEN_HOST}:{LISTEN_PORT}  data={DATA_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
