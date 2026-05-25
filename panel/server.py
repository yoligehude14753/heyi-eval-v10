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
  - POST /api/enqueue  PR#55: manual prioritisation — body {"hf_id": "..."}

The GET surface is read-only. The single write surface (POST /api/enqueue)
only appends an hf_id to the orchestrator queue; everything else
(spawning containers, mutating runs) is exclusively the orchestrator's
responsibility. Validation rejects anything that isn't a strict
``<org>/<name>`` model id so we don't expose a path-injection vector.
"""
from __future__ import annotations

import html
import json
import os
import re
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


# PR#55: strict validator for the POST /api/enqueue surface. HF model
# IDs are ``<org>/<name>`` with a tight charset (alnum + . _ -). Anything
# else is either a bug or a probe and we reject with HTTP 400.
_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9._-]{1,96}$")


def _is_valid_hf_id(s: str) -> bool:
    if not s or len(s) > 200:
        return False
    return bool(_HF_ID_RE.match(s))


# ---------- PR#57: Chinese localization helpers ----------
#
# The orchestrator emits English error strings (e.g. "container exited 1"
# or "disk headroom too low: free=..."). The user explicitly asked for
# the panel to render those reasons in Chinese, so we keep the source
# of truth in English (greppable, machine-friendly, lossless in the
# raw JSON artifacts) and translate at render-time. Patterns are
# matched in order; anything unrecognized falls through to the raw
# English text with a small "原文" prefix so debuggability is preserved.

_STAGE_ZH = {
    "DISCOVER":     "发现",
    "CURATE":       "整理",
    "METADATA":     "元数据",
    "ENGINE_SELECT": "引擎选择",
    "STAGE_MODEL":  "模型暂存",
    "DEPLOY":       "部署",
    "READY_WAIT":   "就绪等待",
    "CAPABILITY":   "能力评测",
    "PERF_BENCH":   "性能基准",
    "SHOWCASE":     "展示评测",
    "CLEANUP":      "清理",
}

_STATUS_ZH = {
    "ok":          "成功",
    "failed":      "失败",
    "aborted":     "已中止",
    "in_progress": "进行中",
    "queued":      "排队中",
    "skipped":     "跳过",
    "no-state":    "无状态",
}


def stage_zh(name: str) -> str:
    return _STAGE_ZH.get(name, name)


def status_zh(s: str) -> str:
    return _STATUS_ZH.get(s, s or "-")


# Failure reason patterns: (regex, lambda match -> Chinese rendering).
# Order matters — most-specific first. Lambdas receive the re.Match.
def _fmt_bytes(s: str) -> str:
    try:
        n = int(s)
    except (TypeError, ValueError):
        return s
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f}GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}MB"
    return f"{n}B"


_FAILURE_RULES: list[tuple[re.Pattern[str], "callable"]] = [  # type: ignore[name-defined]
    # "aborted at <STAGE>: <reason>" — the orchestrator wraps every
    # downstream skip/error this way. Recurse into the wrapped reason
    # so each layer prints in Chinese ("DEPLOY 阶段中止：磁盘空间不足…").
    (re.compile(r"^aborted at (\w+):\s*(.+)$", re.S),
     lambda m: f"{_STAGE_ZH.get(m.group(1), m.group(1))} 阶段中止：{failure_zh(m.group(2).strip())}"),
    (re.compile(r"^not_a_model:\s*(.*)$"),
     lambda m: f"不是可评测的语言模型仓库（{m.group(1).strip()[:120]}）"),
    (re.compile(r"^disk headroom too low.*free=(\d+).*need.?≥(\d+).*"),
     lambda m: f"磁盘空间不足：可用 {_fmt_bytes(m.group(1))}，需要至少 {_fmt_bytes(m.group(2))}"),
    (re.compile(r"^container exited (\d+)(?::|$)\s*(.*)", re.I),
     lambda m: f"容器退出码 {m.group(1)}" + (f"：{m.group(2).strip()}" if m.group(2).strip() else "")),
    (re.compile(r"^READY_WAIT timeout after (\d+)s.*", re.I),
     lambda m: f"就绪等待超时（{m.group(1)} 秒内 /v1/models 未通过）"),
    (re.compile(r"^connection refused.*", re.I),
     lambda m: "连接被拒绝（服务未监听或防火墙拦截）"),
    (re.compile(r"^image not found.*", re.I),
     lambda m: "Docker 镜像不存在或拉取失败"),
    (re.compile(r"^OOM|out of memory.*", re.I),
     lambda m: "显存/内存不足（OOM）"),
    (re.compile(r"^download failed.*"),
     lambda m: "权重下载失败（HF 网络/镜像问题）"),
    (re.compile(r".*safetensors.*not.found.*", re.I),
     lambda m: "未找到 safetensors 权重文件"),
    (re.compile(r"^bad_args.*"),
     lambda m: "参数错误（编排器内部）"),
    (re.compile(r"^missing_artifact.*"),
     lambda m: "缺少上游 stage 产物"),
    (re.compile(r"^engine \w+ not supported.*", re.I),
     lambda m: "所选推理引擎不支持该模型架构"),
    (re.compile(r"^staged size .* exceeds.*", re.I),
     lambda m: "模型权重过大，超出本机磁盘配额"),
    (re.compile(r"^license blocked.*", re.I),
     lambda m: "许可证受限，已拒绝评测"),
    (re.compile(r"^HF repo lacks any.*", re.I),
     lambda m: "HF 仓库缺少有效权重文件（既无 safetensors 也无 GGUF）"),
    (re.compile(r"^skipped:\s*(.*)$"),
     lambda m: f"跳过：{failure_zh(m.group(1).strip()) or m.group(1).strip()}"),
    (re.compile(r".*snapshot_download failed.*No space left on device.*", re.I | re.S),
     lambda m: "权重下载失败：磁盘空间不足（OS Errno 28）"),
    (re.compile(r".*snapshot_download failed.*", re.I | re.S),
     lambda m: "权重下载失败（HF snapshot_download 抛错）"),
    (re.compile(r"^Read timed out.*|.*ReadTimeoutError.*", re.I),
     lambda m: "HF 镜像读取超时（网络/限流问题）"),
    (re.compile(r".*connection reset.*", re.I),
     lambda m: "网络连接被重置（中途断流）"),
]


def failure_zh(text: str | None) -> str:
    """Render an orchestrator failure_reason in Chinese.

    Unrecognized strings are passed through with a "原文" prefix so we
    don't silently swallow novel failure modes — they'll show up in the
    UI as e.g. "原文: foo bar baz" until a rule is added above.
    """
    if not text:
        return ""
    s = text.strip()
    if not s:
        return ""
    for rx, fmt in _FAILURE_RULES:
        m = rx.match(s)
        if m:
            return fmt(m)
    return f"原文：{s}"


# ---------- PR#60: clickable hf_id + multimodal preview helpers ----------
#
# The user explicitly asked: "所有页面里出现了模型名称就要能够跳转".
# Every hf_id rendered anywhere on the panel must link to:
#   (a) the local run-detail page if a run exists, OR
#   (b) the HF Hub page so users can read the model card
# The convention is: clicking the hf_id text opens the model on
# HF Hub in a new tab; a small "📄" icon links to /run/<id> for the
# detail view when applicable. This keeps the surface uniform across
# results / candidates / queue / index / run-detail / showcase cards.

_FIXTURES_ROOT = Path(
    os.environ.get(
        "HEYI_EVAL_FIXTURES_ROOT",
        "/home/ai/heyi-eval-v10/orchestrator/capability_data/fixtures",
    ),
)
_FIXTURE_REL_RE = re.compile(r"^[A-Za-z0-9_/.\-]{1,200}$")
_FIXTURE_BAD = re.compile(r"\.\.|/{2,}|^/")


def _is_safe_fixture_path(rel: str) -> bool:
    """Validate a fixture relative path. Rejects:
    - empty/oversized
    - '..' traversal, leading '/', double slashes
    - characters outside the safe whitelist
    """
    if not rel or len(rel) > 200:
        return False
    if _FIXTURE_BAD.search(rel):
        return False
    if not _FIXTURE_REL_RE.match(rel):
        return False
    return True


def hf_hub_url(hf_id: str | None) -> str:
    if not hf_id or "/" not in hf_id:
        return "#"
    return f"https://huggingface.co/{hf_id}"


def hf_link(
    hf_id: str | None, *,
    run_id: str | None = None,
    show_run_icon: bool = True,
    css_class: str = "",
) -> str:
    """Render an hf_id as: <a target=_blank href=HF>org/name</a> [📄 → /run/<id>].

    - hf_id text → HF Hub (always, opens new tab)
    - 📄 icon → local /run/<id> page (only if run_id given)
    The user can both inspect the model card on HF and dive into our
    evaluation in one glance. ``css_class`` lets callers theme the
    text (e.g. larger header link vs inline table link).
    """
    if not hf_id:
        return "<span class='muted'>-</span>"
    safe = html.escape(hf_id)
    hub = html.escape(hf_hub_url(hf_id), quote=True)
    cls = f" class='{html.escape(css_class, quote=True)}'" if css_class else ""
    link = (
        f"<a{cls} href='{hub}' target='_blank' rel='noopener noreferrer' "
        f"title='在 HuggingFace Hub 查看模型卡'>{safe}</a>"
    )
    if show_run_icon and run_id:
        run_safe = html.escape(run_id, quote=True)
        link += (
            f" <a class='run-icon' href='/run/{run_safe}' "
            f"title='查看本地评测详情'>📄</a>"
        )
    return link


def hf_publisher_link(name: str | None) -> str:
    """Publisher name → https://huggingface.co/<name> (no slash)."""
    if not name:
        return "<span class='muted'>-</span>"
    n = html.escape(name)
    n_attr = html.escape(name, quote=True)
    return (
        f"<a href='https://huggingface.co/{n_attr}' target='_blank' "
        f"rel='noopener noreferrer' title='在 HuggingFace 查看该厂商'>{n}</a>"
    )


# PR#61: shared stylesheet — modern dark theme, card-based, lots of
# breathing room. Used by /run/<id>, /candidates, /results and the
# main /. Kept as a single constant so updates are one-file.
_PANEL_STYLES = """<style>
:root{
  --bg:#0b0b10; --bg-card:#15151c; --bg-card-2:#1a1a22;
  --border:#262630; --border-2:#2f2f3a;
  --text:#e7e7ea; --text-2:#b3b3bf; --text-3:#7a7a86;
  --accent:#7cb7ff; --accent-2:#5ad48d; --warn:#e9b870; --err:#ef5f64;
  --info:#a8aaf7;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;
}
*,*::before,*::after{box-sizing:border-box}
html,body{background:var(--bg);color:var(--text);margin:0;padding:0;font-family:var(--sans);
  font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
code{font-family:var(--mono);font-size:0.9em;background:#0a0a10;padding:1px 5px;border-radius:3px;
  border:1px solid var(--border)}
pre{font-family:var(--mono);font-size:12px;line-height:1.55;background:#08080c;
  border:1px solid var(--border);border-radius:6px;padding:12px 14px;overflow-x:auto;
  max-height:480px;white-space:pre-wrap;word-break:break-word}
.muted{color:var(--text-3)}
.empty-note{padding:24px;text-align:center;font-style:italic}

/* Page chrome */
.page-header{padding:18px 28px;background:var(--bg-card);border-bottom:1px solid var(--border);
  display:flex;flex-direction:column;gap:6px}
.page-header .header-left{font-size:13px}
.page-header .breadcrumb-sep{color:var(--text-3);margin:0 6px}
.page-header .page-title{margin:4px 0 0;font-size:22px;font-weight:600;letter-spacing:-0.01em}
.page-header .header-sub{font-size:12px}
.back-link{font-weight:500}
.hf-header-link{color:#fff !important;font-weight:600}
.hf-header-link:hover{color:var(--accent) !important;text-decoration:underline}
.run-icon{font-size:0.85em;opacity:0.7;margin-left:3px;text-decoration:none !important}
.run-icon:hover{opacity:1}

main{padding:20px 28px 80px;max-width:1500px;margin:0 auto}
.card-section{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;
  padding:18px 22px;margin-bottom:18px}
.card-section h2{margin:0 0 14px;font-size:13px;color:var(--text-2);
  text-transform:uppercase;letter-spacing:0.08em;font-weight:600}
.card-section h3{margin:18px 0 8px;font-size:12px;color:var(--text-2);
  text-transform:uppercase;letter-spacing:0.05em;font-weight:600}
.raw-section summary{cursor:pointer;color:var(--text-2);font-size:13px}
.raw-section summary:hover{color:var(--text)}

/* Pills (unified) */
.pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;
  background:#22222c;color:var(--text-2);white-space:nowrap;font-weight:500;
  border:1px solid var(--border-2)}
.pill-ok{background:#15331f;color:#5ad48d;border-color:#1f4a2d}
.pill-err{background:#3a1820;color:#ef5f64;border-color:#5a232f}
.pill-warn{background:#3a2a18;color:#e9b870;border-color:#5a4322}
.pill-info{background:#1a2245;color:#a8aaf7;border-color:#2f3a6a}
.pill-muted{background:#1a1a22;color:#6a6a76;border-color:#262630}
.pill-big{font-size:14px;padding:6px 14px;border-radius:14px}

/* Overall banner */
.overall-banner{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:14px}
.overall-banner .fail-text{color:var(--err);font-size:13px}

/* Metadata pills (4-column grid) */
.meta-pills{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:8px}
.meta-pill{background:var(--bg-card-2);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;display:flex;flex-direction:column;gap:2px}
.meta-pill label{font-size:10px;color:var(--text-3);text-transform:uppercase;letter-spacing:0.05em}
.meta-pill .val{font-size:14px;color:var(--text);font-weight:500}

/* Stages table */
.stages-table{width:100%;border-collapse:collapse;font-size:13px}
.stages-table th{text-align:left;padding:8px 12px;color:var(--text-2);font-size:11px;
  text-transform:uppercase;letter-spacing:0.05em;font-weight:600;border-bottom:1px solid var(--border-2)}
.stages-table td{padding:9px 12px;border-bottom:1px solid var(--border)}
.stages-table tr:last-child td{border-bottom:none}
.stage-cell{min-width:140px}
.stage-zh{font-weight:500}
.stage-en{font-size:10px;color:var(--text-3);font-family:var(--mono)}
.dur-cell{color:var(--text-2);font-variant-numeric:tabular-nums}
.err-cell{color:var(--err);font-size:12px;max-width:600px}

/* Capability overall */
.cap-overall{margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid var(--border)}
.cap-overall .big-score{font-size:28px;font-weight:600;color:var(--accent-2);font-variant-numeric:tabular-nums}

/* Category blocks */
.cap-cat{background:var(--bg-card-2);border:1px solid var(--border);
  border-radius:8px;padding:12px 16px;margin-bottom:12px}
.cap-cat[open]>summary{margin-bottom:14px;border-bottom:1px solid var(--border);padding-bottom:12px}
.cap-cat>summary{cursor:pointer;list-style:none}
.cap-cat>summary::-webkit-details-marker{display:none}
.cap-cat.cat-na{padding:14px 16px}
.cat-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.cat-name{font-weight:600;font-size:14px;color:var(--text)}
.cat-na .cat-na-reason{margin-top:6px;font-size:12px;font-style:italic}

/* Item card grid */
.item-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
.item-card{background:var(--bg);border:1px solid var(--border-2);border-radius:8px;
  padding:12px 14px;display:flex;flex-direction:column;gap:8px;
  border-left:3px solid var(--border-2)}
.item-card.item-ok{border-left-color:var(--accent-2)}
.item-card.item-err{border-left-color:var(--err)}
.item-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px}
.item-id{font-family:var(--mono);color:var(--text-2);font-size:11px;background:#0a0a10;
  padding:2px 7px;border-radius:4px;border:1px solid var(--border)}
.item-meta{font-size:10px;color:var(--text-3);margin-left:auto;font-family:var(--mono)}
.item-body{display:flex;flex-direction:column;gap:8px}
.item-prompt label,.item-actual label,.show-rationale label,.show-comment label,
.show-summary label,.first-impression label{font-size:10px;color:var(--text-3);
  text-transform:uppercase;letter-spacing:0.05em;font-weight:600;display:block;margin-bottom:4px}
.prompt-text{color:var(--text-2);font-size:13px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
.actual-text{font-family:var(--mono);font-size:12px;background:#08080c;
  border:1px solid var(--border);border-radius:6px;padding:10px 12px;
  white-space:pre-wrap;word-break:break-word;max-height:360px;overflow-y:auto;margin:6px 0 0}
.item-actual details>summary,.item-prompt details>summary{cursor:pointer;font-size:12px;color:var(--accent);padding:3px 0}

/* Fixture image preview */
.fixture-preview img{display:block}

/* Showcase */
.first-impression{margin-bottom:10px;display:flex;align-items:center;gap:10px}
.show-summary{background:var(--bg-card-2);border:1px solid var(--border);
  border-radius:8px;padding:12px 16px;margin-bottom:14px}
.show-summary p{margin:0;line-height:1.8;font-size:14px;color:var(--text)}
.show-card{background:var(--bg);border:1px solid var(--border-2);border-radius:8px;padding:12px 14px;
  display:flex;flex-direction:column;gap:8px}
.show-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px}
.show-body{display:flex;flex-direction:column;gap:8px}
.show-rationale>div,.show-comment>div{color:var(--text-2);font-size:13px;line-height:1.55}
.show-comment{background:#2a230f;border:1px solid #5a4322;border-radius:6px;padding:8px 12px;color:#e9b870}
.show-comment label{color:#e9b870}
</style>"""


def render_fixture_preview(
    fixture: str | None,
    *,
    max_height_px: int = 180,
) -> str:
    """Render an <img> tag for a fixture path. The panel exposes
    fixtures via GET /fixtures/<rel>, validated by _is_safe_fixture_path.
    Returns empty string if fixture is missing or invalid; never raises.
    """
    if not fixture:
        return ""
    if not _is_safe_fixture_path(fixture):
        return ""
    src = "/fixtures/" + fixture
    safe = html.escape(src, quote=True)
    return (
        f"<div class='fixture-preview' style='margin:6px 0'>"
        f"<img src='{safe}' alt='{html.escape(fixture)}' "
        f"style='max-height:{max_height_px}px;max-width:100%;"
        f"border:1px solid #2a2a32;border-radius:4px;background:#fff'/>"
        f"<div class='muted' style='font-size:10px;margin-top:2px'>"
        f"输入图像：{html.escape(fixture)}</div></div>"
    )


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
        # PR#57: showcase summary often contains the raw <think>...</think>
        # block from MiniMax-M2.7 / Qwen3-thinking reasoning models. Strip
        # it so the preview shows the actual answer instead of CoT.
        try:
            from orchestrator.llm_text_utils import strip_think_blocks
            summary_text = strip_think_blocks(summary_text)
        except Exception:
            pass
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

        raw_fr = state.get("failure_reason")
        summaries.append({
            "run_id": state.get("run_id") or run_dir.name,
            "hf_id": state.get("hf_id"),
            "status": state.get("status", "?"),
            "status_zh": status_zh(state.get("status", "?")),
            "stages": stage_status,
            "stage_durations": stage_dur,
            "failure_reason": raw_fr,
            "failure_reason_zh": failure_zh(raw_fr),
            "created_at": started,
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
            "status_zh": r.get("status_zh") or status_zh(r["status"]),
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
            "created_at": r.get("created_at"),
            "ended_at": r.get("ended_at"),
            "failure_reason": r.get("failure_reason"),
            "failure_reason_zh": r.get("failure_reason_zh") or failure_zh(r.get("failure_reason")),
        })
    # PR#58: time-DESC ordering — newest run on top. The user explicitly
    # asked for chronological ordering with the freshest evaluations
    # surfaced first; previously the leaderboard relied on an implicit
    # status-priority sort that buried fresh results below historical
    # successes. Tiebreak on ended_at for runs that share created_at.
    rows.sort(
        key=lambda r: (
            -(r.get("created_at") or 0),
            -(r.get("ended_at") or 0),
        ),
    )
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
    # PR#56: surface the auto-enqueue cadence so the panel can show
    # "下次入队 in 7m" beside "pending: 0" — the user reported the
    # panel showing 0 and assuming the system had stopped monitoring,
    # when in fact the orchestrator had just drained the batch and
    # was waiting for the next 15-min tick.
    next_enqueue = _next_enqueue_eta()
    return {
        "pending_count": len(pending),
        "pending": pending[:50],
        "next_enqueue_eta_s": next_enqueue.get("eta_s"),
        "next_enqueue_iso": next_enqueue.get("iso"),
        "enqueue_interval_s": next_enqueue.get("interval_s"),
    }


def _next_enqueue_eta() -> dict:
    """Best-effort: ask systemd when the enqueue timer fires next.

    Returns an empty dict if systemctl isn't accessible (e.g. dev box)
    so the renderer can fall back to a static label. Cheap — single
    subprocess call, ~30ms; cached implicitly by being called per
    /api/queue request.
    """
    try:
        out = subprocess.check_output(
            ["systemctl", "list-timers",
             "heyi-eval-enqueue.timer",
             "--no-pager", "--output=json"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        rows = json.loads(out)
        if not rows:
            return {}
        row = rows[0]
        # systemctl --output=json reports usec since epoch
        next_us = row.get("next") or 0
        last_us = row.get("last") or 0
        if not next_us:
            return {}
        import time as _t
        eta_s = max(0, int(next_us / 1_000_000 - _t.time()))
        interval_s = (
            int((next_us - last_us) / 1_000_000) if last_us else None
        )
        iso = datetime.fromtimestamp(
            next_us / 1_000_000, tz=UTC,
        ).isoformat(timespec="seconds")
        return {"eta_s": eta_s, "iso": iso, "interval_s": interval_s}
    except Exception:
        return {}


def discover_summary(top_n: int = 20) -> dict:
    """Aggregated counts over discover/candidates.jsonl.

    PR#54 perf: previously loaded the entire file into memory and sorted
    all N rows. With the PR#52 backfill that file grew to 557k rows and
    every request was taking >30s — the single endpoint blocked every
    table on the main panel. Now we stream line-by-line and keep a
    bounded min-heap so total memory is ``O(top_n)`` and total time is
    ``O(n)``, ~120ms for 557k rows on NV8.
    """
    import heapq
    path = DATA_ROOT / "discover" / "candidates.jsonl"
    by_reason: Counter = Counter()
    by_pipeline: Counter = Counter()
    by_org: Counter = Counter()
    top_heap: list[tuple] = []  # (downloads, likes, monotonic_idx, row)
    idx = 0
    total = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                by_reason[c.get("reason", "?")] += 1
                by_pipeline[c.get("pipeline_tag") or "(none)"] += 1
                hf_id = c.get("hf_id") or ""
                org = hf_id.split("/", 1)[0]
                if org:
                    by_org[org] += 1
                downloads = c.get("downloads") or 0
                likes = c.get("likes") or 0
                key = (downloads, likes, idx)
                idx += 1
                row = {
                    "hf_id": hf_id,
                    "reason": c.get("reason"),
                    "pipeline_tag": c.get("pipeline_tag"),
                    "downloads": c.get("downloads"),
                    "likes": c.get("likes"),
                    "last_modified": c.get("last_modified"),
                }
                if len(top_heap) < top_n:
                    heapq.heappush(top_heap, (key, row))
                elif key > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (key, row))
    # Drain heap newest-first.
    top_rows = [r for _, r in sorted(top_heap, key=lambda t: t[0], reverse=True)]
    return {
        "total": total,
        "by_reason": dict(by_reason),
        "by_pipeline": dict(by_pipeline.most_common(15)),
        "by_org_top10": dict(by_org.most_common(10)),
        "top20_by_downloads": top_rows,
    }


def candidates_lifecycle(limit: int = 500) -> dict:
    """PR#50 + PR#52 (perf fix): per-candidate lifecycle view that joins
      - discover/candidates.jsonl  (everything we know exists on HF Hub)
      - store/queue.jsonl          (pending runs)
      - runs/<run_id>/state.json   (in_progress / terminal runs)

    Each candidate gets status ∈ {discovered, queued, in_progress,
    ok, failed, aborted}.

    Performance: with PR#52 backfill, candidates.jsonl now has 500k+
    rows. Naive "build a row dict for everything, then sort" was
    blocking the panel's single-threaded HTTP server for minutes.
    This version streams the candidates file once, keeps small-cardinality
    sets in full (queued / runs are O(hundreds)), and uses a top-N
    heap for the "discovered" tail so memory stays O(limit). Total work
    is O(n) candidates × O(1) lookups, plus a final O(limit·log·limit)
    heap-to-list step.
    """
    import heapq

    # 1) Runs: small set (hundreds), keep in full.
    runs_root = DATA_ROOT / "runs"
    by_hf: dict[str, dict] = {}

    def _epoch_to_iso(ts: Any) -> str | None:
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            try:
                from datetime import UTC, datetime
                return datetime.fromtimestamp(
                    float(ts), tz=UTC,
                ).isoformat(timespec="seconds")
            except (OverflowError, OSError, ValueError):
                return None
        return str(ts)

    if runs_root.exists():
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            state = _read_json(run_dir / "state.json")
            if not state:
                continue
            hf_id = state.get("hf_id")
            if not hf_id:
                continue
            cur = by_hf.get(hf_id)
            if (cur is None or (state.get("created_at") or 0) >
                    (cur.get("_created_raw") or 0)):
                cap = _read_json(run_dir / "capability.json") or {}
                # PR#58: surface param_count + modality from the metadata
                # artifact so the /candidates page can show the size of
                # each model the user explicitly asked for ("必要的参数还
                # 是得有，比如模型参数量大小啥的").
                meta_art = _read_json(run_dir / "_meta" / "metadata.json") or {}
                cur_art = _read_json(run_dir / "_meta" / "curated.json") or {}
                params_b = (
                    meta_art.get("param_count")
                    or cur_art.get("param_count")
                )
                modality = (
                    meta_art.get("modality")
                    or (meta_art.get("modalities") or [None])[0]
                )
                by_hf[hf_id] = {
                    "run_id": state.get("run_id") or run_dir.name,
                    "status": state.get("status") or "?",
                    "_created_raw": state.get("created_at"),
                    "created_at": _epoch_to_iso(state.get("created_at")),
                    "ended_at": _epoch_to_iso(state.get("ended_at")),
                    "failure_reason": state.get("failure_reason"),
                    "pass_rate": cap.get("pass_rate"),
                    "score": cap.get("score"),
                    "params": params_b,
                    "modality": modality,
                }

    # 2) Queue: small (max 1000s), keep in full.
    queued = _read_jsonl(DATA_ROOT / "store" / "queue.jsonl")
    queued_by_id: dict[str, dict] = {}
    for q in queued:
        hf_id = q.get("hf_id")
        if hf_id:
            queued_by_id[hf_id] = q

    # 3) Stream candidates.jsonl, decide status per row, only build
    #    full row dicts for non-"discovered" (small set) + top-N
    #    "discovered" by discovered_at.
    cand_path = DATA_ROOT / "discover" / "candidates.jsonl"
    status_counts: Counter = Counter()
    total_candidates = 0
    non_discovered_rows: list[dict] = []      # all in_progress/ok/failed/.../queued
    discovered_heap: list[tuple] = []         # min-heap on (anchor, hf_id)
    # Track which hf_ids we've consumed (newest-discovered-at wins).
    seen_cand: dict[str, str] = {}            # hf_id -> best discovered_at
    # PR#52 perf: tally statuses during the single stream pass so we
    # don't need a second walk over rows to count them.
    candidate_discovered_count = 0
    candidate_status_tally: Counter = Counter()

    def _push_discovered(row: dict) -> None:
        # PR#54: user wants "最新的模型放到最前面" — i.e. sort by
        # ``last_modified`` (when the model was actually pushed to the
        # Hub), not ``discovered_at`` (when our crawler saw it). With
        # backfill those two diverge by months; with curated-only they
        # are very close, but ``last_modified`` is the right semantic.
        # Fall back to discovered_at so models without a hub timestamp
        # don't sort to "" (which would put them at the very bottom).
        anchor = row.get("last_modified") or row.get("discovered_at") or ""
        if len(discovered_heap) < limit:
            heapq.heappush(discovered_heap, (anchor, row["hf_id"], row))
        else:
            if anchor > discovered_heap[0][0]:
                heapq.heapreplace(
                    discovered_heap, (anchor, row["hf_id"], row),
                )

    if cand_path.exists():
        with cand_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                hf_id = c.get("hf_id")
                if not hf_id:
                    continue
                total_candidates += 1
                # Dedup: newest discovered_at wins (skip if older).
                prev_at = seen_cand.get(hf_id)
                this_at = c.get("discovered_at") or ""
                if prev_at is not None and this_at <= prev_at:
                    continue
                seen_cand[hf_id] = this_at

                run = by_hf.get(hf_id)
                if run:
                    status = run["status"]
                elif hf_id in queued_by_id:
                    status = "queued"
                else:
                    status = "discovered"
                raw_fr = (run or {}).get("failure_reason")
                row = {
                    "hf_id": hf_id,
                    "status": status,
                    "status_zh": status_zh(status),
                    "discovered_at": c.get("discovered_at"),
                    "reason": c.get("reason"),
                    "pipeline_tag": c.get("pipeline_tag"),
                    "downloads": c.get("downloads"),
                    "likes": c.get("likes"),
                    "last_modified": c.get("last_modified"),
                    "run_id": (run or {}).get("run_id"),
                    "pass_rate": (run or {}).get("pass_rate"),
                    "score": (run or {}).get("score"),
                    "failure_reason": raw_fr,
                    "failure_reason_zh": failure_zh(raw_fr),
                    "ended_at": (run or {}).get("ended_at"),
                    "params": (run or {}).get("params"),
                    "modality": (run or {}).get("modality")
                                or c.get("pipeline_tag"),
                }
                if status == "discovered":
                    _push_discovered(row)
                    candidate_discovered_count += 1
                else:
                    non_discovered_rows.append(row)
                    candidate_status_tally[status] += 1

    # 4) Runs / queue with no candidate entry — surface as "manual".
    for hf_id, run in by_hf.items():
        if hf_id in seen_cand:
            continue
        non_discovered_rows.append({
            "hf_id": hf_id,
            "status": run["status"],
            "status_zh": status_zh(run["status"]),
            "discovered_at": None,
            "reason": "manual",
            "pipeline_tag": None,
            "downloads": None,
            "likes": None,
            "last_modified": None,
            "run_id": run["run_id"],
            "pass_rate": run.get("pass_rate"),
            "score": run.get("score"),
            "failure_reason": run.get("failure_reason"),
            "failure_reason_zh": failure_zh(run.get("failure_reason")),
            "ended_at": run.get("ended_at"),
            "params": run.get("params"),
            "modality": run.get("modality"),
        })
        seen_cand[hf_id] = ""
    for hf_id, q in queued_by_id.items():
        if hf_id in seen_cand:
            continue
        non_discovered_rows.append({
            "hf_id": hf_id,
            "status": "queued",
            "status_zh": status_zh("queued"),
            "discovered_at": None,
            "reason": "manual",
            "pipeline_tag": None,
            "downloads": None,
            "likes": None,
            "last_modified": None,
            "run_id": q.get("run_id"),
            "pass_rate": None,
            "score": None,
            "failure_reason": None,
            "failure_reason_zh": "",
            "ended_at": None,
            "params": None,
            "modality": None,
        })
        seen_cand[hf_id] = ""

    # 5) Status counts include EVERY candidate + manual rows. Tallied
    #    during the stream + the manual loop above.
    manual_status_tally: Counter = Counter(
        r["status"] for r in non_discovered_rows
        if r.get("reason") == "manual"
    )
    status_counts.update(candidate_status_tally)
    status_counts.update(manual_status_tally)
    if candidate_discovered_count:
        status_counts["discovered"] = candidate_discovered_count
    manual_count = sum(manual_status_tally.values())

    # 6) Final row list: non-discovered (full) + discovered (top-N
    #    by last_modified). Sort once at the end.
    rows = non_discovered_rows + [t[2] for t in discovered_heap]

    def _sort_key(r: dict) -> tuple[int, str]:
        # PR#54 + PR#55: the user's explicit ask is two-fold —
        #   (a) "最新的模型放到最前面"  → newest by last_modified DESC
        #   (b) "为什么只有标题，没有结果的示例" → real eval results
        #       (pass_rate, capability) must be visible, not buried
        #       under random unrelated rows.
        # Group-then-timestamp ordering achieves both. Groups, top→bot:
        #   5  active work (in_progress / queued)
        #   4  successful evaluations (ok with pass_rate)
        #   3  curated discoveries (reason in whitelist/trending/curated/
        #      backfill/incremental) — including failed/aborted ones
        #   2  ok runs without a candidate record (manual enqueue or
        #      legacy)
        #   1  orphan manual failed/aborted (old runs whose candidate
        #      record was purged — push to the bottom so they don't
        #      crowd out fresh discoveries)
        # Within each group, sort by the most informative timestamp
        # DESC (last_modified for HF freshness, ended_at for completed
        # runs, discovered_at as final fallback).
        status = r.get("status") or ""
        reason = r.get("reason") or ""
        is_manual = (reason == "manual")
        if status in ("in_progress", "queued"):
            group = 5
        elif status == "ok" and not is_manual:
            group = 4
        elif not is_manual:
            group = 3
        elif status == "ok":
            group = 2
        else:
            group = 1
        ts = str(
            r.get("last_modified")
            or r.get("ended_at")
            or r.get("discovered_at")
            or "",
        )
        return (group, ts)
    rows.sort(key=_sort_key, reverse=True)
    rows = rows[:limit]

    cursor = _read_json(DATA_ROOT / "discover" / "cursor.json") or {}
    backfill = {
        "complete": bool(cursor.get("backfill_complete", False)),
        "high_water": cursor.get("backfill_high_water"),
        "last_run_ts": cursor.get("last_run_ts"),
        "seen_size": len(cursor.get("seen") or []),
    }

    return {
        "total": total_candidates + manual_count,
        "status_counts": dict(status_counts),
        "backfill": backfill,
        "rows": rows,
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
    <h2>discover 候选概览 <a href="/candidates" style="font-size:11px;font-weight:normal;margin-left:8px">查看完整候选生命周期 →</a></h2>
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
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error('HTTP ' + r.status + ' on ' + url);
  }
  return await r.json();
}

function pillStatus(s) {
  // PR#57: localized status labels match the server-side status_zh map.
  const ZH = {ok:'成功', failed:'失败', aborted:'已中止',
              in_progress:'进行中', queued:'排队中', skipped:'跳过',
              running:'进行中'};
  const label = ZH[s] || s || '-';
  if (s === 'ok') return '<span class="pill pill-ok" title="'+s+'">'+label+'</span>';
  if (s === 'failed' || s === 'aborted') return '<span class="pill pill-err" title="'+s+'">'+label+'</span>';
  if (s === 'in_progress' || s === 'running' || s === 'queued') return '<span class="pill pill-info" title="'+s+'">'+label+'</span>';
  return '<span class="pill pill-muted" title="'+s+'">'+label+'</span>';
}

// PR#60: client-side equivalent of panel.server.hf_link — every place
// that prints a model id on the dashboard tables uses this so users
// can jump to either the model card on HF or the local run detail.
function escHTML(s){return String(s||'').replace(/[&<>\"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));}
function hfLinkJS(hfId, runId, showIcon) {
  if (!hfId || hfId === '?') return '<span class="muted">-</span>';
  const safe = escHTML(hfId);
  const enc = encodeURIComponent(hfId);
  let out = '<a href="https://huggingface.co/'+enc+'" target="_blank" '
          + 'rel="noopener noreferrer" title="在 HuggingFace Hub 查看模型卡">'+safe+'</a>';
  if (showIcon && runId) {
    out += ' <a class="run-icon" href="/run/'+encodeURIComponent(runId)+'" title="查看本地评测详情">📄</a>';
  }
  return out;
}
function publisherLinkJS(name) {
  if (!name) return '<span class="muted">-</span>';
  const safe = escHTML(name);
  const enc = encodeURIComponent(name);
  return '<a href="https://huggingface.co/'+enc+'" target="_blank" '
       + 'rel="noopener noreferrer" title="在 HuggingFace 查看该厂商">'+safe+'</a>';
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

function renderError(elId, label, err) {
  // PR#54: surface per-endpoint failures inline so the user can see
  // which section failed instead of staring at an "更新中…" that never
  // finishes (the old bug, where a slow /api/discover blocked every
  // table behind it).
  const msg = (err && err.message) ? err.message : String(err);
  const el = document.getElementById(elId);
  if (!el) return;
  el.innerHTML = `<tr><td colspan="20" class="err" style="font-size:12px">
    ${label} 加载失败: ${msg}</td></tr>`;
}

function renderErrorGrid(elId, label, err) {
  const msg = (err && err.message) ? err.message : String(err);
  const el = document.getElementById(elId);
  if (!el) return;
  el.innerHTML = `<div class="stat"><div class="label">${label}</div>
    <div class="value err" style="font-size:14px">加载失败</div>
    <div class="muted" style="font-size:11px">${msg}</div></div>`;
}

async function refresh() {
  document.getElementById('updated').textContent = '更新中… ' + new Date().toLocaleString('zh-CN');

  // PR#54: fire every section as an independent promise so one slow
  // endpoint doesn't blank the whole page. allSettled never throws.
  await Promise.allSettled([
    refreshHealth(),
    refreshBackup(),
    refreshQueue(),
    refreshResults(),
    refreshRuns(),
    refreshDiscover(),
    refreshOutbox(),
  ]);

  document.getElementById('updated').textContent = '已更新 ' + new Date().toLocaleString('zh-CN') + ' · 每 30s 自动刷新';
}

async function refreshHealth() {
  let h;
  try { h = await getJSON('/api/health'); }
  catch (e) { renderErrorGrid('health-grid', 'health', e); return; }
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
}

function _fmtAge(s) {
  if (s == null) return '<span class="muted">无记录</span>';
  if (s < 60) return s + 's 前';
  if (s < 3600) return Math.floor(s/60) + 'm 前';
  if (s < 86400) return (s/3600).toFixed(1) + 'h 前';
  return (s/86400).toFixed(1) + 'd 前';
}
function _fmtGB(b) { return b ? (b / (1024**3)).toFixed(2) + ' GB' : '<span class="muted">0</span>'; }

async function refreshBackup() {
  let bk;
  try { bk = await getJSON('/api/backup'); }
  catch (e) { renderErrorGrid('backup-grid', 'backup', e); return; }
  const bkCls = bk.health === 'ok' ? 'ok' : (bk.health === 'warn' ? 'warn' : 'err');
  document.getElementById('backup-grid').innerHTML = `
    <div class="stat"><div class="label">备份状态</div>
      <div class="value ${bkCls}">${bk.health.toUpperCase()}</div>
      <div class="muted" style="font-size:11px">${bk.backups_root}</div></div>
    <div class="stat"><div class="label">最近成功</div>
      <div class="value" style="font-size:14px">${_fmtAge(bk.last_backup_age_s)}</div>
      <div class="muted" style="font-size:11px">${bk.last_backup_ts || ''}</div></div>
    <div class="stat"><div class="label">snapshot 数量</div>
      <div class="value">${bk.snapshot_count}</div>
      <div class="muted" style="font-size:11px">7d 保留窗口</div></div>
    <div class="stat"><div class="label">总占用</div>
      <div class="value">${_fmtGB(bk.total_size_bytes)}</div>
      <div class="muted" style="font-size:11px">hard-link 共享</div></div>
  `;
  document.getElementById('backup-snapshots').textContent = JSON.stringify(bk.snapshots_tail, null, 2);
}

async function refreshQueue() {
  let q;
  try { q = await getJSON('/api/queue'); }
  catch (e) {
    renderError('queue-table tbody', 'queue', e);
    document.getElementById('queue-summary').innerHTML = '<span class="err">加载失败</span>';
    return;
  }
  // PR#56: show pending count *and* "下次自动入队" ETA. The user
  // assumed pending:0 meant "monitoring stopped" — it actually means
  // the orchestrator drained the last batch and we're waiting for the
  // 15-min auto-enqueue tick. Surfacing the ETA makes the cadence
  // visible so the panel matches user intuition.
  let etaLabel = '';
  if (typeof q.next_enqueue_eta_s === 'number') {
    const m = Math.floor(q.next_enqueue_eta_s / 60);
    const s = q.next_enqueue_eta_s % 60;
    const cadence = q.enqueue_interval_s ? ` · 周期 ${Math.round(q.enqueue_interval_s/60)} min` : '';
    etaLabel = ` · 下次自动入队 ${m}m ${s}s${cadence}`;
  }
  document.getElementById('queue-summary').innerHTML =
    `待评测 <strong>${q.pending_count}</strong> 个${etaLabel}`;
  const qbody = document.querySelector('#queue-table tbody');
  qbody.innerHTML = (q.pending||[]).map(p =>
    `<tr><td><a href="/run/${encodeURIComponent(p.run_id||'')}"><code>${escHTML((p.run_id||'').substring(0,32))}…</code></a></td>`
    + `<td>${hfLinkJS(p.hf_id, p.run_id, false)}</td></tr>`
  ).join('') ||
    '<tr><td colspan="2" class="muted">空 — 等待下次自动入队 / 或在 <a href="/candidates">候选模型</a> 页手动入队</td></tr>';
}

async function refreshResults() {
  let res;
  try { res = await getJSON('/api/results'); }
  catch (e) {
    renderErrorGrid('results-grid', 'results', e);
    const tb = document.querySelector('#results-table tbody');
    if (tb) tb.innerHTML = `<tr><td colspan="11" class="err">加载失败: ${(e.message||e)}</td></tr>`;
    return;
  }
  document.getElementById('results-grid').innerHTML = `
    <div class="stat"><div class="label">总评测 runs</div><div class="value">${res.total}</div></div>
    <div class="stat"><div class="label">完成</div><div class="value ok">${res.completed_ok}</div></div>
    <div class="stat"><div class="label">失败 / 进行中</div><div class="value warn">${res.failed} / ${res.in_progress}</div></div>
    <div class="stat"><div class="label">平均 pass rate</div><div class="value">${res.avg_pass_rate != null ? (res.avg_pass_rate*100).toFixed(0)+'%' : '<span class="muted">N/A</span>'}</div></div>
  `;
  const resBody = document.querySelector('#results-table tbody');
  const visibleRows = (res.rows||[]).filter(r => r.hf_id !== '?').slice(0, 30);
  resBody.innerHTML = visibleRows.map(r => {
    const pr = (typeof r.pass_rate === 'number') ? `<strong class="${r.pass_rate>=0.8?'ok':r.pass_rate>=0.5?'warn':'err'}">${(r.pass_rate*100).toFixed(0)}%</strong>` : '<span class="muted">-</span>';
    const cap = r.capability ? `<span class="pill ok">${r.capability}</span>` : '<span class="muted">-</span>';
    const fi  = r.first_impression ? `<span class="pill">${r.first_impression}</span>` : '<span class="muted">-</span>';
    const sm  = r.summary ? `<span class="muted" style="font-size:11px">${r.summary}…</span>` : (r.failure_reason ? `<span class="err" style="font-size:11px">✗ ${r.failure_reason}</span>` : '<span class="muted">-</span>');
    const ttft = (typeof r.ttft_ms_p50 === 'number') ? `${r.ttft_ms_p50.toFixed(0)}ms` : (r.perf_applicable === false ? '<span class="muted">N/A</span>' : '<span class="muted">-</span>');
    const tps  = (typeof r.tps_p50 === 'number') ? `${r.tps_p50.toFixed(1)} tok/s` : (r.perf_applicable === false ? '<span class="muted">N/A</span>' : '<span class="muted">-</span>');
    return `<tr>
      <td><strong>${hfLinkJS(r.hf_id, r.run_id, true)}</strong>
        <div class="muted" style="font-size:11px">${publisherLinkJS(r.publisher)}</div></td>
      <td>${pillStatus(r.status)}</td>
      <td>${escHTML(r.modality)||'<span class="muted">-</span>'}</td>
      <td>${escHTML(r.engine)||'<span class="muted">-</span>'}</td>
      <td>${cap}</td>
      <td>${pr}</td>
      <td>${ttft}</td>
      <td>${tps}</td>
      <td>${fi}</td>
      <td style="max-width:380px">${sm}</td>
      <td>${formatDuration(r.duration_s)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="11" class="muted">无评测结果</td></tr>';
}

async function refreshRuns() {
  let runs;
  try { runs = await getJSON('/api/runs'); }
  catch (e) {
    const tb = document.querySelector('#runs-table tbody');
    if (tb) tb.innerHTML = `<tr><td colspan="14" class="err">加载失败: ${(e.message||e)}</td></tr>`;
    return;
  }
  const rbody = document.querySelector('#runs-table tbody');
  rbody.innerHTML = (runs||[]).slice(0, 30).map(r => {
    const stages = STAGES.map(s => pillStatus((r.stages||{})[s])).join('</td><td>');
    return `<tr class="stage-row">
      <td><a href="/run/${encodeURIComponent(r.run_id)}"><code>${escHTML(r.run_id.substring(0,28))}…</code></a></td>
      <td>${hfLinkJS(r.hf_id, r.run_id, true)}</td>
      <td>${pillStatus(r.status)}</td>
      <td>${stages}</td>
      <td>${formatDuration(r.duration_s)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="14" class="muted">无 runs</td></tr>';
}

async function refreshDiscover() {
  let d;
  try { d = await getJSON('/api/discover'); }
  catch (e) { renderErrorGrid('discover-grid', 'discover', e); return; }
  const reasons = d.by_reason || {};
  // PR#53: with curated mode, common reasons are whitelist / trending / both / manual.
  const wl = reasons.whitelist || 0;
  const tr = reasons.trending || 0;
  const both = reasons.both || 0;
  document.getElementById('discover-grid').innerHTML = `
    <div class="stat"><div class="label">候选总数</div><div class="value">${d.total}</div></div>
    <div class="stat"><div class="label">whitelist 命中</div><div class="value ok">${wl + both}</div></div>
    <div class="stat"><div class="label">trending 命中</div><div class="value">${tr + both}</div></div>
    <div class="stat"><div class="label">Top org</div><div class="value" style="font-size:14px">${Object.entries(d.by_org_top10||{})[0]?.[0] || '-'}</div></div>
  `;
  document.getElementById('discover-top').textContent = JSON.stringify(d.top20_by_downloads, null, 2);
}

async function refreshOutbox() {
  let o;
  try { o = await getJSON('/api/outbox'); }
  catch (e) {
    const tb = document.querySelector('#outbox-table tbody');
    if (tb) tb.innerHTML = `<tr><td colspan="4" class="err">加载失败: ${(e.message||e)}</td></tr>`;
    return;
  }
  const obody = document.querySelector('#outbox-table tbody');
  obody.innerHTML = (o||[]).slice().reverse().map(e => {
    const lvl = e.level || 'info';
    const cls = lvl === 'error' ? 'err' : (lvl === 'warn' ? 'warn' : 'muted');
    return `<tr><td>${formatTs(e.ts)}</td><td>${e.event_type||'-'}</td><td class="${cls}">${lvl}</td><td>${e.title||'-'}</td></tr>`;
  }).join('') || '<tr><td colspan="4" class="muted">无</td></tr>';
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


def render_candidates_page() -> str:
    """PR#50: lifecycle view of every discovered candidate joined with
    its current run state. Top half is a backfill-progress card so we
    can watch the 2026 history-sweep at a glance; bottom is a sortable
    table of (hf_id, status, pass_rate, last_modified) with status pills.
    """
    data = candidates_lifecycle(limit=500)
    backfill = data.get("backfill") or {}
    status_counts = data.get("status_counts") or {}

    def _status_color(s: str) -> str:
        return {
            "ok": "ok",
            "failed": "err",
            "aborted": "warn",
            "in_progress": "run",
            "queued": "run",
            "discovered": "muted",
        }.get(s, "muted")

    rows_html = []
    for r in data["rows"]:
        hf_raw = r.get("hf_id") or "-"
        hf = html.escape(hf_raw)
        hf_attr = html.escape(hf_raw, quote=True)
        st = r.get("status") or "?"
        st_cls = _status_color(st)
        st_label = html.escape(r.get("status_zh") or status_zh(st))
        pipe = html.escape(r.get("pipeline_tag") or r.get("modality") or "-")
        reason = html.escape(r.get("reason") or "-")
        lm = html.escape((r.get("last_modified") or "-")[:10])
        dl = r.get("downloads")
        dl_s = f"{dl:,}" if isinstance(dl, int) else "-"
        likes = r.get("likes")
        likes_s = f"{likes:,}" if isinstance(likes, int) else "-"
        params_v = r.get("params") or "-"
        params_html = (
            f"<span class='pill'>{html.escape(str(params_v))}</span>"
            if params_v != "-" else "<span class='muted'>-</span>"
        )
        pr = r.get("pass_rate")
        if isinstance(pr, (int, float)):
            pr_pct = f"{pr * 100:.0f}%"
            pr_cls = "ok" if pr >= 0.8 else "warn" if pr >= 0.5 else "err"
        else:
            pr_pct, pr_cls = "-", "muted"
        run_id = r.get("run_id")
        if run_id:
            run_link = (
                f"<a href='/run/{html.escape(run_id)}'>"
                f"{html.escape(run_id[:8])}…</a>"
            )
        else:
            run_link = "<span class='muted'>-</span>"
        # PR#57: render failure reason in Chinese. Keep the English raw
        # text in the tooltip for grepability.
        fail_raw = r.get("failure_reason") or ""
        fail_zh_text = r.get("failure_reason_zh") or failure_zh(fail_raw)
        if len(fail_zh_text) > 120:
            fail_zh_disp = fail_zh_text[:120] + "…"
        else:
            fail_zh_disp = fail_zh_text
        fail_html = (
            f"<div class='err' style='font-size:11px;max-width:280px;"
            f"overflow:hidden;text-overflow:ellipsis' "
            f"title='{html.escape(fail_raw, quote=True)}'>"
            f"{html.escape(fail_zh_disp)}</div>"
            if fail_raw else ""
        )

        # PR#55: action column. Hide the "立即评测" button for rows that
        # are already in-flight; show "重测" instead for terminal runs
        # so the user can rerun a failed/aborted model after fixing
        # whatever blocked it.
        if hf_raw == "-":
            action_html = "<span class='muted'>-</span>"
        elif st in ("queued", "in_progress"):
            action_html = (
                f"<span class='pill run' title='已在队列或评测中，"
                f"无需重复入队'>排队中</span>"
            )
        else:
            label = "重测" if st in ("ok", "failed", "aborted") else "立即评测"
            action_html = (
                f"<button class='enqueue-btn' data-hf=\"{hf_attr}\" "
                f"onclick='enqueueModel(this)'>{label}</button>"
            )

        # PR#60: hf_id text becomes a link to HF Hub + 📄 to run detail
        hf_cell = hf_link(hf_raw if hf_raw != "-" else None,
                          run_id=run_id,
                          show_run_icon=False)  # run_id link is its own column already
        rows_html.append(
            f"<tr>"
            f"<td><strong>{hf_cell}</strong>{fail_html}</td>"
            f"<td><span class='pill {st_cls}' title='{html.escape(st)}'>{st_label}</span></td>"
            f"<td>{params_html}</td>"
            f"<td>{pipe}</td>"
            f"<td>{reason}</td>"
            f"<td>{lm}</td>"
            f"<td style='text-align:right'>{dl_s}</td>"
            f"<td style='text-align:right'>{likes_s}</td>"
            f"<td class='{pr_cls}' style='text-align:right'>"
            f"<strong>{pr_pct}</strong></td>"
            f"<td>{run_link}</td>"
            f"<td>{action_html}</td>"
            f"</tr>"
        )

    table_body = "".join(rows_html) or (
        "<tr><td colspan='11' class='muted'>暂无候选 — 等 discover daemon 抓第一轮</td></tr>"
    )

    bf_complete = backfill.get("complete")
    bf_badge = (
        "<span class='pill ok'>backfill 完成</span>"
        if bf_complete else "<span class='pill run'>backfill 进行中</span>"
    )
    bf_water = html.escape((backfill.get("high_water") or "-")[:19])
    bf_last = html.escape((backfill.get("last_run_ts") or "-")[:19])
    bf_seen = backfill.get("seen_size") or 0

    sc = {k: status_counts.get(k, 0) for k in (
        "discovered", "queued", "in_progress", "ok", "failed", "aborted",
    )}

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>heyi-eval-v10 · 候选模型生命周期</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;background:#0c0c10;color:#e7e7ea;margin:0;padding:0}}
header{{padding:16px 24px;background:#14141a;border-bottom:1px solid #26262e}}
header h1{{margin:0;font-size:18px}} header .sub{{color:#8a8a96;font-size:12px;margin-top:4px}}
main{{padding:18px 24px 60px;max-width:1700px;margin:0 auto}}
.grid{{display:grid;grid-template-columns:repeat(7,1fr);gap:12px;margin-bottom:20px}}
.stat{{padding:12px 16px;background:#14141a;border:1px solid #26262e;border-radius:6px}}
.stat .label{{color:#8a8a96;font-size:11px;text-transform:uppercase;letter-spacing:0.05em}}
.stat .value{{font-size:22px;margin-top:6px;font-weight:500}}
section{{background:#14141a;border:1px solid #26262e;border-radius:8px;padding:14px 18px;margin-bottom:18px}}
section h2{{margin:0 0 10px;font-size:14px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #20202a;vertical-align:top}}
th{{color:#8a8a96;font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:0.05em}}
tr:hover td{{background:#1a1a22}}
.ok{{color:#5ad48d}} .err{{color:#ef5f64}} .warn{{color:#e9b870}} .muted{{color:#6a6a76}} .run{{color:#6ec0ff}}
.pill{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;background:#20202a;color:#b8b8c4;white-space:nowrap}}
.pill.ok{{background:#1a3a26;color:#5ad48d}}
.pill.err{{background:#3a1a1f;color:#ef5f64}}
.pill.warn{{background:#3a2a18;color:#e9b870}}
.pill.run{{background:#1a2a3a;color:#6ec0ff}}
.pill.muted{{background:#1a1a22;color:#6a6a76}}
a{{color:#6ec0ff;text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body>
<header><a href="/">← 返回主面板</a> ·
<h1 style="display:inline">🔭 候选模型生命周期</h1>
<div class="sub">discover/candidates.jsonl × store/queue.jsonl × runs/* — 完整抓取→测试→结果链路</div></header>
<main>
<section>
  <h2>🛰️ Backfill 进度 {bf_badge}</h2>
  <div class="grid" style="grid-template-columns:repeat(4,1fr)">
    <div class="stat"><div class="label">backfill 完成</div>
      <div class="value {'ok' if bf_complete else 'warn'}">{'是' if bf_complete else '否'}</div></div>
    <div class="stat"><div class="label">cursor.seen 数</div>
      <div class="value">{bf_seen:,}</div></div>
    <div class="stat"><div class="label">high_water (已回填到)</div>
      <div class="value" style="font-size:14px">{bf_water}</div></div>
    <div class="stat"><div class="label">last_run_ts</div>
      <div class="value" style="font-size:14px">{bf_last}</div></div>
  </div>
</section>
<div class="grid">
  <div class="stat"><div class="label">候选总数</div><div class="value">{data['total']:,}</div></div>
  <div class="stat"><div class="label">已发现</div><div class="value muted">{sc['discovered']:,}</div></div>
  <div class="stat"><div class="label">排队中</div><div class="value run">{sc['queued']:,}</div></div>
  <div class="stat"><div class="label">进行中</div><div class="value run">{sc['in_progress']:,}</div></div>
  <div class="stat"><div class="label">成功</div><div class="value ok">{sc['ok']:,}</div></div>
  <div class="stat"><div class="label">失败</div><div class="value err">{sc['failed']:,}</div></div>
  <div class="stat"><div class="label">已中止</div><div class="value warn">{sc['aborted']:,}</div></div>
</div>
<section>
  <h2>✋ 手动入队（按 hf_id 指定模型）</h2>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <input id="manual-hf" type="text" placeholder="org/model 例: deepseek-ai/DeepSeek-V3.2"
           style="flex:1;min-width:320px;padding:8px 12px;background:#0c0c10;
           border:1px solid #26262e;border-radius:4px;color:#e7e7ea;font-family:monospace"/>
    <button class="enqueue-btn primary" onclick="enqueueManual()">入队评测</button>
    <span id="manual-status" class="muted" style="font-size:12px"></span>
  </div>
  <p class="muted" style="font-size:11px;margin:8px 0 0">
    任何符合 <code>&lt;org&gt;/&lt;name&gt;</code> 格式的 HF 模型 ID 都可手动指定；
    orchestrator 自动去重，已在队列 / in_progress 的请求会被拒绝。
  </p>
</section>
<section><table>
<thead><tr>
  <th>hf_id / 失败原因（中文）</th><th>状态</th><th>参数量</th><th>模态/类型</th><th>来源</th>
  <th>HF 最后更新</th><th style="text-align:right">下载</th>
  <th style="text-align:right">点赞</th><th style="text-align:right">通过率</th>
  <th>run_id</th><th>操作</th>
</tr></thead><tbody>{table_body}</tbody>
</table>
<p class="muted" style="font-size:11px;margin:10px 0 0">显示最近 500 条；按 last_modified 倒序（最新模型在前）；来源 = discover 来源
(whitelist / trending / curated / manual)；状态 = 已发现 → 排队中 → 进行中 → 成功 / 失败 / 中止。
点击「立即评测 / 重测」立即入队；同一模型并发入队会被 orchestrator 去重。失败原因悬停可看英文原文。</p>
</section>
<style>
.enqueue-btn{{
  padding:4px 10px;border-radius:4px;border:1px solid #2a4a6a;
  background:#1a2a3a;color:#6ec0ff;font-size:11px;cursor:pointer;
}}
.enqueue-btn:hover{{background:#22344a}}
.enqueue-btn.primary{{
  background:#1a3a26;color:#5ad48d;border-color:#2a5a3e;font-size:13px;
  padding:8px 18px;
}}
.enqueue-btn.primary:hover{{background:#225033}}
.enqueue-btn[disabled]{{opacity:0.5;cursor:not-allowed}}
</style>
<script>
async function _postEnqueue(hfId) {{
  const r = await fetch('/api/enqueue', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{hf_id: hfId}}),
  }});
  let body = {{}};
  try {{ body = await r.json(); }} catch (e) {{ /* keep body={{}} */ }}
  return {{status: r.status, body}};
}}

async function enqueueModel(btn) {{
  const hfId = btn.dataset.hf;
  if (!hfId) return;
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = '入队中…';
  try {{
    const {{status, body}} = await _postEnqueue(hfId);
    if (status === 202 && body.ok) {{
      btn.textContent = '已入队 ✓';
      setTimeout(() => location.reload(), 1500);
    }} else if (status === 409) {{
      btn.textContent = '已存在';
      setTimeout(() => {{ btn.disabled = false; btn.textContent = orig; }}, 2500);
    }} else {{
      btn.textContent = '失败';
      alert('入队失败 (HTTP ' + status + '): ' + (body.detail || body.error || 'unknown'));
      btn.disabled = false;
      btn.textContent = orig;
    }}
  }} catch (e) {{
    alert('网络错误: ' + e.message);
    btn.disabled = false;
    btn.textContent = orig;
  }}
}}

async function enqueueManual() {{
  const input = document.getElementById('manual-hf');
  const status = document.getElementById('manual-status');
  const hfId = (input.value || '').trim();
  status.textContent = '';
  status.className = 'muted';
  if (!hfId) {{
    status.textContent = '请输入 hf_id';
    status.className = 'warn';
    return;
  }}
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{{0,95}}\\/[A-Za-z0-9._-]{{1,96}}$/.test(hfId)) {{
    status.textContent = '格式不合法 (期望 org/name)';
    status.className = 'err';
    return;
  }}
  status.textContent = '入队中…';
  try {{
    const {{status: code, body}} = await _postEnqueue(hfId);
    if (code === 202 && body.ok) {{
      status.textContent = '已入队 ✓ run_id=' + body.run_id;
      status.className = 'ok';
      setTimeout(() => location.reload(), 1500);
    }} else if (code === 409) {{
      status.textContent = '已在队列 / in_progress，无需重复入队';
      status.className = 'warn';
    }} else {{
      status.textContent = '失败 (HTTP ' + code + '): ' + (body.detail || body.error || 'unknown');
      status.className = 'err';
    }}
  }} catch (e) {{
    status.textContent = '网络错误: ' + e.message;
    status.className = 'err';
  }}
}}
</script>
</main></body></html>"""


def render_results_page() -> str:
    """Standalone results leaderboard — all-test-results-at-a-glance view.

    PR#32 follow-up: sort rows so the actual evaluation results
    (status=ok with capability + pass_rate data) are at the top —
    previously a single failed run with a 4 KB vLLM traceback in its
    failure_reason pushed every successful run below the fold and the
    user couldn't see any actual evaluation data without scrolling.
    Also: truncate failure_reason at 200 chars in the cell, expose
    full text via title= attr / a details disclosure.
    """
    data = results_leaderboard()

    # PR#58: pure time-DESC ordering. The user explicitly asked
    # "测试结果时间排序，最新的放前面" — newest run on top, regardless of
    # status. results_leaderboard() already sorts this way, but keep
    # the explicit sort here so future readers see the contract; also
    # ensures the page is correct even if upstream ordering changes.
    sorted_rows = sorted(
        data["rows"],
        key=lambda r: (-(r.get("created_at") or 0),
                       -(r.get("ended_at") or 0)),
    )

    rows_html = []
    for r in sorted_rows:
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
        fail_raw = r.get("failure_reason") or ""
        # PR#57: render the failure reason in Chinese (failure_zh()
        # handles known patterns; unknown text passed through with
        # "原文：" prefix). Keep the raw English in the tooltip for
        # debuggability — grepping logs by exact English string is
        # still the fastest way to find a regression.
        fail_zh_text = r.get("failure_reason_zh") or failure_zh(fail_raw)
        if len(fail_zh_text) > 200:
            fail_short = fail_zh_text[:200] + "…"
        else:
            fail_short = fail_zh_text
        fail = html.escape(fail_short)
        if fail_raw:
            fail_html = (
                '<div class="err" '
                'style="font-size:11px;overflow:hidden;max-height:80px" '
                f'title="{html.escape(fail_raw, quote=True)}">'
                f'✗ {fail}</div>'
            )
        else:
            fail_html = ""
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

        # PR#58: time column — show "刚刚" / "Nm 前" / "Nh 前" / ISO so
        # the user can immediately tell which result is freshest.
        created = r.get("created_at") or 0
        if created:
            try:
                import time as _t
                age_s = _t.time() - float(created)
                if age_s < 60:
                    when = "刚刚"
                elif age_s < 3600:
                    when = f"{int(age_s / 60)} 分钟前"
                elif age_s < 86400:
                    when = f"{int(age_s / 3600)} 小时前"
                else:
                    when = f"{int(age_s / 86400)} 天前"
                when_iso = datetime.fromtimestamp(float(created), tz=UTC).strftime("%Y-%m-%d %H:%M")
            except Exception:
                when, when_iso = "-", ""
        else:
            when, when_iso = "-", ""
        when_cell = (
            f"<div>{html.escape(when)}</div>"
            f"<div class='muted' style='font-size:10px'>{html.escape(when_iso)}</div>"
        )

        status_zh_label = html.escape(r.get("status_zh") or status_zh(status))
        # PR#60: hf_id text → HF Hub (new tab), 📄 icon → local run
        # detail; publisher name → HF Hub publisher page.
        pub_link = (
            hf_publisher_link(r.get("publisher"))
            if r.get("publisher") else "<span class='muted'>-</span>"
        )
        hf_cell = hf_link(r.get("hf_id"), run_id=r.get("run_id"))
        rows_html.append(
            f"<tr><td>{when_cell}</td>"
            f"<td><strong>{hf_cell}</strong>"
            f"<div class='muted' style='font-size:11px'>{pub_link}</div></td>"
            f"<td><span class='pill {status_cls}' title='{html.escape(status)}'>{status_zh_label}</span></td>"
            f"<td>{modality}</td><td>{params}</td><td>{license_}</td>"
            f"<td>{engine}</td>"
            f"<td><span class='pill'>{cap}</span><div style='margin-top:3px'>{cats_html}</div></td>"
            f"<td class='{pr_cls}'><strong>{pr_pct}</strong></td>"
            f"<td>{ttft_cell}</td><td>{tps_cell}</td>"
            f"<td>{showcase_n} 条</td>"
            f"<td><span class='pill'>{fi}</span></td>"
            f"<td style='max-width:380px;font-size:12px;color:#b8b8c4;overflow:hidden;text-overflow:ellipsis;max-height:120px'>{summary}{fail_html}</td>"
            f"<td>{dur_s}</td></tr>"
        )

    table_body = "".join(rows_html) or '<tr><td colspan="15" class="muted">无评测结果</td></tr>'

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
  <th>时间</th><th>hf_id / publisher</th><th>状态</th><th>模态</th><th>参数量</th><th>许可证</th>
  <th>引擎</th><th>能力得分 / 类目</th><th>通过率</th>
  <th>TTFT (p50)</th><th>TPS (p50)</th><th>展示题数</th>
  <th>首印象</th><th>评价摘要 / 失败原因</th><th>耗时</th>
</tr></thead><tbody>{table_body}</tbody>
</table></section>
</main></body></html>"""


# ── PR#18: CAPABILITY (multimodal) + PERF_BENCH renderers ─────────────────


def _render_item_card(r: dict) -> str:
    """PR#61: card-based per-item rendering. Replaces the cramped table
    row that lost detail behind 80-char truncation. Cards show:
      - PASS/FAIL pill + latency
      - full prompt (no truncation, wrapped)
      - fixture image preview when present (vision/ocr items)
      - full model output in a monospace block (collapsible if long)
      - scorer attribution + tokens i/o on the footer

    User feedback: "页面太丑了 ... 又图片、音频、视频的，你从测试到结果
    都要能够支持预览这些内容". Visual hierarchy now leads with image
    input + verdict, then prompt, then output."""
    passed = bool(r.get("pass"))
    cls = "ok" if passed else "err"
    verdict = "通过" if passed else "未通过"
    iid = html.escape(r.get("id", ""))
    prompt = html.escape(r.get("prompt") or "")
    actual_raw = r.get("actual") or ""
    actual = html.escape(actual_raw)
    latency = r.get("latency_ms")
    tokens_in = r.get("tokens_in") or 0
    tokens_out = r.get("tokens_out") or 0
    scorer = html.escape(r.get("scorer_used") or r.get("scorer") or "-")
    fixture_html = render_fixture_preview(r.get("fixture"))

    # Collapse very long outputs by default; users can expand.
    long_output = len(actual_raw) > 400
    out_open = "" if long_output else " open"

    return (
        f"<article class='item-card item-{cls}'>"
        f"<header class='item-head'>"
        f"<span class='item-id'>{iid}</span>"
        f"<span class='pill pill-{cls}'>{verdict}</span>"
        f"<span class='item-meta'>"
        f"{latency} ms · in {tokens_in}t · out {tokens_out}t · scorer={scorer}"
        f"</span></header>"
        f"<div class='item-body'>"
        f"<div class='item-prompt'><label>题目</label>"
        f"<div class='prompt-text'>{prompt}</div>"
        f"{fixture_html}</div>"
        f"<div class='item-actual'>"
        f"<details{out_open}><summary>模型作答 ({len(actual_raw)} 字符)</summary>"
        f"<pre class='actual-text'>{actual}</pre></details></div>"
        f"</div></article>"
    )


def _render_item_row(r: dict) -> str:
    """Back-compat helper: PR#18 tests still call _render_item_row.
    Kept as a thin alias so old call sites keep working; new code
    should call _render_item_card directly."""
    return _render_item_card(r)


_CATEGORY_ZH = {
    "text_reasoning":      "文本推理",
    "code_gen":            "代码生成",
    "code_repair":         "代码修复",
    "code_complete":       "代码补全",
    "vision":              "视觉理解",
    "ocr":                 "光学字符识别 (OCR)",
    "asr":                 "语音识别 (ASR)",
    "tts":                 "语音合成 (TTS)",
    "image_gen":           "图像生成",
    "video_gen":           "视频生成",
    "music_gen":           "音乐生成",
    "music_understanding": "音乐理解",
    "video_understanding": "视频理解",
}

# PR#62: when a category is `applicable=False` because the model
# doesn't have the relevant pipeline_tag, surface a clear Chinese
# explanation instead of the cryptic "missing capability_tags: ..."
# string that bleeds the orchestrator's debug log into the UI.
_NA_REASON_ZH = {
    "image_gen":           "该模型未声明图像生成能力（HF pipeline_tag 不含 text-to-image），跳过此类目",
    "video_gen":           "该模型未声明视频生成能力（HF pipeline_tag 不含 text-to-video），跳过此类目",
    "music_gen":           "该模型未声明音频/音乐生成能力，跳过此类目",
    "tts":                 "该模型未声明 TTS 能力（HF pipeline_tag 不含 text-to-speech），跳过此类目",
    "asr":                 "该模型未声明 ASR 能力（HF pipeline_tag 不含 automatic-speech-recognition），跳过此类目",
    "vision":              "该模型未声明视觉理解能力（HF pipeline_tag 不含 image-text-to-text），跳过此类目",
    "ocr":                 "该模型未声明 OCR/视觉能力，跳过此类目",
    "video_understanding": "该模型未声明视频理解能力，跳过此类目",
    "music_understanding": "该模型未声明音乐理解能力，跳过此类目",
}


def _category_label(name: str) -> str:
    zh = _CATEGORY_ZH.get(name)
    return f"{zh}（{name}）" if zh else name


def _na_reason_zh(cat_name: str, raw_reason: str | None) -> str:
    pretty = _NA_REASON_ZH.get(cat_name)
    if pretty:
        return pretty
    if raw_reason and raw_reason.startswith("missing capability_tags"):
        return f"模型未声明此类目所需的 capability_tags（{raw_reason.split(':',1)[-1].strip()}）"
    return raw_reason or "该类目不适用于当前模型"


def _render_capability_html(cap: dict) -> str:
    """Render the CAPABILITY section.

    PR#15+: ``cap.categories`` is a dict
        { category_name: {applicable, scorer, score, pass_rate, items, reason?} }
    Each category renders as a collapsible <details> block with its own
    pass-rate banner + a card grid for items.

    Legacy artifacts (pre-PR#15) only have ``cap.results``: rendered as a
    single card grid for back-compat.
    """
    if not cap:
        return ""

    overall_score = html.escape(cap.get("score") or "?")
    overall_pr = cap.get("pass_rate")
    overall_pct = (
        f"{overall_pr*100:.0f}%" if isinstance(overall_pr, (int, float)) else "?"
    )
    overall = (
        "<div class='cap-overall'>"
        f"<span class='big-score'>{overall_score}</span>"
        f"<span class='muted'> · 总通过率 {overall_pct}</span>"
        "</div>"
    )

    categories = cap.get("categories") or {}
    if categories:
        # Sort so applicable categories come first, then by name
        sorted_cats = sorted(
            categories.items(),
            key=lambda kv: (not (kv[1] or {}).get("applicable"), kv[0]),
        )
        blocks: list[str] = []
        for cat_name, info in sorted_cats:
            info = info or {}
            applicable = bool(info.get("applicable"))
            score = info.get("score") or "0/0"
            pass_rate = info.get("pass_rate")
            scorer = info.get("scorer") or ""
            reason = info.get("reason") or ""
            label = _category_label(cat_name)

            if not applicable:
                blocks.append(
                    f"<div class='cap-cat cat-na'>"
                    f"<div class='cat-head'>"
                    f"<span class='cat-name'>{html.escape(label)}</span> "
                    f"<span class='pill pill-muted'>未适用</span></div>"
                    f"<div class='cat-na-reason muted'>"
                    f"{html.escape(_na_reason_zh(cat_name, reason))}</div>"
                    f"</div>"
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
            open_attr = (
                "open" if (pass_rate is None or pass_rate < 1.0) and items else ""
            )
            cards = "".join(_render_item_card(r) for r in items)
            blocks.append(
                f"<details class='cap-cat' {open_attr}>"
                f"<summary class='cat-head'>"
                f"<span class='cat-name'>{html.escape(label)}</span> "
                f"<span class='pill pill-{pass_cls}'>{html.escape(score)}</span> "
                f"<span class='muted'>"
                f"通过率 {pass_pct} · 评分器 {html.escape(scorer)}"
                f"</span></summary>"
                f"<div class='item-grid'>{cards}</div></details>"
            )
        return overall + "".join(blocks)

    if cap.get("results"):
        cards = "".join(_render_item_card(r) for r in cap["results"])
        return overall + f"<div class='item-grid'>{cards}</div>"

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

    hf_id = state.get("hf_id") or ""

    # PR#61: showcase rendering rebuilt as proper cards with prompt
    # collapsibles and "first impression" headline.
    show_html = ""
    if show.get("items"):
        cards = []
        for it in show["items"]:
            iid = html.escape(it.get("id", ""))
            rationale = html.escape(it.get("rationale", "") or "")
            prompt = html.escape(it.get("prompt", "") or "")
            actual = it.get("actual", "") or ""
            actual_h = html.escape(actual)
            comment = html.escape(it.get("comment", "") or "")
            tok_out = it.get("tokens_out") or 0
            lat = it.get("latency_ms") or 0
            long_output = len(actual) > 600
            out_open = "" if long_output else " open"
            cards.append(
                f"<article class='show-card'>"
                f"<header class='show-head'>"
                f"<span class='item-id'>{iid}</span>"
                f"<span class='item-meta'>{lat} ms · 输出 {tok_out} tokens</span>"
                f"</header>"
                f"<div class='show-body'>"
                f"<div class='show-rationale'>"
                f"<label>题目意图</label>"
                f"<div>{rationale}</div></div>"
                f"<details><summary>题目原文</summary>"
                f"<pre class='prompt-text'>{prompt}</pre></details>"
                f"<details{out_open}>"
                f"<summary>模型作答 ({len(actual)} 字符)</summary>"
                f"<pre class='actual-text'>{actual_h}</pre></details>"
                + (f"<div class='show-comment'>"
                   f"<label>系统注释</label><div>{comment}</div></div>"
                   if comment else "")
                + "</div></article>"
            )
        first = html.escape(show.get("model_first_impression") or "")
        summary_raw = show.get("summary") or ""
        try:
            from orchestrator.llm_text_utils import strip_think_blocks
            summary_raw = strip_think_blocks(summary_raw)
        except Exception:
            pass
        summary = html.escape(summary_raw)
        first_html = (
            f"<div class='first-impression'>"
            f"<label>首印象</label>"
            f"<span class='pill pill-info'>{first}</span></div>"
            if first else
            "<div class='first-impression muted'>"
            "<label>首印象</label>未生成</div>"
        )
        summary_html = (
            f"<div class='show-summary'><label>整体评价（LLM-as-judge）</label>"
            f"<p>{summary}</p></div>"
            if summary else ""
        )
        show_html = (
            first_html + summary_html
            + f"<div class='item-grid'>{''.join(cards)}</div>"
        )

    # Stages table — vertical timeline-style list so users see at a
    # glance which step failed and why.
    stages_rows = []
    for st_name, st_info in (state.get("stages") or {}).items():
        st = (st_info or {}).get("status", "?")
        cls = {"ok": "ok", "failed": "err", "aborted": "err",
               "skipped": "muted", "in_progress": "warn"}.get(st, "muted")
        dur = (st_info or {}).get("duration_s")
        dur_s = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "-"
        err = (st_info or {}).get("error") or ""
        err_zh = failure_zh(err) if err else ""
        stages_rows.append(
            f"<tr><td class='stage-cell'>"
            f"<div class='stage-zh'>{stage_zh(st_name)}</div>"
            f"<div class='stage-en'>{st_name}</div></td>"
            f"<td><span class='pill pill-{cls}'>{status_zh(st)}</span></td>"
            f"<td class='dur-cell'>{dur_s}</td>"
            f"<td class='err-cell'>{html.escape(err_zh)}</td></tr>"
        )
    stages_table = (
        "<table class='stages-table'>"
        "<thead><tr><th>阶段</th><th>状态</th><th>耗时</th>"
        "<th>失败原因（中文）</th></tr></thead>"
        f"<tbody>{''.join(stages_rows) or '<tr><td colspan=4 class=muted>无</td></tr>'}</tbody></table>"
    )

    # Top-level overall banner + key metadata pills
    overall_status = state.get("status") or "?"
    overall_status_zh = status_zh(overall_status)
    overall_fr_raw = state.get("failure_reason") or ""
    overall_fr_zh = failure_zh(overall_fr_raw) if overall_fr_raw else ""
    overall_banner_cls = {
        "ok": "ok", "failed": "err", "aborted": "err",
        "in_progress": "warn",
    }.get(overall_status, "muted")
    overall_banner = (
        f"<div class='overall-banner'>"
        f"<span class='pill pill-{overall_banner_cls} pill-big'>"
        f"{overall_status_zh}</span>"
        + (f"<span class='fail-text'>失败原因：{html.escape(overall_fr_zh)}</span>"
           if overall_fr_zh else "")
        + "</div>"
    )

    params_pill = meta.get("param_count") or cur.get("param_count") or "-"
    modality_pill = meta.get("modality") or "-"
    publisher_dict = cur.get("publisher") or meta.get("publisher") or {}
    publisher_name = (
        publisher_dict.get("name") if isinstance(publisher_dict, dict)
        else (publisher_dict or None)
    )
    license_pill = meta.get("license") or cur.get("license") or "-"
    metadata_pills = (
        "<div class='meta-pills'>"
        f"<span class='meta-pill'><label>参数量</label>"
        f"<span class='val'>{html.escape(str(params_pill))}</span></span>"
        f"<span class='meta-pill'><label>模态</label>"
        f"<span class='val'>{html.escape(str(modality_pill))}</span></span>"
        f"<span class='meta-pill'><label>厂商</label>"
        f"<span class='val'>{hf_publisher_link(publisher_name)}</span></span>"
        f"<span class='meta-pill'><label>许可证</label>"
        f"<span class='val'>{html.escape(str(license_pill))}</span></span>"
        "</div>"
    )

    hf_link_header = hf_link(
        hf_id, run_id=None, show_run_icon=False,
        css_class="hf-header-link",
    )

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>{html.escape(hf_id or run_id)} · heyi-eval</title>
{_PANEL_STYLES}
</head><body>
<header class='page-header'>
  <div class='header-left'>
    <a href="/" class='back-link'>← 返回主面板</a>
    <span class='breadcrumb-sep'>·</span>
    <a href="/results" class='back-link'>评测结果</a>
  </div>
  <h1 class='page-title'>{hf_link_header}</h1>
  <div class='header-sub muted'>
    run_id: <code>{html.escape(run_id)}</code>
  </div>
</header>
<main>

<section class='card-section'>
  <h2>整体概览</h2>
  {overall_banner}
  {metadata_pills}
</section>

<section class='card-section'>
  <h2>阶段执行（按发生顺序）</h2>
  {stages_table}
</section>

<section class='card-section'>
  <h2>能力评测（多模态分轨）</h2>
  {cap_html or '<div class="muted empty-note">无 — 运行未走到此阶段</div>'}
</section>

<section class='card-section'>
  <h2>性能基准（TTFT / TPS / 并发 / VRAM）</h2>
  {perf_html or '<div class="muted empty-note">无 — 运行未走到此阶段</div>'}
</section>

<section class='card-section'>
  <h2>展示评测（LLM 自主设计的题目 + 中文评价）</h2>
  {show_html or '<div class="muted empty-note">无 — 运行未走到此阶段</div>'}
</section>

<details class='card-section raw-section'>
  <summary><strong>原始 JSON 数据</strong>（点击展开，便于排查）</summary>
  <h3>state.json</h3><pre>{esc(state)}</pre>
  <h3>engine.json</h3><pre>{esc(eng)}</pre>
  <h3>metadata.json（HF + curator 合并）</h3><pre>{esc(meta)}</pre>
  <h3>curated.json（LLM 解读）</h3><pre>{esc(cur)}</pre>
  <h3>ready.json</h3><pre>{esc(ready)}</pre>
</details>

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
            elif path == "/api/candidates":
                self._json(candidates_lifecycle(limit=500))
            elif path == "/candidates":
                self._html(render_candidates_page())
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
            elif path.startswith("/fixtures/"):
                # PR#60: serve capability fixture images (vision/ocr
                # input PNGs) so the run detail page can render them
                # inline next to each test item. Strict path validation
                # rejects '..' traversal and characters outside the
                # tight whitelist (see _is_safe_fixture_path).
                self._serve_fixture(path[len("/fixtures/"):])
            else:
                self._json({"error": "not_found", "path": path}, status=404)
        except Exception as e:
            self._json({"error": str(e), "type": type(e).__name__}, status=500)

    def _serve_fixture(self, rel: str) -> None:
        if not _is_safe_fixture_path(rel):
            self._json({"error": "bad_fixture_path"}, status=400)
            return
        full = _FIXTURES_ROOT / rel
        try:
            # Resolve real path then re-check containment — defense in
            # depth in case a symlink lives inside the fixtures tree.
            real = full.resolve(strict=True)
            real.relative_to(_FIXTURES_ROOT.resolve())
        except (OSError, ValueError):
            self._json({"error": "not_found"}, status=404)
            return
        try:
            data = real.read_bytes()
        except OSError:
            self._json({"error": "read_failed"}, status=500)
            return
        ext = real.suffix.lower()
        ctype = {
            ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".gif": "image/gif",
            ".webp": "image/webp", ".svg": "image/svg+xml",
            ".wav": "audio/wav", ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg", ".flac": "audio/flac",
            ".mp4": "video/mp4", ".webm": "video/webm",
        }.get(ext, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # Fixtures are immutable, cache aggressively
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        """PR#55: manual interaction surface.

        ``POST /api/enqueue {"hf_id": "<org>/<name>"}`` — let the user
        prioritise a specific model from the candidates panel. The
        orchestrator's own dedup applies (queue/in-progress + recent
        success), and we forbid arbitrary path-like hf_ids to keep the
        endpoint a tight gate.

        Any other path returns 404 so we don't accidentally expose a
        write surface.
        """
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path == "/api/enqueue":
                self._handle_enqueue()
            else:
                self._json({"error": "not_found", "path": path}, status=404)
        except Exception as e:
            self._json({"error": str(e), "type": type(e).__name__}, status=500)

    def _handle_enqueue(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 4096:
            self._json({"error": "bad_request",
                        "detail": "missing/oversized body"}, status=400)
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as e:
            self._json({"error": "bad_json", "detail": str(e)}, status=400)
            return
        if not isinstance(payload, dict):
            self._json({"error": "bad_request",
                        "detail": "payload must be a JSON object"},
                       status=400)
            return
        hf_id = (payload.get("hf_id") or "").strip()
        if not _is_valid_hf_id(hf_id):
            self._json({"error": "bad_hf_id",
                        "detail": "expected '<org>/<name>' with safe "
                                  "characters only"},
                       status=400)
            return
        try:
            from orchestrator.main import enqueue as _orch_enqueue
            from orchestrator.store import Store
        except Exception as e:
            self._json({"error": "orchestrator_unavailable",
                        "detail": f"{type(e).__name__}: {e}"},
                       status=503)
            return
        store = Store(DATA_ROOT)
        # skip_if_recent=False: manual enqueue is an explicit user
        # action — they may want to re-evaluate a model that succeeded
        # earlier (different weights, vendor pushed update, etc.).
        run_id = _orch_enqueue(store, hf_id, skip_if_recent=False)
        if run_id is None:
            self._json({
                "ok": False,
                "hf_id": hf_id,
                "reason": "duplicate_or_in_progress",
                "detail": "already pending or running; orchestrator dedup",
            }, status=409)
            return
        self._json({"ok": True, "hf_id": hf_id, "run_id": run_id},
                   status=202)


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
