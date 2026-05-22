"""Unit + integration tests for transformers_runner (PR#19).

Tests do not require torch/transformers/diffusers — heavy imports are
guarded inside the pipeline builders, and these tests inject fake
pipeline objects via _PipelineCache so we exercise the full HTTP layer
on CPU-only machines.
"""
from __future__ import annotations

import io
import json
import socket
import threading
import time
import unittest
import urllib.request
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError

from transformers_runner import detect as detect_mod
from transformers_runner import server as server_mod


# ── detect.py ─────────────────────────────────────────────────────────────


class DetectTests(unittest.TestCase):

    def _mkmodel(self, root: Path, files: dict[str, dict]) -> Path:
        for name, obj in files.items():
            (root / name).write_text(json.dumps(obj), encoding="utf-8")
        return root

    def test_d1_text_model_via_model_type(self) -> None:
        with TemporaryDirectory() as d:
            p = self._mkmodel(Path(d), {
                "config.json": {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                }
            })
            det = detect_mod.detect(p)
        self.assertEqual(det.capability, "text")
        self.assertEqual(det.framework, "transformers")
        self.assertEqual(det.model_type, "llama")

    def test_d2_whisper_asr(self) -> None:
        with TemporaryDirectory() as d:
            p = self._mkmodel(Path(d), {
                "config.json": {
                    "model_type": "whisper",
                    "architectures": ["WhisperForConditionalGeneration"],
                }
            })
            det = detect_mod.detect(p)
        self.assertEqual(det.capability, "asr")

    def test_d3_vlm_via_qwen2_vl(self) -> None:
        with TemporaryDirectory() as d:
            p = self._mkmodel(Path(d), {
                "config.json": {
                    "model_type": "qwen2_vl",
                    "architectures": ["Qwen2VLForConditionalGeneration"],
                }
            })
            det = detect_mod.detect(p)
        self.assertEqual(det.capability, "vlm")

    def test_d4_diffusers_image_gen(self) -> None:
        with TemporaryDirectory() as d:
            p = self._mkmodel(Path(d), {
                "model_index.json": {
                    "_class_name": "StableDiffusionXLPipeline",
                }
            })
            det = detect_mod.detect(p)
        self.assertEqual(det.capability, "image_gen")
        self.assertEqual(det.framework, "diffusers")
        self.assertEqual(det.diffusers_class, "StableDiffusionXLPipeline")

    def test_d5_diffusers_video_gen(self) -> None:
        with TemporaryDirectory() as d:
            p = self._mkmodel(Path(d), {
                "model_index.json": {"_class_name": "CogVideoXPipeline"},
            })
            det = detect_mod.detect(p)
        self.assertEqual(det.capability, "video_gen")

    def test_d6_unknown_when_no_metadata(self) -> None:
        with TemporaryDirectory() as d:
            det = detect_mod.detect(Path(d))
        self.assertEqual(det.capability, "unknown")
        self.assertEqual(det.framework, "unknown")

    def test_d7_unknown_when_path_missing(self) -> None:
        det = detect_mod.detect("/nonexistent/path/abc123")
        self.assertEqual(det.capability, "unknown")

    def test_d8_arch_suffix_fallback_for_unknown_model_type(self) -> None:
        with TemporaryDirectory() as d:
            p = self._mkmodel(Path(d), {
                "config.json": {
                    "model_type": "some_brand_new_arch",
                    "architectures": ["MyCustomForCausalLM"],
                }
            })
            det = detect_mod.detect(p)
        self.assertEqual(det.capability, "text")

    def test_d9_supports_vlm_can_also_do_text(self) -> None:
        det = detect_mod.ModelDetection(
            capability="vlm", framework="transformers",
        )
        self.assertTrue(det.supports("text"))
        self.assertTrue(det.supports("vlm"))
        self.assertFalse(det.supports("asr"))

    def test_d10_supports_text_does_not_do_vlm(self) -> None:
        det = detect_mod.ModelDetection(
            capability="text", framework="transformers",
        )
        self.assertTrue(det.supports("text"))
        self.assertFalse(det.supports("vlm"))
        self.assertFalse(det.supports("image_gen"))


# ── _parse_multipart_audio ────────────────────────────────────────────────


class MultipartTests(unittest.TestCase):

    def test_m1_extracts_file_field(self) -> None:
        boundary = "abc123"
        audio = b"\x00\x01\x02\x03\xff\xfe"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"evaluated\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("ascii") + audio + (
            f"\r\n--{boundary}--\r\n"
        ).encode("ascii")
        data, ctype = server_mod._parse_multipart_audio(body, boundary)
        self.assertEqual(data, audio)
        self.assertEqual(ctype, "audio/wav")

    def test_m2_returns_empty_when_no_file(self) -> None:
        boundary = "abc"
        body = (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="model"\r\n\r\n'
                f"evaluated\r\n"
                f"--{boundary}--\r\n").encode("ascii")
        data, _ctype = server_mod._parse_multipart_audio(body, boundary)
        self.assertEqual(data, b"")


# ── HTTP server: routing + 501 gating ─────────────────────────────────────


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(detection: detect_mod.ModelDetection
                  ) -> tuple[ThreadingHTTPServer, int, server_mod.ServerState]:
    state = server_mod.ServerState(model_path="/fake/model",
                                   detection=detection)
    handler_cls = server_mod.make_handler_class(state)
    port = _pick_free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port, state


def _post_json(port: int, path: str, obj: dict,
               timeout: float = 5.0) -> tuple[int, dict]:
    body = json.dumps(obj).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


def _get(port: int, path: str,
         timeout: float = 5.0) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=timeout,
        ) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


class HealthEndpointTests(unittest.TestCase):

    def test_h1_health_reports_capability(self) -> None:
        det = detect_mod.ModelDetection(
            capability="text", framework="transformers",
            model_type="llama",
        )
        srv, port, _state = _start_server(det)
        try:
            status, body = _get(port, "/health")
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 200)
        self.assertEqual(body["capability"], "text")
        self.assertEqual(body["framework"], "transformers")

    def test_h2_models_lists_evaluated(self) -> None:
        det = detect_mod.ModelDetection(
            capability="text", framework="transformers",
        )
        srv, port, _state = _start_server(det)
        try:
            status, body = _get(port, "/v1/models")
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], "evaluated")
        self.assertEqual(body["data"][0]["capability"], "text")

    def test_h3_unknown_route_returns_404(self) -> None:
        det = detect_mod.ModelDetection(
            capability="text", framework="transformers",
        )
        srv, port, _state = _start_server(det)
        try:
            status, body = _get(port, "/v1/totally_made_up")
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["kind"], "not_found")


class GatingTests(unittest.TestCase):
    """Routes return 501 when the loaded model can't serve them."""

    def test_g1_chat_on_asr_model_returns_501(self) -> None:
        det = detect_mod.ModelDetection(
            capability="asr", framework="transformers",
            model_type="whisper",
        )
        srv, port, _state = _start_server(det)
        try:
            status, body = _post_json(port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "hi"}],
            })
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 501)
        self.assertEqual(body["error"]["kind"], "unsupported_modality")
        self.assertEqual(body["error"]["needed"], "text")
        self.assertEqual(body["error"]["loaded"]["capability"], "asr")

    def test_g2_image_gen_on_text_model_returns_501(self) -> None:
        det = detect_mod.ModelDetection(
            capability="text", framework="transformers",
            model_type="llama",
        )
        srv, port, _state = _start_server(det)
        try:
            status, body = _post_json(port, "/v1/images/generations", {
                "prompt": "a cat",
            })
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 501)
        self.assertEqual(body["error"]["needed"], "image_gen")

    def test_g3_video_gen_route_returns_501_deferred_when_loaded(self) -> None:
        # Even when the model SUPPORTS video_gen (diffusers CogVideoX),
        # the endpoint is intentionally deferred to PR#22 so the body
        # should carry "kind": "not_implemented" not "unsupported_modality".
        det = detect_mod.ModelDetection(
            capability="video_gen", framework="diffusers",
            diffusers_class="CogVideoXPipeline",
        )
        srv, port, _state = _start_server(det)
        try:
            status, body = _post_json(port, "/v1/videos/generations", {
                "prompt": "a sunset",
            })
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 501)
        self.assertEqual(body["error"]["kind"], "not_implemented")

    def test_g4_unsupported_payload_returns_400(self) -> None:
        det = detect_mod.ModelDetection(
            capability="text", framework="transformers",
            model_type="llama",
        )
        srv, port, _state = _start_server(det)
        try:
            # Send garbage that is not valid JSON
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=b"{not json",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                self.fail("expected HTTPError")
            except HTTPError as e:
                body = json.loads(e.read())
                self.assertEqual(e.code, 400)
                self.assertEqual(body["error"]["kind"], "bad_json")
        finally:
            srv.shutdown()
            srv.server_close()


# ── happy-path inference with injected fake pipeline ──────────────────────


class _FakeTokenizer:
    """Stand-in for transformers.AutoTokenizer in unit tests."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return "\n".join(m.get("content", "") for m in messages)

    def __call__(self, prompt, return_tensors="pt"):
        # Return a structure that mimics the .to(device) + ["input_ids"] use
        class _T:
            shape = (1, 7)
        ids = _T()

        class _Inputs(dict):
            def to(self, _device):
                return self
        return _Inputs(input_ids=ids)

    def decode(self, _tokens, skip_special_tokens=True):
        return "fake-completion-output"


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):  # noqa: ARG002
        class _Out:
            shape = (1, 12)
            def __getitem__(self, _i):
                return list(range(12))
        return _Out()


class ChatHappyPathTests(unittest.TestCase):

    def test_c1_chat_returns_openai_shape_with_fake_pipeline(self) -> None:
        det = detect_mod.ModelDetection(
            capability="text", framework="transformers",
            model_type="llama",
        )
        srv, port, state = _start_server(det)
        # Pre-seed the cache so no real model load happens
        state.cache._instances["text"] = {
            "tokenizer": _FakeTokenizer(),
            "model": _FakeModel(),
            "kind": "text",
        }
        try:
            status, body = _post_json(port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            })
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(
            body["choices"][0]["message"]["content"],
            "fake-completion-output",
        )
        self.assertEqual(body["model"], "evaluated")
        self.assertIn("usage", body)
        self.assertGreaterEqual(body["usage"]["prompt_tokens"], 1)


class AsrHappyPathTests(unittest.TestCase):

    def test_a1_transcribe_returns_text_field(self) -> None:
        det = detect_mod.ModelDetection(
            capability="asr", framework="transformers",
            model_type="whisper",
        )
        srv, port, state = _start_server(det)

        def _fake_asr_pipeline(_path):
            return "unused"
        # Inject a fake pipeline that the inference adapter would call
        captured = {}

        def _fake_infer(_pipe, audio_bytes):
            captured["bytes"] = audio_bytes
            return "hello world"

        # Pre-seed cache + monkeypatch _infer_asr
        state.cache._instances["asr"] = object()
        original = server_mod._infer_asr
        server_mod._infer_asr = _fake_infer  # type: ignore[assignment]
        try:
            boundary = "xxx"
            audio = b"RIFFAUDIO\x00\x01\x02"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/audio/transcriptions",
                data=body,
                headers={"Content-Type":
                         f"multipart/form-data; boundary={boundary}"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                status = r.status
                resp = json.loads(r.read())
        finally:
            server_mod._infer_asr = original  # type: ignore[assignment]
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"text": "hello world"})
        self.assertEqual(captured["bytes"], audio)


class ImageGenHappyPathTests(unittest.TestCase):

    def test_i1_image_returns_b64_json(self) -> None:
        det = detect_mod.ModelDetection(
            capability="image_gen", framework="diffusers",
            diffusers_class="StableDiffusionPipeline",
        )
        srv, port, state = _start_server(det)
        state.cache._instances["image_gen"] = object()
        fake_png = b"\x89PNG\r\n\x1a\nfakebody"
        original = server_mod._infer_image_gen
        server_mod._infer_image_gen = lambda _p, _prompt: fake_png  # type: ignore[assignment]
        try:
            status, body = _post_json(port, "/v1/images/generations", {
                "prompt": "a starry night",
            })
        finally:
            server_mod._infer_image_gen = original  # type: ignore[assignment]
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 200)
        self.assertEqual(len(body["data"]), 1)
        import base64 as _b64
        decoded = _b64.b64decode(body["data"][0]["b64_json"])
        self.assertEqual(decoded, fake_png)


# ── CLI argv contract ─────────────────────────────────────────────────────


class CliContractTests(unittest.TestCase):
    """Make sure the argv shape orchestrator/stages_py passes still parses."""

    def test_cli_1_serve_parses_required_args(self) -> None:
        from transformers_runner import __main__ as m
        # parse_args under the hood — verify the parser accepts the
        # exact argv that execute_deploy uses.
        argv = ["serve", "--model-path", "/model", "--port", "8000"]
        # We monkeypatch serve_main so no actual server starts.
        called = {}

        def _fake_serve(model_path, host="0.0.0.0", port=8000, max_loaded=1):
            called.update({"model_path": model_path, "host": host,
                           "port": port, "max_loaded": max_loaded})
            return 0
        # Patch the lazy import target
        import transformers_runner.server as _srv
        original = _srv.serve_main
        _srv.serve_main = _fake_serve  # type: ignore[assignment]
        try:
            rc = m.main(argv)
        finally:
            _srv.serve_main = original  # type: ignore[assignment]
        self.assertEqual(rc, 0)
        self.assertEqual(called["model_path"], "/model")
        self.assertEqual(called["port"], 8000)


if __name__ == "__main__":
    unittest.main()
