# SPDX-License-Identifier: Apache-2.0
"""DashScope local-API tests with a fake Trim backend.

No GPU, checkpoint, or FastVideo import is required for the HTTP cases.
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_api.mapping import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    NVFP4_MODEL_PATH,
    APIError,
    GenerationError,
    PreparedGeneration,
    align_frame_count,
    canvas_for_ratio,
    infer_quant,
    parse_synthesis_body,
)
from local_api.generator import generation_request_raw  # noqa: E402
from local_api.server import SYNTHESIS_PATH, App, main  # noqa: E402


def png_header(width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"


def sample_body(**overrides: object) -> dict:
    body: dict = {
        "model": "MiniMax/MiniMax-H3",
        "input": {
            "prompt": "A red fox walks through fresh snow in a pine forest."
        },
        "parameters": {
            "resolution": "768P",
            "ratio": "16:9",
            "duration": 5,
            "watermark": False,
        },
    }
    for key, value in overrides.items():
        if key in ("prompt", "media", "model"):
            if key == "model":
                body["model"] = value
            else:
                body["input"][key] = value
        else:
            body["parameters"][key] = value
    return body


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[PreparedGeneration] = []

    def generate(self, prepared: PreparedGeneration) -> Path:
        self.calls.append(prepared)
        if "FAIL" in prepared.prompt:
            raise GenerationError("decoder exploded", "InternalError")
        prepared.output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return prepared.output_path


class HoldingBackend:
    def __init__(self) -> None:
        self.calls: list[PreparedGeneration] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, prepared: PreparedGeneration) -> Path:
        self.calls.append(prepared)
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("test did not release the holding backend")
        prepared.output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return prepared.output_path


class TestMapping(unittest.TestCase):

    def test_named_ratios_use_trim_480p_canvas(self) -> None:
        self.assertEqual(canvas_for_ratio("16:9"), (832, 480))
        self.assertEqual(canvas_for_ratio("9:16"), (480, 832))
        self.assertEqual(canvas_for_ratio("1:1"), (480, 480))
        self.assertEqual(canvas_for_ratio("4:3"), (640, 480))
        self.assertEqual(canvas_for_ratio("3:4"), (480, 640))
        self.assertEqual(canvas_for_ratio("21:9"), (960, 416))

    def test_duration_maps_onto_trim_frame_lattice(self) -> None:
        expect = {
            4: 124,
            5: 124,
            6: 158,
            10: 243,
            14: 345,
            15: 362,
        }
        for seconds, frames in expect.items():
            spec = parse_synthesis_body(sample_body(duration=seconds))
            self.assertEqual(spec.num_frames, frames, seconds)
            self.assertEqual(spec.num_frames, align_frame_count(spec.num_frames))
        self.assertEqual(parse_synthesis_body(sample_body(duration=5)).width, 832)
        self.assertEqual(parse_synthesis_body(sample_body(duration=5)).height, 480)

    def test_fast_preset_keeps_trim_recipe(self) -> None:
        spec = parse_synthesis_body(sample_body(duration=14, preset="fast", width=256, height=256))
        self.assertEqual(spec.preset, "fast")
        self.assertEqual((spec.width, spec.height), (256, 256))
        self.assertEqual(spec.num_frames, 345)
        self.assertIsNone(spec.steps)
        bare = parse_synthesis_body(sample_body(preset="fast"))
        self.assertEqual((bare.width, bare.height), (832, 480))

    def test_steps_and_frames(self) -> None:
        self.assertEqual(parse_synthesis_body(sample_body(steps=8)).steps, 9)
        self.assertEqual(parse_synthesis_body(sample_body(steps=9)).steps, 9)
        self.assertEqual(parse_synthesis_body(sample_body(frames=120)).num_frames, 124)
        with self.assertRaises(APIError) as caught:
            parse_synthesis_body(sample_body(steps=4))
        self.assertIn("9 sigma-grid points", caught.exception.message)
        with self.assertRaises(APIError) as short:
            parse_synthesis_body(sample_body(frames=100))
        self.assertIn("124..362", short.exception.message)

    def test_validation_matches_dashscope_errors(self) -> None:
        with self.assertRaises(APIError) as resolution:
            parse_synthesis_body(sample_body(resolution="2K"))
        self.assertEqual(resolution.exception.message, "parameters.resolution must be 768P")
        with self.assertRaises(APIError) as duration:
            parse_synthesis_body(sample_body(duration=3))
        self.assertEqual(duration.exception.message, "parameters.duration must be an integer from 4 to 15")
        with self.assertRaises(APIError) as mixed:
            parse_synthesis_body(
                sample_body(media=[
                    {
                        "type": "first_frame",
                        "url": "a.png"
                    },
                    {
                        "type": "reference_image",
                        "url": "b.png"
                    },
                ]))
        self.assertEqual(
            mixed.exception.message,
            "cannot mix first_frame/last_frame with reference_image, "
            "reference_video, or reference_audio in one request",
        )
        with self.assertRaises(APIError) as reference:
            parse_synthesis_body(sample_body(media=[{
                "type": "image_url",
                "url": "b.png"
            }]))
        self.assertIn("Ref2VA", reference.exception.message)
        with self.assertRaises(APIError) as audio:
            parse_synthesis_body(sample_body(media=[{
                "type": "driving_audio",
                "url": "a.wav"
            }]))
        self.assertIn("reference_audio requires", audio.exception.message)

    def test_explicit_768p_canvas_is_allowed(self) -> None:
        spec = parse_synthesis_body(sample_body(width=1344, height=768))
        self.assertEqual((spec.width, spec.height), (1344, 768))
        with self.assertRaises(APIError) as odd:
            parse_synthesis_body(sample_body(width=100, height=100))
        self.assertIn("multiples of 32", odd.exception.message)

    def test_infer_quant_from_model_id(self) -> None:
        self.assertEqual(infer_quant(DEFAULT_MODEL_PATH, "auto"), "none")
        self.assertEqual(infer_quant(NVFP4_MODEL_PATH, "auto"), "nvfp4")
        self.assertEqual(infer_quant("FastVideo/FastVideo-FastH3-Trim-FP8", "auto"), "fp8")
        self.assertEqual(infer_quant(NVFP4_MODEL_PATH, "none"), "none")


class TestServer(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]
    backend: RecordingBackend
    app: App
    thread: threading.Thread
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.backend = RecordingBackend()
        cls.app = App(
            backend=cls.backend,
            output_dir=root / "out",
            work_cwd=root,
            seed=1234,
            host="127.0.0.1",
            port=0,
        )
        httpd = cls.app.serve()
        cls.thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.app.bound_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.stop()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        raw: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict, bytes]:
        data = raw if raw is not None else None if body is None else json.dumps(body).encode()
        hdrs = dict(headers or {})
        if data is not None and "Content-Type" not in hdrs:
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    def wait_status(self, task_id: str, status: str) -> dict:
        deadline = time.time() + 5
        last: dict | None = None
        while time.time() < deadline:
            code, _headers, raw = self.request("GET", f"/api/v1/tasks/{task_id}")
            self.assertEqual(code, 200, raw)
            last = json.loads(raw)
            if last["output"]["task_status"] == status:
                return last
            time.sleep(0.01)
        self.fail(f"timed out waiting for {status}: {last}")

    def test_submit_poll_and_download(self) -> None:
        code, _headers, raw = self.request("POST", SYNTHESIS_PATH, sample_body())
        self.assertEqual(code, 200, raw)
        submitted = json.loads(raw)
        task_id = submitted["output"]["task_id"]
        self.assertEqual(submitted["output"]["task_status"], "PENDING")
        polled = self.wait_status(task_id, "SUCCEEDED")
        output = polled["output"]
        self.assertEqual(output["orig_prompt"], sample_body()["input"]["prompt"])
        self.assertIn("submit_time", output)
        self.assertIn("scheduled_time", output)
        self.assertIn("end_time", output)
        self.assertTrue(output["video_url"].endswith(f"/api/v1/videos/{task_id}.mp4"))
        self.assertTrue(output["video_url"].startswith(self.base))
        usage = polled["usage"]
        self.assertEqual(usage["SR"], "480")
        self.assertEqual(usage["duration"], 5)
        self.assertEqual(usage["output_seconds"], 5)
        self.assertEqual(usage["size"], "832*480")
        self.assertEqual(usage["video_count"], 1)
        self.assertEqual(usage["image_count"], 0)
        video_code, video_headers, video = self.request("GET", f"/api/v1/videos/{task_id}.mp4")
        self.assertEqual(video_code, 200)
        self.assertTrue(video_headers["Content-Type"].startswith("video/mp4"))
        self.assertEqual(video, b"\x00\x00\x00\x18ftypmp42")
        prepared = self.backend.calls[-1]
        self.assertEqual((prepared.width, prepared.height, prepared.num_frames), (832, 480, 124))
        self.assertEqual(prepared.steps, 9)
        self.assertEqual(prepared.seed, 1234)
        self.assertIsNone(prepared.first_frame)
        self.assertIsNone(prepared.last_frame)

    def test_generate_alias_and_optional_authorization(self) -> None:
        code, _headers, raw = self.request(
            "POST",
            "/generate",
            sample_body(seed=7),
            headers={"Authorization": "Bearer ignored", "X-DashScope-Async": "enable"},
        )
        self.assertEqual(code, 200, raw)
        task_id = json.loads(raw)["output"]["task_id"]
        self.wait_status(task_id, "SUCCEEDED")
        self.assertEqual(self.backend.calls[-1].seed, 7)

    def test_unknown_task_and_path(self) -> None:
        code, _headers, raw = self.request("GET", "/api/v1/tasks/not-a-uuid")
        self.assertEqual(code, 404)
        self.assertEqual(json.loads(raw)["message"], "task not found")
        missing = "00000000-0000-0000-0000-000000000000"
        code, _headers, raw = self.request("GET", f"/api/v1/tasks/{missing}")
        self.assertEqual(code, 404)
        code, _headers, raw = self.request("GET", "/nope")
        self.assertEqual(code, 404)
        self.assertEqual(json.loads(raw)["message"], "unknown path")

    def test_failed_generation_stays_http_200(self) -> None:
        code, _headers, raw = self.request("POST", SYNTHESIS_PATH, sample_body(prompt="FAIL please"))
        self.assertEqual(code, 200, raw)
        task_id = json.loads(raw)["output"]["task_id"]
        polled = self.wait_status(task_id, "FAILED")
        self.assertEqual(polled["output"]["code"], "InternalError")
        self.assertIn("decoder exploded", polled["output"]["message"])
        self.assertNotIn("usage", polled)
        video_code, _headers, _video = self.request("GET", f"/api/v1/videos/{task_id}.mp4")
        self.assertEqual(video_code, 404)

    def test_validation_does_not_queue(self) -> None:
        before = len(self.backend.calls)
        code, _headers, raw = self.request(
            "POST",
            SYNTHESIS_PATH,
            sample_body(media=[
                {
                    "type": "first_frame",
                    "url": "a.png"
                },
                {
                    "type": "reference_video",
                    "url": "b.mp4"
                },
            ]),
        )
        self.assertEqual(code, 400)
        body = json.loads(raw)
        self.assertEqual(body["code"], "InvalidParameter")
        self.assertIn("cannot mix", body["message"])
        self.assertNotIn("output", body)
        self.assertEqual(len(self.backend.calls), before)

    def test_reference_image_is_rejected(self) -> None:
        code, _headers, raw = self.request(
            "POST",
            SYNTHESIS_PATH,
            sample_body(media=[{
                "type": "reference_image",
                "url": "ref.png"
            }]),
        )
        self.assertEqual(code, 400)
        self.assertIn("Ref2VA", json.loads(raw)["message"])

    def test_first_and_last_frame_are_passed_through(self) -> None:
        root = Path(self.tmp.name)
        opening = root / "opening.png"
        closing = root / "closing.png"
        opening.write_bytes(png_header(1600, 900))
        closing.write_bytes(png_header(400, 800))
        code, _headers, raw = self.request(
            "POST",
            SYNTHESIS_PATH,
            sample_body(
                ratio="adaptive",
                media=[
                    {
                        "type": "first_frame",
                        "url": opening.as_uri()
                    },
                    {
                        "type": "last_frame",
                        "url": str(closing)
                    },
                ],
            ),
        )
        self.assertEqual(code, 200, raw)
        task_id = json.loads(raw)["output"]["task_id"]
        polled = self.wait_status(task_id, "SUCCEEDED")
        prepared = self.backend.calls[-1]
        self.assertEqual((prepared.width, prepared.height), (832, 480))
        self.assertIsNotNone(prepared.first_frame)
        self.assertIsNotNone(prepared.last_frame)
        assert prepared.first_frame is not None
        assert prepared.last_frame is not None
        self.assertIn("first_frame", prepared.first_frame.name)
        self.assertIn("last_frame", prepared.last_frame.name)
        self.assertTrue(prepared.first_frame.is_file())
        self.assertEqual(polled["usage"]["image_count"], 2)
        self.assertEqual(polled["usage"]["size"], "832*480")

    def test_missing_media_fails_the_task(self) -> None:
        missing = Path(self.tmp.name) / "missing.png"
        code, _headers, raw = self.request(
            "POST",
            SYNTHESIS_PATH,
            sample_body(media=[{
                "type": "last_frame",
                "url": str(missing)
            }]),
        )
        self.assertEqual(code, 200, raw)
        task_id = json.loads(raw)["output"]["task_id"]
        polled = self.wait_status(task_id, "FAILED")
        self.assertEqual(polled["output"]["code"], "InvalidParameter")
        self.assertIn("not found", polled["output"]["message"])

    def test_bad_json_and_content_length(self) -> None:
        code, _headers, raw = self.request("POST", SYNTHESIS_PATH, raw=b"not-json")
        self.assertEqual(code, 400)
        self.assertIn("JSON", json.loads(raw)["message"])
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.app.bound_port, timeout=5)
        connection.putrequest("POST", SYNTHESIS_PATH)
        connection.putheader("Content-Type", "application/json")
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        self.assertIn("Content-Length", json.loads(response.read())["message"])
        connection.close()

    def test_help_exits_cleanly(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--help"])
        self.assertEqual(caught.exception.code, 0)


class TestQueue(unittest.TestCase):

    def test_one_generation_at_a_time(self) -> None:
        backend = HoldingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            app = App(
                backend=backend,
                output_dir=Path(tmp) / "out",
                work_cwd=Path(tmp),
                seed=1234,
                host="127.0.0.1",
                port=0,
            )
            httpd = app.serve()
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{app.bound_port}"
            try:
                first = _post(base, sample_body(prompt="first clip"))
                self.assertTrue(backend.started.wait(5))
                second = _post(base, sample_body(prompt="second clip"))
                pending = json.loads(_get(base, f"/api/v1/tasks/{second}"))
                running = json.loads(_get(base, f"/api/v1/tasks/{first}"))
                self.assertEqual(pending["output"]["task_status"], "PENDING")
                self.assertEqual(running["output"]["task_status"], "RUNNING")
                missing = _status(base, f"/api/v1/videos/{first}.mp4")
                self.assertEqual(missing, 404)
                backend.release.set()
                self.assertEqual(_wait(base, first, "SUCCEEDED")["output"]["task_status"], "SUCCEEDED")
                self.assertEqual(_wait(base, second, "SUCCEEDED")["output"]["task_status"], "SUCCEEDED")
                self.assertEqual([call.prompt for call in backend.calls], ["first clip", "second clip"])
            finally:
                backend.release.set()
                app.stop()
                thread.join(timeout=5)


class TestGeneratorConfig(unittest.TestCase):

    def test_generation_request_raw_pins_trim_recipe(self) -> None:
        opening = Path(self.id().replace(".", "_") + "-should-not-be-read")
        prepared = PreparedGeneration(
            prompt="A quiet pottery studio.",
            duration=5,
            width=832,
            height=480,
            num_frames=124,
            steps=9,
            seed=1234,
            first_frame=opening,
            last_frame=None,
            output_path=Path("/tmp/fasth3-local-api-video.mp4"),
            image_count=1,
        )
        raw = generation_request_raw(prepared)
        self.assertEqual(raw["negative_prompt"], "")
        self.assertEqual(raw["sampling"]["num_frames"], 124)
        self.assertEqual(raw["sampling"]["num_inference_steps"], 9)
        self.assertEqual(raw["sampling"]["guidance_scale"], 1.0)
        self.assertFalse(raw["sampling"]["batch_cfg"])
        self.assertEqual(raw["sampling"]["fps"], 24)
        self.assertEqual((raw["sampling"]["width"], raw["sampling"]["height"]), (832, 480))
        self.assertFalse(raw["output"]["return_frames"])
        self.assertTrue(raw["output"]["save_video"])
        self.assertEqual(raw["inputs"]["image_path"], str(opening))
        self.assertNotIn("last_image", raw["inputs"])
        with self.assertRaises(GenerationError):
            generation_request_raw(
                PreparedGeneration(
                    prompt="x",
                    duration=5,
                    width=832,
                    height=480,
                    num_frames=124,
                    steps=4,
                    seed=1,
                    first_frame=None,
                    last_frame=None,
                    output_path=Path("video.mp4"),
                    image_count=0,
                ))

    def test_parsed_request_and_nvfp4_config(self) -> None:
        try:
            from fastvideo.api.request_metadata import get_explicit_paths

            from local_api.generator import build_generation_request, build_generator_config
        except Exception as exc:  # noqa: BLE001 - CPU images may not import FastVideo
            self.skipTest(f"FastVideo schema import is unavailable: {exc}")

        config = build_generator_config(
            model_path=NVFP4_MODEL_PATH,
            quant="auto",
            num_gpus=1,
            vsa_sparsity=0.8,
            offload=False,
        )
        self.assertEqual(config.model_path, NVFP4_MODEL_PATH)
        self.assertEqual(config.engine.quantization.transformer_quant, "NVFP4")
        self.assertEqual(config.engine.quantization.layer_profile, "h3_dit_vsa")
        self.assertEqual(config.pipeline.experimental["VSA_sparsity"], 0.8)
        self.assertEqual(config.pipeline.experimental["attention_backend"], "VIDEO_SPARSE_ATTN_H3")
        self.assertEqual(config.pipeline.workload_type, "t2v")
        self.assertFalse(config.engine.offload.text_encoder)

        prepared = PreparedGeneration(
            prompt="A quiet pottery studio.",
            duration=5,
            width=832,
            height=480,
            num_frames=124,
            steps=9,
            seed=1234,
            first_frame=None,
            last_frame=None,
            output_path=Path("/tmp/fasth3-local-api-video.mp4"),
            image_count=0,
        )
        request = build_generation_request(prepared)
        paths = get_explicit_paths(request)
        self.assertEqual(request.negative_prompt, "")
        self.assertEqual(request.sampling.num_inference_steps, 9)
        self.assertIn("negative_prompt", paths)
        self.assertIn("sampling.num_inference_steps", paths)
        self.assertIn("output.output_path", paths)


def _post(base: str, body: dict) -> str:
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        base + SYNTHESIS_PATH,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())
    return str(payload["output"]["task_id"])


def _get(base: str, path: str) -> bytes:
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return response.read()


def _status(base: str, path: str) -> int:
    try:
        with urllib.request.urlopen(base + path, timeout=10) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def _wait(base: str, task_id: str, status: str) -> dict:
    deadline = time.time() + 5
    last: dict | None = None
    while time.time() < deadline:
        last = json.loads(_get(base, f"/api/v1/tasks/{task_id}"))
        if last["output"]["task_status"] == status:
            return last
        time.sleep(0.01)
    raise AssertionError(last)


if __name__ == "__main__":
    unittest.main()
