# SPDX-License-Identifier: Apache-2.0
"""Local HTTP server for the DashScope async video API, backed by FastH3 Trim.

AiVideo can point its base URL here and keep posting to
``/api/v1/services/aigc/video-generation/video-synthesis`` and polling
``/api/v1/tasks/{task_id}``. One generation runs at a time. ``Authorization``
is optional and ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol

from local_api.mapping import (
    TRIM_GRID_POINTS,
    TRIM_SEED,
    TRIM_VSA_SPARSITY,
    APIError,
    GenerationError,
    JobSpec,
    MediaItem,
    PreparedGeneration,
    adapt_canvas,
    canvas_for_ratio,
    parse_synthesis_body,
)

SYNTHESIS_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
GENERATE_PATH = "/generate"
TASK_PREFIX = "/api/v1/tasks/"
VIDEO_PREFIX = "/api/v1/videos/"
MAX_BODY_BYTES = 1_000_000
MAX_MEDIA_BYTES = 512 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 180
LOG_PREFIX = "[fasth3-local-api]"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class GenerationBackend(Protocol):

    def generate(self, prepared: PreparedGeneration) -> Path:
        """Write one mp4 and return its path."""


@dataclass
class Task:
    task_id: str
    prompt: str
    spec: JobSpec
    status: str
    submit_time: str
    width: int | None = None
    height: int | None = None
    scheduled_time: str | None = None
    end_time: str | None = None
    video_path: Path | None = None
    error_code: str | None = None
    error_message: str | None = None
    image_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", file=sys.stderr, flush=True)


def sniff_media_size(data: bytes) -> tuple[int, int] | None:
    """Read width and height from a short image header, if one is recognized."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24 and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if data.startswith(b"BM") and len(data) >= 26:
        return int.from_bytes(data[18:22], "little", signed=True), int.from_bytes(data[22:26], "little", signed=True)
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    if len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    index = 2
    limit = len(data)
    while index + 8 < limit:
        if data[index] != 0xFF:
            return None
        while index < limit and data[index] == 0xFF:
            index += 1
        if index >= limit:
            return None
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > limit:
            return None
        segment = int.from_bytes(data[index:index + 2], "big")
        if segment < 2:
            return None
        if marker in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
        ):
            if index + 7 > limit:
                return None
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return width, height
        index += segment
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and len(data) >= 30:
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None


def media_dimensions(path: Path) -> tuple[int, int]:
    blob = path.read_bytes()[:2 * 1024 * 1024]
    sniffed = sniff_media_size(blob)
    if sniffed and sniffed[0] > 0 and sniffed[1] > 0:
        return sniffed
    ffprobe = os.environ.get("FASTH3_FFPROBE", "ffprobe")
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GenerationError(
            f"cannot read media dimensions for {path.name}: {exc}",
            "InvalidParameter",
        ) from exc
    line = proc.stdout.strip().splitlines()
    if proc.returncode == 0 and line and "x" in line[0]:
        width_text, height_text = line[0].split("x", 1)
        try:
            width = int(width_text)
            height = int(height_text)
        except ValueError:
            width = height = 0
        if width > 0 and height > 0:
            return width, height
    detail = proc.stderr.strip() or "unrecognized image"
    raise GenerationError(
        f"cannot read media dimensions for {path.name}: {detail}",
        "InvalidParameter",
    )


def local_media_path(url: str, work_cwd: Path) -> Path | None:
    """Return a filesystem path for file/local media, or None for http(s)."""
    if url.startswith(("http://", "https://")):
        return None
    if "://" in url and not url.startswith("file:"):
        scheme = url.split(":", 1)[0]
        raise GenerationError(f"unsupported media URL scheme: {scheme}", "InvalidParameter")
    if url.startswith("file:"):
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc not in ("", "localhost"):
            raise GenerationError(f"unsupported file URL host: {parsed.netloc}", "InvalidParameter")
        raw = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
        path = Path(raw)
    else:
        path = Path(url)
    if not path.is_absolute():
        path = work_cwd / path
    return path


def suffix_for(url: str, role: str) -> str:
    parsed = urllib.parse.urlparse(url) if "://" in url else None
    raw_path = parsed.path if parsed is not None else url
    suffix = Path(raw_path).suffix.lower()
    if suffix and 1 < len(suffix) <= 8 and suffix[1:].isalnum():
        return suffix
    if role == "reference_video":
        return ".mp4"
    if role == "reference_audio":
        return ".wav"
    return ".img"


def download_http(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "fasth3-local-api"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            remaining = MAX_MEDIA_BYTES
            with dest.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise GenerationError(
                            f"media exceeds {MAX_MEDIA_BYTES} bytes: {url}",
                            "InvalidParameter",
                        )
                    handle.write(chunk)
    except GenerationError:
        raise
    except urllib.error.HTTPError as exc:
        raise GenerationError(
            f"media download failed: {url} HTTP {exc.code}",
            "InvalidParameter",
        ) from exc
    except urllib.error.URLError as exc:
        raise GenerationError(
            f"media download failed: {url} ({exc.reason})",
            "InvalidParameter",
        ) from exc


def materialize_media(url: str, dest: Path, work_cwd: Path) -> None:
    source = local_media_path(url, work_cwd)
    partial = dest.with_name(dest.name + ".partial")
    try:
        if source is None:
            download_http(url, partial)
        else:
            if not source.is_file():
                raise GenerationError(f"media file not found: {source}", "InvalidParameter")
            if source.stat().st_size > MAX_MEDIA_BYTES:
                raise GenerationError(
                    f"media exceeds {MAX_MEDIA_BYTES} bytes: {source}",
                    "InvalidParameter",
                )
            shutil.copyfile(source, partial)
        os.replace(partial, dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _first_path(paths: list[tuple[MediaItem, Path]], roles: set[str]) -> Path | None:
    for item, path in paths:
        if item.role in roles:
            return path
    return None


def resolve_canvas(spec: JobSpec, paths: list[tuple[MediaItem, Path]]) -> tuple[int, int]:
    if spec.width is not None and spec.height is not None:
        return spec.width, spec.height
    probe = _first_path(paths, {"first_frame"})
    if probe is None:
        probe = _first_path(paths, {"last_frame"})
    if probe is None:
        return canvas_for_ratio("16:9")
    width, height = media_dimensions(probe)
    return adapt_canvas(width, height)


class App:

    def __init__(
        self,
        *,
        backend: GenerationBackend,
        output_dir: Path,
        work_cwd: Path,
        seed: int,
        host: str,
        port: int,
    ) -> None:
        self.backend = backend
        self.output_dir = output_dir
        self.work_cwd = work_cwd
        self.seed = seed
        self.host = host
        self.port = port
        self.bound_port = port
        self.tasks: dict[str, Task] = {}
        self.tasks_lock = threading.Lock()
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.httpd: ThreadingHTTPServer | None = None
        self.worker: threading.Thread | None = None

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        load = getattr(self.backend, "load", None)
        if load is not None:
            load()

    def serve(self) -> ThreadingHTTPServer:
        self.prepare()
        handler = handler_factory(self)
        httpd = LocalHTTPServer((self.host, self.port), handler)
        self.httpd = httpd
        self.bound_port = int(httpd.server_address[1])
        self.worker = threading.Thread(target=self._worker, name="fasth3-local-api-worker", daemon=True)
        self.worker.start()
        return httpd

    def stop(self) -> None:
        """Unblock ``serve_forever`` from another thread, then join the worker."""
        if self.httpd is not None:
            self.httpd.shutdown()
        self.close()

    def close(self) -> None:
        self.queue.put(None)
        if self.worker is not None:
            self.worker.join(timeout=5)
        if self.httpd is not None:
            self.httpd.server_close()

    def fallback_host(self) -> str:
        return f"{self.host}:{self.bound_port}"

    def submit(self, spec: JobSpec) -> Task:
        task = Task(
            task_id=str(uuid.uuid4()),
            prompt=spec.prompt,
            spec=spec,
            status="PENDING",
            submit_time=timestamp(),
            width=spec.width,
            height=spec.height,
        )
        with self.tasks_lock:
            self.tasks[task.task_id] = task
        self.queue.put(task.task_id)
        return task

    def get_task(self, task_id: str) -> Task | None:
        with self.tasks_lock:
            return self.tasks.get(task_id)

    def _worker(self) -> None:
        while True:
            task_id = self.queue.get()
            try:
                if task_id is None:
                    return
                self._run_task(task_id)
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                self._fail(task_id, "InternalError", f"internal error: {exc}")
            finally:
                self.queue.task_done()

    def _run_task(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        with task.lock:
            task.status = "RUNNING"
            task.scheduled_time = timestamp()
        try:
            self._generate(task)
        except GenerationError as exc:
            self._fail(task_id, exc.code, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._fail(task_id, "InternalError", f"internal error: {exc}")

    def _fail(self, task_id: str | None, code: str, message: str) -> None:
        if not task_id:
            return
        task = self.get_task(task_id)
        if task is None:
            return
        with task.lock:
            task.status = "FAILED"
            task.error_code = code
            task.error_message = message
            task.end_time = timestamp()
        log(f"{task_id} FAILED {message}")

    def _generate(self, task: Task) -> None:
        task_dir = self.output_dir / task.task_id
        media_dir = task_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        video_path = task_dir / "video.mp4"
        paths: list[tuple[MediaItem, Path]] = []
        for index, item in enumerate(task.spec.media):
            dest = media_dir / f"{index:02d}-{item.role}{suffix_for(item.url, item.role)}"
            materialize_media(item.url, dest, self.work_cwd)
            paths.append((item, dest))
        width, height = resolve_canvas(task.spec, paths)
        image_count = sum(1 for item, _path in paths if item.role in ("first_frame", "last_frame"))
        seed = task.spec.seed if task.spec.seed is not None else self.seed
        steps = task.spec.steps if task.spec.steps is not None else TRIM_GRID_POINTS
        with task.lock:
            task.width = width
            task.height = height
            task.image_count = image_count
        prepared = PreparedGeneration(
            prompt=task.prompt,
            duration=task.spec.duration,
            width=width,
            height=height,
            num_frames=task.spec.num_frames,
            steps=steps,
            seed=seed,
            first_frame=_first_path(paths, {"first_frame"}),
            last_frame=_first_path(paths, {"last_frame"}),
            output_path=video_path,
            image_count=image_count,
        )
        prompt = prepared.prompt if len(prepared.prompt) <= 120 else prepared.prompt[:117] + "..."
        log(f"{task.task_id} {prepared.width}x{prepared.height} frames={prepared.num_frames} "
            f"steps={prepared.steps} seed={prepared.seed} prompt={prompt!r}")
        written = self.backend.generate(prepared)
        with task.lock:
            task.status = "SUCCEEDED"
            task.video_path = written
            task.end_time = timestamp()
        log(f"{task.task_id} SUCCEEDED {written}")


def handler_factory(app: App) -> type[BaseHTTPRequestHandler]:

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write(LOG_PREFIX + " " + (fmt % args) + "\n")

        def do_GET(self) -> None:  # noqa: N802
            self._guard(self._dispatch)

        def do_POST(self) -> None:  # noqa: N802
            self._guard(self._dispatch)

        def _guard(self, method: Callable[[], None]) -> None:
            try:
                method()
            except APIError as exc:
                self._send_json(
                    exc.status,
                    {
                        "request_id": str(uuid.uuid4()),
                        "code": exc.code,
                        "message": exc.message,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    500,
                    {
                        "request_id": str(uuid.uuid4()),
                        "code": "InternalError",
                        "message": str(exc),
                    },
                )
            finally:
                self.close_connection = True

        def _dispatch(self) -> None:
            path = self._path()
            if self.command == "POST" and path in (SYNTHESIS_PATH, GENERATE_PATH):
                self._submit()
                return
            if self.command == "GET" and path.startswith(TASK_PREFIX):
                self._task(path[len(TASK_PREFIX):])
                return
            if self.command == "GET" and path.startswith(VIDEO_PREFIX):
                self._video(path[len(VIDEO_PREFIX):])
                return
            if self.command in ("GET", "POST"):
                raise APIError("unknown path", status=404)
            raise APIError("method not allowed", status=405)

        def _path(self) -> str:
            path = urllib.parse.urlsplit(self.path).path
            if len(path) > 1 and path.endswith("/"):
                path = path[:-1]
            return path

        def _submit(self) -> None:
            _ = self.headers.get("Authorization")
            _ = self.headers.get("X-DashScope-Async")
            length = self.headers.get("Content-Length")
            if length is None or not length.isdigit():
                raise APIError("Content-Length is required")
            size = int(length)
            if size > MAX_BODY_BYTES:
                raise APIError("request body is too large")
            raw = self.rfile.read(size)
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise APIError(f"request body must be JSON: {exc}") from exc
            spec = parse_synthesis_body(body)
            task = app.submit(spec)
            self._send_json(
                200,
                {
                    "request_id": str(uuid.uuid4()),
                    "output": {
                        "task_id": task.task_id,
                        "task_status": "PENDING",
                    },
                },
            )

        def _task(self, task_id: str) -> None:
            if not UUID_RE.match(task_id):
                raise APIError("task not found", status=404)
            task = app.get_task(task_id)
            if task is None:
                raise APIError("task not found", status=404)
            self._send_json(200, task_view(task, self._public_host()))

        def _video(self, name: str) -> None:
            task_id = name[:-4] if name.endswith(".mp4") else name
            if not UUID_RE.match(task_id):
                raise APIError("video not found", status=404)
            task = app.get_task(task_id)
            path = None
            status = None
            if task is not None:
                with task.lock:
                    status = task.status
                    path = task.video_path
            if task is None or status != "SUCCEEDED" or path is None or not path.is_file():
                raise APIError("video not found", status=404)
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                with path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)
            except BrokenPipeError:
                return

        def _public_host(self) -> str:
            host = self.headers.get("Host", "")
            host = host.strip()
            if host and not any(char in host for char in "\r\n /\\"):
                return host
            return app.fallback_host()

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                return

    return Handler


def task_view(task: Task, host: str) -> dict:
    with task.lock:
        output: dict[str, object] = {
            "task_id": task.task_id,
            "task_status": task.status,
            "submit_time": task.submit_time,
        }
        if task.scheduled_time:
            output["scheduled_time"] = task.scheduled_time
        if task.end_time:
            output["end_time"] = task.end_time
        if task.prompt:
            output["orig_prompt"] = task.prompt
        payload: dict[str, object] = {
            "request_id": str(uuid.uuid4()),
            "output": output,
        }
        if task.status == "SUCCEEDED" and task.video_path is not None:
            output["video_url"] = f"http://{host}/api/v1/videos/{task.task_id}.mp4"
            short_side = min(task.width or 0, task.height or 0)
            payload["usage"] = {
                "SR": str(short_side),
                "duration": task.spec.duration,
                "image_count": task.image_count,
                "input_seconds": 0,
                "output_seconds": task.spec.duration,
                "size": f"{task.width}*{task.height}",
                "total_seconds": task.spec.duration,
                "video_count": 1,
            }
        if task.status == "FAILED":
            output["code"] = task.error_code or "InternalError"
            output["message"] = task.error_message or "generation failed"
        return payload


class LocalHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a local DashScope-compatible FastH3 Trim video API.")
    parser.add_argument(
        "--host",
        default=os.environ.get("FASTH3_LOCAL_API_HOST", "127.0.0.1"),
        help="bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("FASTH3_LOCAL_API_PORT", 8765),
        help="bind port (default: 8765)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("FASTH3_MODEL", "FastVideo/FastVideo-FastH3-Trim-8-Step"),
        help="FastH3 Trim checkpoint repo id or local directory",
    )
    parser.add_argument(
        "--quant",
        default=os.environ.get("FASTH3_QUANT", "auto"),
        choices=("auto", "none", "nvfp4", "fp8"),
        help="transformer quantization (default: auto from the model id)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("FASTH3_LOCAL_API_OUTPUT"),
        help="directory for task videos (default: ./outputs/local-api)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("FASTH3_SEED", TRIM_SEED),
        help=f"seed used when the request omits parameters.seed (default: {TRIM_SEED})",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=_env_int("FASTH3_NUM_GPUS", 1),
        help="GPUs for one generator; sequence parallel matches this count (default: 1)",
    )
    parser.add_argument(
        "--vsa-sparsity",
        type=float,
        default=_env_float("FASTH3_VSA_SPARSITY", TRIM_VSA_SPARSITY),
        help=f"VSA sparsity in [0, 1) (default: {TRIM_VSA_SPARSITY})",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        default=os.environ.get("FASTH3_OFFLOAD", "") == "1",
        help="offload the text encoder and VAE between stages",
    )
    return parser


def app_from_args(args: argparse.Namespace, work_cwd: Path | None = None) -> App:
    from local_api.generator import FastVideoTrimBackend

    if args.num_gpus < 1:
        raise SystemExit("--num-gpus must be at least 1")
    if not 0.0 <= args.vsa_sparsity < 1.0:
        raise SystemExit("--vsa-sparsity must be in [0, 1)")
    if args.seed < 0:
        raise SystemExit("--seed must be >= 0")
    cwd = (work_cwd or Path.cwd()).resolve()
    output = Path(args.output_dir) if args.output_dir else cwd / "outputs" / "local-api"
    if not output.is_absolute():
        output = cwd / output
    backend = FastVideoTrimBackend(
        model_path=args.model,
        quant=args.quant,
        num_gpus=args.num_gpus,
        vsa_sparsity=args.vsa_sparsity,
        offload=bool(args.offload),
    )
    return App(
        backend=backend,
        output_dir=output,
        work_cwd=cwd,
        seed=args.seed,
        host=args.host,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        app = app_from_args(args)
        httpd = app.serve()
    except SystemExit as exc:
        message = exc.code
        if isinstance(message, str):
            print(message, file=sys.stderr)
            return 2
        raise
    except Exception as exc:  # noqa: BLE001 - surface model-load failures on startup
        print(f"failed to start FastH3 Trim: {exc}", file=sys.stderr)
        return 1
    raw_host, port = httpd.server_address[:2]
    host = raw_host.decode() if isinstance(raw_host, bytes) else raw_host
    print(f"fasth3 local api listening on http://{host}:{port}", flush=True)
    print("POST /api/v1/services/aigc/video-generation/video-synthesis", flush=True)
    print("GET  /api/v1/tasks/<task_id>", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
