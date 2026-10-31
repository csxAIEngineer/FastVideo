# SPDX-License-Identifier: Apache-2.0
"""Map MiniMax-H3 DashScope requests onto a FastH3 Trim generation.

The HTTP body stays the DashScope shape AiVideo already posts. Geometry does
not. h3.c sizes a 768-short-edge canvas (16:9 is 1344x768). This server sizes
the released Trim recipe: short edge 480, at most 480*832 pixels, so 16:9 is
832x480. ``parameters.resolution`` is still the DashScope token ``768P``; it
does not select those 768p pixels. An explicit width and height pair can
still ask for the native 1344x768 canvas.

Duration stays an integer from 4 to 15 so existing clients do not change.
Frames are 24 fps rounded up onto ``5 + 17*n`` and then clamped to FastVideo's
accepted range, 124 through 362. A 4 second request is 107 frames on h3.c and
124 frames here (about 5.17 seconds), which is Trim's shortest clip.

``preset: fast`` is accepted and does not switch to h3.c's 512/320, 4-step
path. Trim always runs nine sigma-grid points (eight DMD forwards). Reference
media is rejected: this checkpoint is the T2VA/FL2VA pipeline, not Ref2VA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TRIM_SHORT_EDGE = 480
TRIM_MAX_PIXELS = 480 * 832
# Explicit width/height may still request the released H3 768p canvas.
H3_MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
FPS = 24
MIN_DURATION = 4
MAX_DURATION = 15
MIN_ALIGNED_FRAMES = 124
MAX_ALIGNED_FRAMES = 362
TRIM_GRID_POINTS = 9
TRIM_VSA_SPARSITY = 0.8
TRIM_VSA_TILE_SIZE = 64
TRIM_SEED = 1234
DEFAULT_MODEL_PATH = "FastVideo/FastVideo-FastH3-Trim-8-Step"
NVFP4_MODEL_PATH = "FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4"
MAX_PROMPT_CHARS = 7000
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIO = 3
MAX_REFERENCES = 12

CLIENT_MODEL_NAME = "MiniMax/MiniMax-H3"
NAMED_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")

ROLE_ALIASES = {
    "first_frame": "first_frame",
    "last_frame": "last_frame",
    "reference_image": "reference_image",
    "image_url": "reference_image",
    "reference_video": "reference_video",
    "feature": "reference_video",
    "reference_audio": "reference_audio",
    "driving_audio": "reference_audio",
}

FRAME_ROLES = {"first_frame", "last_frame"}
REFERENCE_ROLES = {"reference_image", "reference_video", "reference_audio"}


class APIError(Exception):
    """A request that should be rejected before a task is queued."""

    def __init__(
        self,
        message: str,
        code: str = "InvalidParameter",
        status: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class GenerationError(Exception):
    """A queued task that failed while materializing media or generating."""

    def __init__(self, message: str, code: str = "InternalError") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MediaItem:
    role: str
    url: str


@dataclass(frozen=True)
class JobSpec:
    prompt: str
    duration: int
    ratio: str
    preset: str
    width: int | None
    height: int | None
    num_frames: int
    steps: int | None
    media: tuple[MediaItem, ...]
    seed: int | None


@dataclass(frozen=True)
class PreparedGeneration:
    """One Trim request after media files exist on disk."""

    prompt: str
    duration: int
    width: int
    height: int
    num_frames: int
    steps: int
    seed: int
    first_frame: Path | None
    last_frame: Path | None
    output_path: Path
    image_count: int


def align_frame_count(requested: int) -> int:
    """Round up to ``5 + 17*n``, matching MiniMax-H3's causal VAE lattice."""
    value = requested if requested >= 5 else 5
    remainder = (value - 5) % 17
    if remainder:
        value += 17 - remainder
    return value


def frames_for_duration(duration: int) -> int:
    """Map a DashScope second count onto Trim's legal frame count."""
    aligned = align_frame_count(duration * FPS)
    if aligned < MIN_ALIGNED_FRAMES:
        return MIN_ALIGNED_FRAMES
    if aligned > MAX_ALIGNED_FRAMES:
        raise APIError(f"parameters.duration must align within {MIN_ALIGNED_FRAMES}..{MAX_ALIGNED_FRAMES} frames", )
    return aligned


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """Return a Trim canvas for an aspect ratio expressed as width/height.

    The short side starts at 480 and the result snaps to a multiple of 32.
    Pixel count stays within ``480 * 832``, so 16:9 lands on 832x480.
    """
    if width < 1 or height < 1:
        raise APIError("media dimensions must be positive")
    ratio = width / height
    if ratio >= 1.0:
        nominal_w = float(TRIM_SHORT_EDGE) * ratio
        nominal_h = float(TRIM_SHORT_EDGE)
    else:
        nominal_w = float(TRIM_SHORT_EDGE)
        nominal_h = float(TRIM_SHORT_EDGE) / ratio
    pixels = nominal_w * nominal_h
    if pixels > TRIM_MAX_PIXELS:
        scale = (TRIM_MAX_PIXELS / pixels)**0.5
        nominal_w *= scale
        nominal_h *= scale
    out_w = int(round(nominal_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    out_h = int(round(nominal_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    out_w = max(out_w, CANVAS_MULTIPLE)
    out_h = max(out_h, CANVAS_MULTIPLE)
    while out_w * out_h > TRIM_MAX_PIXELS and (out_w > CANVAS_MULTIPLE or out_h > CANVAS_MULTIPLE):
        if out_w >= out_h and out_w > CANVAS_MULTIPLE:
            out_w -= CANVAS_MULTIPLE
        elif out_h > CANVAS_MULTIPLE:
            out_h -= CANVAS_MULTIPLE
        else:
            break
    return out_w, out_h


def canvas_for_ratio(ratio: str) -> tuple[int, int]:
    if ratio not in NAMED_RATIOS:
        raise APIError("ratio must be adaptive, " + ", ".join(NAMED_RATIOS))
    left, right = ratio.split(":")
    return adapt_canvas(int(left), int(right))


def validate_canvas(
    width: int,
    height: int,
    render_width: int | None,
    render_height: int | None,
) -> None:
    if (width < CANVAS_MULTIPLE or height < CANVAS_MULTIPLE or width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE):
        raise APIError("width and height must be multiples of 32 and at least 32")
    if width * height > H3_MAX_PIXELS:
        raise APIError("canvas exceeds the released 768*1344 pixel limit")
    if (render_width is None) != (render_height is None):
        raise APIError("render_width and render_height must be set together")
    if render_width is None or render_height is None:
        return
    if (render_width < CANVAS_MULTIPLE or render_height < CANVAS_MULTIPLE or render_width % CANVAS_MULTIPLE
            or render_height % CANVAS_MULTIPLE):
        raise APIError("render width and height must be multiples of 32 and at least 32")
    if render_width != width or render_height != height:
        raise APIError(
            "FastH3 Trim renders at the output canvas; parameters.render_width and "
            "parameters.render_height must match width and height", )


def infer_quant(model: str, quant: str) -> str:
    """Pick ``none``, ``nvfp4``, or ``fp8`` from an explicit flag or the repo id."""
    choice = quant.strip().lower()
    if choice not in ("auto", "none", "nvfp4", "fp8"):
        raise ValueError("quant must be auto, none, nvfp4, or fp8")
    if choice != "auto":
        return choice
    lowered = model.lower()
    if "nvfp4" in lowered:
        return "nvfp4"
    if "fp8" in lowered:
        return "fp8"
    return "none"


def parse_synthesis_body(body: object) -> JobSpec:
    """Validate a video-synthesis JSON body and return a Trim job."""
    data = _as_object(body, "body")
    model = data.get("model", CLIENT_MODEL_NAME)
    if not isinstance(model, str) or not _accepted_model(model):
        raise APIError(f"model must be {CLIENT_MODEL_NAME}")

    prompt_input = _as_object(data.get("input"), "input")
    prompt = prompt_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise APIError("input.prompt is required")
    if "\x00" in prompt:
        raise APIError("input.prompt must not contain NUL")
    prompt = prompt[:MAX_PROMPT_CHARS]

    raw_media = prompt_input.get("media", [])
    if raw_media is None:
        raw_media = []
    if not isinstance(raw_media, list):
        raise APIError("input.media must be an array")
    media = tuple(_parse_media_item(item, index) for index, item in enumerate(raw_media))
    _validate_media_roles(media)
    if any(item.role in REFERENCE_ROLES for item in media):
        raise APIError(
            "FastH3 Trim supports first_frame and last_frame; reference_image, "
            "reference_video, and reference_audio require a Ref2VA checkpoint", )

    parameters = data.get("parameters", {})
    if parameters is None:
        parameters = {}
    parameters = _as_object(parameters, "parameters")

    if "duration" not in parameters:
        raise APIError("parameters.duration is required")
    duration = _as_int(parameters.get("duration"), "parameters.duration")
    if duration < MIN_DURATION or duration > MAX_DURATION:
        raise APIError(f"parameters.duration must be an integer from {MIN_DURATION} to {MAX_DURATION}")

    resolution = parameters.get("resolution", "768P")
    if not isinstance(resolution, str) or resolution.strip().upper() != "768P":
        raise APIError("parameters.resolution must be 768P")

    ratio_value = parameters.get("ratio", "adaptive")
    if ratio_value is None:
        ratio_value = "adaptive"
    if not isinstance(ratio_value, str):
        raise APIError("parameters.ratio must be a string")
    ratio = ratio_value.strip().lower()
    if ratio == "":
        ratio = "adaptive"
    named_canvas = None if ratio == "adaptive" else canvas_for_ratio(ratio)

    preset = _preset(parameters.get("preset", "default"))
    steps = _steps(parameters)
    frames = _optional_bounded_int(parameters, "frames", 5, MAX_ALIGNED_FRAMES)
    num_frames = _resolve_frames(duration, frames)
    width, height = _dimension_pair(parameters, "width", "height")
    render_width, render_height = _dimension_pair(parameters, "render_width", "render_height")
    if width is None and named_canvas is not None:
        width, height = named_canvas
    if render_width is not None and width is None:
        raise APIError("parameters.render_width requires parameters.width and parameters.height")
    if width is not None and height is not None:
        validate_canvas(width, height, render_width, render_height)

    seed = _optional_seed(parameters.get("seed")) if "seed" in parameters else None
    return JobSpec(
        prompt=prompt,
        duration=duration,
        ratio=ratio,
        preset=preset,
        width=width,
        height=height,
        num_frames=num_frames,
        steps=steps,
        media=media,
        seed=seed,
    )


def _accepted_model(model: str) -> bool:
    name = model.strip()
    if name in (CLIENT_MODEL_NAME, "MiniMax-H3"):
        return True
    return name.startswith("FastVideo/FastVideo-FastH3-Trim")


def _resolve_frames(duration: int, frames: int | None) -> int:
    if frames is None:
        return frames_for_duration(duration)
    aligned = align_frame_count(frames)
    if aligned < MIN_ALIGNED_FRAMES or aligned > MAX_ALIGNED_FRAMES:
        raise APIError(
            "parameters.frames must align within the FastH3 Trim range "
            f"{MIN_ALIGNED_FRAMES}..{MAX_ALIGNED_FRAMES} (5 + 17*n at 24 fps)", )
    return aligned


def _steps(parameters: dict) -> int | None:
    if "steps" not in parameters or parameters["steps"] is None:
        return None
    value = _as_int(parameters["steps"], "parameters.steps")
    if value in (8, TRIM_GRID_POINTS):
        return TRIM_GRID_POINTS
    raise APIError("parameters.steps must be 9 sigma-grid points (8 DMD forwards) for FastH3 Trim")


def _preset(value: object) -> str:
    if value is None:
        return "default"
    if not isinstance(value, str):
        raise APIError("parameters.preset must be fast or default")
    preset = value.strip().lower()
    if preset == "":
        return "default"
    if preset not in ("fast", "default"):
        raise APIError("parameters.preset must be fast or default")
    return preset


def _optional_bounded_int(parameters: dict, name: str, low: int, high: int) -> int | None:
    if name not in parameters or parameters[name] is None:
        return None
    value = _as_int(parameters[name], f"parameters.{name}")
    if value < low or value > high:
        raise APIError(f"parameters.{name} must be in [{low}, {high}]")
    return value


def _dimension_pair(parameters: dict, width_name: str, height_name: str) -> tuple[int | None, int | None]:
    present = [name for name in (width_name, height_name) if name in parameters and parameters[name] is not None]
    if not present:
        return None, None
    if len(present) != 2:
        raise APIError(f"parameters.{width_name} and parameters.{height_name} must be set together")
    width = _as_int(parameters[width_name], f"parameters.{width_name}")
    height = _as_int(parameters[height_name], f"parameters.{height_name}")
    return width, height


def _parse_media_item(item: object, index: int) -> MediaItem:
    entry = _as_object(item, f"input.media[{index}]")
    raw_type = entry.get("type")
    url = entry.get("url")
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise APIError(f"input.media[{index}].type is required")
    role = ROLE_ALIASES.get(raw_type.strip())
    if role is None:
        known = "first_frame, last_frame, reference_image, reference_video, reference_audio"
        raise APIError(f"input.media[{index}].type must be one of {known} "
                       "(aliases: image_url, feature, driving_audio)")
    if not isinstance(url, str) or not url.strip():
        raise APIError(f"input.media[{index}].url is required")
    if any(char in url for char in "\r\n\x00"):
        raise APIError(f"input.media[{index}].url is invalid")
    return MediaItem(role=role, url=url.strip())


def _validate_media_roles(media: tuple[MediaItem, ...]) -> None:
    counts = {role: 0 for role in (*FRAME_ROLES, *REFERENCE_ROLES)}
    for item in media:
        counts[item.role] += 1
    frames = counts["first_frame"] + counts["last_frame"]
    references = counts["reference_image"] + counts["reference_video"] + counts["reference_audio"]
    if frames and references:
        raise APIError("cannot mix first_frame/last_frame with reference_image, "
                       "reference_video, or reference_audio in one request")
    if counts["first_frame"] > 1:
        raise APIError("at most one first_frame is supported")
    if counts["last_frame"] > 1:
        raise APIError("at most one last_frame is supported")
    if counts["reference_image"] > MAX_REFERENCE_IMAGES:
        raise APIError(f"at most {MAX_REFERENCE_IMAGES} reference_image inputs are supported")
    if counts["reference_video"] > MAX_REFERENCE_VIDEOS:
        raise APIError(f"at most {MAX_REFERENCE_VIDEOS} reference_video inputs are supported")
    if counts["reference_audio"] > MAX_REFERENCE_AUDIO:
        raise APIError(f"at most {MAX_REFERENCE_AUDIO} reference_audio inputs are supported")
    if references > MAX_REFERENCES:
        raise APIError(f"at most {MAX_REFERENCES} reference inputs are supported")
    if counts["reference_audio"] and not (counts["reference_image"] or counts["reference_video"]):
        raise APIError("reference_audio requires a reference_image or reference_video")


def _as_object(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise APIError(f"{name} must be an object")
    return value


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise APIError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("+", "-")):
            digits = text[1:]
            sign = text[0]
        else:
            digits = text
            sign = ""
        if digits.isdigit():
            return int(sign + digits)
    raise APIError(f"{name} must be an integer")


def _optional_seed(value: object) -> int | None:
    if value is None:
        return None
    seed = _as_int(value, "parameters.seed")
    if seed < 0:
        raise APIError("parameters.seed must be >= 0")
    return seed
