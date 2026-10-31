# SPDX-License-Identifier: Apache-2.0
"""FastH3 Trim backend for the local DashScope server.

Weights load only when :meth:`FastVideoTrimBackend.load` runs. Importing this
module does not import FastVideo, so the HTTP tests can fake the backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_api.mapping import (
    DEFAULT_MODEL_PATH,
    TRIM_GRID_POINTS,
    TRIM_VSA_SPARSITY,
    TRIM_VSA_TILE_SIZE,
    GenerationError,
    PreparedGeneration,
    infer_quant,
)


def build_generator_config(
    *,
    model_path: str,
    quant: str,
    num_gpus: int,
    vsa_sparsity: float,
    offload: bool,
) -> Any:
    """Build the typed config for one resident FastH3 Trim generator.

    The field set follows ``examples/serving/openai_fasth3_trim_spark.yaml``:
    VSA-H3, sparsity 0.8, 64-token tiles, and the light H3 VAE. NVFP4 and FP8
    are selected here; the eight-forward ladder is a per-request step count.
    """
    from fastvideo.api.schema import (
        CompileConfig,
        EngineConfig,
        GeneratorConfig,
        OffloadConfig,
        ParallelismConfig,
        PipelineSelection,
        QuantizationConfig,
    )

    resolved = infer_quant(model_path, quant)
    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")
    if not 0.0 <= vsa_sparsity < 1.0:
        raise ValueError("vsa_sparsity must be in [0, 1)")

    quantization = None
    if resolved == "nvfp4":
        quantization = QuantizationConfig(transformer_quant="NVFP4", layer_profile="h3_dit_vsa")
    elif resolved == "fp8":
        quantization = QuantizationConfig(transformer_quant="FP8")

    if offload:
        offload_config = OffloadConfig(
            dit=False,
            dit_layerwise=False,
            text_encoder=True,
            image_encoder=False,
            vae=True,
            pin_cpu_memory=True,
            lazy_module_load=None,
        )
    else:
        offload_config = OffloadConfig(
            dit=False,
            dit_layerwise=False,
            text_encoder=False,
            image_encoder=False,
            vae=False,
            pin_cpu_memory=False,
            lazy_module_load=False,
        )

    return GeneratorConfig(
        model_path=model_path,
        engine=EngineConfig(
            num_gpus=num_gpus,
            execution_backend="mp",
            use_fsdp_inference=False,
            quantization=quantization,
            parallelism=ParallelismConfig(tp_size=1, sp_size=num_gpus),
            offload=offload_config,
            compile=CompileConfig(enabled=False, vae_enabled=False),
        ),
        pipeline=PipelineSelection(
            workload_type="t2v",
            vae_tiling=True,
            experimental={
                "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                "VSA_sparsity": vsa_sparsity,
                "VSA_tile_size": TRIM_VSA_TILE_SIZE,
                "h3_sequential_load": False,
                "inference_torch_compile": False,
                "vae_parallel_decode": False,
                "video_decode_backend": "h3-vae",
            },
        ),
    )


def generation_request_raw(prepared: PreparedGeneration) -> dict[str, Any]:
    """Return the explicit GenerationRequest mapping for one Trim clip.

    Text-only requests use the T2VA path. ``first_frame`` is ``image_path``
    and ``last_frame`` is a PIL image on ``last_image``. Both go through
    ``MiniMaxH3ModularPipeline``, which already implements FL2VA keyframes.
    The empty negative prompt and ``guidance_scale`` of 1 are required:
    Trim is guidance-distilled.
    """
    if prepared.steps != TRIM_GRID_POINTS:
        raise GenerationError(
            f"FastH3 Trim requires {TRIM_GRID_POINTS} sigma-grid points, got {prepared.steps}",
            "InvalidParameter",
        )

    raw: dict[str, Any] = {
        "prompt": prepared.prompt,
        "negative_prompt": "",
        "sampling": {
            "seed": prepared.seed,
            "height": prepared.height,
            "width": prepared.width,
            "num_frames": prepared.num_frames,
            "fps": 24,
            "num_inference_steps": prepared.steps,
            "num_videos_per_prompt": 1,
            "guidance_scale": 1.0,
            "batch_cfg": False,
        },
        "output": {
            "output_path": str(prepared.output_path),
            "save_video": True,
            "return_frames": False,
        },
    }
    inputs: dict[str, Any] = {}
    if prepared.first_frame is not None:
        inputs["image_path"] = str(prepared.first_frame)
    if prepared.last_frame is not None:
        inputs["last_image"] = _load_pil(prepared.last_frame)
    if inputs:
        raw["inputs"] = inputs
    return raw


def build_generation_request(prepared: PreparedGeneration) -> Any:
    """Parse :func:`generation_request_raw` into a tracked GenerationRequest."""
    from fastvideo.api.parser import parse_config
    from fastvideo.api.schema import GenerationRequest

    return parse_config(GenerationRequest, generation_request_raw(prepared))


def _load_pil(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.copy()


class FastVideoTrimBackend:
    """One long-lived VideoGenerator. The HTTP worker calls it serially."""

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_MODEL_PATH,
        quant: str = "auto",
        num_gpus: int = 1,
        vsa_sparsity: float = TRIM_VSA_SPARSITY,
        offload: bool = False,
    ) -> None:
        self.model_path = model_path
        self.quant = infer_quant(model_path, quant)
        self.num_gpus = num_gpus
        self.vsa_sparsity = vsa_sparsity
        self.offload = offload
        self._generator: Any = None

    def load(self) -> None:
        if self._generator is not None:
            return
        _require_model_path(self.model_path)
        from fastvideo.entrypoints.video_generator import VideoGenerator

        config = build_generator_config(
            model_path=self.model_path,
            quant=self.quant,
            num_gpus=self.num_gpus,
            vsa_sparsity=self.vsa_sparsity,
            offload=self.offload,
        )
        self._generator = VideoGenerator.from_config(config)

    def generate(self, prepared: PreparedGeneration) -> Path:
        if self._generator is None:
            self.load()
        request = build_generation_request(prepared)
        try:
            result = self._generator.generate(request)
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - the HTTP task should fail closed
            raise GenerationError(f"FastH3 Trim generation failed: {exc}", "InternalError") from exc
        return _video_path(result, prepared.output_path)


def _require_model_path(model: str) -> None:
    path = Path(model).expanduser()
    if path.exists():
        if not path.is_dir():
            raise SystemExit(f"model path is not a directory: {path}")
        return
    looks_local = model.startswith(("/", ".", "~")) or path.is_absolute()
    if looks_local:
        raise SystemExit(f"model directory not found: {path}")


def _video_path(result: Any, fallback: Path) -> Path:
    if isinstance(result, list):
        if not result:
            raise GenerationError("FastH3 Trim returned no video", "InternalError")
        result = result[0]
    video_path = getattr(result, "video_path", None)
    path = Path(video_path) if video_path else fallback
    if not path.is_file() or path.stat().st_size <= 0:
        raise GenerationError("FastH3 Trim exited without writing a video", "InternalError")
    return path
