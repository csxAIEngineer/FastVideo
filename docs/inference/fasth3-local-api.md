# Local FastH3 Trim API

`python -m local_api` is a loopback HTTP server that speaks the DashScope-style
async API AiVideo already uses for `MiniMax/MiniMax-H3`. Point the video
client at this process and leave the request body unchanged. No Qianwen or
MiniMax cloud key is required, and the server does not call the C `h3`
binary. Generation goes through FastVideo's `VideoGenerator` and the FastH3
Trim checkpoint.

The default clip is Trim's released recipe: **832×480**, **124 frames** at
24 fps (about 5.17 seconds) with the model's stereo audio, **9 sigma-grid
points** (8 DMD forwards), and **VSA sparsity 0.8**.

## Start

From the repository root, after FastVideo is installed:

```sh
python -m local_api --model FastVideo/FastVideo-FastH3-Trim-8-Step
```

The default listen address is `http://127.0.0.1:8765`. The same entry point
is `scripts/fasth3-local-api` and, after an editable install, the
`fasth3-local-api` command.

NVFP4 Trim (the Spark serving checkpoint) is selected by the repo id.
`--quant auto` reads `NVFP4` or `FP8` out of the model id; pass `--quant`
when the path does not contain that token.

```sh
python -m local_api \
  --model FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4 \
  --quant nvfp4
```

`--quant fp8` sets `transformer_quant: FP8` for a local FP8 export. The BF16
repo above uses no transformer quant.

AiVideo: set the video client's base URL to `http://127.0.0.1:8765` (no path
suffix, no API key). The client already posts to
`/api/v1/services/aigc/video-generation/video-synthesis` and polls
`/api/v1/tasks/{task_id}`.

`Authorization` is optional. A missing header or any `Bearer` value is
accepted and ignored. `X-DashScope-Async: enable` is accepted and not
required. One generation runs at a time. Further submits stay `PENDING`
until the current request finishes. Outputs stay under the output directory
until you delete them.

The checkpoint loads through `VideoGenerator.from_config` before the process
prints its listen line. Startup does not run a warmup generation. A Hugging
Face repo id is downloaded by FastVideo on that load; a local directory is
used as-is.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `FastVideo/FastVideo-FastH3-Trim-8-Step` | Hub repo id or a local checkpoint directory. `FASTH3_MODEL` overrides the default. |
| `--quant` | `auto` | `auto`, `none`, `nvfp4`, or `fp8`. |
| `--host` / `--port` | `127.0.0.1:8765` | Bind address. |
| `--output-dir` | `./outputs/local-api` | Task files and mp4s. |
| `--seed` | `1234` | Used when the request omits `parameters.seed`. |
| `--num-gpus` | `1` | One process group. Sequence parallel matches this count; tensor parallel stays 1. |
| `--vsa-sparsity` | `0.8` | Trim's trained VSA sparsity. Tile size stays 64. |
| `--offload` | off | Offload the text encoder and VAE between stages. |

Optional Spark performance variables (`FASTVIDEO_VSA_TRITON`,
`FASTVIDEO_ATTENTION_BACKEND`, and the rest of the Trim recipe in
`docs/getting_started/installation/spark_performance.md`) are read by
FastVideo, not by this server. Export them in the environment before
starting the process when you want that recipe. The server's own engine
config already sets `VIDEO_SPARSE_ATTN_H3`, sparsity 0.8, and the light
`h3-vae` decoder, matching
`examples/serving/openai_fasth3_trim_spark.yaml`.

## Submit and poll

`POST /api/v1/services/aigc/video-generation/video-synthesis`

`POST /generate` is the same submit with a shorter path.

```sh
curl -sS -X POST http://127.0.0.1:8765/api/v1/services/aigc/video-generation/video-synthesis \
  -H 'Content-Type: application/json' \
  -H 'X-DashScope-Async: enable' \
  -d '{
    "model": "MiniMax/MiniMax-H3",
    "input": {
      "prompt": "A red fox walks through fresh snow in a pine forest."
    },
    "parameters": {
      "resolution": "768P",
      "ratio": "16:9",
      "duration": 5,
      "watermark": false
    }
  }'
```

The submit response is `PENDING`. Poll `GET /api/v1/tasks/{task_id}` until
`task_status` is `SUCCEEDED` or `FAILED`. Success includes an absolute
`output.video_url` on this server. Download that URL with a plain `GET`;
the body is `video/mp4` and, on the Trim path, contains the joint stereo
audio the pipeline muxes into the file. Failure stays HTTP 200 so a poller
can read `output.code` and `output.message`. An unknown task id is HTTP 404.

`model` may be `MiniMax/MiniMax-H3` (what AiVideo sends), `MiniMax-H3`, or
a `FastVideo/FastVideo-FastH3-Trim…` id. The loaded checkpoint is the
server `--model`, not the JSON field.

`parameters.resolution` must be `768P`. That token is the DashScope value
AiVideo already sends. It does **not** select h3.c's 768-short-edge canvas.
`parameters.watermark` is ignored. `parameters.seed` is optional.

## Canvas, duration, and preset

Named ratios use a 480 short side, snapped to a multiple of 32, with at
most `480 * 832` pixels. `adaptive` with no media is `16:9`. With a first
frame, otherwise a last frame, the same 480 rule follows that image.
h3.c instead uses a 768 short side (`h3_adapt_canvas`).

| Ratio | This server | h3.c 768P |
| --- | --- | --- |
| `16:9` | 832×480 | 1344×768 |
| `9:16` | 480×832 | 768×1344 |
| `1:1` | 480×480 | 768×768 |
| `4:3` | 640×480 | 1024×768 |
| `3:4` | 480×640 | 768×1024 |
| `21:9` | 960×416 | 1536×672 |

`duration` is an integer from 4 to 15. It becomes `duration * 24` frames,
rounded up to `5 + 17*n`, then clamped to FastVideo's 124–362 frame range.
`usage.duration` still reports the requested seconds. `usage.SR` is the
short side of the canvas that was actually rendered (`480` for the default
16:9), and `usage.size` is `width*height`.

| Seconds | Frames | File length at 24 fps |
| --- | --- | --- |
| 4 | 124 | about 5.17s (h3.c would use 107 frames) |
| 5 | 124 | about 5.17s |
| 6 | 158 | about 6.58s |
| 10 | 243 | about 10.13s |
| 14 | 345 | about 14.38s |
| 15 | 362 | about 15.08s |

`parameters.frames`, when set, replaces that second count. The value is
rounded up onto the same lattice and must land in 124–362.
`parameters.width` and `parameters.height` (both required) replace the
ratio canvas. They must be multiples of 32 and at most `768 * 1344`
pixels, so an explicit `1344×768` request still runs. There is no separate
internal render canvas: `render_width` and `render_height`, if present,
must equal the output size.

`parameters.preset` may be `fast` or `default`. Both run the Trim recipe.
`fast` does not switch to h3.c's 512×512 output, 320×320 render canvas, or
4 steps. An explicit size such as 256×256 is kept. `parameters.steps` may
be omitted, `8`, or `9`; `8` and `9` both mean nine sigma-grid points
(eight DMD forwards). Any other step count is `InvalidParameter`, because
the Trim ladder is fixed. `parameters.layers` and `parameters.reuse` are
ignored.

## Media

`input.media` is optional. Each item is `{"type", "url"}`. URLs may be
`http://`, `https://`, `file://`, or a local path. Remote objects are
downloaded and local files are copied into the task directory.

| `type` | Trim behavior |
| --- | --- |
| `first_frame` | At most one. Passed as `image_path` on the T2VA/FL2VA pipeline. |
| `last_frame` | At most one. May be used alone or with `first_frame`. Loaded as `last_image`. |
| `reference_image` (`image_url`) | Rejected. Ref2VA is a different checkpoint. |
| `reference_video` (`feature`) | Rejected. |
| `reference_audio` (`driving_audio`) | Rejected. Audio on Trim is generated with the video, not driven by a reference clip. |

Frame roles and reference roles cannot be combined. That mix is rejected
with HTTP 400, `code` `InvalidParameter`, and the same message as h3.c's
local API, and no task is created. A reference-only request is also HTTP
400, with a message that names the Ref2VA checkpoint.

## Tests

```sh
python -m unittest tests.test_local_api
pytest tests/test_local_api.py -q
```

The tests use a fake generator, so they do not need a GPU or a checkpoint.
They cover submit, poll, download, validation, the 832×480 Trim canvas,
duration-to-frame mapping, one-at-a-time execution, and first/last frame
paths. The Trim request mapping is checked without a GPU. Parsing that
mapping into a `GenerationRequest`, and building the NVFP4 generator
config, runs when FastVideo imports and is skipped otherwise.
