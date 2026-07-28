# Krea 2 Outpaint — Docker, REST API, Gradio UI, continuous mode

A self-hosted GPU service around `yijunwang2/krea2-outpaint` and `krea/Krea-2-Turbo`.

It provides:

- a Gradio web UI at `/ui`;
- a FastAPI REST API and OpenAPI docs at `/docs`;
- one-shot directional outpainting;
- continuous recursive outpainting with live Server-Sent Events (SSE);
- automatic saving of every completed frame;
- Node.js and PHP client examples;
- one model instance and a process-wide GPU lock to avoid overlapping calls that commonly cause OOM errors.

## Important behavior

Continuous mode chooses from `left`, `right`, `up`, and `down`. After each generation, the completed image is saved and becomes the next source.

The model's canvas is intentionally capped by `MAX_CANVAS` (default `1280`). When a previous result is too large to add the requested new strip, it is resized down just enough to leave room. This permits a long-running loop, but it means older content gradually becomes smaller. Set a finite `max_steps` unless that rolling zoom-out behavior is what you want.

## Requirements

- Linux host with an NVIDIA GPU and current driver.
- Docker Engine and Docker Compose.
- NVIDIA Container Toolkit configured for Docker.
- A Hugging Face account that has accepted access to the gated `krea/Krea-2-Turbo` model.
- A Hugging Face read token in `HF_TOKEN`.
- Substantial disk space for the model cache and generated PNG files.

The base model is large. Full BF16 placement normally needs a high-memory GPU; lower-memory cards may require `CPU_OFFLOAD=1` or `SEQUENTIAL_CPU_OFFLOAD=1`, plus ample system RAM, and will run more slowly.

## Start

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN.

docker compose up --build
```

Open:

- UI: `http://localhost:8000/ui`
- REST docs: `http://localhost:8000/docs`
- health: `http://localhost:8000/api/v1/health/live`

The first start downloads the gated base model and the outpaint adapter into the named `hf-cache` Docker volume.

## Verify Docker GPU access

Before building the service, this command should show your GPU:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

## One-shot REST request

```bash
curl -X POST http://localhost:8000/api/v1/outpaint \
  -F image=@input.png \
  -F 'prompt=a vast cinematic desert landscape, consistent lighting, complete composition' \
  -F direction=right \
  -F expand_pixels=256 \
  -F steps=8 \
  -F seed=-1
```

Example response:

```json
{
  "image_url": "/outputs/single/....png",
  "seed": 123456,
  "direction": "right",
  "canvas_size": [1280, 768],
  "bbox": [0, 0, 1024, 768],
  "actual_expand": 256,
  "source_was_resized": false,
  "elapsed_seconds": 9.41
}
```

## Start a continuous job

```bash
curl -X POST http://localhost:8000/api/v1/jobs/continuous \
  -F image=@input.png \
  -F 'prompt=a surreal endless landscape, coherent perspective and lighting, complete composition' \
  -F directions=left,right,up,down \
  -F expand_pixels=256 \
  -F steps=8 \
  -F max_steps=20 \
  -F delay_seconds=0 \
  -F randomize_seed=true \
  -F seed=42
```

The response includes:

- `status_url`
- `events_url`
- `stop_url`

Stream live updates:

```bash
curl -N http://localhost:8000/api/v1/jobs/JOB_ID/events
```

Stop the loop:

```bash
curl -X POST http://localhost:8000/api/v1/jobs/JOB_ID/stop
```

Set `max_steps=0` for an unlimited job. Every image is retained under `data/outputs/jobs/JOB_ID`, so unlimited jobs can fill the disk.

## Node.js client

Requires Node.js 18 or newer.

```bash
IMAGE=input.png \
PROMPT='an endless alien valley, coherent full scene' \
MAX_STEPS=30 \
node clients/node/continuous.mjs
```

The client starts the job and parses the SSE stream.

## PHP client

Requires PHP with the cURL extension.

```bash
IMAGE=input.png \
PROMPT='an endless alien valley, coherent full scene' \
MAX_STEPS=30 \
php clients/php/continuous.php
```

The PHP example starts a job and polls its status.

## Authentication

Set an API key:

```dotenv
API_KEY=replace-with-a-long-random-value
```

Then include it on REST requests:

```bash
-H 'x-api-key: replace-with-a-long-random-value'
```

Set Gradio basic authentication independently:

```dotenv
UI_USERNAME=admin
UI_PASSWORD=replace-me
```

For internet exposure, put the container behind a TLS reverse proxy, restrict request size, and add rate limiting. The model license also requires deployment safeguards appropriate to the use case.

## GPU-memory controls

Default, fastest, and highest VRAM use:

```dotenv
CPU_OFFLOAD=0
SEQUENTIAL_CPU_OFFLOAD=0
```

Lower VRAM, slower:

```dotenv
CPU_OFFLOAD=1
SEQUENTIAL_CPU_OFFLOAD=0
```

Lowest VRAM, slowest:

```dotenv
CPU_OFFLOAD=0
SEQUENTIAL_CPU_OFFLOAD=1
```

Do not enable both offload modes.

Other useful settings:

```dotenv
MAX_CANVAS=1280
SOURCE_MAX_EDGE=384
SEAM_PX=32
TORCH_DTYPE=auto
```

`MAX_CANVAS` has the largest effect on speed and memory. Larger values can increase OOM risk sharply.

## Architecture

```text
Browser / Node / PHP
        |
        v
FastAPI :8000
  |-- /api/v1/outpaint
  |-- /api/v1/jobs/continuous
  |-- /api/v1/jobs/{id}/events   (SSE)
  |-- /outputs                   (saved frames)
  `-- /ui                        (mounted Gradio)
        |
        v
OutpaintEngine
  |-- Krea 2 Turbo
  |-- registered outpaint custom pipeline
  |-- rank-32 outpaint LoRA
  `-- single inference lock
```

The custom pipeline revision is pinned in `.env.example`. It is loaded with `trust_remote_code=True`, so inspect upstream changes before moving the pin.

## Production notes

- Run exactly one Uvicorn worker per GPU. Multiple workers load separate copies of the model.
- For multiple GPUs, run one container per GPU and route requests externally.
- A proper production queue such as Redis/RQ, Celery, or a database-backed worker is preferable if jobs must survive container restarts.
- Current in-memory job metadata is lost after restart, although generated files remain on disk.
- The service does not include a moderation model. Add prompt/output policy checks before public deployment.
- Back up or rotate `data/outputs`; continuous generation can produce a large number of PNGs.

## Files

```text
app/main.py               FastAPI app and mounted Gradio UI
app/api.py                REST and SSE endpoints
app/engine.py             model loading and inference serialization
app/jobs.py               continuous background jobs
app/outpaint_helpers.py   directional canvas planning and compositing
app/ui.py                 Gradio frontend
clients/node              Node SSE client
clients/php               PHP polling client
```

## Licensing

See `LICENSE_NOTICE.md`. This project downloads upstream model weights and custom pipeline code at runtime and does not bundle them.
