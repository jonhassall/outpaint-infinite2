from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from .config import Settings
from .engine import OutpaintEngine
from .jobs import JobManager, TERMINAL_STATES
from .outpaint_helpers import Direction


def _read_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


def _parse_directions(value: str) -> list[Direction]:
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parts:
        raise HTTPException(status_code=400, detail="At least one direction is required")
    try:
        return list(dict.fromkeys(Direction(part) for part in parts))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in Direction)
        raise HTTPException(
            status_code=400, detail=f"Invalid direction. Allowed: {allowed}"
        ) from exc


def build_router(
    settings: Settings, engine: OutpaintEngine, jobs: JobManager
) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/health/live")
    async def live() -> dict:
        return {"status": "ok"}

    @router.get("/health/ready")
    async def ready() -> dict:
        return {
            "status": "ready" if engine.loaded else "loading",
            "model_loaded": engine.loaded,
        }

    @router.post("/model/load")
    async def load_model() -> dict:
        try:
            await asyncio.to_thread(engine.load)
            return {"model_loaded": True}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/outpaint")
    async def outpaint(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        direction: Direction = Form(Direction.right),
        expand_pixels: int = Form(256),
        steps: int = Form(8),
        seed: int = Form(-1),
    ) -> dict:
        data = await image.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds upload limit")
        source = _read_image(data)
        try:
            result = await asyncio.to_thread(
                engine.outpaint,
                source,
                prompt,
                direction=direction,
                expand_pixels=expand_pixels,
                steps=steps,
                seed=seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        output_dir = settings.output_dir / "single"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}_{result.direction.value}_{result.seed}.png"
        path = output_dir / filename
        result.image.save(path, format="PNG")
        return {
            "image_url": f"/outputs/single/{filename}",
            "seed": result.seed,
            "direction": result.direction.value,
            "canvas_size": list(result.canvas_size),
            "bbox": list(result.bbox),
            "actual_expand": result.actual_expand,
            "source_was_resized": result.source_was_resized,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
        }

    @router.post("/jobs/continuous")
    async def start_continuous(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        directions: str = Form("left,right,up,down"),
        expand_pixels: int = Form(256),
        steps: int = Form(8),
        max_steps: int = Form(20),
        delay_seconds: float = Form(0.0),
        randomize_seed: bool = Form(True),
        seed: int = Form(-1),
    ) -> dict:
        data = await image.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds upload limit")
        source = _read_image(data)
        parsed_directions = _parse_directions(directions)
        try:
            job = jobs.create(
                source,
                prompt=prompt,
                directions=parsed_directions,
                expand_pixels=expand_pixels,
                steps=steps,
                max_steps=max_steps,
                delay_seconds=delay_seconds,
                randomize_seed=randomize_seed,
                seed=seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **job.snapshot(),
            "status_url": f"/api/v1/jobs/{job.id}",
            "events_url": f"/api/v1/jobs/{job.id}/events",
            "stop_url": f"/api/v1/jobs/{job.id}/stop",
        }

    @router.get("/jobs/{job_id}")
    async def job_status(job_id: str) -> dict:
        try:
            return jobs.snapshot(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @router.post("/jobs/{job_id}/stop")
    async def stop_job(job_id: str) -> dict:
        try:
            return jobs.stop(job_id).snapshot()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @router.get("/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")

        async def stream():
            version = -1
            while True:
                try:
                    snapshot = await asyncio.to_thread(
                        jobs.wait_for_update, job_id, version, 15.0
                    )
                except KeyError:
                    yield "event: error\ndata: {\"error\":\"job not found\"}\n\n"
                    return

                if snapshot is None:
                    yield ": keep-alive\n\n"
                    continue

                version = snapshot["version"]
                yield f"event: update\ndata: {json.dumps(snapshot)}\n\n"
                if snapshot["status"] in TERMINAL_STATES:
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return router
