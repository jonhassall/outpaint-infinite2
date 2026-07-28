from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import random
import threading
import time
import uuid

from PIL import Image

from .config import Settings
from .engine import OutpaintEngine
from .outpaint_helpers import Direction

TERMINAL_STATES = {"completed", "stopped", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ContinuousJob:
    id: str
    prompt: str
    directions: list[Direction]
    expand_pixels: int
    steps: int
    max_steps: int
    delay_seconds: float
    randomize_seed: bool
    seed: int | None
    output_dir: Path

    status: str = "queued"
    current_step: int = 0
    latest_path: Path | None = None
    latest_url: str | None = None
    latest_seed: int | None = None
    latest_direction: str | None = None
    latest_canvas_size: list[int] | None = None
    latest_source_was_resized: bool | None = None
    latest_elapsed_seconds: float | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    version: int = 0

    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def snapshot(self) -> dict:
        with self.condition:
            return {
                "id": self.id,
                "status": self.status,
                "current_step": self.current_step,
                "max_steps": self.max_steps,
                "latest_url": self.latest_url,
                "latest_seed": self.latest_seed,
                "latest_direction": self.latest_direction,
                "latest_canvas_size": self.latest_canvas_size,
                "latest_source_was_resized": self.latest_source_was_resized,
                "latest_elapsed_seconds": self.latest_elapsed_seconds,
                "error": self.error,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "version": self.version,
            }

    def update(self, **values) -> None:
        with self.condition:
            for key, value in values.items():
                setattr(self, key, value)
            self.updated_at = _now()
            self.version += 1
            self.condition.notify_all()


class JobManager:
    def __init__(self, settings: Settings, engine: OutpaintEngine):
        self.settings = settings
        self.engine = engine
        self._jobs: dict[str, ContinuousJob] = {}
        self._lock = threading.Lock()

    def create(
        self,
        source: Image.Image,
        *,
        prompt: str,
        directions: list[Direction],
        expand_pixels: int,
        steps: int,
        max_steps: int,
        delay_seconds: float,
        randomize_seed: bool,
        seed: int | None,
    ) -> ContinuousJob:
        if not directions:
            raise ValueError("at least one direction is required")
        if max_steps < 0:
            raise ValueError("max_steps must be 0 (unlimited) or a positive integer")
        if delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative")

        job_id = uuid.uuid4().hex
        job_dir = self.settings.output_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        initial_path = job_dir / "step_0000_source.png"
        source.convert("RGB").save(initial_path, format="PNG")

        job = ContinuousJob(
            id=job_id,
            prompt=prompt,
            directions=directions,
            expand_pixels=expand_pixels,
            steps=steps,
            max_steps=max_steps,
            delay_seconds=delay_seconds,
            randomize_seed=randomize_seed,
            seed=seed,
            output_dir=job_dir,
            latest_path=initial_path,
            latest_url=f"/outputs/jobs/{job_id}/{initial_path.name}",
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run,
            args=(job, source.convert("RGB")),
            name=f"outpaint-job-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: ContinuousJob, current: Image.Image) -> None:
        job.update(status="running")
        step = 0
        try:
            while not job.stop_event.is_set():
                if job.max_steps and step >= job.max_steps:
                    job.update(status="completed")
                    return

                step += 1
                direction = random.choice(job.directions)
                if job.randomize_seed or job.seed is None or job.seed < 0:
                    step_seed = random.randint(0, 2**31 - 1)
                else:
                    step_seed = int(job.seed) + step - 1

                result = self.engine.outpaint(
                    current,
                    job.prompt,
                    direction=direction,
                    expand_pixels=job.expand_pixels,
                    steps=job.steps,
                    seed=step_seed,
                )
                filename = (
                    f"step_{step:04d}_{result.direction.value}_seed_{result.seed}.png"
                )
                path = job.output_dir / filename
                result.image.save(path, format="PNG")
                current = result.image

                job.update(
                    current_step=step,
                    latest_path=path,
                    latest_url=f"/outputs/jobs/{job.id}/{filename}",
                    latest_seed=result.seed,
                    latest_direction=result.direction.value,
                    latest_canvas_size=list(result.canvas_size),
                    latest_source_was_resized=result.source_was_resized,
                    latest_elapsed_seconds=round(result.elapsed_seconds, 3),
                )

                if job.delay_seconds > 0 and job.stop_event.wait(job.delay_seconds):
                    break

            job.update(status="stopped")
        except Exception as exc:  # keep the full message visible to API/UI
            job.update(status="failed", error=f"{type(exc).__name__}: {exc}")

    def get(self, job_id: str) -> ContinuousJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stop(self, job_id: str) -> ContinuousJob:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        job.stop_event.set()
        if job.status not in TERMINAL_STATES:
            job.update(status="stopping")
        return job

    def snapshot(self, job_id: str) -> dict:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job.snapshot()

    def wait_for_update(
        self, job_id: str, previous_version: int, timeout: float = 15.0
    ) -> dict | None:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        with job.condition:
            if job.version <= previous_version and job.status not in TERMINAL_STATES:
                job.condition.wait(timeout=timeout)
            if job.version > previous_version or job.status in TERMINAL_STATES:
                return job.snapshot()
            return None
