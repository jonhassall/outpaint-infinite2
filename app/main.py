from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

from .api import build_router
from .config import settings
from .engine import OutpaintEngine
from .jobs import JobManager
from .ui import build_ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

engine = OutpaintEngine(settings)
jobs = JobManager(settings, engine)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.preload_model:
        try:
            await asyncio.to_thread(engine.load)
        except Exception:
            logger.exception("Model preload failed; the API remains up for diagnostics/retry")
    yield


app = FastAPI(
    title="Krea 2 Outpaint API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def optional_api_key(request: Request, call_next):
    if (
        settings.api_key
        and request.url.path.startswith("/api/v1")
        and not request.url.path.startswith("/api/v1/health/")
    ):
        supplied = request.headers.get("x-api-key")
        if supplied != settings.api_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")


app.include_router(build_router(settings, engine, jobs))
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")

ui_auth = None
if settings.ui_username and settings.ui_password:
    ui_auth = (settings.ui_username, settings.ui_password)

app = gr.mount_gradio_app(
    app,
    build_ui(engine, jobs),
    path="/ui",
    auth=ui_auth,
    max_file_size=f"{settings.max_upload_mb}mb",
    allowed_paths=[str(settings.output_dir)],
    show_error=True,
    footer_links=["api", "gradio"],
)
