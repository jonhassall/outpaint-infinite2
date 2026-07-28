from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000)

    base_model: str = os.getenv("BASE_MODEL", "krea/Krea-2-Turbo")
    outpaint_repo: str = os.getenv("OUTPAINT_REPO", "yijunwang2/krea2-outpaint")
    outpaint_revision: str = os.getenv(
        "OUTPAINT_REVISION", "8e1be6098331d7650398c806f23f4163431cbe68"
    )
    weight_name: str = os.getenv("OUTPAINT_WEIGHT", "krea2_outpaint_rank32.safetensors")
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    hf_home: Path = Path(os.getenv("HF_HOME", "/models/huggingface"))

    device: str = os.getenv("DEVICE", "cuda")
    torch_dtype: str = os.getenv("TORCH_DTYPE", "auto")
    cpu_offload: bool = _bool("CPU_OFFLOAD", False)
    sequential_cpu_offload: bool = _bool("SEQUENTIAL_CPU_OFFLOAD", False)
    preload_model: bool = _bool("PRELOAD_MODEL", True)

    max_canvas: int = _int("MAX_CANVAS", 1280)
    source_max_edge: int = _int("SOURCE_MAX_EDGE", 384)
    seam_px: int = _int("SEAM_PX", 32)

    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "/data/outputs"))
    api_key: str | None = os.getenv("API_KEY") or None
    ui_username: str | None = os.getenv("UI_USERNAME") or None
    ui_password: str | None = os.getenv("UI_PASSWORD") or None
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 30)


settings = Settings()
settings.hf_home.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
