from __future__ import annotations

from dataclasses import dataclass
import logging
import random
import threading
import time

import torch
from diffusers import DiffusionPipeline
from PIL import Image

from .config import Settings
from .outpaint_helpers import (
    Direction,
    composite,
    plan_directional_canvas,
    prepare_source,
)

logger = logging.getLogger(__name__)


def _patch_qwen3_vl_rope_config() -> None:
    """Handle Krea's null Qwen3-VL RoPE config with Transformers 4.57.x.

    The Krea checkpoint's text config has ``rope_scaling: null``.  Transformers
    4.57 ships Qwen3-VL support but later assumes this value is a mapping while
    constructing the rotary embedding.  The model's default MRoPE layout is the
    same fallback used by Transformers when the field is omitted.
    """
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
        )
    except ImportError:
        return

    original_init = Qwen3VLTextRotaryEmbedding.__init__
    if getattr(original_init, "_krea_null_rope_patch", False):
        return

    def compatible_init(self, config, *args, **kwargs):
        if getattr(config, "rope_scaling", None) is None:
            config.rope_scaling = {
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
            }
        original_init(self, config, *args, **kwargs)

    compatible_init._krea_null_rope_patch = True
    Qwen3VLTextRotaryEmbedding.__init__ = compatible_init


@dataclass(frozen=True)
class GenerationResult:
    image: Image.Image
    seed: int
    direction: Direction
    canvas_size: tuple[int, int]
    bbox: tuple[int, int, int, int]
    actual_expand: int
    source_was_resized: bool
    elapsed_seconds: float


class OutpaintEngine:
    """Lazy-loaded, single-GPU inference engine.

    A process-wide lock serializes model calls. This is intentional: the pipeline is
    large, and overlapping requests usually causes OOM rather than useful throughput.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe: DiffusionPipeline | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def _resolve_dtype(self) -> torch.dtype:
        requested = self.settings.torch_dtype.lower()
        if requested == "float16":
            return torch.float16
        if requested == "bfloat16":
            return torch.bfloat16
        if requested == "float32":
            return torch.float32

        if self.settings.device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def load(self) -> None:
        if self._pipe is not None:
            return
        with self._load_lock:
            if self._pipe is not None:
                return

            if self.settings.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available inside the container. Check NVIDIA Container "
                    "Toolkit and run Docker with GPU access."
                )
            if not self.settings.hf_token:
                logger.warning(
                    "HF_TOKEN is not set. Krea-2-Turbo is gated, so model loading will "
                    "normally fail unless the cache is already populated."
                )

            dtype = self._resolve_dtype()
            logger.info(
                "Loading %s with custom pipeline %s at revision %s (%s)",
                self.settings.base_model,
                self.settings.outpaint_repo,
                self.settings.outpaint_revision,
                dtype,
            )

            _patch_qwen3_vl_rope_config()

            pipe = DiffusionPipeline.from_pretrained(
                self.settings.base_model,
                custom_pipeline=self.settings.outpaint_repo,
                custom_revision=self.settings.outpaint_revision,
                revision="main",
                trust_remote_code=True,
                torch_dtype=dtype,
                token=self.settings.hf_token,
                cache_dir=str(self.settings.hf_home),
                low_cpu_mem_usage=True,
            )
            pipe.load_lora_weights(
                self.settings.outpaint_repo,
                weight_name=self.settings.weight_name,
                adapter_name="outpaint",
                token=self.settings.hf_token,
                revision=self.settings.outpaint_revision,
            )
            pipe.set_adapters(["outpaint"], weights=[1.0])

            if self.settings.sequential_cpu_offload:
                pipe.enable_sequential_cpu_offload()
            elif self.settings.cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.settings.device)

            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass

            self._pipe = pipe
            logger.info("Model loaded")

    def outpaint(
        self,
        image: Image.Image,
        prompt: str,
        *,
        direction: Direction | str = Direction.right,
        expand_pixels: int = 256,
        steps: int = 8,
        seed: int | None = None,
    ) -> GenerationResult:
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")
        if not 1 <= int(steps) <= 50:
            raise ValueError("steps must be between 1 and 50")

        self.load()
        assert self._pipe is not None

        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        plan = plan_directional_canvas(
            image,
            direction=direction,
            expand_pixels=expand_pixels,
            max_canvas=self.settings.max_canvas,
        )
        prepared = prepare_source(
            plan.source,
            plan.canvas_size,
            plan.bbox,
            source_max_edge=self.settings.source_max_edge,
            seam_px=self.settings.seam_px,
        )

        start = time.perf_counter()
        with self._inference_lock, torch.inference_mode():
            generator = torch.Generator(device=self.settings.device).manual_seed(seed)
            generated = self._pipe(
                prompt=prompt.strip(),
                image=prepared.condition,
                width=plan.canvas_size[0],
                height=plan.canvas_size[1],
                num_inference_steps=int(steps),
                guidance_scale=0.0,
                generator=generator,
                reference_max_pixels=self.settings.source_max_edge
                * self.settings.source_max_edge,
                reference_placements=[{"bbox_normalized": prepared.bbox_normalized}],
                encode_reference_in_prompt=False,
                kv_cache=True,
            ).images[0]
        result = composite(generated, prepared)
        elapsed = time.perf_counter() - start

        return GenerationResult(
            image=result,
            seed=seed,
            direction=plan.direction,
            canvas_size=plan.canvas_size,
            bbox=plan.bbox,
            actual_expand=plan.actual_expand,
            source_was_resized=plan.source_was_resized,
            elapsed_seconds=elapsed,
        )
